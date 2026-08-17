"""
scripts/generate_counterfactuals.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generates stochastic counterfactual reward samples for underrepresented state-action pairs.
"""
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from blackjack.engine.game import GameEngine
from blackjack.engine.outcomes import HandOutcome
from blackjack.rules.rules import BlackjackRules
from blackjack.cards.card import Card, Rank, Suit
from blackjack.cards.hand import Hand
from blackjack.strategies.base import Action
from blackjack.rules.legal_actions import LegalActionsCalculator
from ml.features.state import GameState
from ml.features.extractor import FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_player_cards(state_str: str) -> list[Card]:
    """Approximates the player's starting hand from the canonical state string."""
    # Examples: "Hard 19 vs T", "Soft 18 vs 2", "Pair 8,8 vs 6"
    parts = state_str.split(" vs ")[0]
    cards = []
    
    if parts.startswith("Pair"):
        rank_str = parts.replace("Pair ", "").split(",")[0]
        rank = Rank.ACE if rank_str == "11" else Rank(rank_str)
        cards = [Card(rank, Suit.HEARTS), Card(rank, Suit.DIAMONDS)]
        
    elif parts.startswith("Soft"):
        total = int(parts.replace("Soft ", ""))
        other = total - 11
        rank_str = "T" if other >= 10 else str(other)
        cards = [Card(Rank.ACE, Suit.HEARTS), Card(Rank(rank_str), Suit.DIAMONDS)]
        
    else: # Hard
        total = int(parts.replace("Hard ", ""))
        if total <= 10:
            cards = [Card(Rank(str(total - 2)), Suit.HEARTS), Card(Rank("2"), Suit.DIAMONDS)]
        elif total <= 19:
            val = total - 10
            rank_str = "2" if val < 2 else str(val) # Approximation
            cards = [Card(Rank("T"), Suit.HEARTS), Card(Rank(rank_str), Suit.DIAMONDS)]
        else:
            cards = [Card(Rank("T"), Suit.HEARTS), Card(Rank("9" if total == 19 else "T"), Suit.DIAMONDS)]
            
    return cards


def simulate_action(player_cards: list[Card], dealer_rank_str: str, action: Action, num_sims: int) -> list[float]:
    """Simulates a specific action N times from a cloned starting state."""
    rules = BlackjackRules.standard()
    dealer_upcard = Card(Rank(dealer_rank_str), Suit.SPADES)
    
    rewards = []
    
    from blackjack.cards.deck import Shoe
    from blackjack.strategies.random_strategy import RandomStrategy
    for _ in range(num_sims):
        # Fresh shoe for each stochastic rollout to prevent determinism overlap
        engine = GameEngine(shoe=Shoe(rules.num_decks), rules=rules, strategy=RandomStrategy())
        player_hand = Hand(player_cards.copy())
        
        # Override the engine's initial state
        # In a real counterfactual, we'd clone the shoe exactly. Here we approximate:
        engine.dealer_hand = Hand([dealer_upcard])
        engine.player_hands = [player_hand]
        
        # For Hit/Stand/Double/Surrender
        if action == Action.SURRENDER:
            rewards.append(-0.5)
            continue
            
        elif action == Action.STAND:
            from blackjack.engine.state import PlayerHand, HandContext
            ph = PlayerHand(hand=player_hand, context=HandContext(is_first_action=True))
            engine._play_dealer(engine.dealer_hand)
            outcome = engine._determine_outcome(ph, engine.dealer_hand)
            rewards.append(engine._payout.net(outcome, 1.0))
            
        elif action == Action.HIT:
            from blackjack.strategies.basic import BasicStrategy
            strat = BasicStrategy()
            engine._strategy = strat
            
            card = engine._shoe.deal()
            player_hand.add_card(card)
            
            # Recreate state for engine
            from blackjack.engine.state import PlayerHand, HandContext
            ph = PlayerHand(hand=player_hand, context=HandContext(is_first_action=False))
            player_hands = [ph]
            
            if player_hand.is_bust:
                engine._play_dealer(engine.dealer_hand)
                outcome = engine._determine_outcome(ph, engine.dealer_hand)
                rewards.append(engine._payout.net(outcome, 1.0))
            else:
                engine._play_player_hand(ph, dealer_upcard, player_hands, 0, 1.0)
                engine._play_dealer(engine.dealer_hand)
                total = sum(engine._payout.net(engine._determine_outcome(h, engine.dealer_hand), 1.0) for h in player_hands)
                rewards.append(total)
                
        elif action == Action.DOUBLE:
            card = engine._shoe.deal()
            player_hand.add_card(card)
            
            from blackjack.engine.state import PlayerHand, HandContext
            ph = PlayerHand(hand=player_hand, context=HandContext(is_first_action=False, doubled=True))
            
            if not player_hand.is_bust:
                engine._play_dealer(engine.dealer_hand)
                
            outcome = engine._determine_outcome(ph, engine.dealer_hand)
            rewards.append(engine._payout.net(outcome, bet=2.0))
            
        elif action == Action.SPLIT:
            from blackjack.strategies.basic import BasicStrategy
            engine._strategy = BasicStrategy()
            
            from blackjack.engine.state import PlayerHand, HandContext
            ph = PlayerHand(hand=player_hand, context=HandContext(is_first_action=True, split_count=0))
            player_hands = [ph]
            engine._execute_split(ph, player_hands, 0, 1.0)
            
            # Play out both hands
            engine._play_player_hand(player_hands[0], dealer_upcard, player_hands, 0, 1.0)
            if len(player_hands) > 1:
                engine._play_player_hand(player_hands[1], dealer_upcard, player_hands, 1, 1.0)
                
            engine._play_dealer(engine.dealer_hand)
            total = sum(engine._payout.net(engine._determine_outcome(h, engine.dealer_hand), 1.0) for h in player_hands)
            rewards.append(total)
            
    return rewards


def process_gap(state_str: str, missing_actions: list[str], sims_per_action: int = 1000):
    rules = BlackjackRules.standard()
    calc = LegalActionsCalculator(rules)
    
    player_cards = get_player_cards(state_str)
    dealer_rank_str = state_str.split(" vs ")[1]
    dealer_upcard = Card(Rank(dealer_rank_str), Suit.SPADES)
    
    # Generate canonical features for this state
    state_obj = GameState(
        player_ranks=[c.rank.value for c in player_cards],
        dealer_upcard_rank=dealer_upcard.rank.value,
        shoe_total_cards=6*52,
        shoe_cards_remaining=4*52,  # Approximation
        running_count=0,
        rules=rules.__dict__
    )
    features = FeatureExtractor.to_vector(state_obj)
    
    # Check true legality
    # Context mock:
    from collections import namedtuple
    Context = namedtuple('Context', ['is_first_action', 'split_count', 'from_split_aces'])
    ctx = Context(is_first_action=True, split_count=0, from_split_aces=False)
    legal = calc.get_legal_actions(Hand(player_cards), dealer_upcard, ctx) # type: ignore
    
    new_rows = []
    
    for act_str in missing_actions:
        act = Action(act_str)
        if act not in legal:
            continue
            
        rewards = simulate_action(player_cards, dealer_rank_str, act, sims_per_action)
        
        for r in rewards:
            new_rows.append({
                "trajectory_id": f"cf_{state_str}_{act_str}_{np.random.randint(10000)}",
                "state_id": f"cf_{state_str}",
                "player_total": Hand(player_cards).value,
                "dealer_upcard": dealer_rank_str,
                "is_soft": Hand(player_cards).is_soft,
                "is_pair": Hand(player_cards).is_pair,
                "action": act_str,
                "reward_sample": float(r),
                "features": features,
                "source": "counterfactual_approximation"
            })
            
    return new_rows


def main():
    coverage_path = Path("reports/phase11/action_coverage_before.parquet")
    if not coverage_path.exists():
        logging.error("Coverage matrix not found.")
        return 1
        
    df_cov = pd.read_parquet(coverage_path)
    
    gaps_to_process = []
    MIN_SAMPLES = 1000
    
    for state, row in df_cov.iterrows():
        # Skip 21s
        if "Hard 21" in state or "Hard 20" in state:
            continue
            
        missing = []
        for act in ["hit", "stand", "double", "split", "surrender"]:
            if row.get(act, 0) < MIN_SAMPLES:
                missing.append(act)
                
        if missing:
            gaps_to_process.append((state, missing))
            
    # Cap to 50 states for the pilot to avoid blowing up compute
    gaps_to_process = gaps_to_process[:50]
    
    logging.info(f"Generating counterfactuals for {len(gaps_to_process)} states (1000 sims per missing action)...")
    
    all_new_rows = []
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_gap, state, missing, 1000) for state, missing in gaps_to_process]
        for i, future in enumerate(futures):
            res = future.result()
            all_new_rows.extend(res)
            if (i+1) % 10 == 0:
                logging.info(f"Processed {i+1}/{len(gaps_to_process)} states.")
                
    cf_df = pd.DataFrame(all_new_rows)
    logging.info(f"\nGenerated {len(cf_df):,} counterfactual samples.")
    
    out_dir = Path("data/processed/counterfactual")
    out_dir.mkdir(parents=True, exist_ok=True)
    cf_df.to_parquet(out_dir / "counterfactual_samples.parquet")
    
    logging.info(f"Saved to {out_dir}/counterfactual_samples.parquet")
    return 0

if __name__ == "__main__":
    sys.exit(main())
