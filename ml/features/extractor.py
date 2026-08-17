"""
ml/features/extractor.py
~~~~~~~~~~~~~~~~~~~~~~~~
Extracts derived features and ML-ready vectors from GameState.
"""
from typing import Any

from ml.features.state import GameState
from ml.features.derived import DerivedFeatures


class FeatureExtractor:
    """Converts raw GameState into structured features for machine learning."""
    
    @staticmethod
    def get_card_value(card_str: str) -> int:
        """Returns the Blackjack value of a card string (e.g. 'TH' -> 10)."""
        rank = card_str[0]
        if rank in "TJQK":
            return 10
        if rank == "A":
            return 11
        return int(rank)
        
    @staticmethod
    def extract_derived(state: GameState) -> DerivedFeatures:
        """Extract logical derived features from raw GameState."""
        
        ranks = state.get_player_ranks()
        
        # 1. Player Total, Soft, Pair
        num_aces = sum(1 for r in ranks if r == "A")
        hard_total = sum(FeatureExtractor.get_card_value(r) if r != "A" else 1 for r in ranks)
        
        # Calculate best total
        best_total = hard_total  # hard_total already counts every ace as 1
        is_soft = False
        if num_aces > 0 and best_total + 10 <= 21:
            best_total += 10
            # A soft hand is one where an ace is counted as 11, AND it hasn't busted, AND total != 21
            if best_total < 21:
                is_soft = True
                
        is_pair = False
        if len(ranks) == 2 and ranks[0] == ranks[1]:
            is_pair = True
                
        # 2. Shoe Info
        decks_total = state.shoe_total_cards / 52.0
        decks_remaining = state.shoe_cards_remaining / 52.0
        
        # 3. True Count
        true_count = 0.0
        if decks_remaining > 0:
            true_count = state.running_count / decks_remaining
            
        # 4. Penetration
        penetration = 1.0 - (state.shoe_cards_remaining / state.shoe_total_cards)
        
        return DerivedFeatures(
            player_total=best_total,
            is_soft=is_soft,
            is_pair=is_pair,
            num_cards=len(ranks),
            decks_total=decks_total,
            decks_remaining=decks_remaining,
            true_count=true_count,
            penetration=penetration,
        )

    @staticmethod
    def to_vector(state: GameState) -> list[float]:
        """Convert a GameState into a deterministic, flat numeric vector for ML models."""
        derived = FeatureExtractor.extract_derived(state)
        
        vector = []
        
        # Feature 1: Normalized Player Total (0 to 1, scaled by 30)
        vector.append(derived.player_total / 30.0)
        
        # Feature 2: Is Soft? (0.0 or 1.0)
        vector.append(1.0 if derived.is_soft else 0.0)
        
        # Feature 3: Is Pair? (0.0 or 1.0)
        vector.append(1.0 if derived.is_pair else 0.0)
        
        # Feature 4: Normalized Num Cards
        vector.append(derived.num_cards / 10.0)
        
        # Feature 5: Dealer Upcard Value (Normalized 0 to 1, scaled by 11)
        dealer_val = FeatureExtractor.get_card_value(state.get_dealer_rank())
        vector.append(dealer_val / 11.0)
        
        # Feature 6: True Count (Normalized approx -10 to +10 range -> scaled roughly to [-1, 1])
        vector.append(derived.true_count / 10.0)
        
        # Feature 7: Penetration (already 0.0 to 1.0)
        vector.append(derived.penetration)
        
        # Additional features can be appended here (e.g. specific rule configurations)
        
        return vector
