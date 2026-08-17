"""
tests/unit/ml/test_features.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for ML state representation and feature extraction.
"""
import pytest
from pydantic import ValidationError

from ml.features.state import GameState
from ml.features.derived import DerivedFeatures
from ml.features.extractor import FeatureExtractor


def test_game_state_validation():
    # Valid state
    state = GameState(
        player_cards=["TH", "AS"],
        dealer_upcard="2D",
        shoe_total_cards=312,
        shoe_cards_remaining=150,
        running_count=-3,
        rules={"peek": True}
    )
    assert state.player_cards == ["TH", "AS"]
    
    # Invalid card length
    with pytest.raises(ValidationError):
        GameState(
            player_cards=["10H"],  # Should be TH
            dealer_upcard="2D",
            shoe_total_cards=312,
            shoe_cards_remaining=150,
            running_count=-3,
            rules={}
        )
        
    # Invalid rank
    with pytest.raises(ValidationError):
        GameState(
            player_cards=["1H"],
            dealer_upcard="2D",
            shoe_total_cards=312,
            shoe_cards_remaining=150,
            running_count=-3,
            rules={}
        )


def test_feature_extractor_hard_hand():
    state = GameState(
        player_cards=["TH", "7S"],
        dealer_upcard="9D",
        shoe_total_cards=52,
        shoe_cards_remaining=26,
        running_count=2,
        rules={}
    )
    
    derived = FeatureExtractor.extract_derived(state)
    assert derived.player_total == 17
    assert not derived.is_soft
    assert not derived.is_pair
    assert derived.num_cards == 2
    assert derived.true_count == 4.0  # 2 / (26/52) = 2 / 0.5 = 4
    assert derived.penetration == 0.5


def test_feature_extractor_soft_hand():
    state = GameState(
        player_cards=["AS", "2H", "3D"],
        dealer_upcard="TH",
        shoe_total_cards=312,
        shoe_cards_remaining=156,
        running_count=-5,
        rules={}
    )
    
    derived = FeatureExtractor.extract_derived(state)
    assert derived.player_total == 16
    assert derived.is_soft
    assert not derived.is_pair
    assert derived.num_cards == 3
    assert derived.true_count == pytest.approx(-5 / 3.0)


def test_feature_extractor_pair():
    state = GameState(
        player_cards=["8S", "8D"],
        dealer_upcard="AS",
        shoe_total_cards=52,
        shoe_cards_remaining=52,
        running_count=0,
        rules={}
    )
    
    derived = FeatureExtractor.extract_derived(state)
    assert derived.player_total == 16
    assert not derived.is_soft
    assert derived.is_pair


def test_feature_extractor_busted_ace():
    # A, 5, 8 = 14 (hard)
    state = GameState(
        player_cards=["AS", "5H", "8D"],
        dealer_upcard="2S",
        shoe_total_cards=52,
        shoe_cards_remaining=52,
        running_count=0,
        rules={}
    )
    
    derived = FeatureExtractor.extract_derived(state)
    assert derived.player_total == 14
    assert not derived.is_soft
    
def test_feature_extractor_blackjack_is_not_soft():
    state = GameState(
        player_cards=["AS", "TH"],
        dealer_upcard="2S",
        shoe_total_cards=52,
        shoe_cards_remaining=52,
        running_count=0,
        rules={}
    )
    derived = FeatureExtractor.extract_derived(state)
    assert derived.player_total == 21
    assert not derived.is_soft


def test_to_vector():
    state = GameState(
        player_cards=["TH", "7S"],
        dealer_upcard="9D",
        shoe_total_cards=52,
        shoe_cards_remaining=26,
        running_count=2,
        rules={}
    )
    
    vec = FeatureExtractor.to_vector(state)
    assert len(vec) == 7
    # 1. Player Total (17 / 30)
    assert vec[0] == 17 / 30.0
    # 2. Is Soft (0.0)
    assert vec[1] == 0.0
    # 3. Is Pair (0.0)
    assert vec[2] == 0.0
    # 4. Num Cards (2 / 10)
    assert vec[3] == 0.2
    # 5. Dealer Upcard (9 / 11)
    assert vec[4] == 9 / 11.0
    # 6. True Count (4.0 / 10)
    assert vec[5] == 0.4
    # 7. Penetration
    assert vec[6] == 0.5
