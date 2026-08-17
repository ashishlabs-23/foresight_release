"""
backend.app.services.card_accounting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 20: Engine to track physical card limits and reject impossible states.
"""
from typing import List, Dict

def validate_card_availability(decks: int, all_visible_cards: List[str]) -> None:
    """Calculates remaining cards and raises ValueError if limits exceeded."""
    valid_ranks = {'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'}
    max_per_rank = decks * 4
    counts: Dict[str, int] = {}
    
    for c in all_visible_cards:
        if c not in valid_ranks:
            raise ValueError(f"Invalid card rank: {c}. Must be one of {valid_ranks}")
        counts[c] = counts.get(c, 0) + 1
        if counts[c] > max_per_rank:
            raise ValueError(f"Impossible state: Found {counts[c]} instances of '{c}' with only {decks} decks.")

def get_remaining_cards(decks: int, all_visible_cards: List[str]) -> Dict[str, int]:
    max_per_rank = decks * 4
    remaining = {r: max_per_rank for r in ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']}
    
    for c in all_visible_cards:
        if c in remaining:
            remaining[c] -= 1
            
    return remaining

# Backward compatibility alias
class CardAvailabilityEngine:
    validate_availability = staticmethod(validate_card_availability)
    get_remaining_cards = staticmethod(get_remaining_cards)

