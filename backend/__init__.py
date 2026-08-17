"""
backend — FastAPI application layer.

Dependency rule: backend/ may import from blackjack/ and ml/ interfaces,
but blackjack/ and ml/ must NEVER import from backend/.
"""
