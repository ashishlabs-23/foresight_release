"""
backend.app.services.rule_validator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 20: Validates Blackjack rule configurations.
"""
from typing import Dict, Any

def validate_rule_configuration(rules: Dict[str, Any]) -> None:
    """Validates rule combinations. Raises ValueError if invalid."""
    split_allowed = rules.get("split_allowed", True)
    resplit_allowed = rules.get("resplit_allowed", True)
    double_allowed = rules.get("double_allowed", True)
    double_after_split = rules.get("double_after_split", True)
    
    if resplit_allowed and not split_allowed:
        raise ValueError("Invalid configuration: resplit cannot be allowed if split is not allowed.")
        
    if double_after_split and not double_allowed:
        raise ValueError("Invalid configuration: double after split cannot be allowed if double is not allowed.")
        
    if double_after_split and not split_allowed:
        raise ValueError("Invalid configuration: double after split cannot be allowed if split is not allowed.")
        
    surrender_allowed = rules.get("surrender_allowed", True)
    late_surrender = rules.get("late_surrender", True)
    early_surrender = rules.get("early_surrender", False)
    
    if early_surrender and late_surrender:
        raise ValueError("Invalid configuration: Cannot have both early and late surrender enabled simultaneously.")
        
    if (early_surrender or late_surrender) and not surrender_allowed:
        raise ValueError("Invalid configuration: Surrender variants require surrender_allowed to be true.")

# Backward compatibility alias
class RuleConfigurationValidator:
    validate = staticmethod(validate_rule_configuration)

