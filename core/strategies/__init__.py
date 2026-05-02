def __getattr__(name):
    if name == "ContractSellingAnalyst":
        from .contract_selling_analyst import ContractSellingAnalyst
        return ContractSellingAnalyst
    if name == "OptionStrikeOptimizer":
        from .option_strike_optimizer import OptionStrikeOptimizer
        return OptionStrikeOptimizer
    raise AttributeError(f"module 'core.strategies' has no attribute {name!r}")
