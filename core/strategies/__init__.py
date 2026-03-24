def __getattr__(name):
    if name == "ContractSellingAnalyst":
        from .contract_selling_analyst import ContractSellingAnalyst
        return ContractSellingAnalyst
    raise AttributeError(f"module 'core.strategies' has no attribute {name!r}")
