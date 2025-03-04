import copy

from torch_geometric.explain.explanation import Explanation
from torch import concat

# deep copy explanation
def merge_pe_with_x(exp: Explanation, **kwargs) -> Explanation:
    exp = copy.deepcopy(exp)
    pe = list(kwargs.values())[0]
    
    exp.x = concat((exp.x, pe), dim=1)
    return exp
