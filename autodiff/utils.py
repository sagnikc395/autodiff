from var import Var 
from typing import List 

def get_topological_order(output: Var) -> List[Var]:
    """Return nodes in topological order (inputs → output)."""
    order, visited = [], set()

    def dfs(node: Var):
        if id(node) in visited:
            return
        visited.add(id(node))
        for parent in node.parents:
            dfs(parent)
        order.append(node)

    dfs(output)
    return order
