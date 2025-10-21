def get_topological_order(output):
    visited = set()
    topo_order = []

    def build_topo(node):
        if node in visited:
            return

        visited.add(node)
        for parent in node.parents:
            build_topo(parent)
        topo_order.append(node)

    build_topo(output)
    return topo_order


