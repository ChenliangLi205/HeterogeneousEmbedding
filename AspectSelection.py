# -*- coding: utf-8 -*-
import networkx as nx
import itertools


def is_subset(node_types):
    """Judge if the given aspect is a subset of the Selected ones"""
    global Selected_Aspects
    nt_set = set(node_types)
    for sa in Selected_Aspects:
        if nt_set.issubset(sa):
            return True
    return False


def is_rational(type_graph):
    """The rationality of the given aspect is determined by its connectivity"""
    return nx.is_connected(type_graph)


def center_nodes(type_graph):
    """Return the center node types of an aspect"""
    centers = []
    for node in type_graph.nodes():
        if len([n for n in type_graph[node]]) > 1:
            centers.append(node)
    return centers


def Incompatibility(graph, node_types, edge_types, center_types):
    """Calculate Incompatitance for the given aspect
       Each bloody aspect is determined by its node types"""
    center_nodes_dict = {}
    for c_type in center_types:
        center_nodes_dict[c_type] = []
    for node in graph.nodes():
        if node[0] in center_nodes_dict.keys():
            center_nodes_dict[node[0]].append(node)
    inc = 0.
    num_nonzero = 0
    for c_type, node_list in center_nodes_dict.items():
        accessable_nodetypes = extract_accessable_edgetypes(c_type, node_types, edge_types)
        count = 0
        total = len(node_list)
        for u in node_list:
            if count % 1000 == 0:
                print('{} / {}'.format(count, total))
            inc_u, nonzero = Inc_score(graph, u, node_list, accessable_nodetypes)
            inc += inc_u
            num_nonzero += nonzero
            count += 1
    return inc / num_nonzero


def extract_accessable_edgetypes(c, node_types, edge_types):
    a_types = []
    for e_t in edge_types:
        if c == e_t[0] and e_t[-1] in node_types:
            a_types.append(e_t[-1])
            continue
        if c == e_t[-1] and e_t[0] in node_types:
            a_types.append(e_t[0])
    return a_types


def Inc_score(graph, u, node_list, accessable):
    """Calculate gamma(u) for a single node u"""
    numerator = 0.
    denominator = 0.
    for v in node_list:
        if u == v:
            continue
        # compute the reachability through all accessable edge types
        reachability = Num_Cn(graph, u, v, accessable)
        numerator += max(reachability)
        denominator += min(reachability)
    if -0.1 <= denominator <= 0.1:
        return 0, 0
    else:
        return numerator / denominator - 1, 1


def Num_Cn(graph, u, v, accessable):
    neighbors_u = set([n for n in graph[u]])
    neighbors_v = set([n for n in graph[v]])
    cn = neighbors_u & neighbors_v
    count = [0] * len(accessable)
    for n in cn:
        assert n[0] in accessable
        count[accessable.index(n[0])] += 1
    return count


# node types : ['A', 'P', 'P', 'V'], P appears multiple times because the P-P edge type
# edge types : ['A-P', 'P-P', 'P-V', ...]
def Select_Aspect(graph, node_types, edge_types, threshold):
    """Se个粑粑"""
    global Selected_Aspects
    if is_subset(node_types):
        return
    type_graph = nx.Graph()
    for et in edge_types:
        if et[0] in node_types and et[-1] in node_types:
            type_graph.add_edge(et[0], et[-1])
    if is_rational(type_graph):
    # whether it is a valid aspect
        center_types = center_nodes(type_graph)
        Inc = Incompatibility(graph, node_types, edge_types, center_types)
        if Inc > threshold:
            Selected_Aspects.append(node_types)
            return
        if len(node_types) <= 3:
        # It takes at least 3 node types to make an aspect
            return
        else:
            for c in itertools.combinations(node_types, len(node_types)-1):
                Select_Aspect(graph, list(c), edge_types, threshold)


def show_Inc_aspects(graph, node_types, edge_types, aspects):
    for a in aspects:
        type_graph = nx.Graph()
        for et in edge_types:
            if et[0] in a and et[-1] in a:
                type_graph.add_edge(et[0], et[-1])
        center_types = center_nodes(type_graph)
        print(Incompatibility(graph, node_types, edge_types, center_types))


if __name__ == '__main__':
    datasets = ['dblp/']
    using = datasets[0]
    graph = nx.read_edgelist(
        'data/' + using + 'graph.edgelist', delimiter=',',
        create_using=nx.Graph(), nodetype=str, data=False
    )
    Select_Aspect(
        graph=graph,
        node_types=['A', 'P', 'P', 'V'],
        edge_types=['A-P', 'P-V', 'P-P'],
        threshold=1.
    )
