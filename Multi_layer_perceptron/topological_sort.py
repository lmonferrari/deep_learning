from entrada import Input


def topological_sort(feed_dict):

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]

    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.nodes_output:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()
        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.nodes_output:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_and_backward(graph):
    """
    Executa uma passagem para frente através de uma lista de nós ordenados.
    :param output_node: Um nó no grafo, deve ser o nó de saída
    :param sorted_nodes: Uma lista topologicamente ordenada
    :return: O valor do nó de saída
    """
    for n in graph:
        n.forward()

    for n in graph[::-1]:
        n.backward()


def sgd_update(params, learning_rate=1e-2):
    for t in params:
        partial = t.gradients[t]
        t.value -= learning_rate * partial

