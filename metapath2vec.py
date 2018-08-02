import networkx as nx
from random import choice
from gensim.models import word2vec


# metapath format: {'A':['P'], 'P':['A', 'V'], 'V':['P']}
def random_walk(graph, metapath, walks_per_node=10, walk_length=40):
    walks = []
    for i in range(walks_per_node):
        for node in graph.nodes():
            walk = [node]
            this_node = node
            while len(walk) < walk_length:
                this_type = this_node[0]
                neighbors = [n for n in graph.neighbors() if n[0] in metapath[this_type] and this_node != n]
                next_node = choice(neighbors)
                walk.append(next_node)
                this_node = next_node
            walks.append(walk)
    return walks


def learn_embeddings(walks, output, dimensions=100, window_size=5, min_count=0, sg=1,
                     iterations=3, alpha=.1, min_alpha=.01, workers=4):
    model = word2vec.Word2Vec(sentences=walks, size=dimensions,
                              window=window_size, min_count=min_count, sg=sg, workers=workers,
                              iter=iterations, alpha=alpha, min_alpha=min_alpha)
    model.save_word2vec_format(output)


if __name__ == '__main__':
    datasets = ['dblp']
    using = datasets[0]
    metapath = {'A': ['P'], 'P': ['A', 'V'], 'V': ['P']}
    graph = nx.read_edgelist('data/'+using+'/graph.edgelist', delimiter=',',
                             create_using=nx.Graph(), nodetype=str, data=False)
    walks = random_walk(graph, metapath,
                        walks_per_node=10, walk_length=40)
    learn_embeddings(walks, 'data/'+using+'m2v.emb', dimensions=100, window_size=5,
                     min_count=0, sg=1, iterations=3, alpha=.1, min_alpha=.01, workers=4)
