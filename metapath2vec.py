import networkx as nx
import time
from random import choice
from gensim.models import word2vec
from concurrent.futures import ProcessPoolExecutor, as_completed


# metapath format: {'A':['P'], 'P':['A', 'V'], 'V':['P']}
def random_walk(graph, metapath, walks_per_node=10, walk_length=40, workers=3):
    """random walk using metapath, starts multiple times at each node. Returns a 2d list"""
    walks = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for walk_iter in range(walks_per_node):
            job = executor.submit(_random_walk, graph, metapath, walk_length)
            futures[job] = walk_iter
        for job in as_completed(futures):
            walk_one_time = job.result()
            walks.extend(walk_one_time)
            del futures[job]
    return walks


def _random_walk(graph, metapath, walk_length):
    """returns a 2d list"""
    walks_one_time = []
    for node in graph.nodes():
        walk = [node]
        this_node = node
        while len(walk) < walk_length:
            this_type = this_node[0]
            neighbors = [n for n in graph.neighbors(this_node) if n[0] in metapath[this_type] and this_node != n]
            next_node = choice(neighbors)
            walk.append(next_node)
            this_node = next_node
        walks_one_time.append(walk)
    return walks_one_time


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
    start = time.time()
    graph = nx.read_edgelist('data/'+using+'/graph.edgelist', delimiter=',',
                             create_using=nx.Graph(), nodetype=str, data=False)
    walks = random_walk(graph, metapath,
                        walks_per_node=10, walk_length=40)
    learn_embeddings(walks, 'data/'+using+'m2v.emb', dimensions=100, window_size=5,
                     min_count=0, sg=1, iterations=3, alpha=.1, min_alpha=.01, workers=4)
    print "time consuming: {}" .format(int(time.time()-start) / 60)
