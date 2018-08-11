# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
from random import choice, choices, sample
from torch.optim import SGD, Adam
from model.skipgram import SkipGram
from copy import deepcopy
from gensim.models import word2vec
from concurrent.futures import ProcessPoolExecutor, as_completed


def random_walk(aspect_graph, walks_per_node=10, walk_length=40, workers=3):
    """random walk using metapath, starts multiple times at each node. Returns a 2d list"""
    walks = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for walk_iter in range(walks_per_node):
            job = executor.submit(_random_walk, aspect_graph, walk_length)
            futures[job] = walk_iter
        for job in as_completed(futures):
            walk_one_time = job.result()
            walks.extend(walk_one_time)
            del futures[job]
    return walks


def _random_walk(aspect_graph, walk_length):
    """returns a 2d list"""
    walks_one_time = []
    for node in graph.nodes():
        walk = [node]
        this_node = node
        while len(walk) < walk_length:
            neighbors = [n for n in aspect_graph.neighbors(this_node)]
            next_node = choice(neighbors)
            walk.append(next_node)
            this_node = next_node
        walks_one_time.append(walk)
    return walks_one_time


def build_dict(graph):
    """Build node2id and id2node dictionaries"""
    count = 0
    node2id, id2node = {}, {}
    for node in graph.nodes():
        node2id[node] = count
        id2node[count] = node
        count += 1
    return node2id, id2node


def construct_aspect_graph(original_graph, aspect):
    """Remove links that does not belong to this aspect"""
    aspect_graph = deepcopy(original_graph)
    for node in original_graph.nodes():
        if node[0] not in aspect:
            aspect_graph.remove_node(node)
    return aspect_graph


def sample4training(aspect_graph, num_samples, node2id_dict, batch_size=100):
    """Sample u v and negative samples for the training of the skipgram model"""
    u_list, v_list = [], []
    sampled_edges = choices([e for e in aspect_graph.edges()], k=int(num_samples))
    for e in sampled_edges:
        u = node2id_dict[e[0]]
        v = node2id_dict[e[1]]
        # if len(u_list[-1]) < batch_size:
        #     u_list[-1].append(u)
        #     v_list[-1].append(v)
        # else:
        u_list.append([u])
        v_list.append([v])
    return u_list, v_list


def build_aspect_vector(num_nodes, aspect_graph, node2id_dict):
    """ Build a vector whose entries with value 1 represent the
        nodes in the aspect"""
    aspect_vector = np.zeros(num_nodes)
    for node in aspect_graph.nodes():
        aspect_vector[node2id_dict[node]] = 1.
    return aspect_vector


def fit(model, us, vs, k, initial_lr=0.0025, max_iters=2):
    """Optimize the skipgram model"""
    lr = initial_lr
    optimizer = Adam(params=model.parameters(), lr=lr)
    for j in range(len(us)):
        u = us[j]
        v = vs[j]
        for i in range(max_iters):
            optimizer.zero_grad()
            loss = model.forward(u, v, k)
            loss.backward()
            optimizer.step()


def learn_embeddings_skipgram(graph, aspects, dimensions=100, initial_lr=0.1, max_iters=1, k=3):
    """
    :param graph:
    :param aspects:
    :param dimensions:
    :param initial_lr:
    :param max_iters:
    :param k: number of negative samples per sample
    :return:
    """
    node2id, id2node = build_dict(graph)
    embedding_matrix = np.zeros((graph.number_of_nodes(), dimensions))
    for aspect in aspects:
        aspect_graph = construct_aspect_graph(graph, aspect)
        aspect_vector = build_aspect_vector(
            graph.number_of_nodes(),
            aspect_graph, node2id
        )
        num_samples = aspect_graph.number_of_edges() / 10
        us, vs = sample4training(
            aspect_graph, num_samples,
            node2id_dict=node2id
        )
        model = SkipGram(
            graph.number_of_nodes(),
            dimensions
        )
        fit(
            model, us, vs, k,
            initial_lr=initial_lr,
            max_iters=max_iters,
        )
        embedding_matrix += np.dot(aspect_vector, model.out_embeddings())
    return embedding_matrix / len(aspects), id2node


def learn_embeddings_randomwalk(aspect_graphs, output, dimensions=100, window_size=5, min_count=0, sg=1,
                     iterations=3, alpha=.1, min_alpha=.01, workers=4):
    for i, aspect_graph in enumerate(aspect_graphs):
        walks = random_walk(aspect_graph,
                            walks_per_node=10, walk_length=40)
        model = word2vec.Word2Vec(sentences=walks, size=dimensions,
                                  window=window_size, min_count=min_count, sg=sg, workers=workers,
                                  iter=iterations, alpha=alpha, min_alpha=min_alpha)
        model.wv.save_word2vec_format('%s%s.emb' % (output, i))


def save_embeddings(embedding_matrix, path, id2node_dict):
    """ Output the embedding matrix into a .emb file"""
    with open(path, 'w') as fo:
        for i in range(embedding_matrix.shape[0]):
            v = embedding_matrix[i]
            v = ' '.join(map(str, v))
            fo.write('%s %s\n' % (id2node_dict[i], v))


if __name__ == '__main__':
    aspects = ['APV']
    graph = nx.read_edgelist('data/dblp/graph.edgelist', delimiter=',',
                             create_using=nx.Graph(), nodetype=str, data=False)
    e, id2node = learn_embeddings_skipgram(
        graph,
        aspects,
        dimensions=100,
        initial_lr=0.025,
        max_iters=1,
        k=3,
    )
    save_embeddings(e, 'data/dblp/AspEm.emb', id2node)
