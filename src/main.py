'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import pickle
import json

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/karate.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	parser.add_argument('--resume', dest='resume', default=False, action='store_true',
						help='Generate prob only for another parallel program, resume when load from a generated walk list')

	parser.add_argument('--end2end', default=False, dest='end2end', action='store_true',
						help='')

	parser.add_argument('--walk-list', nargs='?',
	                    help='Input generated walk list path for resume')

	parser.add_argument('--probs-graph', nargs='?',
	                    help='')

	return parser.parse_args()

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	# if args.weighted:
	# 	G = nx.read_edgelist(args.input, nodetype=str, data=(('weight',float),), create_using=nx.DiGraph())
	# else:
	# 	G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
	# 	for edge in G.edges():
	# 		G[edge[0]][edge[1]]['weight'] = 1
	G = pickle.load(open(args.input,"rb"))

	# if not args.directed:
	# 	G = G.to_undirected()

	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [list(map(str, walk)) for walk in walks]
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.wv.save(args.output+".vec")
	model.save(args.output+".model")
	
	return

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	if args.resume == False:
		print("Reading graph.")
		nx_G = read_graph()
		print("Passthrough graph and construct class.")
		G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
		print("Generate probs.")
		G.preprocess_transition_probs()
		if args.end2end:
			print("Generate path.")
			walks = G.simulate_walks(args.num_walks, args.walk_length)
		else:
			print("Dump probs.")
			pickle.dump([G.alias_nodes, G.alias_edges],open(args.probs_graph,"wb"))
			# nx.write_weighted_edgelist(G, args.probs_graph)

	if args.end2end or args.resume:
		if args.resume:
			walks = pickle.load(open(args.walk_list,"rb"))
		print("Pass to word2vec.")
		learn_embeddings(walks)

if __name__ == "__main__":
	args = parse_args()
	main(args)
