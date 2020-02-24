
from graphviz import render
from graphviz import Digraph,Graph
def g_get_current_id():
	global index_couches
	return str(index_couches)
def g_get_past_id(back=1):
	global index_couches
	return str(index_couches-back)
def g_new_id():
	global index_couches
	index_couches += 1
	return str(index_couches)
global Llink
Llink = []
def g_link(graph,id1,id2):
    global Llink
    if not([id1,id2] in Llink):
        graph.edge(id1,id2)
        Llink += [id1,id2]
    else:
        print("Link already present : %s"%[id1,id2])
global index_graph
global index_couches
index_graph = 0
index_couches = 0
def new_graph(bgcolor='transparent'):
    """Return the graph"""
    global index_graph
    global index_couches
    index_couches = 0
    graph = Digraph(name="cluster_Graph%d"%(index_graph),format='png')
    graph.attr(bgcolor=bgcolor)
    graph.attr(rankdir="LR")
    index_graph += 1
    return graph
def end_graph(graph,name):
    """Create the png file of the graph"""
    graph.render(name)
def begin_cluster(past_couche,name,color):
	orig_graph = past_couche[1]
	past_couche[1] = Digraph(name="cluster_%s"%(name))
	past_couche[1].attr(style='filled', color=color, label=name)
	return orig_graph, past_couche
def end_cluster(last_couche,orig_graph):
	orig_graph.subgraph(last_couche[1])
	last_couche[1] = orig_graph
	return last_couche