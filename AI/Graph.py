from graphviz import render
from graphviz import Digraph,Graph
import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'

graph = Digraph(comment='Graph', format='png')#,engine="neato")
graph.node("0","Image")
previous_id = "0"
id = 0
def get_current_id():
	global id
	return str(id)
def get_past_id(back=1):
	global id
	return str(id-back)
def new_id():
	global id
	id += 1
	return str(id)
def link(graph,id1,id2):
	graph.edge(id1,id2)
def auto_link(graph):
	graph.edge(get_past_id(),get_current_id())

def conv(graph,noyau, filtres, strides = 1, auto_connect = True):
	label = "{Convolution | {Noyau | %d} | {Filtres | %d} | {Strides | %d}}"%(noyau,filtres,strides)
	graph.node(new_id(),shape="record",label=label,color="black",fillcolor="white",style="filled")
	if auto_connect == True:
		auto_link(graph)
def dense(graph,filtres, strides = 1, auto_connect = True):
	label = "{Dense | {Filtres | %d} | {Strides | %d}}"%(filtres,strides)
	graph.node(new_id(),shape="record",label=label,color="black",fillcolor="white",style="filled")
	if auto_connect == True:
		auto_link(graph)

def dropout(graph,taux, auto_connect = True):
	label = "{Dropout | {Rate\n(taux désactivation) | %.3f}}"%(taux)
	graph.node(new_id(),shape="record",label=label,color="black",fillcolor="white",style="filled")
	if auto_connect == True:
		auto_link(graph)
def regLoc(graph,noyau = 20, k = 2, alpha= 10**-4,beta = 0.75, auto_connect = True):
	label = "{Regularisation\nRéponse\nLocale | {Noyau | %d} | {k | %.2f} | {alpha | %.2e} | {beta | %.2e}}"%(noyau,k,alpha,beta)
	graph.node(new_id(),shape="record",label=label,color="black",fillcolor="white",style="filled")
	if auto_connect == True:
		auto_link(graph)
def activation(graph,type = "SELU", auto_connect = True):
	label = "{Activation | {Type | %s}}"%(type)
	graph.node(new_id(),shape="record",label=label,color="black",fillcolor="white",style="filled")
	if auto_connect == True:
		auto_link(graph)
def batch_norm(graph, auto_connect = True):
	label = "{Normalisation\nPar\nBatch}"
	graph.node(new_id(),shape="record",label=label,color="black",fillcolor="white",style="filled")
	if auto_connect == True:
		auto_link(graph)
input_id = get_current_id()

with graph.subgraph(name='Generator') as gen:
	#Input
	gen.attr(style='filled', color='deepskyblue3')
	gen.node(new_id(),"FFT")
	gen.edge(get_past_id(),get_current_id())
	gen.node(new_id(),"Magnitude")
	gen.node(new_id(),"Angle")
	gen.edge(get_past_id(back=2),get_current_id())
	gen.edge(get_past_id(back=2),get_past_id(back=1))
	angle_id,magn_id = get_current_id(),get_past_id(back=1)
	#Analyse angle
	conv(gen,2,500)
	dropout(gen,0.25)
	regLoc(gen)
	activation(gen)
	conv(gen,2,100)
	batch_norm(gen)
	activation(gen)
	conv(gen,2,100)
	batch_norm(gen)
	activation(gen)
	dense(gen,100)
	batch_norm(gen)
	activation(gen)
	end_angle = get_current_id()

	#Analyse magnitude
	conv(gen,2,500,auto_connect=False)
	link(gen,magn_id,get_current_id())
	dropout(gen,0.25)
	regLoc(gen)
	activation(gen)
	conv(gen,2,100)
	batch_norm(gen)
	activation(gen)
	conv(gen,2,100)
	batch_norm(gen)
	activation(gen)
	dense(gen,100)
	batch_norm(gen)
	activation(gen)
	end_magn = get_current_id()

	#Analyse image
	conv(gen,2,500,auto_connect=False)
	link(gen,input_id,get_current_id())
	dropout(gen,0.25)
	regLoc(gen)
	activation(gen)
	conv(gen,2,100)
	batch_norm(gen)
	activation(gen)
	conv(gen,2,100)
	batch_norm(gen)
	activation(gen)
	gen.node(new_id(),"Concatenate")
	auto_link(gen)
	link(gen,end_angle,get_current_id())
	link(gen,end_magn,get_current_id())

	#Analyse globale
	conv(gen,2,100)
	batch_norm(gen)
	activation(gen)
	dense(gen,3)
	batch_norm(gen)
	activation(gen)

os.chdir("D:/Projets/Github/Astronomy/AI/")
graph.render("Modele000001")
os.system("start D:/Projets/Github/Astronomy/AI/Modele000001.png")
import time
time.sleep(10)
os.system("taskkill /f /im Microsoft.Photos.exe")