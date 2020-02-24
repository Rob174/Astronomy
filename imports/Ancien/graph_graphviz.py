
from graph_global import * 
def g_conv(graph,prev,noyau, filtres, strides = 1, auto_connect = True, identifier=False):
	label = None
	if identifier == True:
		print("Received identifier")
	if type(noyau) == int:
		label = "{Convolution %s | {Noyau | %d} | {Filtres | %d} | {Strides | %d}}"%(g_get_current_id(),noyau,filtres,strides)
	else:
		label = "{Convolution %s | {Noyau | %dx%d} | {Filtres | %d} | {Strides | %d}}"%(g_get_current_id(),noyau[0],noyau[1],filtres,strides)
	graph.node(g_get_current_id(),shape="record",label=label,color="black",fillcolor="white" if identifier==False else "red",style="filled")
	if auto_connect == True:
		g_link(graph,id1=prev,id2=g_get_current_id())
def g_max_p(graph,prev,noyau, auto_connect = True, identifier=False):
	label = "{MaxPooling | {Noyau | %d}}"%(noyau)
	graph.node(g_get_current_id(),shape="record",label=label,color="black",fillcolor="white" if identifier==False else "red",style="filled")
	if auto_connect == True:
		g_link(graph,id1=prev,id2=g_get_current_id())
def g_deconv(graph,prev,noyau,filtres, auto_connect = True, identifier=False):
    label = "{Déconvolution %s | {Noyau | %d} | {Filtres | %d}}"%(g_get_current_id(),noyau,filtres)
    graph.node(g_get_current_id(),shape="record",label=label,color="black",fillcolor="white" if identifier==False else "red",style="filled")
    if auto_connect == True:
        g_link(graph,id1=prev,id2=g_get_current_id())
def g_dense(graph,prev,filtres, strides = 1, auto_connect = True, identifier=False):
	label = "{Dense | {Filtres | %d} | {Strides | %d}}"%(filtres,strides)
	graph.node(g_get_current_id(),shape="record",label=label,color="black",fillcolor="white" if identifier==False else "red",style="filled")
	if auto_connect == True:
		g_link(graph,id1=prev,id2=g_get_current_id())

def g_dropout(graph,prev,taux, auto_connect = True, identifier=False):
	label = ""
	if type(taux) == float:
		label = "{Dropout %s | {Rate\n(taux désactivation) | %.3f}}"%(g_get_current_id(),taux)
	else:
		label = "{Dropout %s | {Rate\n(taux désactivation) | Adapté}}"%(g_get_current_id())

	graph.node(g_get_current_id(),shape="record",label=label,color="black",fillcolor="white" if identifier==False else "red",style="filled")
	if auto_connect == True:
		g_link(graph,id1=prev,id2=g_get_current_id())
def g_regLoc(graph,prev,noyau = 20, k = 2, alpha= 10**-4,beta = 0.75, auto_connect = True, identifier=False):
	label = "{Regularisation\nRéponse\nLocale\n%s | {Noyau | %d} | {k | %.2f} | {alpha | %.2e} | {beta | %.2e}}"%(g_get_current_id(),noyau,k,alpha,beta)
	if identifier == True:
		print(identifier)
	graph.node(g_get_current_id(),shape="record",label=label,color="black",fillcolor="white" if identifier==False else "red",style="filled")
	if auto_connect == True:
		g_link(graph,id1=prev,id2=g_get_current_id())
def g_activation(graph,prev,type = "SELU", auto_connect = True, identifier=False):
	label = "{Activation %s | {Type | %s}}"%(g_get_current_id(),type)
	graph.node(g_get_current_id(),shape="record",label=label,color="black",fillcolor="white" if identifier==False else "red",style="filled")
	if auto_connect == True:
		g_link(graph,id1=prev,id2=g_get_current_id())
def g_batch_norm(graph,prev, auto_connect = True, identifier=False):
	label = "{Normalisation\nPar\nBatch\n%s}"%(g_get_current_id())
	graph.node(g_get_current_id(),shape="record",label=label,color="black",fillcolor="white" if identifier==False else "red",style="filled")
	if auto_connect == True:
		g_link(graph,id1=prev,id2=g_get_current_id())
def g_flat(graph,prev, auto_connect = True, identifier=False):
	label = "{Flatten}"
	graph.node(g_get_current_id(),shape="record",label=label,color="black",fillcolor="white" if identifier==False else "red",style="filled")
	if auto_connect == True:
		g_link(graph,id1=prev,id2=g_get_current_id())
def g_concat(graph,index, identifier=False):
    graph.node(g_get_current_id(),"Concatenate\n%s"%(g_get_current_id()),fillcolor="white" if identifier==False else "red")
    for id in index:
        graph.edge(id,g_get_current_id())
def g_add(graph,index, identifier=False):
    graph.node(g_get_current_id(),"+",fillcolor="white" if identifier==False else "red")
    for id in index:
        graph.edge(id,g_get_current_id())
def g_subtract(graph,index, identifier=False):
    graph.node(g_get_current_id(),"-",fillcolor="white" if identifier==False else "red")
    for id in index:
        graph.edge(id,g_get_current_id())
def g_proba(graph,prev, auto_connect = True, identifier=False):
	graph.node(g_get_current_id(),"Probabilite",fillcolor="white" if identifier==False else "red")
	if auto_connect == True:
		g_link(graph,id1=prev,id2=g_get_current_id())