
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense,Convolution2D,Activation,Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D,AveragePooling2D
from tensorflow.keras.layers import Dropout,BatchNormalization
from tensorflow.keras.layers import Concatenate,Subtract,Add
from tensorflow.keras.layers import Flatten, Reshape
from custom_layers import * 
from graphviz import render
from graphviz import Digraph,Graph
import numpy as np
class G_Graph:
    """
    Couche ajoutant les fonctionnalités<br/>
    - Création d'un graph Graphviz<br/>
    - Utilisation de sous-cluster pour séparer des sous-ensemble comme le générateur du discriminateur<br/>
    #Arguments
    - <i>list_nodes</i>: cf attribut node 
    - <i>list_names</i>: cf attribut names
    #Attributs
    - <i>index_couche</i>: <b>int</b>, index de la couche courante
    - <i>index_graph</i>: <b>int</b>, index du graph courant
    - <i>graph</i>: <b>Digraph</b>, graph parent
    - <i>nodes</i>: <b>list de G_Layer</b>, couches du modèle <b>entrées dans l'ordre de définition</b>
    - <i>names</i>: <b>list de string</b>, noms des couches <b>entrés dans le même ordre que pour nodes</b> sous la forme (indiqué sous forme regex) [+*]?nom1_color1  [+*]?nom2_color2 [+*]?nom3_color3 du plus large graph au plus petit 
    - <i>graph_list</i>: <b>list de Digraph</b>, liste des graphs successivement créés
    - <i>graph_names_list</i>: <b>list de string</b>, liste des noms de ces graph
    """
    def __init__(self,list_nodes,list_names):
        assert type(list_nodes)==list , 'Veuillez entrer une liste et non %s'%(list_nodes)
        assert type(list_names)==list , 'Veuillez entrer une liste et non %s'%(list_names)
        
        self.index_couche = 0
        self.index_graph = 0
        self.graph = self.new_graph("transparent","LR","")
        self.nodes = list_nodes
        self.names = list_names
        self.graph_list = [self.graph]
        self.graph_names_list = []
        
    def build_graph_clusters(self):
        """Exploite les nom des couches pour construire les clusters indiqués (cf attribut names)"""
        _,_,_,tmp_names = self.graph_list[:],self.graph_names_list[:],self.nodes[:],self.names[:]
        for name_node in tmp_names:
            if name_node != "":#Si on a un sous-cluster
                clusters = name_node.split(" ")#On choisit espace comme séparateur de sous-clusters ; + comme début de sous-cluster * comme fin (à placer au début du sous-cluster)
                
                for i,clust_name_bg in enumerate(clusters):
                    clust_name = clust_name_bg.split("_")[0]
                    clust_bg = clust_name_bg.split("_")[1]
                    if clust_name[0] == "+":
                        self.new_graph(clust_bg,"LR",clust_name)#Création d'un sous-graph
                    elif clust_name[0] == "*":
                        index_graph = self.graph_names_list.index(clust_name[1:])
                        index_graph_prec = self.graph_names_list.index(clusters[i-1])
                        self.graph_list[index_graph_prec].subgraph(self.graph_list[index_graph])#Ajout du cluster au graph plus élevé
                        
    def new_graph(self,bg_clr,dir,name):
        """Crée un nouveau cluster avec les paramètres et l ajoute à graph_list et graph_names_list
        #Arguments
        - <i>bg_clr</i>: <b>string</b>, couleur de fond du cluster
        - <i>dir</i>: <b>enum string</b>, LR ou TB direction du graph
        - <i>name</i>: <b>string</b>, nom du graph 
        """
        graph = Digraph(name="cluster_Graph%d"%(self.index_graph),format='png')
        graph.attr(style='filled',label=name)
        graph.attr(bgcolor=bg_clr)
        graph.attr(rankdir=dir)
        self.graph_list.append(graph)
        self.graph_names_list.append(name)
        self.index_graph += 1
        return graph
    def render_graph(self,name):
        """Fait le rendu du graph avec le nom de graph indiqué"""
        assert type(name)==str , 'Veuillez entrer une chaine de caractères et non %s'%(name)
        self.graph.render(name)
class G_Layer:
    """Représente un layer quelconque
        #Arguments
        - <i>dico_param</i>: <b>dictionnaire</b>, dépend du layer
        - <i>index</i>: <b>int</b>, index du précédent layer
        - <i>cluster_name</i>: <b>string</b>, nom du cluster
        - <i>cluster_bg</i>: <b>string</b>, couleur du cluster graphviz (cf doc graphviz pour valeurs autorisées)
        - <i>draw</i>: <b>bool</b>, Indique si le layer doit être dessiné par graphviz
        #Attributs
        - <i>index</i>: <b>int</b>, index de la couche courante
        - <i>draw</i>: <b>bool</b>, Indique si le layer doit être dessiné par graphviz
        - <i>layer</i>: <b>layer Keras</b>, à évaluer pour avoir la sortie
        - <i>param</i>: <b>dictionnaire</b>, paramètres passés en argument, dépendent de la couche, vérifiés par verif_param
        - <i>output</i>: <b>None ou sortie de couche Keras</b>, None = non évalué ; après appel de <b>eval</b> contient la sortie de la couche évaluée
        - <i>name</i>: <b>string</b>, nom du layer keras ; défini après appel de <b>build_layer</b>
        - <i>input</i>: <b>G_Layer</b>, entrée du modèle
        - <i>output_shape</i>: <b>tuple</b>, forme de sortie calculée théoriquement
        - <i>label</i>: <b>string</b>, texte affiché par graphviz pour la couche
        - <i>graph</i>: <b>Digraph</b>, graph dans lequel est compris la couche
    """
    def __init__(self,dico_param,index,cluster_name, cluster_bg,draw=True):
        
        assert type(index) == str,"Veuillez entrer un entier comme index de la couche et non %s"%(index)
        self.index = index+1
        self.draw = draw
        self.layer = None
        self.param = dico_param
        self.output = None
        self.name = None
        self.input = None
        self.input_shape = None
        self.output_shape = None
        #Graphviz parameter
        self.label = ""
        self.graph = None
    def set_input(self,input):
        """Specifie l entree de la couche, sa forme. Verifie que les arguments sont bien ceux attendus : <b>Fonction à appeler après build_layer et donc build_and_graph</b>
        #Arguments
        - <i>input_dic</i>: <b>G_Layer</b>, entree
        """
        self.input = input
        self.input_shape = input.output_shape
        self.output_shape = self.input_shape
        self.verif_param()
        return self
    def set_graph(self,graph):
        """Specifie le graph de la couche : <b>Fonction à appeler avant build_and_graph</b>
        #Arguments
        - <i>graph</i>: <b>Digraph</b>, graph de la couche
        """
        self.graph = graph

    def build_and_graph(self):
        """Fait le lien sur le graph entre la couche précédante et la couche courante si la couche doit être dessinée"""
        if self.draw == True:
            self.build_layer()
            #Graphviz links
            if type(self.input)==list:#Pour les layers Concatenate, Add, Subtract
                for layer in self.input:#Permet d'avoir un nb qlconque de layers en entrée notamment pour le Concatenate
                    self.graph.edge(layer.index,self.index)
            else:
                self.graph.edge(self.input.index,self.index)
    def build_layer(self):
        """Construit la couche : Surchargée par les fils"""
        print("Erreur de définition de build_layer pour l'index %d"%(self.index))
        pass
    def verif_param(self):
        """Verifie que les arguments sont bien ceux attendus : Surchargée par les fils"""
        print("Erreur de définition de verif_param pour l'index %d"%(self.index))
        pass
    def identifie(self):
        """Affiche le nom de la couche ou un message d'erreur"""
        if type(self.name) == str:
            print("Je suis ",self.name)
        elif type(self.name) == list:
            print("Je suis ",self.name[0]," et ",self.name[1])
        else:
            print("Je suis quelque chose .... d'inconnu : ",self.name)
    def trainable(self,is_trainable):
        """Indique si la couche est entrainable ou non <b>is_trainable</b> booleen"""
        if type(self.layer) == list:
            for l in self.layer:
                l.trainable = False
        else:
            l.trainable = False
    def eval(self):
        """Utilise l input pour evaluer la sortie de la couche. (Fonctionnalité en + voir le code)"""
        sortie = self.input.layer
        if type(self.layer) == list:
            for l in self.layer:
                sortie = l(sortie)
            self.output = sortie
        else:
            self.output = l(sortie)
        return self.output

#Improvements at the upper level
class G_Input(G_Layer):
    """Construit une entree du reseau
    #Arguments
    - <i>name</i>: <b>string</b>, nom
    """
    def verif_param(self):
        assert type(self.param["name"])==str , 'Veuillez entrer une chaine de caractere et non %s'%(self.param["name"])
    def build_layer(self):
        """Construit la couche"""
        self.name = "%d_%s"%(self.index,self.param["name"])
        self.layer = Input(shape=self.input_shape, name=self.name)
        self.label = '{Input\n%s|{Shape | %s}}'%(self.param["name"],self.input_shape)
        self.graph.node(self.index,self.label)
    def eval(self):
        """Adapte eval à la couche input"""
        self.output = self.layer
        self.output_shape = self.input_shape
    def build_and_graph(self):
        """Adapte build_and_graph à la couche input"""
        if self.draw == True:
            self.build_layer()

class G_Convolution(G_Layer):
    """
    #Arguments
    - <i>k</i>: <b>int</b>, > 0 noyau
    - <i>f</i>: <b>int</b>, > 0 nb de filtres
    - <i>s</i>: <b>int</b>, > 0 pas
    #Illustration
    Avec strides à 1 et le padding SAME choisi ici :
    ![](https://github.com/vdumoulin/conv_arithmetic/raw/master/gif/full_padding_no_strides.gif)
    Source : [vdumoulin github](https://github.com/vdumoulin/conv_arithmetic)
    Avec strides différent de 1 en descendant
    ![](https://github.com/vdumoulin/conv_arithmetic/raw/master/gif/padding_strides_odd.gif)
    Source : [vdumoulin github](https://github.com/vdumoulin/conv_arithmetic)
    """
    def build_layer(self):
        """Construit la couche"""
        self.name = [   "%d_conv_k%d_1_f%d"%(self.index,self.param["k"],self.param["f"]),
                        "%d_conv_k1_%d_f%d"%(self.index,self.param["k"],self.param["f"])]
        couche1 = Convolution2D(filters=self.param["f"],
                                kernel_size=(self.param["k"],1),
                                activation=None,
                                strides=(self.param["s"],1),
                                padding='SAME',
                                name=self.name[0],
                                trainable=True)
        couche2 = Convolution2D(filters=self.param["f"],
                                kernel_size=(1,self.param["k"]),
                                activation=None,
                                strides=(1,self.param["s"]),
                                padding='SAME',
                                name=self.name[1],
                                trainable=True)
        self.layer = [couche1,couche2]
        self.label = '{Convolution %s | {Shape | %s} | {Noyau | %d} | {Filtres | %d} | {Strides | %d}}'%(self.index, self.input_shape,self.param["k"],self.param["f"],self.param["s"])
        self.graph.node(self.index,self.label)
    def verif_param(self):
        assert type(self.param["k"])==int , 'Veuillez entrer entier et non %s'%(self.param["k"])
        assert type(self.param["f"])==int , 'Veuillez entrer entier et non %s'%(self.param["f"])
class G_Dropout(G_Layer):
    """
    #Arguments
    - <i>r</i>: <b>float</b>, 0 < r < 1 ou Input
    """
    def build_layer(self):
        """Construit la couche"""
        if type(self.param['rate']) != float:
            self.rate = K.cast(self.param['rate'][0,0],K.floatx())
            self.name = '%d_dropout_r_adaptative'%(self.index)
            self.label = '{Dropout %d | {Rate | Adapté}}'%(self.index)
        else:
            self.rate = self.param['rate']
            self.name = '%d_dropout_r%.2f'%(self.index,self.rate)
            self.label = '{Dropout %d | {Rate | %f}}'%(self.index,self.rate)
        self.layer = Dropout(name=self.name,rate=self.rate,trainable=True)
        self.graph.node(self.index,self.label)
    def verif_param(self):
        print("ATTENTION : parametre non vérifié : le type du taux de dropout est il correct ? %s"%(self.param["rate"]))
class G_Lrn(G_Layer):
    """![](https://miro.medium.com/max/1918/1*MFl0tPjwvc49HirAJZPhEA.png)
    #Arguments
    - <i>n</i>: <b>int >0</b>, neighborhood length i.e. how many consecutive pixel values need to be considered while carrying out the normalization
    - <i>k</i>: <b>float</b>, offset (usually positive to avoid dividing by 0).
    - <i>a</i>: <b>float</b>, scale factor, usually positive
    - <i>b</i>: <b>float</b>, exponent
    #Reference
    [Difference between Local Response Normalization and Batch Normalization](https://towardsdatascience.com/difference-between-local-response-normalization-and-batch-normalization-272308c034ac)

    """
    def build_layer(self):
        """Construit la couche"""
        self.name = '%d_lrn_n%d_k%d_a%.2e_b%.2f'%(self.index,self.param["n"],self.param["k"],self.param["a"],self.param["b"])
        self.layer = LRN2D(n=self.param["n"],k=self.param["k"],alpha=self.param["a"],beta=self.param["b"],name=self.name)
        self.label = '{Regularisation\nRéponse\nLocale\n%s | {Noyau | %d} | {k | %.2f} | {alpha | %.2e} | {beta | %.2e}}'%(self.index,self.param["n"],self.param["k"],self.param["a"],self.param["b"])
        self.graph.node(self.index,self.label)
    def verif_param(self):
        assert type(self.param["n"])==int , 'Veuillez entrer un entier et non %s'%(self.param["n"])
        assert type(self.param["k"])==int , 'Veuillez entrer un entier et non %s'%(self.param["k"])
        assert type(self.param["a"])==float , 'Veuillez entrer un flotant et non %s'%(self.param["a"])
        assert type(self.param["b"])==float , 'Veuillez entrer un flotant et non %s'%(self.param["b"])
class G_Activation(G_Layer):
    """Applique la fonction d'activation SELU"""
    def build_layer(self):
        """Construit la couche"""
        self.name = '%d_activation_SELU'%(self.index)
        self.layer = Activation(SELU,name=self.name)
        self.label = '{Activation %d | {Type | SELU}}'%(self.index)
        self.graph.node(self.index,self.label)
class G_BatchNorm(G_Layer):
    """Normalisation par batch"""
    def build_layer(self):
        """Construit la couche"""
        self.name = '%d_batchnorm'%(self.index)
        self.layer = BatchNormalization(name=self.name,trainable=True)
        self.label = '{Normalisation\npar\nBatch\n%d}'%(self.index)
        self.graph.node(self.index,self.label)
class G_Dense(G_Layer):
    """Couche totalement connectée
    #Arguments
    - <i>f</i>: <b>int > 0</b>, nombre de filtres
    """
    def build_layer(self):
        """Construit la couche"""
        self.name = '%d_dense_f%d'%(self.index,self.param["f"])
        self.layer = Dense(self.param["f"],activation=None,name=self.name,trainable=True)
        self.label = '{Dense %d | {Shape | %s} | {Filtres | %d}}'%(self.index,self.input_shape,self.param["f"])
        self.graph.node(self.index,self.label)
    def verif_param(self):
        assert type(self.param["f"])==int , 'Veuillez entrer un entier et non %s'%(self.param["f"])
class G_Pool(G_Layer):
    """Couche de max-pooling uniquement pour le moment. Par défaut le pas (strides) est de deux avec keras.
    #Arguments
    - <i>f</i>: <b>int > 0</b>, nombre de filtres
    - <i>k</i>: <b>int > 0</b>, noyau
    #Illustration
    En appliquant soit le max soit la moyenne
    ![](https://github.com/vdumoulin/conv_arithmetic/raw/master/gif/no_padding_no_strides.gif)
    Source : [vdumoulin github](https://github.com/vdumoulin/conv_arithmetic)
    """
    def build_layer(self):
        """Construit la couche"""
        self.name = '%d_max_p_k%d'%(self.index,self.param["k"])
        self.layer = MaxPooling2D(name=self.name,pool_size=self.param["k"],padding='VALID',trainable=True)
        self.output_shape = (self.input_shape[0],(self.param["k"] - self.input_shape[-1]) + 1,(self.param["k"] - self.input_shape[-1]) + 1,self.input_shape[-1])
        self.label = '{MaxPooling %d | { Input Shape | %s} | {Output Shape | %s} | {Noyau | %d}}'%(self.index,self.input_shape,self.output_shape,self.param["k"])
        self.graph.node(self.index,self.label)
    def verif_param(self):
        assert type(self.param["k"])==int , 'Veuillez entrer un entier et non %s'%(self.param["k"])
class G_Deconv(G_Layer):
    """Couche inverse de la couche de convolution
    #Arguments
    - <i>k</i>: <b>int > 0</b>, noyau
    - <i>f</i>: <b>int > 0</b>, nombre de filtres
    - <i>strides</i>: <b>int > 0 en général > 1</b>, pas
    #Illustration
    ![](https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/padding_strides_transposed.gif)
    Source : [vdumoulin github](https://github.com/vdumoulin/conv_arithmetic)
    """
    def build_layer(self):
        self.name = '%d_deconv_k%d_f%d'%(self.index,self.param["k"],self.param["f"])
        self.layer = Conv2DTranspose(filters=self.param["f"],kernel_size=self.param["k"], strides=self.param["strides"],name=self.name,padding='VALID',trainable=True)
        self.output_shape = (self.input_shape[0], (self.input_shape[1]-1)*self.param["strides"]+self.param["k"], (self.input_shape[1]-1)*self.param["strides"]+self.param["k"],self.input_shape[-1])
        self.label = '{Deconvolution %d | {Input Shape | %s} | {Output Shape | %s} | {Noyau | %d} | {Filtres | %d} | {Strides | %s}}'%(self.index, self.input_shape, self.output_shape, self.param["k"], self.param["f"], self.param["strides"])
        self.graph.node(self.index,self.label)
    def verif_param(self):
        assert type(self.param["k"])==int , 'Veuillez entrer un entier et non %s'%(self.param["k"])
        assert type(self.param["f"])==int , 'Veuillez entrer un entier et non %s'%(self.param["f"])
        assert type(self.param["strides"])==int , 'Veuillez entrer un entier et non %s'%(self.param["strides"])
class G_Flat(G_Layer):
    """Mise en vecteur"""
    def build_layer(self):
        self.name = '%d_flatten'%(self.index)
        self.layer = Flatten(name=self.name)
        self.output_shape = (self.input_shape[0],np.prod(self.input_shape[1:]))
        self.label = '{Flatten %d | {Input Shape | %s} | {Output Shape | %s}}'%(self.index,self.input_shape,self.output_shape)
        self.graph.node(self.index,self.label)
class G_Proba(G_Layer):
    """Application de a fonction sigmoide interprétée comme probabilité"""
    def build_layer(self):
        """Construit la couche"""
        self.name = '%d_sigmoid_proba'%(self.index)
        self.layer = Activation('sigmoid',name=self.name)
        self.label = '{Sigmoïde %d}'%(self.index)
        self.graph.node(self.index,self.label)

class G_Add(G_Layer):
    """Ajoute le résultat de plusieurs couches"""
    def build_layer(self):
        """Construit la couche"""
        self.name = '%d_add'%(self.index)
        self.layer = Add(name=self.name)
        self.label = '{Addition %d | {Shape | %s} | {Couche %d - Couche %d}}'%(self.index,self.input_shape[0],self.input[0].index,self.input[1].index)
        self.graph.node(self.index,self.label)
    def verif_param(self):
        assert type(self.input)==list , 'Veuillez entrer une liste et non %s'%(self.param["input"])
        assert False not in list(map(lambda x:x.output_shape==self.input[0].output_shape,self.input)) , '2 layers du Add %d ne sont pas de même dimension %s'%(self.index,list(map(lambda x:x.output_shape,self.input)))
class G_Subtract(G_Layer):
    """Soustrait le résultat de 2 couches"""
    def build_layer(self):
        """Construit la couche"""
        self.name = '%d_subtract'%(self.index)
        self.layer = Subtract(name=self.name)
        self.label = '{Soustraction %d | {Shape | %s} | {Couche %d - Couche %d}}'%(self.index,self.input_shape[0],self.input[0].index,self.input[1].index)
        self.graph.node(self.index,self.label)
    def verif_param(self):
        assert type(self.input)==list , 'Veuillez entrer une liste et non %s'%(self.param["input"])
        assert False not in list(map(lambda x:x.output_shape==self.input[0].output_shape,self.input)) , '2 layers du Subtract %d ne sont pas de même dimension %s'%(self.index,list(map(lambda x:x.output_shape,self.input)))
class G_Concatenate(G_Layer):
    """Concatène les entrées suivant le dernier axed des tenseurs"""
    def build_layer(self):
        """Construit la couche"""
        self.name = '%d_concatenate'%(self.index)
        self.layer = Concatenate(axis=-1,name=self.name)
    def verif_param(self):
        assert type(self.input)==list , 'Veuillez entrer une liste et non %s'%(self.param["input"])
        assert False not in list(map(lambda x:x.output_shape==self.input[0].output_shape,self.input)) , '2 layers du Concatenate %d ne sont pas de même dimension %s'%(self.index,list(map(lambda x:x.output_shape,self.input)))
class G_Reshape(G_Layer):
    """Remet en forme l'entree"""
    def build_layer(self):
        """Construit la couche"""
        self.name = '%d_reshape'%(self.index)
        self.layer = Reshape(self.param["shape"],name=self.name)
        self.output_shape = self.param["shape"]
    def verif_param(self):
        assert type(self.param["shape"])==tuple , 'Veuillez entrer une liste et non %s'%(self.param["shape"])
    