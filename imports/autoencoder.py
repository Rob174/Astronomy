from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from graph_keras import *
#SOURCE : https://keras.io/examples/mnist_denoising_autoencoder/

###################################################################
###################    ENCODER     ###############################
##################################################################
def build_autoencoder():
    """Construit un autoencodeur. Retourne la liste des couches
    #References
    [Exemple keras](https://keras.io/examples/mnist_denoising_autoencoder/)
    """
    index_couche = 0
    list_layers = []
    layer_filters = [32,64,128]
    kernel_size = 3

    list_layers.append(G_Input({"name":"encoder-input"},index_couche,"+Autoencodeur","darkgreen"))
    for filters in layer_filters:
        list_layers.append(G_Convolution({"k":kernel_size,"f":filters,"s":2},list_layers[-1].index,"Autoencodeur","darkgreen"))
        list_layers[-1].set_input(list_layers[-1])
        list_layers.append(G_BatchNorm({},list_layers[-1].index,"Autoencodeur","darkgreen"))
        list_layers[-1].set_input(list_layers[-1])
        list_layers.append(G_Activation({},list_layers[-1].index,"Autoencodeur","darkgreen"))
        list_layers[-1].set_input(list_layers[-1])

    # Shape info needed to build Decoder Model
    shape = list_layers[-1].output_shape

    # Generate the latent vector
    list_layers.append(G_Flat({},list_layers[-1].index,"Autoencodeur","darkgreen"))
    list_layers[-1].set_input(list_layers[-1])
    list_layers.append(G_Dense({"f":16},list_layers[-1].index,"Autoencodeur","darkgreen"))
    list_layers[-1].set_input(list_layers[-1])
    list_layers.append(G_Dense({"f":shape[0]*shape[1]*shape[2]},list_layers[-1].index,"Autoencodeur","darkgreen"))
    list_layers[-1].set_input(list_layers[-1])


    ##################################################################
    ###################    DECODER     ###############################
    ##################################################################
    # Build the Decoder Model
    list_layers.append(G_Reshape({"shape":(shape[0], shape[1], shape[2])},list_layers[-1].index,"Autoencodeur","darkgreen"))
    list_layers[-1].set_input(list_layers[-1])

    # Stack of Transposed Conv2D blocks
    # Notes:
    # 1) Use Batch Normalization before ReLU on deep networks
    # 2) Use UpSampling2D as alternative to strides>1
    # - faster but not as good as strides>1
    for filters in layer_filters[::-1]:
        list_layers.append(G_Deconv({"k":kernel_size,"f":filters,"strides":2},list_layers[-1].index,"Autoencodeur","darkgreen"))
        list_layers[-1].set_input(list_layers[-1])
        list_layers.append(G_BatchNorm({},list_layers[-1].index,"Autoencodeur","darkgreen"))
        list_layers[-1].set_input(list_layers[-1])
        list_layers.append(G_Activation({},list_layers[-1].index,"Autoencodeur","darkgreen"))
        list_layers[-1].set_input(list_layers[-1])

    list_layers.append(G_Proba({},list_layers[-1].index,"*Autoencodeur","darkgreen"))
    list_layers[-1].set_input(list_layers[-1])
    return list_layers