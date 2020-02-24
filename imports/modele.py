
from tensorflow.keras import models
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import os
import cv2
import re
class Modele_Keras:
    """
    Couche ajoutant les fonctionnalités suivantes :
    - Entrainement et validation automatique
    - Sauvegarde des Métriques d'entrainement et ed validation
    - Early stoping
    """
    def __init__(self, dic_param):
        """
        #Arguments - Mise en place du modèle
            model_directory, dossier du modèle avec / à la fin
            type_opt, str adam pour le moment
            beta1, float valeur Adam
            beta2, float valeur Adam
            lr, float valeur Adam
            type_model, enum gen, gan ou disc
            modele, le modèle keras
            modele_init_weights, chemin du modèle d initialisation
        #Arguments - Metriques et evaluation        
            layers_name_to_evaluate, list layers intermédiaire à prédire en plus lors de l évaluation
            layers_evaluate_is_metric, list indique si la prediction indiquée ci-dessus devra être considérée comme une métrique ou une image
            layers_evaluate_names, list les noms donnés à chaque évaluation intermédiaire
            metriques, list les fonctions métriques keras (leur noms sera extrait des noms de fonction pour indiquer à quoi les valeurs de métrique correspondent)
            validation_step, int le nombre d étape entre chaque validation
        #Arguments - Sauvegarde
            backup_step, int le nombre d iteration entre chaque enregistrement
        #Arguments - Early stoping
            nb_iter_degrad, int le nb d iterations telle que le jeu d entrainement à une erreur plus élevée que le jeu de validation
            taille_buffer, int le nb d iteration considérée pour calculer la pente de la courbe d entrainement
            pente_limite, float pente limite que la médiane de la liste ci-dessous ne doit pas dépasser
            taille_list_pente, int nb de pentes conservées en mémoire
        """
        assert dic_param["type_opt"] == 'adam', "Only adam supported up today"
        assert type(dic_param["beta1"])==float, "Must be float not %s"%(dic_param["beta1"])
        assert type(dic_param["beta2"])==float, "Must be float not %s"%(dic_param["beta2"])
        assert type(dic_param["lr"])==float, "Must be float not %s"%(dic_param["lr"])
        assert dic_param["type_model"] == 'gen' or dic_param["type_model"] == 'disc'or dic_param["type_model"] == 'gan', "Only suited for gen, disc or gan"
        assert type(dic_param["backup_path"]) == str, "Invalid backup path %s"%(dic_param["backup_path"])
        assert type(dic_param["backup_step"]) == int, "Invalid backup step %s"%(dic_param["backup_step"])
        assert dic_param["metriques"] != None and type(dic_param["metriques"]) == list, "On doit spécifier si il y a des metriques ou non et non %s"%(type(dic_param["metriques"]))
        assert type(dic_param["validation_step"]) == int,"Invalid validation step %s"%(dic_param["validation_step"])
        assert type(dic_param["layers_name_to_evaluate"])==list,"Veuillez passer sous forme de liste tous les noms de layers intermédiaires à évaluer (liste vide si rien en dehors des sorties par défaut) et non %s"%(dic_param["layers_name_to_evaluate"])
        assert type(dic_param["layers_evaluate_names"])==list,"Veuillez passer sous forme de liste tous les noms des layers intermédiaires évalués (liste vide si rien en dehors des sorties par défaut) et non %s"%(dic_param["layers_evaluate_names"])
        assert type(dic_param["model_directory"]) == str, 'Veuillez entrer où seront sauvegarder les donnees du modèle et non %s'%(dic_param["model_directory"])
        assert dic_param["model_directory"][-1] == '/', "N'oubliez pas le / à la fin du chemin"
        assert type(dic_param["layers_evaluate_is_metric"])==list, "Veuillez entrer une liste pas %s"%(type(dic_param["layers_evaluate_is_metric"]))
        assert len(dic_param["layers_evaluate_is_metric"])==len(dic_param["layers_evaluate_names"]), "La liste doit indiquer quelle prediction n'est pas une image et devra être enregistrée avec les metriques donc chaque element de layers_evaluate_names doit être etiqueté et ici on a 2 liste de longueurs différentes avec %d et %d"%(len(dic_param["layers_evaluate_is_metric"]),len(dic_param["layers_evaluate_names"]))
        assert False not in list(map(lambda x:type(x) == bool,dic_param["layers_evaluate_is_metric"])),"On doit indiquer par un booleen si la prediction sera ajoutee aux metriques ou non"
        assert type(dic_param["modele_init_weights"]) == str, "Veuillez entrer un chemin de restoration ou '' si il n'y a pas de restoration à faire et pas %s"%(dic_param["modele_init_weights"])
        assert type(dic_param["nb_iter_degrad"])==int,"Veuillez entrer un nb_iter_degrad entier et non %s"%(dic_param["nb_iter_degrad"])
        assert type(dic_param["taille_buffer"])==int,"Veuillez entrer une taille_buffer entiere et non %s"%(dic_param["taille_buffer"])
        assert type(dic_param["pente_limite"])==float,"Veuillez entrer une pente_limite flotante et non %s"%(dic_param["pente_limite"])
        assert type(dic_param["taille_list_pente"])==int,"Veuillez entre une taille entière de nombre de mesurre de pente à considérer et non %s"%(dic_param["taille_list_pente"])

        self.model_backup = dic_param["model_directory"]+'Model'+".h5"
        self.model_directory = dic_param["model_directory"]
        self.dataset = dic_param["dataset"]
        #Initialisation du modèle
        if dic_param["type_opt"] == 'adam':
            self.optimisateur = tf.keras.optimizers.Adam(learning_rate=10**-dic_param["lr"],
                                                            beta_1=dic_param["beta1"],
                                                            beta_2=dic_param["beta2"])
        self.metriques_fcts = dic_param["metriques"]
        self.type_model = dic_param["type_model"]
        if dic_param["type_model"] == 'gen':
            self.loss = "MSE"
        elif dic_param["type_model"] == 'disc':
            self.loss = "binary_crossentropy"
        elif dic_param["type_model"] == 'gan':
            self.loss = "binary_crossentropy"
        #Compilation du modèle
        self.modele = dic_param["modele"]
        self.modele.compile(loss=self.loss, metrics=self.metriques_fcts,optimizer=self.optimisateur)
        
        if dic_param["modele_init_weights"] != "":
            self.modele.load_weights(dic_param["modele_init_weights"])
        #Sauvegarde des metriques
        self.iteration_modele = 0
        self.liste_entrainement_iteration_globale = []#Commune à disc, gen et gan
        #Sauvegarde du modèle
        self.backup_path = dic_param["backup_path"]
        self.backup_step = dic_param["backup_step"]
        if os.path.exists(self.backup_path):
            self.modele.load_weights(self.backup_path)
        #Evaludation du modèle
        self.validation_step = dic_param["validation_step"]
        self.validation_metrics_path = dic_param["model_directory"]+"validation_metrics.txt"
        self.validation_metriques_list = []
        self.deja_evaluee = False
        layers_output = [l.output  for l in self.modele.layers if l.name in dic_param["layers_name_to_evaluate"]]

        modele_eval = Model(inputs=self.modele.inputs,outputs=self.modele.outputs+layers_output)
        modele_eval.compile(loss='binary_crossentropy',metrics=[],optimizer='Adam')
        self.modele_eval = modele_eval
        output_type = None
        if self.type_model == 'gen':
            output_type = ["Image_generee"]
        elif self.type_model == 'disc':
            output_type = ["Probabilite_bruitee"]
        elif self.type_model == 'gan':
            output_type = ["Probabilite_bruitee"]
        self.modele_eval_names_predictions = output_type+dic_param["layers_evaluate_names"]
        #Sauvegarde des metriques
        self.metriques_names = [self.loss]+[metric.__name__ for metric in dic_param["metriques"]] + [name for name,metric in zip(dic_param["layers_evaluate_names"],dic_param["layers_evaluate_is_metric"]) if metric == True]
        if self.type_model == 'gan' or self.type_model=='disc':
            self.metriques_names += ["Prediction"]
        self.path_backup_metriques = dic_param["model_directory"]+"training_metrics.txt"
        self.metriques_list = []
        self.deja_enregistre = False
        #Early stoping
        self.nb_iter_degrad = dic_param["nb_iter_degrad"]
        self.nb_degradation = 0
        self.taille_buffer = dic_param["taille_buffer"]
        self.buffer_early_stoping = []
        self.pente_limite = dic_param["pente_limite"]
        self.taille_list_pente = dic_param["taille_list_pente"]
        self.list_pente = []
        self.tr_loss = 0
        self.break_training = False

    def change_opt_param(self,beta1,beta2,lr):
        self.optimisateur = tf.keras.optimizers.Adam(learning_rate=10**-lr,
                                                            beta_1=beta1,
                                                            beta_2=beta2)
        self.modele.optimizer = self.optimisateur

    def normalisation(self,array):
        return np.array((array-np.min(array))/(np.max(array)-np.min(array)),np.float32)

    def new_epoch(self):
        self.dataset.new_epoch()

    def train(self,iteration_globale):

        if self.break_training == True:
            self.new_epoch()
            return False#Fini l'epoch
        batchs = self.dataset.next_gan_batch('tr')
        if batchs == None:
            self.new_epoch()
            return False#Fini l'epoch
        gen_input, gen_output,disc_input,disc_output,gan_input,gan_output = batchs
        input = None
        output = None
        if self.type_model == "gen":
            print("ATTENTION, générateur à 2 entrées (batch_input et output pour l'entrée) !'")
            input = [gen_input,gen_output]#ATTENTION !!! Valable pour ce générateur uniquement
            output = gen_output
        elif self.type_model == "disc":
            input = disc_input
            output = disc_output
        elif self.type_model == "gan":
            input = [gen_input,gen_output]
            output = gan_output
        metriques = self.modele.train_on_batch(input,output)
        if type(metriques) != list:
            metriques = [metriques]
        print("Iteration %s Loss : "%(self.iteration_modele),metriques[0])
        #Early stoping
        if len(self.buffer_early_stoping) < self.taille_buffer:
            self.buffer_early_stoping.append(metriques[0])
        else:
            self.buffer_early_stoping = self.buffer_early_stoping[1:]+[metriques[0]]
            pente = abs(max(self.buffer_early_stoping)-min(self.buffer_early_stoping))/self.taille_buffer
            if len(self.list_pente) < self.taille_list_pente:
                self.list_pente.append(pente)
            else:
                self.list_pente = self.list_pente[1:]+[pente]
        self.tr_loss = metriques[0]
        #Enregistrement et validation
        if self.iteration_modele % self.backup_step:
            self.modele.save_weights(self.model_backup)
        if self.iteration_modele % self.validation_step:
            self.validation()
        #self.save_metrics(metriques,type_entr='tr') ------> Pose problème si des sorties intermédiaires ajoutées.
        #Incrementation locale et globale
        self.iteration_modele += 1
        self.liste_entrainement_iteration_globale.append(iteration_globale)
        return iteration_globale+1

    def save_metrics(self,metriques,type_entr):
        assert len(metriques) == len(self.metriques_names) or (type(metriques)==float and len(self.metriques_names)==1), "Pas autant de metriques à l'entrainement qu'attendu avec %d à l'entrainement contre %d normalement"%(len(metriques), len(self.metriques_names))
        assert type_entr == 'tr' or type_entr =='v' or type_entr == 'tst', "Veuillez spécifier dans quelle type de situation on se trouve : un entrainement (tr), une évaluation pour validation (v) ou un test final (tst)"
        path = None
        statut = None
        metrics_list = None
        if type_entr == 'tr':
            path = self.path_backup_metriques
            self.metriques_list.append(metriques)
            statut = self.deja_enregistre
            metrics_list = self.liste_entrainement_iteration_globale
        elif type_entr == 'v':
            path = self.validation_metrics_path
            self.validation_metriques_list.append(metriques)
            statut = self.deja_evaluee
            metrics_list = self.validation_metriques_list
        if statut == False:
            with open(path,"w") as f:
                f.write("Iteration_globale,"+",".join(self.metriques_names)+"\n")
            self.deja_enregistre = True
        with open(path,"a") as f: 
            if type(metriques)==float:
                f.write(str(self.liste_entrainement_iteration_globale[-1])+","+metrics_list+"\n")
            else:
                f.write(str(self.liste_entrainement_iteration_globale[-1])+","+",".join(list(map(lambda x:str(x),metrics_list)))+"\n")


    def save_img(self,img_normalized,path):
        assert type(img_normalized) == np.ndarray, "Pass a numpy array"
        assert len(img_normalized.shape) == 3, "Pass an img"
        assert img_normalized.shape[-1] == 3, "Pass a img with 3 channels at the end to build rgb images"
        assert np.max(img_normalized) <=1, "Max tensor value must be 1 no %f"%np.max(img_normalized)
        individual_path = path + "_iteration_modele_%d"%(self.iteration_modele)
        cv2.imwrite(individual_path+'.jpg', np.uint8(img_normalized*255))


    def validation(self):
        batchs = self.dataset.next_gan_batch('v')
        gen_input_clean, gen_output_clean, gen_input_noise, gen_output_noise, disc_input_clean, disc_output_clean, disc_input_noise, disc_output_noise, gan_input_clean, gan_output_clean, gan_input_noise, gan_output_noise = batchs
        #Choix des input output en fonction du modèle
        if self.type_model == "gen":
            print("ATTENTION, générateur à 2 entrées (batch_input et output pour l'entrée) !'")
            input_clean = [gen_input_clean,gen_output_clean]#ATTENTION !!! Valable pour ce générateur uniquement
            input_noise = [gen_input_noise,gen_input_noise]#ATTENTION !!! Valable pour ce générateur uniquement
            output_clean = gen_output_clean
            output_noise = gen_output_noise
        elif self.type_model == "disc":
            input_clean = disc_input_clean
            input_noise = disc_input_noise
            output_clean = disc_output_clean
            output_noise = disc_output_noise
        elif self.type_model == "gan":
            input_clean = [gen_input_clean,gen_output_clean]#ATTENTION !!! Valable pour ce générateur uniquement
            input_noise = [gen_input_noise,gen_input_noise]#ATTENTION !!! Valable pour ce générateur uniquement
            output_clean = gan_output_clean
            output_noise = gan_output_noise
        metriques_clean = self.test_and_save(input_clean,output_clean)
        metriques_noise = self.test_and_save(input_noise,output_noise)
        #Early stoping
        if max(metriques_clean[0],metriques_noise[0]) > self.tr_loss:
            self.nb_degradation += 1
        else:
            self.nb_degradation = max(self.nb_degradation-1,0)
        if self.nb_degradation > self.nb_iter_degrad and np.median(self.list_pente) > self.pente_limite:
            self.break_training = True

    def test_and_save(self,input,output):
        metriques = self.modele.test_on_batch(input,output)
        if type(metriques) != list:
            metriques = [metriques]
        #Prediction avec le modèle de test
        prediction = self.modele_eval.predict(input)
        if len(prediction) != len(self.modele_eval_names_predictions) and type(prediction)!=np.ndarray:
            raise Exception("Invalid prediction : %d names was given for the ouputs but %d has been generated"%(len(self.modele_eval_names_predictions),len(prediction)))
        for predic,name in zip(prediction,self.modele_eval_names_predictions):
            if len(predic.shape) == 4: #Si c'est un batch d'img
                for img_index in range(predic.shape[0]):
                    self.save_img(self.normalisation(predic[img_index,:,:,:]),self.model_directory+'_'+name+'iter_modele_'+str(self.iteration_modele))
            elif len(predic.shape) == 2:#C'est une prediction numérique
                metriques.append(array_to_str(x))
        print("metriques : ",metriques)
        print("longueur : ",len(metriques))
        print("Outputs : ",self.modele.output)
        self.save_metrics(metriques,type_entr='v')
        return metriques
    def array_to_str(array):
        """
        Met en string une array. Applatit l'array et met la forme en début
        """
        array = np.array(array)
        s = list(array.shape)
        flat =  list(array.flatten())
        chaine = "__"
        print(s)
        chaine +="_".join(list(map(lambda x:str(x),s)))+"__"
        chaine +="|".join(list(map(lambda x:str(x),flat)))
        return chaine
    def str_to_array(chaine):
        [_,shape,array] = chaine.split("__")
        shape = list(map(lambda x:int(x),shape.split("_")))
        array = list(map(lambda x:float(x),array.split("|")))
        print(array)
        array = np.array(array).reshape(shape)
        return array