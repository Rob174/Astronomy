import os
import cv2
import numpy as np
import time
class Dataset:
    """Gère les 3 dataset de test, validation et entrainement"""
    def __init__(self, localisation_annexe_txt,train_rate=0.7, validation_rate=0.2, test_rate=0.1, batch_size=10, taille_img=256, nb_open_err=10, verification_err=True):
        """
        #Arguments
            localisation_annexe_txt: string chemin relatif depuis le dossier TIPE
            train_rate: float entre 0 et 1 pourcentage des données consacrées à l'entrainement
            validation_rate: float entre 0 et 1 pourcentage des données consacrées à la validation
            test_rate: float entre 0 et 1 pourcentage des données consacrées aux tests
            batch_size: int > 0 nombre d'image passées simultannément dans le réseau
            taille_img: int > 0 fortement recommendé d'être une puissance de 2 pour ne pas avoir de problème si on a besoin d'utiliser des couches de 'déconvolution'
            nb_open_err: int > 0 nb de fois où pour une même image, le programme échoue à l'ouvrir
            verification_err: bool indique s'il faut vérifier que toutes les images puissent être ouvertes
        """
        # Spliting into train, validation, test
        self.train_rate=train_rate
        self.validation_rate=validation_rate
        self.test_rate=test_rate
        self.annexe_img = []
        with open(localisation_annexe_txt,'r') as f:
                for l in f:
                    self.annexe_img.append(["/content/drive/My Drive/TIPE/"+l.split(",")[0].strip(),"/content/drive/My Drive/TIPE/"+l.split(",")[1].strip()])#img noise, file
        self.liste_files = ["/content/drive/My Drive/TIPE/Galaxies_resized/"+f for f in os.listdir("/content/drive/My Drive/TIPE/Galaxies_resized/")]+list(map(lambda x:x[1],self.annexe_img))
        train,validation,test = self.index_split_into(len(self.liste_files))
        self.train_dataset = [self.liste_files[i] for i in train]
        self.type_train_dataset = [i-len(self.liste_files) if i >= len(self.liste_files)-len(self.annexe_img) else 'orig' for i in train]
        self.validation_dataset = [self.liste_files[i] for i in validation]
        self.type_validation_dataset = [i-len(self.liste_files) if i >= len(self.liste_files)-len(self.annexe_img) else 'orig' for i in validation]
        self.test_dataset = [self.liste_files[i] for i in test]
        self.type_test_dataset = [i-len(self.liste_files) if i > len(self.liste_files)-len(self.annexe_img) else 'orig' for i in test]

        #buffer to save current batch type img (coming from previous training or artificially noised
        self.type_img = ['orig' for i in range(batch_size)]
        #Initialisasing batch count
        if verification_err == True:
            self.verification_dataset()
        self.batch_size = batch_size
        self.batch_nb_tr = 0
        self.max_batch_nb_tr = int(len(self.train_dataset)/self.batch_size)
        self.batch_nb_v = 0
        self.max_batch_nb_v = int(len(self.validation_dataset)/self.batch_size)
        self.batch_nb_tst = 0
        self.max_batch_nb_tst = int(len(self.test_dataset)/self.batch_size)
        #img parameters
        self.taille_img = taille_img
        self.nb_open_err = nb_open_err
    def new_epoch(self):
        self.batch_nb_tr = 0
        self.batch_nb_v = 0
        self.batch_nb_tst = 0
    def index_split_into(self,longueur_data):
        """Retourne une liste contenant la liste des index des images d'entrainement, celle de ceux de vérification et celel de ceux de test
            #Arguments
                    longueur_data, int > 0 longueur de la liste des données
        """
        indexes = np.arange(longueur_data)
        np.random.shuffle(indexes)
        L_int = [0]
        for prct in [self.train_rate,self.validation_rate,self.test_rate]:
            nvl_val = int(L_int[-1]+prct*longueur_data)
            if nvl_val < longueur_data:
                L_int.append(nvl_val+1)
            else:
                L_int.append(longueur_data)
        return [indexes[L_int[i-1]:L_int[i]+1] for i in range(1,len(L_int))]
    def verification_dataset(self):
        """Vérifie si toutes les images du dataset peuvent être ouvertes"""
        print("Vérification")
        valide = True
        result = None
        for img in self.liste_files:
            try:
                result = cv2.imread(img)
                if type(result)!=np.ndarray and result == None:
                    valide = False
                    raise Exception("Erreur image nulle : %s"%(img))
                del result
            except Exception as e:
                print("Can't open img %s"%(img))
                print(result)
                print("Fin")
                print(e)
                valide = False
        return valide
    def normalisation(self,array):
        assert type(array) == np.ndarray, "Doit être une array et non %s"%array
        
        return np.array((array-np.min(array))/(np.max(array)-np.min(array)),np.float32)
    def next_batch_dataset(self,dataset):
        """Renvoi le dataset correspondant à l'identificateur passé. Après avoir passé toutes les image : Renvoie None si le dataset d'entrainement est fini, revient au début pour les autres datasets
            #Arguments
                    dataset: enum tr, v ou tst pour entrainement (train), validation et test choix du dataset
        """
        assert dataset == 'tr' or dataset == 'v' or dataset == 'tst', "%s n'est pas un des type tr, v ou tst autorisés"%(dataset)

        data = None
        if dataset == 'tr':
            data = self.train_dataset,self.type_train_dataset
            if self.batch_nb_tr > self.max_batch_nb_tr or self.batch_nb_tr+1 > self.max_batch_nb_tr:
                return None # fin du lot d'entrainement
        elif dataset == 'v':
            data = self.validation_dataset,self.type_validation_dataset
            if self.batch_nb_v > self.max_batch_nb_v or self.batch_nb_v+1 > self.max_batch_nb_v:
                self.batch_nb_v = 0# repart du début
        elif dataset == 'tst':
            data = self.test_dataset,self.type_test_dataset
            if self.batch_nb_tst > self.max_batch_nb_tst or self.batch_nb_tst+1 > self.max_batch_nb_tst:
                self.batch_nb_tst = 0# repart du début
        return data
    def open_img(self,path):
        """Ouvre l'image au format float32 en acceptant self.nb_open_err erreurs
            #Arguments
                path: string, chemin de l image
        """
        assert type(path) == str, "Objet de type %s invalide"%(type(path))

        succes = False
        compteur_erreurs = 0
        img = None
        while succes == False and compteur_erreurs < self.nb_open_err:
            try:
                image = cv2.imread(path)
                resized_image = cv2.resize(image,(self.taille_img,self.taille_img))
                img = np.array(resized_image,dtype=np.float32)
                succes = True
            except Exception as e:
                print("Error in next_batch")
                compteur_erreurs += 1
                if compteur_erreurs == self.nb_open_err:
                    
                    raise Exception("image %s raised error "%img,e)
        return img
    def next_batch_original_img(self,dataset):
        """Renvoie le batch suivant dans le lot total d'entrainement. Retourne None si il n'y a plus de batch disponible et que tout le lot d'entrainement a été passé
            #Arguments
                dataset, enum tr, v ou tst choix du dataset
        """
        assert dataset == 'tr' or dataset == 'v' or dataset == 'tst', "%s n'est pas un des type tr, v ou tst autorisés"%(dataset)
        
        dataset_type = self.next_batch_dataset(dataset)
        if dataset_type == None:
            return None,None
        data,type_data = dataset_type
        batch_nb = None
        if dataset == 'tr':
            batch_nb = self.batch_nb_tr
            if (batch_nb+1)*self.batch_size > len(data):
                self.batch_nb_tr = 0
        elif dataset == 'v':
            batch_nb = self.batch_nb_v
            if (batch_nb+1)*self.batch_size > len(data):
                self.batch_nb_v = 0
        else:
            batch_nb = self.batch_nb_tst
            if (batch_nb+1)*self.batch_size > len(data):
                self.batch_nb_tst = 0

        img_path = [img for img in data[batch_nb*self.batch_size:(batch_nb+1)*self.batch_size]]
        type_img = [typ for typ in type_data[batch_nb*self.batch_size:(batch_nb+1)*self.batch_size]]

        batch = []
        batch_deja_bruit = []
        for path,typ_index in zip(img_path,type_img):
            batch.append(self.open_img(path))
            if typ_index == 'orig':
                batch_deja_bruit.append(None)
            else:
                batch_deja_bruit.append(self.normalisation(np.array(self.open_img(self.annexe_img[typ_index][0]))))
        batch = self.normalisation(np.array(batch))
        if dataset_type == 'tr':
            self.batch_nb_tr += 1
        elif dataset_type == 'v':
            self.batch_nb_v += 1
        else:
            self.batch_nb_tst += 1
        return batch,batch_deja_bruit
    def next_batch_bruit_voile(self,voile_pow_neg_10_val1=0.4,voile_pow_neg_10_val2=1.3,μ_bruit=1.779,σ_bruit=1.779, typ='norm', dataset='tr'):
        """Renvoie un tuple avec le batch (groupe) d'images et son equivalent bruité artificellement avec les paramètres spécifiés
            #Arguments
                voile_pow_neg_10_val1,voile_pow_neg_10_val2 : définissent l'intervalle [a,b] tel que le facteur sera entre 10**-b et 10**-a
                bruit : gaussien des paramètres ci-dessus suivant N(10e-μ_bruit,10e-σ_bruit) 
                typ : enum clip ou autre string (pour normaliser) indique si après éventuel bruitage on doit normaliser ou couper le batch au bon intervalle
        """
        assert type(voile_pow_neg_10_val1) == float or type(voile_pow_neg_10_val1) == int, "Veuillez entrer une valeur entière ou flotante pour l'argument 1"
        assert type(voile_pow_neg_10_val1) == float or type(voile_pow_neg_10_val1) == int, "Veuillez entrer une valeur entière ou flotante pour l'argument 2"
        assert dataset == 'tr' or dataset == 'v' or dataset == 'tst', "%s n'est pas un des type tr, v ou tst autorisés"%(dataset)
        assert typ == 'norm' or typ == 'clip', "Méthode d'ajustement à l'intervalle inconnue : %s n'est pas clip ou norm"%(typ)
        
        batch_clean,batch_bruit = self.next_batch_original_img(dataset)
        
        if type(batch_clean) != np.ndarray:
            print("End of epoch")
            return None
        batch_noise = np.copy(batch_clean)
        #Calcul des facteurs de voile
        random_in_interval = np.random.rand(self.batch_size,3)*abs(voile_pow_neg_10_val1-voile_pow_neg_10_val2) +min(voile_pow_neg_10_val1,voile_pow_neg_10_val2)
        random_factor = 10**-(random_in_interval)
        # Calcul du bruit
        bruit = np.random.normal(μ_bruit,σ_bruit,size=(self.batch_size,self.taille_img,self.taille_img,3))
        #Applique au batch
        for img in range(self.batch_size):
            if type(batch_bruit[img]) != np.ndarray:
                for rgb_index in range(3):
                    batch_noise[img,:,:,rgb_index] *= random_factor[img,rgb_index]
                    batch_noise[img,:,:,rgb_index] += bruit[img,:,:,rgb_index]
            else:
                batch_noise[img,:,:,:] = batch_bruit[img]

        #Choix de la méthode d'ajustement à l'intervalle
        if typ == 'clip':
            batch_noise = np.clip(batch_noise,0,1)
        else:
            batch_noise = self.normalisation(batch_noise)
        
        return batch_clean,batch_noise
    def next_gan_batch(self,dataset):
        """Construction des batch d'entrainement suivant le type de modèle\n
        NB discriminateur : 1 annote une image bruitée\n
        En phase d'entrainement :\n
        - entree generateur partiellement bruitee et sortie débruitee\n
        - entree discriminateur partiellement bruitee et sortie telle que pour chaque img bruitee on ait un 1
        - entree gan partiellement bruitee et sortie telle que pour chaque img on ait 0 (toutes les images débruitées)\n
        En phase de validation ou test : Pour chaque ensemble (generateur, discriminateur, gan)\n
        - Entree débruitée et sortie correspondante\n
        - Entrée totalement bruitée et sortie correspondante\n
            #Arguments
                dataset: enum tr v ou tst pour entrainement, validation ou test
        """
        assert dataset == 'tr' or dataset == 'v' or dataset == 'tst', "%s n'est pas un des type tr, v ou tst autorisés"%(dataset)

        data_batch = self.next_batch_bruit_voile(voile_pow_neg_10_val1=0.4, voile_pow_neg_10_val2=1.3, μ_bruit=1.779, σ_bruit=1.779, typ='norm', dataset=dataset)
        if data_batch == None:
            return None
        batch_clean,batch_noise = data_batch
        if dataset == 'tr':
            #Generateur
            gen_input = np.copy(batch_clean)
            gen_output = batch_clean

            random_index = np.arange(0,self.batch_size)
            np.random.shuffle(random_index)
            nb_changt = np.random.randint(0,self.batch_size)

            random_index_to_chg = random_index[:nb_changt]
            for index_chgt in random_index_to_chg:
                gen_input[index_chgt,:,:,:] = batch_noise[index_chgt,:,:,:]
            #Discriminateur
            disc_input = gen_input
            disc_output = np.zeros(self.batch_size)

            for i in random_index_to_chg:
                disc_output[i] = 1
            #GAN
            gan_input = gen_input
            gan_output = np.zeros(self.batch_size)
            return gen_input, gen_output,disc_input,disc_output,gan_input,gan_output
        else:
            #Generateur
            gen_input_clean = batch_clean
            gen_output_clean = batch_clean
            gen_input_noise = batch_noise
            gen_output_noise = batch_clean
            #Discriminateur
            disc_input_clean = batch_clean
            disc_output_clean = np.zeros(self.batch_size)
            disc_input_noise = batch_noise
            disc_output_noise = np.ones(self.batch_size)
            #GAN
            gan_input_clean = batch_clean
            gan_output_clean = np.zeros(self.batch_size)
            gan_input_noise = batch_noise
            gan_output_noise = np.ones(self.batch_size)
            return gen_input_clean, gen_output_clean, gen_input_noise, gen_output_noise, disc_input_clean, disc_output_clean, disc_input_noise, disc_output_noise, gan_input_clean, gan_output_clean, gan_input_noise, gan_output_noise