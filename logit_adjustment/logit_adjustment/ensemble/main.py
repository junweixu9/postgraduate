from tensorflow.keras import backend as backend
from tensorflow.keras.utils import to_categorical
from sca_metrics import SCAMetrics
from loaddataset import LoadDatasets
from datasets import SCADatasets
from neural_networks import NeuralNetwork
from datetime import datetime
import numpy as np
import random
import time
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt

class EnsembleAES:

    def __init__(self):
        self.number_of_models = 50
        self.number_of_best_models = [10, 10, 10, 10]
        self.ge_all_validation = []
        self.ge_all_attack = []
        self.sr_all_validation = []
        self.sr_all_attack = []
        self.k_ps_all = []
        self.ge_ensemble = None
        self.ge_ensemble_best_models = None
        self.ge_ensemble_best_models0 = None  # added
        self.ge_ensemble_best_models1 = None  # added
        self.ge_ensemble_best_models2 = None  # added
        self.ge_ensemble_best_models3 = None  # added
        self.ge_best_model_validation = None
        self.ge_best_model_attack = None
        self.sr_ensemble = None
        self.sr_ensemble_best_models = None
        self.sr_ensemble_best_models0 = None  # added
        self.sr_ensemble_best_models1 = None  # added
        self.sr_ensemble_best_models2 = None  # added
        self.sr_ensemble_best_models3 = None  # added
        self.sr_best_model_validation = None
        self.sr_best_model_attack = None
        self.target_dataset = None
        self.l_model = None
        self.target_byte = None
        self.classes = None
        self.epochs = None
        self.mini_batch = None

    def set_dataset(self, target):  # function for setting up dataset - ASCAD | ches_ctf
        self.target_dataset = target

    def set_leakage_model(self, leakage_model):  # function for setting up leakage model
        self.l_model = leakage_model
        if leakage_model == "HW":
            self.classes = 9
        else:
            self.classes = 256

    def set_target_byte(self, target_byte):  # target byte to attack: most of the time, this is 3
        self.target_byte = target_byte

    def set_epochs(self, epochs):  # we set no of epochs for our training model
        self.epochs = epochs

    def set_mini_batch(self, mini_batch):  # not sure of mini_batch. I guess it means batch
        self.mini_batch = mini_batch

    def __add_if_one(self, value):  # don't know when this will be useful
        return 1 if value == 1 else 0

    def get_best_models(self, n_models, result_models_validation, n_traces):
        result_number_of_traces_val = []  # best models is ranked by guessing entropy traces (when it's equal 1)
        for model_index in range(n_models):
            if result_models_validation[model_index][n_traces - 1] == 1:  # checking for the best model starting from the last (min) trace
                for index in range(n_traces - 1, -1, -1):  # decrementing and moving up the trace
                    if result_models_validation[model_index][index] != 1:
                        result_number_of_traces_val.append(
                            [result_models_validation[model_index][n_traces - 1], index + 1,
                             model_index])
                        break
            else:
                result_number_of_traces_val.append(
                    [result_models_validation[model_index][n_traces - 1], n_traces,
                     model_index])

        sorted_models = sorted(result_number_of_traces_val, key=lambda l: l[:])  # sort by

        list_of_best_models = []
        for model_index in range(n_models):
            list_of_best_models.append(sorted_models[model_index][2])

        return list_of_best_models


    def run_cnn(self, X_profiling, Y_profiling, X_validation, Y_validation, X_attack, Y_attack, plt_validation,
                plt_attack, params,
                step, fraction):
        X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
        X_validation = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], 1))
        X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))

        mini_batch = random.randrange(500, 1000, 100)
        learning_rate = random.uniform(0.0001, 0.001)
        activation = ['relu', 'tanh', 'elu', 'selu'][random.randint(0, 3)]
        dense_layers = random.randrange(2, 8, 1)
        neurons = random.randrange(500, 800, 100)
        conv_layers = random.randrange(1, 2, 1)
        filters = random.randrange(8, 32, 4)
        kernel_size = random.randrange(10, 20, 2)
        stride = random.randrange(5, 10, 5)

        model = NeuralNetwork().cnn_random(self.classes, params["number_of_samples"], activation, neurons, conv_layers,
                                           filters,
                                           kernel_size, stride, dense_layers, learning_rate)
        model.fit(
            x=X_profiling,
            y=Y_profiling,
            batch_size=self.mini_batch,
            verbose=1,
            epochs=self.epochs,
            shuffle=True,
            validation_data=(X_validation, Y_validation),
            callbacks=[])
        # 100 is for number of runs
        ge_validation, sr_validation, kp_krs = SCAMetrics().ge_and_sr(100, model, params, self.l_model,
                                                                      self.target_byte,
                                                                      X_validation, plt_validation,
                                                                      step, fraction)
        ge_attack, sr_attack, _ = SCAMetrics().ge_and_sr(100, model, params, self.l_model, self.target_byte, X_attack,
                                                         plt_attack, step,
                                                         fraction)

        backend.clear_session()

        return ge_validation, ge_attack, sr_validation, sr_attack, kp_krs

    def compute_ensembles(self, kr_nt, correct_key):
        # nm = number_of_models
        list_of_best_models = self.get_best_models(self.number_of_models, self.ge_all_validation, kr_nt)  # i added

        self.ge_best_model_validation = self.ge_all_validation[list_of_best_models[0]]
        self.ge_best_model_attack = self.ge_all_attack[list_of_best_models[0]]
        self.sr_best_model_validation = self.sr_all_validation[list_of_best_models[0]]
        self.sr_best_model_attack = self.sr_all_attack[list_of_best_models[0]]

        for i in range(4):  # added

            kr_ensemble = np.zeros(kr_nt)
            krs_ensemble = np.zeros((100, kr_nt))
            kr_ensemble_best_models = np.zeros(kr_nt)
            krs_ensemble_best_models = np.zeros((100, kr_nt))

            for run in range(100):

                key_p_ensemble = np.zeros(256)
                key_p_ensemble_best_models = np.zeros(256)

                for index in range(kr_nt):
                    for model_index in range(self.number_of_models):
                        key_p_ensemble += np.log(self.k_ps_all[list_of_best_models[model_index]][run][index] + 1e-36)
                    for model_index in range(self.number_of_best_models[i]):  # i added
                        key_p_ensemble_best_models += np.log(
                            self.k_ps_all[list_of_best_models[model_index]][run][index] + 1e-36)

                    key_p_ensemble_sorted = np.argsort(key_p_ensemble)[::-1]
                    key_p_ensemble_best_models_sorted = np.argsort(key_p_ensemble_best_models)[::-1]

                    kr_position = list(key_p_ensemble_sorted).index(correct_key) + 1
                    kr_ensemble[index] += kr_position
                    krs_ensemble[run][index] = kr_position

                    kr_position = list(key_p_ensemble_best_models_sorted).index(correct_key) + 1
                    kr_ensemble_best_models[index] += kr_position
                    krs_ensemble_best_models[run][index] = kr_position

                print("Run {} - GE {} models: {} | GE {} models: {} | ".format(run, self.number_of_models,
                                                                               int(kr_ensemble[kr_nt - 1] / (run + 1)),
                                                                               self.number_of_best_models[i],  # i added
                                                                               int(kr_ensemble_best_models[
                                                                                       kr_nt - 1] / (run + 1))))

            ge_ensemble = kr_ensemble / 100
            ge_ensemble_best_models = kr_ensemble_best_models / 100

            sr_ensemble = np.zeros(kr_nt)
            sr_ensemble_best_models = np.zeros(kr_nt)

            for index in range(kr_nt):
                for run in range(100):
                    sr_ensemble[index] += self.__add_if_one(krs_ensemble[run][index])
                    sr_ensemble_best_models[index] += self.__add_if_one(krs_ensemble_best_models[run][index])
            if (i == 0):
                ge_ensemble_best_models0 = ge_ensemble_best_models
                sr_ensemble_best_models0 = sr_ensemble_best_models
            if (i == 1):
                ge_ensemble_best_models1 = ge_ensemble_best_models
                sr_ensemble_best_models1 = sr_ensemble_best_models
            if (i == 2):
                ge_ensemble_best_models2 = ge_ensemble_best_models
                sr_ensemble_best_models2 = sr_ensemble_best_models
            if (i == 3):
                ge_ensemble_best_models3 = ge_ensemble_best_models
                sr_ensemble_best_models3 = sr_ensemble_best_models
                ge_ensemble_best_models = ge_ensemble_best_models0
                sr_ensemble_best_models = sr_ensemble_best_models0

        # return ge_ensemble, ge_ensemble_best_models, sr_ensemble/100, sr_ensemble_best_models/100
        # next line added
        return ge_ensemble, ge_ensemble_best_models, ge_ensemble_best_models1, ge_ensemble_best_models2, ge_ensemble_best_models3, sr_ensemble / 100, sr_ensemble_best_models / 100, sr_ensemble_best_models1 / 100, sr_ensemble_best_models2 / 100, sr_ensemble_best_models3 / 100

    def create_z_score_norm(self, dataset):
        z_score_mean = np.mean(dataset, axis=0)
        z_score_std = np.std(dataset, axis=0)
        return z_score_mean, z_score_std

    def apply_z_score_norm(self, dataset, z_score_mean, z_score_std):
        for index in range(len(dataset)):
            dataset[index] = (dataset[index] - z_score_mean) / z_score_std

    def run_ensemble(self, number_of_models, number_of_best_models):

        self.number_of_models = number_of_models
        self.number_of_best_models = number_of_best_models

        target_params = SCADatasets().get_trace_set(self.target_dataset)

        # root_folder = "D:/traces/"
        # root_folder = '/content/drive/MyDrive/Colab Notebooks/QEResearch/ches_ctf/
        root_folder = "C:/workspace/code/logit_adjustment/logit_adjustment/ASCAD_data/ASCAD_databases/"

        (X_profiling, Y_profiling), (X_validation, Y_validation), (X_attack, Y_attack), (
            _, plt_validation, plt_attack) = LoadDatasets().load_dataset(
            root_folder + target_params["file"], target_params["n_profiling"], target_params["n_attack"],
            self.target_byte, self.l_model)

        print("train{}".format(len(X_profiling)))
        print("val{}".format(len(X_validation)))
        print("att{}".format(len(X_attack)))


        # oversample = SMOTE()
        # X_profilinga, Y_profilinga = oversample.fit_resample(X_profiling, Y_profiling)
        # counter = Counter(Y_profilinga)  # added switch betweenn Y_profiling & Y_profilinga
        # for k, v in counter.items():  # added
        #     per = v / len(Y_profilinga) * 100  # added
        #     print('Class=%d, n=%d (%.3f%%)' % (k, v, per))  # added
        #
        # X_profiling, Y_profiling = X_profilinga, Y_profilinga
        # plot the distribution

        # normalize with z-score
        z_score_mean, z_score_std = self.create_z_score_norm(X_profiling)
        self.apply_z_score_norm(X_profiling, z_score_mean, z_score_std)
        self.apply_z_score_norm(X_validation, z_score_mean, z_score_std)
        self.apply_z_score_norm(X_attack, z_score_mean, z_score_std)

        # convert labels to categorical labels
        Y_profiling = to_categorical(Y_profiling, num_classes=self.classes)
        Y_validation = to_categorical(Y_validation, num_classes=self.classes)
        Y_attack = to_categorical(Y_attack, num_classes=self.classes)

        X_profiling = X_profiling.astype('float32')
        X_validation = X_validation.astype('float32')
        X_attack = X_attack.astype('float32')

        kr_step = 10  # key rank processed for each kr_step traces
        kr_fraction = 1  # validation or attack sets are divided by kr_fraction before computing key rank

        self.ge_all_validation = []
        self.sr_all_validation = []
        self.ge_all_attack = []
        self.k_ps_all = []

        kr_nt = int(len(X_validation) / (kr_step * kr_fraction))

        # train random CNN
        for model_index in range(self.number_of_models):
            ge_validation, ge_attack, sr_validation, sr_attack, kp_krs = self.run_cnn(X_profiling, Y_profiling,
                                                                                      X_validation, Y_validation,
                                                                                      X_attack, Y_attack,
                                                                                      plt_validation, plt_attack,
                                                                                      target_params, kr_step,
                                                                                      kr_fraction)
            self.ge_all_validation.append(ge_validation)
            self.ge_all_attack.append(ge_attack)
            self.sr_all_validation.append(sr_validation)
            self.sr_all_attack.append(sr_attack)
            self.k_ps_all.append(kp_krs)  # kr_nt = int(len(X_validation) / (kr_step * kr_fraction))

        ge_ensemble, ge_ensemble_best_models, ge_ensemble_best_models1, ge_ensemble_best_models2, ge_ensemble_best_models3, sr_ensemble, sr_ensemble_best_models, sr_ensemble_best_models1, sr_ensemble_best_models2, sr_ensemble_best_models3 = self.compute_ensembles(
            kr_nt,
            target_params["good_key"])

        self.ge_ensemble = ge_ensemble
        self.ge_ensemble_best_models = ge_ensemble_best_models
        self.ge_ensemble_best_models1 = ge_ensemble_best_models1  # added
        self.ge_ensemble_best_models2 = ge_ensemble_best_models2  # added
        self.ge_ensemble_best_models3 = ge_ensemble_best_models3  # added

        self.sr_ensemble = sr_ensemble
        self.sr_ensemble_best_models = sr_ensemble_best_models
        self.sr_ensemble_best_models1 = sr_ensemble_best_models1  # added
        self.sr_ensemble_best_models2 = sr_ensemble_best_models2  # added
        self.sr_ensemble_best_models3 = sr_ensemble_best_models3  # added

        self.compute_ensembles(
            kr_nt,
            target_params["good_key"])


    def get_ge_ensemble(self):
        return self.ge_ensemble

    def get_ge_ensemble_best_models(self):
        return self.ge_ensemble_best_models

    def get_ge_ensemble_best_models1(self):  # added
        return self.ge_ensemble_best_models1

    def get_ge_ensemble_best_models2(self):  # added
        return self.ge_ensemble_best_models2

    def get_ge_ensemble_best_models3(self):  # added
        return self.ge_ensemble_best_models3

    def get_ge_best_model_validation(self):
        return self.ge_best_model_validation

    def get_ge_best_model_attack(self):
        return self.ge_best_model_attack

    def get_sr_ensemble(self):
        return self.sr_ensemble

    def get_sr_ensemble_best_models1(self):  # added
        return self.sr_ensemble_best_models1

    def get_sr_ensemble_best_models2(self):  # added
        return self.sr_ensemble_best_models2

    def get_sr_ensemble_best_models3(self):  # added
        return self.sr_ensemble_best_models3

    def get_sr_ensemble_best_models(self):
        return self.sr_ensemble_best_models

    def get_sr_best_model_validation(self):
        return self.sr_best_model_validation

    def get_sr_best_model_attack(self):
        return self.sr_best_model_attack


ensemble_aes = EnsembleAES()
#ensemble_aes.set_dataset("ches_ctf")  # "ascad_fixed_key", "ascad_random_key" or "ches_ctf"
ensemble_aes.set_dataset("ascad_fixed_key")  #changed from ches_ctf
ensemble_aes.set_leakage_model("HW")
ensemble_aes.set_target_byte(2)
ensemble_aes.set_mini_batch(400)
ensemble_aes.set_epochs(10)
ensemble_aes.run_ensemble(number_of_models=50, number_of_best_models=[1,5,10,20])

plt.rc('figure', figsize=(15, 8))

# plotting GE and SR
plt.subplot(1, 2, 1)
#plt.plot(ensemble_aes.get_ge_best_model_validation(), label="GE best validation")
#plt.plot(ensemble_aes.get_ge_best_model_attack(), label="GE best attack")
plt.plot(ensemble_aes.get_ge_ensemble(), label="GE Ensemble All Models (50)")
plt.plot(ensemble_aes.get_ge_ensemble_best_models(), label="GE Ensemble Best Models (1)")
plt.plot(ensemble_aes.get_ge_ensemble_best_models1(), label="GE Ensemble Best Models1 (10)")
plt.plot(ensemble_aes.get_ge_ensemble_best_models2(), label="GE Ensemble Best Models2 (20)")
plt.plot(ensemble_aes.get_ge_ensemble_best_models3(), label="GE Ensemble Best Models3 (45)")

plt.xlabel("Traces")
plt.ylabel("Guessing Entropy")
plt.legend()
plt.grid()
plt.subplot(1, 2, 2)
#plt.plot(ensemble_aes.get_sr_best_model_validation(), label="SR best validation")
#plt.plot(ensemble_aes.get_sr_best_model_attack(), label="SR best attack")
plt.plot(ensemble_aes.get_sr_ensemble(), label="SR Ensemble All Models (50)")
plt.plot(ensemble_aes.get_sr_ensemble_best_models(), label="SR Ensemble Best Models (1)")
plt.plot(ensemble_aes.get_sr_ensemble_best_models1(), label="SR Ensemble Best Models1 (10)")
plt.plot(ensemble_aes.get_sr_ensemble_best_models2(), label="SR Ensemble Best Models2 (20)")
plt.plot(ensemble_aes.get_sr_ensemble_best_models3(), label="SR Ensemble Best Models3 (45)")
plt.xlabel("Traces")
plt.ylabel("Success Rate")
plt.legend()
plt.grid()
#plt.show()
plt.savefig('CNN_ASCAD_Ensembles_Tuning.png')
plt.savefig('CNN_ASCAD_Ensembles_Tuning.pdf')

