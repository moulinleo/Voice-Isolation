import yaml
import librosa
import numpy as np
import os


sr = 22050
namesong = 'LizNelson_Rainfall'


def merge_stems(namesong):
    # Merge all instrumental stems into 1 mix and all vocal stems into 1 mix

    stream = open("./MedleyDB_sample/Audio/" + namesong + "/" + namesong + "_METADATA.yaml", "r")

    docs = yaml.load_all(stream)
    list_vocal = []
    list_instru = []
    for doc in docs:
        for k, v in doc.items():
            if k == 'stems':
                for cle, valeur in v.items():
                    for items in valeur.items():
                        if items[0] == 'instrument':
                            if "singer" in items[1]:
                                y, sr = librosa.load("./MedleyDB_sample/Audio/" + namesong + "/" + namesong + "_STEMS/" + namesong + "_STEM_" + cle[1:3] + ".wav")
                                if max(abs(y)) != 0:
                                    y = y / max(abs(y))
                                list_vocal.append(y)
                            else:
                                y, sr = librosa.load("./MedleyDB_sample/Audio/" + namesong + "/" + namesong + "_STEMS/" + namesong + "_STEM_" + cle[1:3] + ".wav")
                                if max(abs(y)) != 0:
                                    y = y / max(abs(y))
                                list_instru.append(y)

    vocal_sum = np.zeros(len(y))
    instru_sum = np.zeros(len(y))

    for i in range(len(list_vocal)):
        vocal_sum += list_vocal[i]
    for j in range(len(list_instru)):
        instru_sum += list_instru[j]

    if max(abs(vocal_sum)) != 0:
        vocal_sum = vocal_sum / max(abs(vocal_sum))
    if max(abs(instru_sum)) != 0:
        instru_sum = instru_sum / max(abs(instru_sum))
    mix_sum = np.zeros(len(y))
    for k in range(len(y)):
        mix_sum[k] = vocal_sum[k] + instru_sum[k]

    librosa.output.write_wav("./MedleyDB_sample/Audio/" + namesong + "/" + namesong + "_VOCALMIX.wav", vocal_sum, sr)
    librosa.output.write_wav("./MedleyDB_sample/Audio/" + namesong + "/" + namesong + "_INSTRUMIX.wav", instru_sum, sr)
    librosa.output.write_wav("./MedleyDB_sample/Audio/" + namesong + "/" + namesong + "_MIX_SUM.wav", mix_sum, sr)

    return 0


def show_intrumental():
    a = 0
    for namesong in os.listdir("./MedleyDB_sample/Audio/"):
        stream = open("./MedleyDB_sample/Audio/" + namesong + "/" + namesong + "_METADATA.yaml", "r")
        docs = yaml.load_all(stream)
        for doc in docs:
            for k, v in doc.items():
                if k == "instrumental":
                    if v == "yes":
                        print(namesong)
                #if k == "genre":
                    #if v == "Singer/Songwriter":
                        #print(namesong, v)

#show_intrumental()