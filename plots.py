import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os


def plot_audio():
    # Plot waveform and spectrogram
    y, sr = librosa.load('LizNelson_Rainfall.wav', duration=10)
    plt.figure()
    plt.subplot(2, 1, 1)
    librosa.display.waveplot(y)
    plt.title('Audio signal')
    plt.subplot(2, 1, 2)
    D = librosa.stft(y)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
    plt.title('Phase')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()


def plot_spectro():
    # Plot magnitude and phase spectrogram
    y, sr = librosa.load('LizNelson_Rainfall.wav', duration=10)
    plt.figure()
    mag, phase = librosa.magphase(librosa.stft(y))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(mag, ref=np.max), y_axis='log', x_axis='time')

    plt.subplot(2, 1, 2)
    librosa.display.specshow(phase)
    librosa.display.specshow(librosa.amplitude_to_db(phase, ref=np.max), y_axis='log', x_axis='time')
    plt.show()


def draw_pie():
    # Camembert drawer
    genres = {}
    for namesong in os.listdir("./MedleyDB_sample/Audio/"):
        stream = open("./MedleyDB_sample/Audio/" + namesong + "/" + namesong + "_METADATA.yaml", "r")
        docs = yaml.load_all(stream)
        for doc in docs:
            for k, v in doc.items():
                if k == "genre":
                    if v in genres:
                        genres[v] = genres[v] + 1
                    else:
                        genres[v] = 1
    labels = genres.keys()
    fracs = genres.values()
    explode = (0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02)
    plt.pie(fracs, explode=explode, labels=labels, autopct='%.0f%%')
    plt.show()


def draw_pie2():
    # Camembert drawer #2
    ct = 0
    for namesong in os.listdir("./MedleyDB_sample/Audio/"):
        stream = open("./MedleyDB_sample/Audio/" + namesong + "/" + namesong + "_METADATA.yaml", "r")
        docs = yaml.load_all(stream)
        for doc in docs:
            for k, v in doc.items():
                if k == "instrumental":
                    if v == "no":
                        ct = ct + 1
    ct2 = 122 - ct
    labels = ['Instrumental', 'Vocal+Instrumental']
    fracs = [ct, ct2]
    explode = (0.02, 0.02)
    # Make square figures and axes
    plt.pie(fracs, explode=explode, labels=labels, autopct='%.0f%%')
    plt.show()


plot_spectro()
draw_pie()
draw_pie2()