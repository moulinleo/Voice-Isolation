from CNN import relu_advanced
from keras.models import model_from_json
from utils import compute_spectrogram, concatenate
import librosa
import numpy as np


def import_model():
    '''
    Import pretrained CNN model
    '''
    print("Importing model..")
    json_file = open('CNN.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects={'relu_advanced': relu_advanced})
    # load weights into new model
    model.load_weights("model.h5")
    return model


def prepare_inputdata(mix):
    '''
    Prepare the input data to feed the neural network. Compute the spectrogram and divide it into chunks of data.
    '''
    interval = 0
    nb_time_samples = 64

    print('Preparing data..')
    spectro_mix, phase = compute_spectrogram(mix)
    total_time_samples = spectro_mix.shape[1]
    nb_samples_per_song = int(round(total_time_samples / (nb_time_samples + interval)))

    x_tot = np.zeros((nb_samples_per_song, spectro_mix.shape[0], nb_time_samples))
    phases = np.zeros((nb_samples_per_song, spectro_mix.shape[0], nb_time_samples), dtype=complex)
    x = 0
    for j in range(nb_samples_per_song - 1):
        x = x + interval + nb_time_samples
        x_tot[j, :, :] = spectro_mix[:, x - nb_time_samples:x]
        phases[j, :, :] = phase[:, x-nb_time_samples:x]

    # Drop the 1025 freq bin
    x_tot = x_tot[:, :1024, :]
    phases = phases[:, :1024, :]
    x_tot = np.expand_dims(x_tot, axis=3)

    return x_tot, phases


def prepare_outputdata(mask_pred, spectro_mix, phase, th):
    '''
    Apply the estimated mask to the input spectrogram and make the IFFT to produce the final vocal-only song
    '''

    print('Preparing output data..')
    # Apply threshold
    mask_pred = 1 * (mask_pred > th)

    # Remove last dimension
    mask_pred = np.reshape(mask_pred, (mask_pred.shape[0], mask_pred.shape[1], mask_pred.shape[2]))

    # Concatenate
    mask = concatenate(mask_pred)
    phase = concatenate(phase)
    spectro_mix = np.reshape(spectro_mix, (spectro_mix.shape[0], spectro_mix.shape[1], spectro_mix.shape[2]))
    spec_mix = concatenate(spectro_mix)

    # Apply mask
    spec_vocal = np.multiply(mask, spec_mix)
    spec_instru = np.multiply(1 - mask, spec_mix)

    # IFFT
    vocal_mix = librosa.istft(np.multiply(spec_vocal, np.exp(1j * phase)))
    instru_mix = librosa.istft(np.multiply(spec_instru, np.exp(1j * phase)))

    return vocal_mix, instru_mix