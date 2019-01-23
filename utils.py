import numpy as np
import librosa
import librosa.display


sr = 22050
interval = 0  # Interval between kept samples
nb_time_samples = 64


def compute_spectrogram(signal):
    mag, phase = librosa.magphase(librosa.stft(signal))
    #s = librosa.stft(signal)
    #librosa.display.specshow(p)
    return mag, phase


def creation_mask(spec_vocal, spec_instru):
    # Creates a mask of 1 (voice) and 0 (instru)
    mask = 1*(spec_vocal > spec_instru)
    return mask


def continuous_mask(spec_vocal, spec_instru):
    # Creates a mask of 1 (voice) and 0 (instru)
    mask = np.zeros((spec_vocal.shape[0], spec_vocal.shape[1]))
    for i in range(spec_vocal.shape[0]):
        for j in range(spec_vocal.shape[1]):
            if spec_vocal[i][j] + spec_instru[i][j] != 0:
                mask[i][j] = spec_vocal[i][j]/(spec_instru[i][j]+spec_vocal[i][j])
    return mask


def apply_mask(mask, mix):
    # Compute spectrogram
    spec_mix = compute_spectrogram(mix)
    # Apply mask
    spec_vocal = np.multiply(mask,spec_mix)
    spec_instru = np.multiply(1-mask,spec_mix)
    # IFFT
    vocal = librosa.istft(spec_vocal, length=len(mix))
    instru = librosa.istft(spec_instru, length=len(mix))
    return vocal, instru


def save_wav(signal, namesong, namefile):
    librosa.output.write_wav("./MedleyDB_sample/Audio/" + namesong + "/" + namesong + "_" + namefile + ".wav", signal, sr)
    print(namesong + "_" + namefile + " saved.")


def load_data(namesong, interval, nb_time_samples):

    # Load data
    vocal_mix, sr = librosa.load("./MedleyDB_sample/Audio/" + namesong + "/" + namesong + "_VOCALMIX.wav")
    instru_mix, sr = librosa.load("./MedleyDB_sample/Audio/" + namesong + "/" + namesong + "_INSTRUMIX.wav")
    mix, sr = librosa.load("./MedleyDB_sample/Audio/" + namesong + "/" + namesong + "_MIX.wav")

    spectro_instru, zz = compute_spectrogram(instru_mix)
    spectro_vocal, zz = compute_spectrogram(vocal_mix)
    spectro_mix, phase = compute_spectrogram(mix)

    total_time_samples = spectro_mix.shape[1]
    nb_samples_per_song = int(round(total_time_samples/(nb_time_samples+interval)))

    x = 0
    x_tot = np.zeros((nb_samples_per_song,spectro_mix.shape[0], nb_time_samples))
    y_tot = np.zeros((nb_samples_per_song,spectro_mix.shape[0], nb_time_samples))
    phases = np.zeros((nb_samples_per_song, spectro_mix.shape[0], nb_time_samples),dtype=complex)

    full_mask = creation_mask(spectro_vocal, spectro_instru)

    for j in range(nb_samples_per_song-1):
        x = x + interval + nb_time_samples
        y_tot[j, :, :] = full_mask[:, x-nb_time_samples:x]
        x_tot[j, :, :] = spectro_mix[:, x-nb_time_samples:x]
        phases[j, :, :] = phase[:, x-nb_time_samples:x]

    # y_train shape : (sample,1025,64)
    return x_tot, y_tot, phases


def concatenate(y):
    for j in range(y.shape[0]):
        if j == 0:
            y_conc = y[j, :, :]
        else:
            a = y[j, :, :]
            y_conc = np.concatenate((y_conc, a), axis=1)
    return y_conc


def load_multiple_song(interval, nb_time_samples):
    # Load songs from list 'train.txt', prepare input data and

    # interval : interval between samples in nb of time steps
    # nb_time_samples :
    with open('train.txt') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for i in range(len(content)):
        namesong = content[i]
        print(namesong)
        vocal_mix, sr = librosa.load("./MedleyDB_sample/Audio/" + namesong + "/" + namesong + "_VOCALMIX.wav")
        instru_mix, sr = librosa.load("./MedleyDB_sample/Audio/" + namesong + "/" + namesong + "_INSTRUMIX.wav")
        mix, sr = librosa.load("./MedleyDB_sample/Audio/" + namesong + "/" + namesong + "_MIX.wav")
        spectro_instru, pp = compute_spectrogram(instru_mix)
        spectro_vocal, pp = compute_spectrogram(vocal_mix)
        x = 0
        if i == 0:
            spectro_mix, phase = compute_spectrogram(mix)
            total_time_samples = spectro_mix.shape[1]
            nb_samples_per_song = int(round(total_time_samples / (nb_time_samples + interval)))

            full_mask = creation_mask(spectro_vocal, spectro_instru)
            i = 1
        else:
            a, p = compute_spectrogram(mix)
            b = a.shape[1]
            d = int(round(b / (nb_time_samples + interval)))
            c = creation_mask(spectro_vocal, spectro_instru)
            spectro_mix = np.concatenate((spectro_mix, a), axis=1)
            full_mask = np.concatenate((full_mask, c), axis=1)
            phase = np.concatenate((phase, p), axis=1)
            nb_samples_per_song = nb_samples_per_song + d

    x_train = np.zeros((nb_samples_per_song, spectro_mix.shape[0], nb_time_samples))
    y_train = np.zeros((nb_samples_per_song, spectro_mix.shape[0], nb_time_samples))
    phases = np.zeros((nb_samples_per_song, spectro_mix.shape[0], nb_time_samples),dtype='complex')
    for j in range(nb_samples_per_song - 1):
        x = x + interval + nb_time_samples
        y_train[j, :, :] = full_mask[:, x - nb_time_samples:x]
        x_train[j, :, :] = spectro_mix[:, x - nb_time_samples:x]
        phases[j, :, :] = phase[:, x - nb_time_samples:x]

    return x_train, y_train, phases


def save_result(x, mask, phase, filename):
    # Apply mask and save vocal song as .wav file
    xc = concatenate(x)
    spec_mix = xc

    # Apply mask
    spec_vocal = np.multiply(mask, spec_mix)
    spec_instru = np.multiply(1 - mask, spec_mix)

    # IFFT
    vocal = librosa.istft(np.multiply(spec_vocal, np.exp(1j*phase)))
    instru = librosa.istft(np.multiply(spec_instru, np.exp(1j*phase)))
    librosa.output.write_wav(filename, vocal, sr)


def compare_errors():
    # Compare 3 masks
    namesong = 'MusicDelta_Country2'
    vocal_mix, sr = librosa.load("./MedleyDB_sample/Audio/" + namesong + "/" + namesong + "_VOCALMIX.wav")
    instru_mix, sr = librosa.load("./MedleyDB_sample/Audio/" + namesong + "/" + namesong + "_INSTRUMIX.wav")
    mix, sr = librosa.load("./MedleyDB_sample/Audio/" + namesong + "/" + namesong + "_MIX.wav")


    spec_vocal = compute_spectrogram(vocal_mix)
    spec_instru = compute_spectrogram(instru_mix)
    # Creates mask
    mask_bin = creation_mask(spec_vocal, spec_instru)
    mask = continuous_mask(spec_vocal, spec_instru)


    # Separate
    vocal_recon, instru_recon = apply_mask(mask, mix)
    vocal_recon2, instru_recon2 = apply_mask(mask_bin, mix)
    #save_wav(vocal_recon, namesong='LizNelson_Rainfall', namefile='VOCAL_RECONcont')
    #save_wav(instru_recon, namesong='LizNelson_Rainfall', namefile='INSTRU_RECONcont')

    error_cont1 = error(vocal_recon, vocal_mix)
    error_cont2 = error(instru_recon, instru_mix)
    error_bin1 = error(vocal_recon2, vocal_mix)
    error_bin2 = error(instru_recon2, instru_mix)

    S_full, phase = librosa.magphase(librosa.stft(mix))


    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))

    S_filter = np.minimum(S_full, S_filter)
    margin_i, margin_v = 2, 10
    power = 2
    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)
    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)
    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    # IFFT
    vocal_recon3 = librosa.istft(S_foreground)
    instru_recon3 = librosa.istft(S_background)

    error31 = error(vocal_recon3, vocal_mix)
    error32 = error(instru_recon3, instru_mix)
    print("Error Continuous mask (vocal):", error_cont1)
    print("Error Continuous mask (instru):", error_cont2)
    print("Error Binary mask (vocal):", error_bin1)
    print("Error Binary mask (instru):", error_bin2)
    print("Error Soft mask (vocal):", error31)
    print("Error Soft mask (instru):", error32)


def error(y_pred, y_true):
    if y_pred.shape != y_true.shape:
        print ("Must be same length")
        return 0
    else:
        err = np.mean(abs(y_pred-y_true))
        return err