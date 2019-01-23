from utils import concatenate, load_multiple_song, save_result
import matplotlib.pyplot as plt
import numpy as np
from CNN import conv_nn
import librosa


# Data parameters
namesong = 'LizNelson_Rainfall'
interval = 0  # Interval between kept samples
nb_time_samples = 64
sr = 22050

# Training parameters
batch_size = 16
epochs = 50

# Load the magnitude of the spectrogram
x_tot, y_tot, phase = load_multiple_song(interval, nb_time_samples)
#x_tot, y_tot, phase = load_data(namesong, interval, nb_time_samples)

# Drop the 1025 freq bin
x_tot = x_tot[:, :1024, :]
y_tot = y_tot[:, :1024, :]
phase = phase[:, :1024, :]

# (1024, 64, 190)
N_train = int(0.8 * x_tot.shape[0])

x_train1 = x_tot[:N_train, :, :]
y_train1 = y_tot[:N_train, :, :]
x_test1 = x_tot[N_train:, :, :]
y_test1 = y_tot[N_train:, :, :]
phase_test = phase[N_train:, :, :]

print('x_train : ', x_train1.shape)
print('x_test : ', x_test1.shape)
print('y_train : ', y_train1.shape)
print('y_test : ', y_test1.shape)

y_train = np.expand_dims(y_train1, axis=3)
x_train = np.expand_dims(x_train1, axis=3)
y_test = np.expand_dims(y_test1, axis=3)
x_test = np.expand_dims(x_test1, axis=3)


model = conv_nn()

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

y_pred = model.predict(x_test)

print("y_pred shape :", y_pred.shape)
y_pred = np.reshape(y_pred,(y_pred.shape[0],y_pred.shape[1],y_pred.shape[2]))
print("y_pred shape after reshape:", y_pred.shape)
y_pred_bin = 1*(y_pred > 0.5)


th = np.linspace(0.3, 0.7, num=21)
accs = np.zeros((21, 1))
for k in range(len(th)):
    y_pred_bin2 = y_pred_bin > th[k]
    acc = format(np.mean(y_pred_bin2 == y_train1))
    accs[k] = acc

plt.figure()
plt.plot(th, accs)
plt.show()
plt.xlabel('Threshold')
plt.ylabel('Accuracy')


y_fig = concatenate(y_test1)
y_fig2 = concatenate(y_pred)
y_fig3 = concatenate(y_pred_bin)
phase_test_tot = concatenate(phase_test)

plt.figure()
plt.subplot(311)
plt.imshow(y_fig, cmap='Greys')
plt.title("Target Mask")
plt.subplot(312)
plt.imshow(y_fig2, cmap='Greys')
plt.title("Estimated Mask with CNN")
plt.subplot(313)
plt.imshow(y_fig3, cmap='Greys')
plt.title("Estimated Mask with CNN with threshold")
plt.show()

print("Accuracy on testing set ={:.2f}".format(np.mean(y_pred_bin == y_test1)))

y_pred_train = model.predict(x_train)
y_pred_train = np.reshape(y_pred_train, (y_pred_train.shape[0], y_pred_train.shape[1], y_pred_train.shape[2]))
y_pred_bin2 = y_pred_train > 0.4
print("Accuracy on training set (th = 0.4) ={:.2f}".format(np.mean(y_pred_bin2 == y_train1)))
y_pred_bin2 = y_pred_train > 0.5
print("Accuracy on training set (th = 0.5) ={:.2f}".format(np.mean(y_pred_bin2 == y_train1)))
y_pred_bin2 = y_pred_train > 0.6
print("Accuracy on training set (th = 0.6) ={:.2f}".format(np.mean(y_pred_bin2 == y_train1)))


save_result(x_test1, y_fig, phase_test_tot, 'estim_true.wav')
save_result(x_test1, y_fig2, phase_test_tot, 'estim_cont.wav')
save_result(x_test1, y_fig3, phase_test_tot, 'estim_bin.wav')


# Save le mix test de base
xc = concatenate(x_test1)
spec_mix = xc

# IFFT
testsignal = librosa.istft(np.multiply(spec_mix, np.exp(1j*phase_test_tot)))
librosa.output.write_wav('xtest.wav', testsignal, sr)



