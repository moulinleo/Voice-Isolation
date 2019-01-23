from utils import load_data
import matplotlib.pyplot as plt
import numpy as np


namesong = "LizNelson_Rainfall"
interval = 0  # Interval between kept samples
nb_time_samples = 64
sr = 22050


def show_mask(y):
    for j in range(y.shape[0]):
        if j == 0:
            y_fig = y[j, :, :, 0]
        else:
            a = y[j, :, :, 0]
            y_fig = np.concatenate((y_fig, a), axis=1)
    plt.figure()
    plt.imshow(y_fig, cmap='Greys')
    plt.title("Target Mask")
    plt.show()


x_tot, y_tot, total_time_samples = load_data(namesong, interval, nb_time_samples)


# Drop the 1025 freq bin
x_tot = x_tot[:1024, :]
y_tot = y_tot[:1024, :]

x_tot = np.reshape(x_tot, (x_tot.shape[2], x_tot.shape[0], x_tot.shape[1], 1))
y_tot = np.reshape(y_tot, (y_tot.shape[2], y_tot.shape[0], y_tot.shape[1], 1))
N_train = int(0.8 * x_tot.shape[0])

print('y_tot :', y_tot.shape)

x_train = x_tot[:N_train, :]
y_train = y_tot[:N_train, :]
x_test = x_tot[N_train:, :]
y_test = y_tot[N_train:, :]

print('x_train : ', x_train.shape)
print('x_test : ', x_test.shape)
print('y_train : ', y_train.shape)
print('y_test : ', y_test.shape)

show_mask(y_test)
show_mask(y_tot)

