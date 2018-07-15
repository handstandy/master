import numpy as np
import random
import copy
import time
import matplotlib.pyplot as plt


def transfer_matrix(matrix, horizontal_resistors, vertical_resistors):
    v_matrix = np.zeros((N, N))
    i = np.arange(0, (N - 1))
    j = np.arange(0, N)

    matrix[i, i] += 1.0 / horizontal_resistors[i]
    matrix[i, i + 1] += -1.0 / horizontal_resistors[i]
    matrix[i + 1, i] += -1.0 / horizontal_resistors[i]
    matrix[i + 1, i + 1] += 1.0 / horizontal_resistors[i]

    for l in range(L):
        v_matrix[j, j] = 1.0 / vertical_resistors[l * N + j]
        matrix = matrix.dot(np.linalg.solve(v_matrix + matrix, v_matrix))

        matrix[i, i] += 1.0 / horizontal_resistors[l * (N - 1) + i + (N - 1)]
        matrix[i, i + 1] += -1.0 / horizontal_resistors[l * (N - 1) + i + (N - 1)]
        matrix[i + 1, i] += -1.0 / horizontal_resistors[l * (N - 1) + i + (N - 1)]
        matrix[i + 1, i + 1] += 1.0 / horizontal_resistors[l * (N - 1) + i + (N - 1)]

    return matrix, horizontal_resistors, vertical_resistors

def metropolis(measured_matrix, max_time):
    initial_energy = (N * L) ** 2
    temperature = 10
    k = 0
    t = 0

    resistors_h = np.random.uniform(low, high, (N - 1) * L + (N - 1))
    resistors_v = np.random.uniform(low, high, (N * L))

    best_resistors_h = copy.copy(resistors_h)
    best_resistors_v = copy.copy(resistors_v)

    while t < max_time:
        random_resistor_value = random.uniform(low, high)
        random_element = random.randrange(0, (2 * N - 1) * L + (N - 1))
        if random_element >= ((N - 1) * L + (N - 1)):
            resistors_v[random_element - ((N - 1) * L + (N - 1))] = random_resistor_value
        else:
            resistors_h[random_element] = random_resistor_value


        trial_matrix = np.zeros((N, N))
        trial_matrix, resistors_h,resistors_v = transfer_matrix(trial_matrix, resistors_h, resistors_v)

        energy = np.linalg.norm(measured_matrix - trial_matrix) ** 2
        p = np.exp(-(energy - initial_energy) / temperature)
        alpha = min(1, p)

        probability_number = random.random()

        if alpha > probability_number:
            initial_energy = energy
            best_resistors_h = copy.copy(resistors_h)
            best_resistors_v = copy.copy(resistors_v)
            k += 1
        else:
            energy = initial_energy
            resistors_h = copy.copy(best_resistors_h)
            resistors_v = copy.copy(best_resistors_v)
        if k != 0 and t % 1 == 0:  # how often T should be lowered
            #print (energy)
            temperature = temperature * 0.95  # controll parameter, annealing schedule that tells how it is lowered from high to low values.
            if temperature < 10 ^ (-30):
                print('Temperature is at zero')
                temperature = 0
        t += 1

    return trial_matrix, best_resistors_h, best_resistors_v

def pattern():
    pass


def imageMatrix(resistors_h, resistors_v):
    image_matrix = np.zeros((2 * L + 1, N))
    l = np.arange(0, (N * L))
    image_matrix[2 * L - 1 - 2 * (l // N % N), l % N] = resistors_v[l]
    m = np.arange(0, (N - 1) * L + (N - 1))
    image_matrix[2 * L - 2 * (m // (N - 1)), m % (N - 1)] = resistors_h[m]
    return image_matrix

if __name__ == '__main__':
    start = time.time()
    random.seed(12345)
    np.random.seed(12345)

    N = 12  # number of "node points", a resistor is placed between two nodes
    L = 3  # number of layers, how many resistor layers ( cannot exceed number of nodes)
    low = 0.5  # smallest resistor value
    high = 1.5  # largest resistor value
    max_time = 100000

    starting_resistors_h = np.full((N - 1) * L + (N - 1), low)
    starting_resistors_v = np.full((N * L), low)

    starting_resistors_h[83:88] = starting_resistors_h[64:69] = starting_resistors_v[67:72] = 1.5
    starting_resistors_h[15] = starting_resistors_h[26] = 1.5
    starting_resistors_v[4] = starting_resistors_v[5] = starting_resistors_v[6] = starting_resistors_v[7] = starting_resistors_v[16] = starting_resistors_v[17] = 1.5
    starting_resistors_v[18] = starting_resistors_v[28] = starting_resistors_v[29] = starting_resistors_v[30] = starting_resistors_v[31] = 1.5

    matrix = np.zeros((N, N))

    measured_matrix, starting_resistors_h, starting_resistors_v  = transfer_matrix(matrix, starting_resistors_h, starting_resistors_v)

    A_matrix_trial, best_resistors_h, best_resistors_v = metropolis(measured_matrix, max_time)
    
    start_image = imageMatrix(starting_resistors_h, starting_resistors_v)
    image = imageMatrix(best_resistors_h, best_resistors_v)

    end = time.time()
    print(end - start)

    fig, (ax0, ax1) = plt.subplots(figsize=(10, 5), ncols=2)
    im1 = ax0.imshow(start_image, cmap="copper", interpolation="nearest")
    im2 = ax1.imshow(image, cmap="copper", interpolation="nearest")
    plt.colorbar(im2)
    plt.show()