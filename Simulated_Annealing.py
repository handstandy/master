import numpy as np
import random
import copy
import time
import matplotlib.pyplot as plt


def transfer_matrix(horizontal_resistors, vertical_resistors):
    matrix = np.zeros((N,N))
    v_matrix = np.zeros((N, N))
    i = np.arange(0, (N - 1))
    j = np.arange(0, N)

    matrix[i, i] += 1.0 / horizontal_resistors[i]
    matrix[i, i + 1] -= 1.0 / horizontal_resistors[i]
    matrix[i + 1, i] -= 1.0 / horizontal_resistors[i]
    matrix[i + 1, i + 1] += 1.0 / horizontal_resistors[i]

    for l in range(L):
        v_matrix[j, j] = 1.0 / vertical_resistors[l * N + j]
        matrix = matrix.dot(np.linalg.solve(v_matrix + matrix, v_matrix))

        matrix[i, i] += 1.0 / horizontal_resistors[l * (N - 1) + i + (N - 1)]
        matrix[i, i + 1] -= 1.0 / horizontal_resistors[l * (N - 1) + i + (N - 1)]
        matrix[i + 1, i] -= 1.0 / horizontal_resistors[l * (N - 1) + i + (N - 1)]
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
    
    random_resistor_value = np.random.uniform(low, high, size=max_time)
    random_element = np.random.randint(0, (2 * N - 1) * L + (N - 1), size=max_time)

    while t < max_time:

        if random_element[t] >= ((N - 1) * L + (N - 1)):
            resistors_v[random_element[t] - ((N - 1) * L + (N - 1))] = random_resistor_value[t]
        else:
            resistors_h[random_element[t]] = random_resistor_value[t]
    
        trial_matrix, resistors_h,resistors_v = transfer_matrix(resistors_h, resistors_v)

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
        if k != 0 and t % 100 == 0:  # how often T should be lowered
            #print (energy)
            temperature *= 0.95  # controll parameter, annealing schedule that tells how it is lowered from high to low values.
            if temperature < 10 ^ (-30):
                print('Temperature is at zero')
                temperature = 0
        t += 1

    return trial_matrix, best_resistors_h, best_resistors_v

def pattern(arg):

    if arg == 'E':
        initial_resistors_h = np.full((N - 1) * L + (N - 1), low)
        initial_resistors_v = np.full((N * L), low)

        initial_resistors_h[83:88] = initial_resistors_h[64:69] = initial_resistors_v[67:72] = 1.5
        initial_resistors_h[15] = initial_resistors_h[26] = 1.5
        initial_resistors_v[4:7] = initial_resistors_v[16:18] = initial_resistors_v[28:31] = 1.5

    elif arg == 'C':
        initial_resistors_h = np.full((N - 1) * L + (N - 1), low)
        initial_resistors_v = np.full((N * L), low)
        initial_resistors_h[1:((N - 1) * L + (N - 1)):2] = high

    return initial_resistors_h, initial_resistors_v

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
    
    initial_resistors_h, initial_resistors_v = pattern('E')
    
    measured_matrix, starting_resistors_h, starting_resistors_v  = transfer_matrix(starting_resistors_h, starting_resistors_v)

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
