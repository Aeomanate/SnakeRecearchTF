from dataclasses import dataclass
from snake_game import SnakeGame
import matplotlib as mpl
import tflearn, math
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from random import randint as rand
import numpy as np
from os import path, remove
from statistics import mean
from scipy import interpolate
import matplotlib.pyplot as plt
import tensorflow
tf = tensorflow.compat.v1


def collect_data_for_nn(turn_variants, turn_code):
    return np.append([turn_code], turn_variants)


def snake_vector(snake):
    return np.array(snake[0]) - np.array(snake[1])


def vector_to_food(snake, food):
    return np.array(food) - np.array(snake[0])


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def distance(snake, food):
    return np.linalg.norm(vector_to_food(snake, food))


def is_direction_wrong(snake, direction):
    point = np.array(snake[0]) + np.array(direction)
    return point.tolist() in snake or point[0] == 0 or point[1] == 0 or point[0] == 21 or point[1] == 21


def rotate_left(vector):
    return np.array([-vector[1], vector[0]])


def rotate_right(vector):
    return np.array([vector[1], -vector[0]])


def angle(a, b):
    a = normalize_vector(a)
    b = normalize_vector(b)
    #return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi
    return (math.atan2(a[1], a[0]) - math.atan2(b[1], b[0])) /  math.pi


class Main:
    def __init__(self):
        self.total_games = 30000
        self.max_steps = 2000
        self.filename_nn_data = 'network_data'
        self.filename_samples_data = 'training_samples.txt'
        self.move_codes = [
            [[-1, 0], 0],
            [[0, 1], 1],
            [[1, 0], 2],
            [[0, -1], 3]
        ]

    def create_training_samples(self):
        samples = []
        for _ in range(self.total_games):
            game = SnakeGame()
            _, prev_score, snake, food, _ = game.start()
            prev_turn_variants = self.turn_variants(snake, food)
            prev_food_distance = distance(snake, food)
            for _ in range(self.max_steps):
                turn_code, turn = self.rand_move_vector(snake)
                is_game_end, score, snake, food, _ = game.step(turn)
                if is_game_end:
                    samples.append([collect_data_for_nn(prev_turn_variants, turn_code), -1])
                    break
                else:
                    food_distance = distance(snake, food)
                    if score > prev_score or food_distance < prev_food_distance:
                        samples.append([collect_data_for_nn(prev_turn_variants, turn_code), 1])
                    else:
                        samples.append([collect_data_for_nn(prev_turn_variants, turn_code), 0])
                    prev_turn_variants = self.turn_variants(snake, food)
                    prev_food_distance = food_distance
        return samples

    def rand_move_vector(self, snake):
        model_true_return = rand(0, 2) - 1
        return model_true_return, self.move_code(snake, model_true_return)

    def move_code(self, snake, model_predicted_value):
        view_direction = snake_vector(snake)
        new_direction = view_direction
        if model_predicted_value == -1:
            new_direction = rotate_left(view_direction)
        elif model_predicted_value == 1:
            new_direction = rotate_right(view_direction)
        for vector_code in self.move_codes:
            if vector_code[0] == new_direction.tolist():
                move_code = vector_code[1]
        return move_code

    def turn_variants(self, snake, food):
        view_vector = snake_vector(snake)
        food_vector = vector_to_food(snake, food)
        left = is_direction_wrong(snake, rotate_left(view_vector))
        front = is_direction_wrong(snake, view_vector)
        right = is_direction_wrong(snake, rotate_right(view_vector))
        return np.array([int(left), int(front), int(right), angle(view_vector, food_vector)])

    def create_model(self, layers, neurons, activation):
        self.model = input_data(shape=[None, 5, 1], name='input')
        for _ in range(layers):
            self.model = fully_connected(self.model, neurons, activation=activation)
        self.model = fully_connected(self.model, 1, activation='linear')
        self.model = regression(self.model, optimizer='adam', loss='mean_square', name='target')
        self.model = tflearn.DNN(self.model, tensorboard_dir='log', tensorboard_verbose=0)

    def train_model(self, training_data):
        class EarlyStoppingCallback(tflearn.callbacks.Callback):
            def on_epoch_end(self, training_state):
                if training_state.loss_value is None: return
                if training_state.loss_value > 0.25:
                    raise StopIteration

        x = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)

        try:
            self.model.fit(
                x, y, n_epoch=8, shuffle=True, batch_size=128,
                run_id=self.filename_nn_data, snapshot_step=False,
                callbacks=EarlyStoppingCallback()
            )
            self.model.save(self.filename_nn_data)
        except StopIteration:
            pass

    def run_game_session(self, is_gui):
        game = SnakeGame(gui=is_gui)
        _, score, snake, food, _ = game.start()
        for _ in range(self.max_steps):
            predictions = []
            for turn_code in range(-1, 2):
                predictions.append(
                    self.model.predict(
                        collect_data_for_nn(self.turn_variants(snake, food), turn_code).reshape(-1, 5, 1)
                    )
                )
            turn_code = np.argmax(np.array(predictions))
            turn = self.move_code(snake, turn_code - 1)
            is_game_end, score, snake, food, steps = game.step(turn)
            if is_game_end: break
        return score, steps

    def load_training_samples(self):
        training_samples = []
        with open(self.filename_samples_data, 'r') as file:
            for line in file:
                str_numbers = line.split()
                numbers = [float(number) for number in str_numbers[0:-1]]
                training_samples.append([np.array(numbers), int(str_numbers[-1])])

        return training_samples

    def save_training_samples(self, samples):
        with open(self.filename_samples_data, 'w') as file:
            for sample in samples:
                file.write(" ".join([str(turn) for turn in sample[0]] + [str(sample[1]) + "\n"]))

    def start(self, layers, neurons, activation, is_gui):
        if not hasattr(self, 'training_samples'):
            if path.exists(self.filename_samples_data):
                self.training_samples = self.load_training_samples()
            else:
                self.training_samples = self.create_training_samples()
                self.save_training_samples(self.training_samples)

        if not hasattr(self, 'model'):
            self.create_model(layers, neurons, activation)

            if not path.exists(self.filename_nn_data + ".meta"):
                self.train_model(self.training_samples)
            else:
                self.model.load(self.filename_nn_data)

        return self.run_game_session(is_gui)


class Max:
    value = 0
    layer = None
    neuron = None


class Result:
    def __init__(self, layers, neurons):
        self.matrix = [[0 for _ in range(neurons)] for _ in range(layers)]
        self.max = Max()


def save_surface(r: Result, filename, cmap):
    r.matrix = np.array(r.matrix)
    max_z = int(r.max.value * 100) / 100
    #
    x = np.array([[i+1 for i in range(r.matrix.shape[1])] for _ in range(r.matrix.shape[0])])
    y = np.array([[i+1 for _ in range(r.matrix.shape[1])] for i in range(r.matrix.shape[0])])
    #
    fig = plt.figure(figsize=[18, 8])
    plt.subplot(221)
    ax1 = plt.subplot(221, projection='3d')
    ax2 = plt.subplot(222, projection='3d')
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    plt.subplots_adjust(left=0.03, bottom=0.1, right=1.05, top=0.9, wspace=0, hspace=0.1)
    title  = filename + f" - L{r.max.layer}, N{r.max.neuron}, Max " + str(max_z)
    xlabel = 'N e u r o n s'
    ylabel = 'L a y e r s'
    zlabel = 'S c o r e'
    #
    norm = mpl.colors.Normalize(vmin=0, vmax=max_z)
    colorbar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(colorbar, ax=ax1)
    fig.colorbar(colorbar, ax=ax2)
    fig.colorbar(colorbar, ax=ax3)
    fig.colorbar(colorbar, ax=ax4)
    #
    ax1.set_zlim(0, max_z)
    ax1.plot_surface(x, y, r.matrix, cmap=cmap)
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_zlabel(zlabel)
    ax1.view_init(azim=-165, elev=40)
    #
    ax3.pcolormesh(np.arange(0, r.matrix.shape[1]+1), np.arange(0, r.matrix.shape[0]+1), r.matrix, cmap=cmap, vmin=0, vmax=max_z, shading='nearest')
    ax3.set_title(title)
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel(ylabel)
    #
    xnew, ynew = np.mgrid[1:64:64j, 1:8:64j]
    tck = interpolate.bisplrep(x, y, r.matrix, s=6000)
    znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)
    ax2.plot_surface(xnew, ynew, znew, cmap=cmap, alpha=None, antialiased=True)
    ax2.set_title(title)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.set_zlabel(zlabel)
    ax2.set_zlim(0, max_z)
    ax2.view_init(azim=-165, elev=40)
    #
    xnew, ynew = np.mgrid[0:64:65j, 0:8:65j]
    ax4.pcolormesh(xnew, ynew, znew, cmap=cmap, vmin=0, vmax=max_z, shading='nearest')
    ax4.set_title(title)
    ax4.set_xlabel(xlabel)
    ax4.set_ylabel(ylabel)
    #
    ax1.text(-13, 10, z=-11, s="a", style='oblique', fontsize=16)
    ax2.text(-13, 10, z=-11, s="b", style='oblique', fontsize=16)
    ax3.text(0, -1.5, s="c", style='oblique', fontsize=16)
    ax4.text(0, -1.5, s="d", style='oblique', fontsize=16)
    #
    # plt.show()
    plt.savefig(title.replace(':', '') + " " + cmap + ".png", dpi=500)


def research():
    controller = Main()

    activations = [
        'elu',
        'hard_sigmoid',
        'linear',
        'relu',
        'selu',
        'sigmoid',
        'softmax',
        'softplus',
        'softsign',
        'tanh'
    ]

    max_layers = 8
    max_neurons = 64
    tests = 150

    results = [Result(max_layers, max_neurons) for _ in range(len(activations))]

    best_layers = 0
    best_neurons = 0
    best_score = 0
    best_activation = ""

    for activation, result in zip(activations, results):
        filename = "results" + "_" + activation

        cur_line = 0
        cur_neuron = 0
        if not path.exists(filename + ".txt"):
            with open(filename + ".txt", 'w') as file:
                file.write("[[ ")
        else:
            with open(filename + ".txt", 'r') as file:
                for cur_line, line in enumerate(file):
                    line = [float(x) for x in line.replace('[', ' ').replace(']', ' ').replace(',', ' ').split()]
                    if cur_line == max_layers:
                        assert(result.max.value == line[0])
                        break
                    if len(line) == 0:
                        if cur_line != 0:
                            cur_line -= 1
                        break
                    for cur_neuron, value in enumerate(line):
                        result.matrix[cur_line][cur_neuron] = value
                        if value > result.max.value:
                            result.max.value = value
                            result.max.layer = cur_line + 1
                            result.max.neuron = cur_neuron + 1
        if result.max.value > best_score:
            best_score = result.max.value
            best_activation = activation
            best_layers = result.max.layer
            best_neurons = result.max.neuron

        from_layers = cur_line+1
        from_neurons = cur_neuron+2  # +1 as number of last neuron фтв +1 as next neuron

        if cur_line == max_layers:
            continue

        for layers in range(from_layers, max_layers+1):
            for neurons in range(from_neurons, max_neurons+1):

                from_layers = 1
                from_neurons = 1

                with tf.Graph().as_default():
                    del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
                    if path.exists(controller.filename_nn_data + ".meta"):
                        remove(controller.filename_nn_data + ".meta")

                    scores = [0 for _ in range(tests)]
                    for test in range(tests):
                        score, steps = controller.start(layers, neurons, activation, is_gui=False)
                        scores[test] = score
                        print(f"\n\n{activation} L{layers} N{neurons}",
                              str(int((test+1)/tests * 100)) + "%", "| Sc: ", score, " | St: ", steps)
                        if steps == controller.max_steps:
                            break

                    test_result = mean(scores)
                    if test_result > result.max.value:
                        result.max.value = test_result
                        result.max.layer = layers
                        result.max.neuron = neurons
                    result.matrix[layers-1][neurons-1] = test_result
                    with open(filename + ".txt", 'a') as file:
                        file.write(str(test_result) + ", ")
                    del controller.model

            with open(filename + ".txt", 'a') as file:
                file.write("],\n" + (" [ " if layers+1 != max_layers+1 else "], "))

        if result.max.value > best_score:
            best_score = result.max.value
            best_activation = activation

        with open(filename + ".txt", 'a') as file:
            file.write(str(result.max.value))

    with open("MAX OF MAX", 'w') as file:
        file.write(best_activation + " - L" + str(best_layers) + " N" + str(best_neurons) + ", Max: " + str(best_score))

    for activation, result in zip(activations, results):
        # cmap = 'Greys' #  'Greys'  # 'gnuplot'
        save_surface(result, activation, 'Greys')
        save_surface(result, activation, 'gnuplot')


def example():
    s = []
    controller = Main()
    tests = 200
    for test in range(1, tests + 1):
        score, steps = controller.start(2, 20, 'softsign', True)
        assert(steps != controller.max_steps)
        print(f"\n\nrelu L5 N48", "%:",
              int(test/tests * 100), "| Sc: ", score, " | St: ", steps)
        s.append(score)
    print("\n\nMean: " + str(mean(s)) + "\n\n")

if __name__ == "__main__":
    example()
    #research()
