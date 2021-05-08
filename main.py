import math
import random
import numpy as np
from tqdm import tqdm
import json


def linear_interpolation(start, end, coeff):
    return (1 - coeff) * start + coeff * end


# od 0 do 1,
def sphere_func(x: np.array):
    return np.sum(x ** 2)


# Rastrigin
def f5_func(x: np.array):
    return np.sum(x ** 2 - 10 * np.cos(math.pi * x) + 10)


def f2_func(x: np.array):
    indexes = np.arange(1, x.size + 1, 1)
    return np.sum((x - indexes) ** 2)


def griewank_func(x: np.array):
    indexes = np.arange(1, x.size + 1, 1)
    return 1 + (1 / 4000) * np.sum(x ** 2) - np.prod(np.cos(x / np.sqrt(indexes)))


def ackley_func(x: np.array):
    n = x.size
    return -20 * np.exp(-0.2 * np.sqrt((1 / n) * np.sum(x ** 2))) - np.exp(
        (1 / n) * np.sum(np.cos(2 * math.pi * x))) + 20 + math.e


def schwefel_func(x: np.array):
    return np.sum(x ** 2) + np.prod(np.abs(x))


def u(z):
    a = 10
    k = 100
    m = 4
    result = 0
    size = z.size
    for cnt in range(size):
        if z[cnt] > a:
            result = result + k * (z[cnt] - a) ** m
        elif z[cnt] < (-1) * a:
            result = result + k * ((-1) * z[cnt] * (-1) * a) ** m
        else:
            result = result + 0

    return result


def leeyao_func(x: np.array):
    n = x.size
    xi = x[0:n - 1]
    xi_plus_1 = x[1:n]
    sigma1 = np.sum(((xi - 1) ** 2) * (1 + 10 * (np.sin(math.pi * xi_plus_1)) ** 2))

    return (math.pi / n) * (10 * ((np.sin(math.pi * x[1])) ** 2) + sigma1 + (x[n - 1] - 1) ** 2) + u(x)


class Butterfly:
    def __init__(self, func, dimensions: int, x_from: int, x_to: int, a_min: float, a_max: float, c: float):
        self.func = func
        self.position = np.random.uniform(low=x_from, high=x_to, size=dimensions)
        self.fitness = self.func(self.position)
        self.a_min = a_min
        self.a_max = a_max
        self.c = c
        # c*I^a
        self.fragrance = self.c * self.fitness ** self.a_min

    def step(self, iteration_ratio, best_solution, position_jth, position_kth, p, global_best_position):
        r = np.random.uniform(0, 1)
        if r > p:
            swarm_global_update_probability = np.random.uniform(0, 1)
            if swarm_global_update_probability > 0.6:
                self.position = self.position + (r ** 2 * global_best_position - self.position) * self.fragrance
            else:
                self.position = self.position + (r ** 2 * best_solution - self.position) * self.fragrance
        else:
            self.position = self.position + (r ** 2 * position_jth.position - position_kth.position) * self.fragrance

        self.fitness = self.func(self.position)
        a = linear_interpolation(self.a_min, a_max, iteration_ratio)
        self.fragrance = self.c * self.fitness ** a
        return self.fragrance, self.position


class SubSwarm:
    def __init__(self, func, dimension, x_from: int, x_to: int, butterflies_number: int, p: float, a_min: float,
                 a_max: float, c: float):
        self.butterflies_number = butterflies_number
        self.p = p
        self.butterflies = [Butterfly(func, dimension, x_from, x_to, a_min, a_max, c) for _ in
                            range(butterflies_number)]
        self.best_scores = np.copy(self.butterflies[0].fragrance)
        self.best_position = np.copy(self.butterflies[0].position)

    def step(self, iteration_ratio, global_best_position):
        for butterfly in self.butterflies:
            j = random.randint(0, self.butterflies_number - 1)
            k = random.randint(0, self.butterflies_number - 1)
            score, position = butterfly.step(iteration_ratio, self.best_position, self.butterflies[j],
                                             self.butterflies[k], self.p, global_best_position)

            if score < self.best_scores:
                self.best_scores = score
                self.best_position = position

        return self.best_scores, self.best_position


class Swarm:
    def __init__(self, func, dimension, x_from: int, x_to: int, butterflies_number: int, subswarms_number: int,
                 p: float, a_min: float, a_max: float, c: float):
        self.butterflies_number = butterflies_number
        self.p = p

        self.subswarms = [
            SubSwarm(func, dimension, x_from, x_to, int(butterflies_number / subswarms_number), p, a_min, a_max, c) for
            _ in range(subswarms_number)]

        self.best_scores = np.copy(self.subswarms[0].best_scores)
        self.best_position = np.copy(self.subswarms[0].best_position)

    def step(self, iteration_ratio):
        for subswarm in self.subswarms:
            best_score, best_position = subswarm.step(iteration_ratio, self.best_position)
            if best_score < self.best_scores:
                self.best_scores = best_score
                self.best_position = best_position

        return self.best_scores, self.best_position


class BoaSubSwarmsAlgorithm:
    def __init__(self, swarm: Swarm, iterations: int, epsilon: float, stop_criterion):
        self.swarm = swarm
        self.iterations = iterations
        self.epsilon = epsilon
        self.stop_criterion = stop_criterion

        self.max_iter = 10000

    def iteration_run(self):
        history = []
        for i in tqdm(range(self.iterations)):
            best_score, _ = self.swarm.step(i / self.iterations)
            history.append(best_score)
            # print(f'{i}: {best_score}')
        return best_score, self.iterations, history

    def epsilon_run(self):
        best_score = math.inf
        history = []
        for i in tqdm(range(self.max_iter)):
            best_score, _ = self.swarm.step(i / self.iterations)
            # print(f'{i}: {best_score}')
            history.append(best_score)
            if best_score < self.epsilon:
                return best_score, i, history
        return best_score, self.max_iter, history

    def run(self):
        if self.stop_criterion == 'iteration':
            return self.iteration_run()
        else:
            return self.epsilon_run()


# if __name__ == '__main__':
#     c = 0.01
#     a_min = 0.1
#     a_max = 0.3
#     swarm = Swarm(sphere_func, 5, -100, 100, 500, 4, 0.3, a_min, a_max, c)
#     boa = BoaSubSwarmsAlgorithm(swarm, 1000, 0.001, 'epsilon')
#     result = boa.run()
#     print(f'Score: {result[0]} Iterations: {result[1]}')


if __name__ == '__main__':
    functions = [
        {'function': sphere_func,
         'low_range': -100,
         'high_range': 100,
         'epsilon': 0.001,
         },
        # {
        #     'function': leeyao_func,
        #     'low_range': -10,
        #     'high_range': 10,
        #     'epsilon': 0.01,
        # },
        # {
        #     'function': schwefel_func,
        #     'low_range': -10,
        #     'high_range': 10,
        #     'epsilon': 0.000001,
        # },
        # {
        #     'function': f2_func,
        #     'low_range': -100,
        #     'high_range': 100,
        #     'epsilon': 0.0001,
        # },
        # {
        #     'function': griewank_func,
        #     'low_range': -600,
        #     'high_range': 600,
        #     'epsilon': 0.1,
        # },
    ]

    dimensions = [5, 20]
    populations = [100, 500]
    subwarms_numbers = [5, 20]
    stop_criterions = ['iteration', 'epsilon']

    probabilities = [0.4, 0.7]

    c = 0.01
    a_min = 0.1
    a_max = 0.3

    for _fun in functions:
        epsilon = _fun['epsilon']
        low_range = _fun['low_range']
        high_range = _fun['high_range']
        function = _fun['function']

        json_content = {"results": []}

        for dimension in dimensions:
            for population in populations:
                for subwarms_number in subwarms_numbers:
                    for probability in probabilities:
                        for stop_criterion in stop_criterions:

                            dict_result = {
                                "function": function.__name__,
                                "dimensions": dimension,
                                "population_size": population,
                                "subwarms": subwarms_number,
                                "criterion": stop_criterion,
                                "probability": probability
                            }

                            scores = []
                            iterations = []
                            histories = []
                            for i in range(1):
                                print(f'[{i}] Fitness: {function.__name__}, variant: {stop_criterion}, dimensions: {dimension}, population: {population}, subwarms: {subwarms_number}, probability: {probability}')
                                swarm = Swarm(function, dimension, low_range, high_range, population, subwarms_number, probability, a_min, a_max, c)

                                mpso_algorithm = BoaSubSwarmsAlgorithm(swarm, 1000, epsilon, stop_criterion)
                                result = mpso_algorithm.run()
                                scores.append(result[0])
                                iterations.append(result[1])
                                histories.append(result[2])
                                print(f'Score: {result[0]} Iterations: {result[1]}')

                            dict_result['mean_score'] = sum(scores) / len(scores)
                            dict_result['mean_iterations'] = sum(iterations) / len(iterations)
                            dict_result['histories'] = histories
                            json_content['results'].append(dict_result)

        with open(f'{function.__name__}.json', 'w') as outfile:
            json.dump(json_content, outfile)
