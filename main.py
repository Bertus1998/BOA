import math
import random
import numpy as np


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
    def __init__(self, func, dimensions: int, x_from: int, x_to: int, a: float, c: float):
        self.func = func
        self.position = np.random.uniform(low=x_from, high=x_to, size=dimensions)
        self.fitness = self.func(self.position)
        self.a = a
        self.c = c
        # c*I^a
        self.fragrance = self.c * self.fitness ** self.a

    def step(self, best_solution, position_jth, position_kth, p):
        r = np.random.uniform(0, 1)
        if r > p:
            self.position = self.position + (r ** 2 * best_solution - self.position) * self.fragrance
        else:
            self.position = self.position + (r ** 2 * position_jth.position - position_kth.position) * self.fragrance

        self.fitness = self.func(self.position)
        self.fragrance = self.c * self.fitness ** self.a
        score = np.copy(self.fragrance)
        return score, self.position


class Swarm:
    def __init__(self, func, dimension, x_from: int, x_to: int, butterflies_number: int, p: float, a: float, c: float):
        self.butterflies_number = butterflies_number
        self.p = p
        self.butterflies = [Butterfly(func, dimension, x_from, x_to, a, c) for _ in range(butterflies_number)]
        self.best_scores = np.copy(self.butterflies[0].fragrance)
        self.best_position = np.copy(self.butterflies[0].position)

    def step(self, iteration):
        for i in range(iteration):
            for butterfly in self.butterflies:
                j = random.randint(0, self.butterflies_number - 1)
                k = random.randint(0, self.butterflies_number - 1)
                score, position = butterfly.step(self.best_position, self.butterflies[j], self.butterflies[k], self.p)

                if score < self.best_scores:
                    self.best_scores = score
                    self.best_position = position

            print(f'Iteration: {i}, Score: {self.best_scores}')

        return self.best_scores


if __name__ == '__main__':
    c = 0.001
    a = 0.1
    swarm = Swarm(sphere_func, 20, -100, 100, 100, 0.5, a, c)
    x = swarm.step(10000)
    print(x)
