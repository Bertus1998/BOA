import math
import random

import numpy as np
from openpyxl import Workbook

c = 0.1
a = 0.5


def const_function(start, end, coeff):
    return start + end / 2


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
    def __init__(self, func, dimensions: int, x_from: int, x_to: int):
        self.func = func
        self.position = np.random.uniform(low=x_from, high=x_to, size=dimensions)
        self.fitness = 1 / self.func(self.position)
        # c*I^a
        self.fragrance = c * self.fitness ** a

    def step(self, best_solution, position_jth, position_kth, p):
        x = np.random.uniform(0, 1)
        if x > p:
            z = np.random.uniform(0, 1)
            self.position = self.position + np.subtract(
                z ** 2 * best_solution, self.position) * self.fragrance
        else:
            z = np.random.uniform(0, 1)
            self.position = self.position + np.subtract(
                z ** 2 * position_jth.position, position_kth.position) * self.fragrance

        self.fitness = 1 / self.func(self.position)
        self.fragrance = c * self.fitness ** a
        score = np.copy(self.fragrance)
        return score


class Swarm:
    def __init__(self, func, dimension, x_from: int, x_to: int, butterflies_number: int, p: float):
        self.butterflies_number = butterflies_number
        self.p = p
        self.butterflies = [Butterfly(func, dimension, x_from, x_to) for _ in range(butterflies_number)]
        self.best_scores = np.copy(self.butterflies[0].fragrance)

    def step(self, iteration):
        edge = self.butterflies_number - 3
        for i in range(iteration):
            x = 0
            for butterfly in self.butterflies:
                score = butterfly.step(self.best_scores, self.butterflies[x + 1],
                                       self.butterflies[x + 2], self.p)

                x = x + 1
                if score < self.best_scores:
                    self.best_scores = score
                if x == edge:
                    x = 0

        return self.best_scores


if __name__ == '__main__':
    swarm = Swarm(sphere_func, 20, 10, 100, 100, 0.9)
    x = swarm.step(10000)
    print(x)
