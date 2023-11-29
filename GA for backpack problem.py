# Project name: Genetic algorithm for backpack problem
# Born: 2023-11-20
# Update: 2023-11-27
# Author: Dylan

import random
import numpy as np
import matplotlib.pyplot as plt
import os


def initialize_population(m, num_items):
    return np.random.randint(2, size=(m, num_items))


def calculate_fitness(chromosomes, items, capacity):
    weight = np.dot(chromosomes, items[:, 0])
    fitness = np.dot(chromosomes, items[:, 1])
    fitness[weight > capacity] = 0
    return fitness, weight


def crossover(chromosomes, size_best):
    new_population = np.zeros_like(chromosomes) # Create a new population with the same size as the initial population
    for i in range(len(chromosomes)):
        parent1, parent2 = random.sample(range(size_best), 2)
        crossover_point = random.randint(0, len(chromosomes[i]))
        new_population[i, :crossover_point] = chromosomes[parent1, :crossover_point]
        new_population[i, crossover_point:] = chromosomes[parent2, crossover_point:]

    return new_population


def mutate(chromosomes, p):
    mutation_mask = np.random.rand(*chromosomes.shape) < p # Create a mask for mutation
    return chromosomes ^ mutation_mask


def genetic_algorithm(m, p, t_max, items, capacity):
    '''
    Args:
        m: number of choromosome
        p: probability of mutation
        t_max: number of generation
        items: 2-dimensional variable: weight and value
        capacity: knapsack capacity

    Returns:
    '''
    population = initialize_population(m, len(items))
    fitness_list = []

    for t in range(t_max):
        # fitness, _ = calculate_fitness(population, items, capacity)
        # sorted_indices = np.argsort(fitness)[::-1]  # Sort in descending order
        # population = population[sorted_indices]
        # fitness_list.append(fitness[sorted_indices][0])

        population = crossover(population, 20) # Select the best 10 chromosomes
        population = mutate(population, p)

        fitness, _ = calculate_fitness(population, items, capacity)
        sorted_indices = np.argsort(fitness)[::-1]  # Sort in descending order
        population = population[sorted_indices]
        fitness_list.append(fitness[sorted_indices][0])

    return fitness_list, population[0]


if __name__ == "__main__":
    random.seed(2023)

    # Load data from files
    dirname = os.path.dirname(__file__)
    idx_dataset = 4
    file_path_c = os.path.join(dirname, f'p0{idx_dataset}/p0{idx_dataset}_c.txt')
    file_path_p = os.path.join(dirname, f'p0{idx_dataset}/p0{idx_dataset}_p.txt')
    file_path_w = os.path.join(dirname, f'p0{idx_dataset}/p0{idx_dataset}_w.txt')
    file_path_s = os.path.join(dirname, f'p0{idx_dataset}/p0{idx_dataset}_s.txt')

    capacity = int(open(file_path_c, 'r').read())
    items = np.column_stack((
        np.loadtxt(file_path_w, dtype=int),
        np.loadtxt(file_path_p, dtype=int)
    ))
    optima_sol = np.loadtxt(file_path_s, dtype=int)
    optima_value = np.dot(optima_sol, items[:, 1])

    num_chromosome = 300
    prob_mutation = 0.1
    generation_max = 20

    fitness_list, ga_sol = genetic_algorithm(num_chromosome, prob_mutation, generation_max, items, capacity)

    print("Capacity: ", capacity)
    print("Value: ", items[:, 1])
    print("Weight: ", items[:, 0])
    print("Optimal selection: ", optima_sol)
    print("Optimal value:", optima_value)
    print("GA selection: ", ga_sol)
    print('GA value: ', np.dot(ga_sol, items[:, 1]))

    random_fitness_list = []
    for t in range(generation_max):
        random_population = initialize_population(num_chromosome, len(items))
        random_fitness, _ = calculate_fitness(random_population, items, capacity)
        random_fitness_list.append(np.mean(random_fitness))

    # Plot the fitness
    plt.figure()
    plt.scatter(0, fitness_list[0], c='r', marker='o')
    plt.plot(fitness_list, c='b', marker='*')
    plt.plot(random_fitness_list, c='g', marker='^')
    plt.axhline(y=optima_value, color='k', linestyle='-')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(["Initial fitness", "GA fitness", "Random fitness", "Maximum"])
    plt.savefig("GA_for_backpack_problem.png")
    plt.show()
