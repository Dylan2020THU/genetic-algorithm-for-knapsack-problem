# Project name: Genetic algorithm for backpack problem
# Born: 2023-11-20
# Update: 2023-11-27
# Author: Dylan

import random
import numpy as np
import matplotlib.pyplot as plt
import os


def ga(m, p, t_max):
    # Generate random chromosomes
    chromosomes = np.zeros((m, num_item))
    for i in range(m):
        chromosomes[i, :] = np.random.randint(2, size=num_item)

    # Calculate fitness and weight of chromosomes
    weight = np.zeros(m)
    fitness = np.zeros(m)
    for i in range(m):
        weight[i] = np.dot(chromosomes[i, :], items[:, 0])
        fitness[i] = np.dot(chromosomes[i, :], items[:, 1])
        # judge if weight is over capacity
        if weight[i] > capacity:
            fitness[i] = 0
    # Sort chromosomes by fitness
    idx = np.argsort(fitness)  # Get indices of sorted chromosomes
    chromosomes_sort = chromosomes[idx, :]  # Sort chromosomes
    init_fitness_sort = fitness[idx]  # Sort fitness
    weight_sort = weight[idx]  # Sort weight

    print('Initial fitness:\n', init_fitness_sort)

    fitness_list = np.empty(t_max)
    # start with an initial time t = 0
    for t in range(t_max):
        print("Generation: ", t + 1)
        # generate a new population with the same size as the initial population
        chromosomes_new = np.zeros((m, num_item))  # m chromosomes with n genes in each chromosome

        # Crossover
        for i in range(m):
            # Select a sub-population for offspring production
            # Select the best 20 chromosomes
            size_best = 20
            chromosomes_best = chromosomes_sort[-size_best:, :]
            # repeat the crossover to get the new population using the best 20 chromosomes
            for j in range(size_best):
                # randomly select two chromosomes from the best 20 chromosomes
                idx_parent1 = random.randint(0, size_best - 1)
                # select another parent chromosome except the first
                best_list = list(range(size_best))
                best_list.remove(idx_parent1)
                idx_parent2 = random.choice(best_list)
                # randomly select a crossover point
                crossover_point = random.randint(0, num_item - 1)
                # crossover
                chromosomes_new[i, :crossover_point] = chromosomes_best[idx_parent1, :crossover_point]
                chromosomes_new[i, crossover_point:] = chromosomes_best[idx_parent2, crossover_point:]

        # Mutation
        # randomly select a chromosome
        for i in range(m):
            # randomly select a gene
            d = random.randint(0, num_item - 1)
            # mutation
            if random.random() < p:
                chromosomes_new[i, d] = 1 - chromosomes_new[i, d]
        # Update chromosomes
        chromosomes = chromosomes_new

        # Calculate and sort fitness of chromosomes in the new population
        weight = np.zeros(m)
        fitness = np.zeros(m)
        for i in range(m):
            weight[i] = np.dot(chromosomes[i, :], items[:, 0])
            fitness[i] = np.dot(chromosomes[i, :], items[:, 1])
            # judge if weight is over capacity
            if weight[i] > capacity:
                fitness[i] = 0
        # Sort chromosomes by fitness
        idx = np.argsort(fitness)  # Get indices of sorted chromosomes
        chromosomes_sort = chromosomes[idx, :]  # Sort chromosomes
        fitness_sort = fitness[idx]  # Sort fitness
        weight_sort = weight[idx]  # Sort weight

        fitness_list[t] = fitness_sort[-1]

        print(f"{t + 1}-th GA fitness: ", fitness_sort[-1])
    return init_fitness_sort[-1], chromosomes_sort[-1], fitness_list


if __name__ == "__main__":

    random.seed(2023)

    # Problem settings
    dirname = os.path.dirname(__file__)
    file_path_c = os.path.join(dirname, 'p04\p04_c.txt')
    file_path_p = os.path.join(dirname, 'p04\p04_p.txt')
    file_path_w = os.path.join(dirname, 'p04\p04_w.txt')  # Replace with your actual file path
    file_path_s = os.path.join(dirname, 'p04\p04_s.txt')

    # Load the capacity
    with open(file_path_c, 'r') as file:
        capacity = int(file.read())

    # Load the amount of objects
    num_item = 0
    with open(file_path_p, 'r') as file:  # Open the file and read the lines
        for line in file:
            num_item += 1

    # Load the values and weights of items
    items = np.zeros((num_item, 2))
    with open(file_path_w, 'r') as file:
        items[:, 0] = [int(line.strip()) for line in file]  # Convert each line to an integer and store in a list
    with open(file_path_p, 'r') as file:
        items[:, 1] = [int(line.strip()) for line in file]

    # Load the optimal solution
    with open(file_path_s, 'r') as file:
        optima_sol = [int(line.strip()) for line in file]

    optima_value = np.dot(optima_sol, items[:,1])
    print("Capacity: ", capacity)
    print("Value: ", items[:,1])
    print("Weight: ", items[:,0])
    print("Optimal selection: ", optima_sol)
    print("Optimal value:", optima_value)

    # Generate random items
    # items = np.zeros((num_item, 2))
    # for i in range(num_item):
    #     items[i, 0] = random.randint(1, 5)  # Weight
    #     items[i, 1] = random.randint(100, 200)  # Value

    # GA fitness
    num_chromosome = 100  # Number of chromosomes or solutions
    prob_mutation = 0.05  # Probability of mutation
    generation_max = 30  # Number of generations

    init_sol, ga_sol, ga_fitness_list = ga(num_chromosome, prob_mutation, generation_max)

    print("Allocation scheme: ", ga_sol)
    print('Offical optima: ', optima_sol)

    # Random fitness
    random_fitness_list = np.empty(shape=generation_max)
    for t in range(generation_max):
        # Generate random chromosomes
        random_chromosomes = np.zeros((num_chromosome, num_item))
        random_weight = np.zeros(num_chromosome)
        random_fitness = np.zeros(num_chromosome)
        for i in range(num_chromosome):
            random_chromosomes[i, :] = np.random.randint(2, size=num_item)
            random_fitness[i] = np.dot(random_chromosomes[i, :], items[:, 1])
            random_weight[i] = np.dot(random_chromosomes[i, :], items[:, 0])
            # judge if weight is over capacity
            if random_weight[i] > capacity:
                random_fitness[i] = 0
        random_fitness_list[t] = np.average(random_fitness)
        # print(f"{t}-th random fitness: ", random_fitness_list[t])

    # Plot the fitness
    plt.figure()
    plt.scatter(0, init_sol, c='r', marker='o')
    plt.plot(ga_fitness_list, c='b', marker='*')
    plt.plot(random_fitness_list, c='g', marker='^')
    plt.axhline(y=optima_value, color='k', linestyle='-')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(["Initial fitness", "GA fitness", "Random fitness", "Maximum"])
    plt.savefig("GA for backpack problem.png")
    plt.show()
