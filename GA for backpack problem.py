# Born: 2023-11-20
# Update: 2023-11-25
# Dylan
# Genetic algorithm for backpack problem

import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(2023)

# Parameters
n = 70  # Number of items
w_max = 100  # Weight capacity of backpack
m = 100  # Number of chromosomes or solutions
p = 0.5  # Probability of mutation
t_max = 300  # Number of generations

# Generate random items
items = np.zeros((n, 2))
for i in range(n):
    items[i, 0] = random.randint(1, 5)  # Weight
    items[i, 1] = random.randint(100, 200)  # Value

# print(items)

# Generate random chromosomes
chromosomes = np.zeros((m, n))
for i in range(m):
    chromosomes[i, :] = np.random.randint(2, size=n)

# Calculate fitness and weight of chromosomes
weight = np.zeros(m)
fitness = np.zeros(m)
for i in range(m):
    weight[i] = np.dot(chromosomes[i, :], items[:, 0])
    fitness[i] = np.dot(chromosomes[i, :], items[:, 1])
    # judge if weight is over w_max
    if weight[i] > w_max:
        fitness[i] = 0
# Sort chromosomes by fitness
idx = np.argsort(fitness)  # Get indices of sorted chromosomes
chromosomes_sort = chromosomes[idx, :]  # Sort chromosomes
init_fitness_sort = fitness[idx]  # Sort fitness
weight_sort = weight[idx]  # Sort weight

print('Initial fitness:\n', init_fitness_sort)

list_best_chromosom_fitness = np.empty(t_max)
# start with an initial time t = 0
for t in range(t_max):
    print("Generation: ", t + 1)
    # generate a new population with the same size as the initial population
    chromosomes_new = np.zeros((m, n))  # m chromosomes with n genes in each chromosome

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
            crossover_point = random.randint(0, n - 1)
            # crossover
            chromosomes_new[i, :crossover_point] = chromosomes_best[idx_parent1, :crossover_point]
            chromosomes_new[i, crossover_point:] = chromosomes_best[idx_parent2, crossover_point:]

    # Mutation
    # randomly select a chromosome
    for i in range(m):
        # randomly select a gene
        d = random.randint(0, n - 1)
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
        # judge if weight is over w_max
        if weight[i] > w_max:
            fitness[i] = 0
    # Sort chromosomes by fitness
    idx = np.argsort(fitness)  # Get indices of sorted chromosomes
    chromosomes_sort = chromosomes[idx, :]  # Sort chromosomes
    fitness_sort = fitness[idx]  # Sort fitness
    weight_sort = weight[idx]  # Sort weight

    list_best_chromosom_fitness[t] = fitness_sort[-1]

    print("Best fitness: ", fitness_sort[-1])

# Plot best fitness
plt.figure()
plt.scatter(0, init_fitness_sort[-1], c='r', marker='o')
plt.plot(list_best_chromosom_fitness, c='b', marker='^')
plt.xlabel("Generation")
plt.ylabel("Best fitness")
plt.show()


