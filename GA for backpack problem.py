# Born: 2023-11-20
# Update: 2023-11-26
# Dylan
# Genetic algorithm for backpack problem

import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(2023)

# Parameters
file_path_c = r'C:\Users\Administrator\Desktop\genetic-algorithm-for-V2X-resource-allocation-main\p01\p01_c.txt'
file_path_p = r'C:\Users\Administrator\Desktop\genetic-algorithm-for-V2X-resource-allocation-main\p01\p01_p.txt'
file_path_w = r'C:\Users\Administrator\Desktop\genetic-algorithm-for-V2X-resource-allocation-main\p01\p01_w.txt'  # Replace with your actual file path
file_path_s = r'C:\Users\Administrator\Desktop\genetic-algorithm-for-V2X-resource-allocation-main\p01\p01_s.txt'

with open(file_path_c, 'r') as file:
    capacity = int(file.read())

n = 0  # the amount of objects
# Open the file and read line by line
with open(file_path_p, 'r') as file:
    for line in file:
        n += 1

items = np.zeros((n, 2))
# Open the file and read the lines
with open(file_path_w, 'r') as file:
    # Convert each line to an integer and store in a list
    items[:, 0] = [int(line.strip()) for line in file]
with open(file_path_p, 'r') as file:
    # Convert each line to an integer and store in a list
    items[:, 1] = [int(line.strip()) for line in file]

with open(file_path_s, 'r') as file:
    # Convert each line to an integer and store in a list
    optima_sol = [int(line.strip()) for line in file]

m = 100  # Number of chromosomes or solutions
p = 0.5  # Probability of mutation
t_max = 30  # Number of generations

# Generate random items
# items = np.zeros((n, 2))
# for i in range(n):
#     items[i, 0] = random.randint(1, 5)  # Weight
#     items[i, 1] = random.randint(100, 200)  # Value

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
    # judge if weight is over capacity
    if weight[i] > capacity:
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
        # judge if weight is over capacity
        if weight[i] > capacity:
            fitness[i] = 0
    # Sort chromosomes by fitness
    idx = np.argsort(fitness)  # Get indices of sorted chromosomes
    chromosomes_sort = chromosomes[idx, :]  # Sort chromosomes
    fitness_sort = fitness[idx]  # Sort fitness
    weight_sort = weight[idx]  # Sort weight

    list_best_chromosom_fitness[t] = fitness_sort[-1]

    print("Best fitness: ", fitness_sort[-1])
print("Allocation scheme: ", chromosomes_sort[-1])
print('Offical optima: ', optima_sol)

# Plot best fitness
plt.figure()
plt.scatter(0, init_fitness_sort[-1], c='r', marker='o')
plt.plot(list_best_chromosom_fitness, c='b', marker='^')
plt.xlabel("Generation")
plt.ylabel("Best fitness")
plt.legend(["Initial fitness", "Best fitness"])
plt.savefig("GA for backpack problem.png")
plt.show()
