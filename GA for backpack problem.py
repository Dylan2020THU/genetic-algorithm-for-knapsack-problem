# 2023-11-20
# Dylan
# Genetic algorithm for backpack problem

import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(2023)

# Parameters
n = 100 # Number of items
w_max = 100 # Weight capacity of backpack
m = 100 # Number of chromosomes
p = 0.5 # Probability of mutation
t = 100 # Number of generations

# Generate random items
items = np.zeros((n,2))
for i in range(n):
    items[i,0] = random.randint(1,10) # Weight
    items[i,1] = random.randint(1,20) # Value

# print(items)

# Generate random chromosomes
chromosomes = np.zeros((m,n))
for i in range(m):
    chromosomes[i,:] = np.random.randint(2, size=n)

# Calculate fitness
fitness = np.zeros(m)
for i in range(m):
    fitness[i] = np.dot(chromosomes[i,:],items[:,1])

# Sort chromosomes by fitness
idx = np.argsort(fitness) # Get indices of sorted chromosomes
chromosomes = chromosomes[idx,:] # Sort chromosomes
fitness_sort = fitness[idx] # Sort fitness

print(fitness_sort)
print(chromosomes)
print(idx)

# Plot initial population
plt.figure()
# plot fitness_sort and fitness variables simultaneously on the same y-axis using different colors
plt.plot(fitness_sort, 'r', fitness, 'b')
plt.legend(['Sorted fitness', 'Unsorted fitness'])
plt.title('Initial population')
plt.xlabel('Chromosome')
plt.ylabel('Fitness')
plt.show()
