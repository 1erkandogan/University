import cv2
import numpy as np
import random
import math
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt


source_image = cv2.imread("painting.png")
width = source_image.shape[1]
height = source_image.shape[0]

max_radius = 40

class Individual:
    def __init__(self, num_genes):
        self.chromosome = []  # List of genes representing circles
        self.fitness = 0  # Fitness value of the individual
        self.elite = False  # Flag to indicate if the individual is an elite
        self.parent = False  # Flag to indicate if the individual is a parent
        for _ in range(num_genes):
            outside = True
            while outside:
                gene = {
                    'x': random.randint(0 - max_radius, width + max_radius),  # Random x-coordinate
                    'y': random.randint(0 - max_radius, height + max_radius),  # Random y-coordinate
                    'radius': random.randint(1, max_radius),  # Random radius
                    'R': random.randint(0, 255),  # Random red value
                    'G': random.randint(0, 255),  # Random green value
                    'B': random.randint(0, 255),  # Random blue value
                    'A': random.uniform(0, 1),  # Random alpha value
                    }
                outside = is_outside(gene['x'], gene['y'], gene['radius'])
            self.chromosome.append(gene)

        self.chromosome.sort(key=lambda gene: gene['radius'], reverse=True)
    
    def length(self):
        return len(self.chromosome)

    def chromosome_list(self):
        return self.chromosome

def evaluate_individual(individual, source_image):
    individual.chromosome.sort(key=lambda gene: gene['radius'], reverse=True)
    image = np.zeros_like(source_image)  # Initialize image with zeros
    for gene in individual.chromosome:
        overlay = deepcopy(image)  # Create a copy of the image
        
        # Extract gene attributes
        x = gene['x']
        y = gene['y']
        radius = gene['radius']
        R = gene['R']
        G = gene['G']
        B = gene['B']
        A = gene['A']
        
        # Draw the circle on the overlay
        cv2.circle(overlay, (x, y), radius, (B, G, R), -1)
        
        # Apply alpha blending to overlay the circle on the image
        image = cv2.addWeighted(overlay, A, image, 1 - A, 0)
    

    # Calculate fitness value by comparing the generated image with the source image
    srcimg_img=np.subtract(np.array(source_image, dtype=np.int64), np.array(image, dtype=np.int64))
    fitness = np.sum(-1*np.power(srcimg_img, 2))

    # Update the individual's fitness attribute
    individual.fitness = fitness

def is_outside(x, y, radius):
    outside = True
    # Check if the circle is outside the image
    # 1. inside, middle, middle
    if (x >= 0 and x <= width) and (y >= 0 and y <= height):
        outside = False
    # 2. outside, left, middle
    elif (x < 0) and (y > 0 and y < height):
        if (x + radius<0):
            outside = True
    # 3. outside, right, middle
    elif (x > width) and (y > 0 and y < height):
        if (x - radius>width):
            outside = True
    # 4. outside, bottom, middle
    elif (y < 0) and (x > 0 and x < width):
        if (y + radius<0):
            outside = True
    # 5. outside, top, middle
    elif (y > height) and (x > 0 and x < width):
        if (y - radius>height):
            outside = True
    # 6. outside, left, bottom
    elif (x < 0) and (y < 0):
        if (radius**2 < (x - 0)**2 + (y - 0)**2):
            outside = True
    # 7. outside, left, top
    elif (x < 0) and (y > height):
        if (radius**2 < (x - 0)**2 + (y - height)**2):
            outside = True
    # 8. outside, right, bottom
    elif (x > width) and (y < 0):
        if (radius**2 < (x - width)**2 + (y - 0)**2):
            outside = True
    # 9. outside, right, top
    elif (x > width) and (y > height):
        if (radius**2 < (x - width)**2 + (y - width)**2):
            outside = True
    else:
        outside = False
    return outside

def selection(population, elites, num_parents):
    selected_parents = []
    parent_candidates = [ind for ind in population if ind not in elites]
    for _ in range(num_parents):
        while True:
            tournament = random.sample(parent_candidates, min(tm_size, num_parents, len(parent_candidates)))
            tournament.sort(key=lambda ind: ind.fitness, reverse=True)
            winner = tournament[0]
            if winner not in selected_parents:
                selected_parents.append(winner)
                winner.parent = True
                parent_candidates.remove(winner)
                break
    return selected_parents

def crossover(parent1, parent2):
    chromosome_length = len(parent1.chromosome)
    child1 = Individual(chromosome_length)
    child2 = Individual(chromosome_length)

    # Perform crossover
    for gene in range(chromosome_length):
        coinflip = random.randint(0, 1)
        if coinflip == 0:
            child1.chromosome[gene] = parent1.chromosome[gene]
            child2.chromosome[gene] = parent2.chromosome[gene]
        else:
            child1.chromosome[gene] = parent2.chromosome[gene]
            child2.chromosome[gene] = parent1.chromosome[gene]

    return child1, child2

def mutate(individual):
    prev_fitness = individual.fitness
    for gene in individual.chromosome:
        if random.random() < mutation_prob:
            if mutation_type == "unguided":
                mutate_unguided(gene)
            elif mutation_type == "guided":
                mutate_guided(gene)
    evaluate_individual(individual, source_image)
    if individual.fitness < prev_fitness:
        mutate(individual)

def mutate_unguided(gene):
    outside = True
    while outside:
        gene['x'] = random.randint(0 - max_radius, width + max_radius)
        gene['y'] = random.randint(0 - max_radius, height + max_radius)
        gene['radius'] = random.randint(1, max_radius)
        outside = is_outside(gene['x'], gene['y'], gene['radius'])
    gene['R'] = random.randint(0, 255)
    gene['G'] = random.randint(0, 255)
    gene['B'] = random.randint(0, 255)
    gene['A'] = random.uniform(0, 1)

def mutate_guided(gene):
    # Mutate the gene attributes without exceeding the boundaries
    x = gene['x']
    y = gene['y']
    radius = gene['radius']
    R = gene['R']
    G = gene['G']
    B = gene['B']
    A = gene['A']
    temp_x = x
    temp_y = y
    temp_radius = radius
    outside = True
    while outside:
        while True:
            temp_x = x + random.randint(-width // 4, width // 4)
            if (temp_x >= 0 - max_radius and temp_x <= width + max_radius):
                break

        while True:
            temp_y = y + random.randint(-height // 4, height // 4)
            if (temp_y >= 0 - max_radius and temp_y <= height + max_radius):
                break

        while True:
            temp_radius = radius + random.randint(-10, 10)
            if (temp_radius > 0):
                break
        
        outside = is_outside(temp_x, temp_y, temp_radius)
    gene['x'] = temp_x
    gene['y'] = temp_y
    gene['radius'] = temp_radius

    while True:
        R = gene['R'] + random.randint(-64, 64)
        if (R >= 0 and R <= 255):
            gene['R'] = R
            break

    while True:
        G = gene['G'] + random.randint(-64, 64)
        if (G >= 0 and G <= 255):
            gene['G'] = G
            break
    
    while True:
        B = gene['B'] + random.randint(-64, 64)
        if (B >= 0 and B <= 255):
            gene['B'] = B
            break

    while True:
        A = gene['A'] + random.uniform(-0.25, 0.25)
        if (A >= 0 and A <= 1):
            gene['A'] = A
            break

def draw_circle(individual, name, value, generation):
    individual.chromosome.sort(key=lambda gene: gene['radius'], reverse=True)
    image = np.ones_like(source_image)*255  # Initialize image with zeros
    for gene in individual.chromosome:
        overlay = deepcopy(image)
        x = gene['x']
        y = gene['y']
        radius = gene['radius']
        color = (gene['B'], gene['G'], gene['R'])
        A = gene['A']
        thickness = -1  # Filled circle

        cv2.circle(overlay, (x, y), radius, color, thickness)
        cv2.addWeighted(overlay, A, image, 1 - A, 0, image)
    cv2.imwrite(f"C:/Users/erkan/Desktop/EE/e2022_2/EE449/2023/HW2/Code/{name}/{name}_{value}img_gen{generation}.png", image)

def draw_fig(fitness_list, name, value):
    part1 = int(num_generations/10)
    part2 = num_generations
    parts = [part1, part2]
    print(len(fitness_list[0:10]))
    print(len(fitness_list[10:]))
    for part in parts:
        if part == part1:
            generations = range(1, part1+1)
            plt.plot(generations, fitness_list[0:part1])
            plt.title(f'Fitness vs Generation {name}={value} {1}-{part1}')
        else:
            generations = range(part1+1, part2+1)
            plt.plot(generations, fitness_list[part1:])
            plt.title(f'Fitness vs Generation {name}={value} {part1}-{part2}')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid()
        plt.savefig(f"C:/Users/erkan/Desktop/EE/e2022_2/EE449/2023/HW2/Code/{name}/{name}_{value}_fig_{part}.png")
        plt.close()

def population_func():
    population = []
    for _ in range(num_inds):
        individual = Individual(num_genes)
        population.append(individual)
    population.sort(key=lambda ind: ind.fitness)
    return population

def genetic_algorithm(name, item):
    # Step 6.1: Initialize the population with random individuals
    population = population_func()
    fitness_list = []

    # Step 6.2: Iterate over the specified number of generations
    for generation in range(num_generations):
        #print(f"Generation {generation+1}/{num_generations}, Best Fitness: {population[0].fitness}")
        if (generation+1) % 100 == 0:
            print(f"Generation {generation+1}/{num_generations}, Best Fitness: {population[0].fitness}")
        if (generation+1) % (num_generations/10) == 0:
            draw_circle(population[0], name, item, generation+1)

        # Step 6.3: Evaluate all individuals in the population
        for individual in population:
            individual.elite, individual.parent = False, False
            evaluate_individual(individual, source_image)
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        fitness_list.append(population[0].fitness)
        # Step 6.4: Select elites to directly pass to the next generation
        num_elites = int(frac_elites * num_inds)
        elites = (population[:num_elites])
        for individual in elites:
            individual.elite = True

        # Step 6.5: Perform tournament selection to select parents for crossover
        num_parents = int(frac_parents * num_inds)
        if num_parents % 2 != 0:
            num_parents += 1
        parents = selection(population, elites, num_parents)
        nonparents = [ind for ind in population if ind not in parents]
        nonparents = [ind for ind in nonparents if ind not in elites]
        for individual in parents:
            individual.parent = True

        # Step 6.6: Apply crossover to create new individuals
        offspring = []
        for i in range(0, num_parents, 2):
            # Perform crossover on adjacent parents
            parent1 = parents.pop(random.randint(0,len(parents)-1))
            parent2 = parents.pop(random.randint(0,len(parents)-1))
            # parent2 = parents[i+1]
            child1, child2 = crossover(parent1, parent2)
            offspring.extend([child1, child2])

        # Step 6.7: Update the population with elites, offspring, and mutated individuals
        population = elites + offspring + nonparents

        # Step 6.8: Perform mutation on some individuals
        mutation_candidates = [ind for ind in population if ind not in elites]
        for individual in mutation_candidates:
            mutate(individual)
        # Sort the final population based on fitness values in descending order
        population.sort(key=lambda ind: ind.fitness, reverse=True)
    # Return the final population
    return population, fitness_list


num_generations = 10000

num_inds = 20
num_genes = 50
tm_size = 5
frac_elites = 0.2
frac_parents = 0.6
mutation_prob = 0.2
mutation_type = "guided"
default_list = [num_inds, num_genes, tm_size, frac_elites, frac_parents, mutation_prob, mutation_type]

num_inds_list = [5, 10, 20, 40, 60]
num_genes_list = [15, 30, 50, 80, 120]
tm_size_list = [2, 5, 8, 16]
frac_elites_list = [0.04, 0.2, 0.35]
frac_parents_list = [0.15, 0.3, 0.6, 0.75]
mutation_prob_list = [0.1, 0.2, 0.4, 0.75]
mutation_type_list = ["unguided", "guided"]
parameters = [num_inds_list,num_genes_list,tm_size_list,frac_elites_list,frac_parents_list,mutation_prob_list,mutation_type_list]
names = ["num_inds","num_genes","tm_size","frac_elites","frac_parents","mutation_prob","mutation_type"]

print(f"Running for default parameters")
population, fitness_list = genetic_algorithm("default_parameters", "default")
best_individual = population[0]
draw_fig(fitness_list, "default_parameters", "default")

for param, name in zip(parameters,names):
    num_inds, num_genes, tm_size, frac_elites, frac_parents, mutation_prob, mutation_type = default_list
    for item in param:
        print(f"Running for {name} = {item}")
        population, fitness_list = genetic_algorithm(name, item)

        # Find the best individual from the final population
        best_individual = population[0]

        # Plot the fitness graph
        draw_fig(fitness_list, name, item)

# Done
print("Done")
