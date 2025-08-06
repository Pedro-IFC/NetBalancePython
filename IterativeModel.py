import numpy as np
from typing import List
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

from methods import generate_tester, select_parent_torneio, Individual

load_dotenv()

POP_SIZE = int(os.getenv("POP_SIZE", 10))
GEN = int(os.getenv("GEN", 1000))
MU_TAX = float(os.getenv("MU_TAX", 0.05))

NUM_ISLAND = int(os.getenv("NUM_ISLAND", 3))
MIGRATION_INT = int(os.getenv("MIGRATION_INT", 10))
MIGRATION_SIZE = int(os.getenv("MIGRATION_SIZE", 3))


def run_ag(initial_positions: List[List[float]], min_matrix: List[List[float]], 
           max_matrix: List[List[float]], b_vector: List[float]):
    
    template = Individual(initial_positions, min_matrix, max_matrix, b_vector)
    tester = generate_tester(initial_positions, min_matrix, max_matrix)
    
    population = [template.copy()]
    for _ in range(1, POP_SIZE):
        population.append(template.randomize())
    
    best_individual = None
    best_fitness = float('-inf')
    history = []
    melhoria_indices = []  # Gerações onde houve melhora
    melhoria_valores = []  # Valores de fitness nos pontos de inflexão
    
    plt.ion()
    fig, ax = plt.subplots()
    linha, = ax.plot([], [], label='Melhor Fitness')
    pontos, = ax.plot([], [], 'ro', label='Pontos de Inflexão')  # pontos em vermelho
    ax.set_xlabel('Geração')
    ax.set_ylabel('Fitness')
    ax.set_title('Evolução do Fitness ao Longo das Gerações')
    ax.legend()
    
    for gen in range(GEN):
        fitness_list = [ind.fitness(tester) for ind in population]
        
        current_best_idx = np.argmax(fitness_list)
        current_best = population[current_best_idx]
        current_fitness = fitness_list[current_best_idx]
        
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_individual = current_best.copy()
            melhoria_indices.append(gen)
            melhoria_valores.append(current_fitness)
        
        history.append(current_fitness)
        
        if gen % 20 == 0 or gen == 0 or gen == GEN - 1:
            linha.set_ydata(history)
            linha.set_xdata(range(len(history)))
            pontos.set_xdata(melhoria_indices)
            pontos.set_ydata(melhoria_valores)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
        
        new_population = []
        tentativas = 0
        while len(new_population) < POP_SIZE and tentativas < 10 * POP_SIZE:
            parents = select_parent_torneio(population, fitness_list, 2)
            child = parents[0].cross(parents[1])
            child = child.mutate(MU_TAX)

            fitness_p1 = parents[0].fitness(tester)
            fitness_p2 = parents[1].fitness(tester)
            fitness_child = child.fitness(tester)

            if fitness_child > max(fitness_p1, fitness_p2):
                new_population.append(child)
            tentativas += 1

        while len(new_population) < POP_SIZE:
            new_population.append(best_individual.copy())
        
        population = new_population

    plt.ioff()
    plt.show()
    
    return best_individual, history


initial_positions = [
    [1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1]
]
min_matrix = [
    [100, 80, 0, 0, 0],
    [80, 100, 80, 0, 0],
    [0, 80, 100, 80, 0],
    [0, 0, 80, 100, 80],
    [0, 0, 0, 80, 100]
]
max_matrix = [
    [100, 80, 0, 0, 0],
    [80, 100, 80, 0, 0],
    [0, 80, 100, 80, 0],
    [0, 0, 80, 100, 80],
    [0, 0, 0, 80, 100]
]
b = [200, 75, 175, 90, 100]

print("Inicial:", Individual(initial_positions, min_matrix, max_matrix, b))

tester = [
    [100, 80, 0, 0, 0],
    [80, 100, 80, 0, 0],
    [0, 80, 100, 80, 0],
    [0, 0, 80, 100, 80],
    [0, 0, 0, 80, 100]
]

ind_initial = Individual(initial_positions, min_matrix, max_matrix, b)
print("Fitness:", ind_initial.fitness(tester))

best, history_simple = run_ag(initial_positions, min_matrix, max_matrix, b)
print("Melhor indivíduo encontrado:", best)
print("Fitness do melhor indivíduo:", best.fitness(tester))
