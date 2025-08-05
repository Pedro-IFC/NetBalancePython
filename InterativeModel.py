import numpy as np
import random
import copy
from typing import List
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()

POP_SIZE = int(os.getenv("POP_SIZE", 10))
GEN = int(os.getenv("GEN", 1000))
MU_TAX = float(os.getenv("MU_TAX", 0.05))

NUM_ISLAND = int(os.getenv("NUM_ISLAND", 3))
MIGRATION_INT = int(os.getenv("MIGRATION_INT", 10))
MIGRATION_SIZE = int(os.getenv("MIGRATION_SIZE", 3))

class Individual:
    def __init__(self, initial_positions: List[List[float]], min_matrix: List[List[float]], 
                 max_matrix: List[List[float]], b_vector: List[float]):
        self.initial_positions = initial_positions
        self.min_matrix = min_matrix
        self.max_matrix = max_matrix
        self.b_vector = b_vector
        self.n = len(initial_positions)
        self.graph = self._initialize_graph()
    
    def _initialize_graph(self) -> List[List[float]]:
        graph = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if self.max_matrix[i][j] > 0:
                    val = self.initial_positions[i][j] * random.uniform(
                        self.min_matrix[i][j], self.max_matrix[i][j])
                    graph[i][j] = val
                    graph[j][i] = val
        return graph

    def fitness(self, tester: List[List[float]]) -> float:
        n = len(self.initial_positions)
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                A[i][j] = self.initial_positions[i][j] * tester[i][j]
        
        try:
            solution = np.linalg.solve(A, self.b_vector.copy())
            total_abs = sum(abs(x) for x in solution) 
            if total_abs == 0:
                return 0.0
            return 10.0 * n / total_abs
        except ValueError:
            return 0.0

    def mutate(self, mutation_rate: float) -> 'Individual':
        new_positions = copy.deepcopy(self.initial_positions)
        for i in range(self.n):
            for j in range(self.n):
                if i != j and random.random() < MU_TAX:
                    new_positions[i][j] = 1 - new_positions[i][j]
                    new_positions[j][i] = new_positions[i][j]
        return Individual(new_positions, self.min_matrix, self.max_matrix, self.b_vector)

    def randomize(self) -> 'Individual':
        new_positions = [[1 if i==j else 0 for j in range(self.n)] for i in range(self.n)]
        for i in range(self.n):
            for j in range(i+1, self.n):
                if random.random() < 0.5:
                    new_positions[i][j] = 1
                    new_positions[j][i] = 1
                else:
                    new_positions[i][j] = 0
                    new_positions[j][i] = 0
        return Individual(new_positions, self.min_matrix, self.max_matrix, self.b_vector)

    def cross(self, other: 'Individual') -> 'Individual':
        new_positions = copy.deepcopy(self.initial_positions)
        for i in range(self.n):
            for j in range(self.n):
                if i != j and random.random() < 0.3:
                    new_positions[i][j] = other.initial_positions[i][j]
                    new_positions[j][i] = new_positions[i][j]
        return Individual(new_positions, self.min_matrix, self.max_matrix, self.b_vector)

    def copy(self) -> 'Individual':
        return Individual(
            copy.deepcopy(self.initial_positions), 
            self.min_matrix,
            self.max_matrix,
            copy.deepcopy(self.b_vector)
        )

    def get_graph(self) -> List[List[float]]:
        return self.graph

    def get_b(self) -> List[float]:
        return self.b_vector

    def __str__(self) -> str:
        return f"Individual:\n{self.initial_positions}"
def generate_tester(initial_positions: List[List[float]], min_matrix: List[List[float]], 
                    max_matrix: List[List[float]]) -> List[List[float]]:
    n = len(initial_positions)
    tester = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if max_matrix[i][j] > 0:
                val = random.uniform(min_matrix[i][j], max_matrix[i][j])
                tester[i][j] = val
                tester[j][i] = val
    return tester
def select_parent_torneio(population: List[Individual], fitness: List[float], 
                          tournament_size: int) -> List[Individual]:
    selected = []
    for _ in range(2):
        tournament = random.sample(list(zip(population, fitness)), tournament_size)
        best = max(tournament, key=lambda x: x[1])[0]
        selected.append(best)
    return selected
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
