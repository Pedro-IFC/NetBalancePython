import numpy as np
import random
import json
import copy
from typing import List, Tuple

def eliminacao_gauss(A, b, max_iter):
    n = len(A)
    for i in range(n):
        max_index = max(range(i, n), key=lambda k: abs(A[k][i]))
        if A[max_index][i] == 0:
            raise ValueError(f"A matriz é singular, não é possível resolver o sistema para a linha {i}.")
        if max_index != i:
            A[i], A[max_index] = A[max_index], A[i]
            b[i], b[max_index] = b[max_index], b[i]
        for k in range(i + 1, n):
            fator = A[k][i] / A[i][i]
            for j in range(i, n):
                A[k][j] -= fator * A[i][j]
            b[k] -= fator * b[i]
    x = [0] * n
    for i in range(n - 1, -1, -1):
        soma = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - soma) / A[i][i]
    for _ in range(max_iter):
        r = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
        delta_x = [0] * n
        for i in range(n - 1, -1, -1):
            soma = sum(A[i][j] * delta_x[j] for j in range(i + 1, n))
            delta_x[i] = (r[i] - soma) / A[i][i]
        x = [x[i] + delta_x[i] for i in range(n)]
    
    return x
def metodo_jacobi(A, b, max_iter, tol=1e-6):
    n = len(A)
    x = [0] * n 
    x_novo = x[:]
    for _ in range(max_iter):
        for i in range(n):
            soma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_novo[i] = (b[i] - soma) / A[i][i]
        if max(abs(x_novo[i] - x[i]) for i in range(n)) < tol:
            return x_novo
        x = x_novo[:]
    return x_novo
def metodo_gauss_seidel(A, b, max_iter, tol=1e-6):
    n = len(A)
    x = [0] * n 
    for _ in range(max_iter):
        x_old = x[:]
        for i in range(n):
            soma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - soma) / A[i][i]
        if max(abs(x[i] - x_old[i]) for i in range(n)) < tol:
            return x
    return x
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
            solution = eliminacao_gauss(A, self.b_vector.copy(), 5)
            total_abs = sum(abs(x) for x in solution) 
            if total_abs == 0:
                return 0.0
            return 10.0 * n / total_abs
        except ValueError:  # Capturar erro de matriz singular
            return 0.0

    def mutate(self, mutation_rate: float) -> 'Individual':
        new_positions = copy.deepcopy(self.initial_positions)
        for i in range(self.n):
            for j in range(self.n):
                if i != j and random.random() < mutation_rate:
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
        return Individual(new_positions, self.min_matrix, self.max_matrix, self.b_vector)

    def cross(self, other: 'Individual') -> 'Individual':
        new_positions = copy.deepcopy(self.initial_positions)
        for i in range(self.n):
            for j in range(self.n):
                if i != j and random.random() < 0.5:
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
def weighted_random_choice(population: List['Individual'], fitness: List[float]) -> 'Individual':
    total = sum(fitness)
    if total == 0:
        return random.choice(population)
    probabilities = [f / total for f in fitness]
    return random.choices(population, weights=probabilities, k=1)[0]

def select_parent_roleta_viciada(population: List['Individual'], fitness: List[float]) -> List['Individual']:
    pai1 = weighted_random_choice(population, fitness)
    pai2 = weighted_random_choice(population, fitness)
    return [pai1, pai2]

def select_parent_roleta(population: List['Individual']) -> List['Individual']:
    pai1 = random.choice(population)
    pai2 = random.choice(population)
    return [pai1, pai2]
def run_ag(initial_positions: List[List[float]], min_matrix: List[List[float]], 
           max_matrix: List[List[float]], b_vector: List[float]) -> Individual:
    template = Individual(initial_positions, min_matrix, max_matrix, b_vector)
    tester = generate_tester(initial_positions, min_matrix, max_matrix)
    
    history = []  # Armazenará fitness por geração
    population = [template.copy()]
    for _ in range(1, POP_SIZE):
        population.append(template.randomize())
    
    best_individual = None
    best_fitness = float('-inf')
    history = []
    
    for gen in range(GEN):
        fitness_list = [ind.fitness(tester) for ind in population]
        
        current_best_idx = np.argmax(fitness_list)
        current_best = population[current_best_idx]
        current_fitness = fitness_list[current_best_idx]
        
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_individual = current_best.copy()
        
        history.append({
            "generation": gen,
            "graph": best_individual.get_graph(),
            "b": best_individual.get_b(),
            "fitness": best_fitness
        })
        
        new_population = []
        while len(new_population) < POP_SIZE:
            parents = select_parent_torneio(population, fitness_list, 3)
            child = parents[0].cross(parents[1])
            child = child.mutate(MU_TAX)
            new_population.append(child)
        
        population = new_population
        history.append(best_fitness)
    
    with open("historico_individuos.json", "w") as f:
        json.dump(history, f, indent=2)
    
    return best_individual, history
def best_realocate(initial_positions, min_matrix, max_matrix, b):
    return run_ag(initial_positions, min_matrix, max_matrix, b)
def run_ag_island(initial_positions: List[List[float]], min_matrix: List[List[float]], 
                 max_matrix: List[List[float]], b_vector: List[float], 
                 num_islands: int, migration_interval: int, migration_size: int) -> Individual:
    template = Individual(initial_positions, min_matrix, max_matrix, b_vector)
    tester = generate_tester(initial_positions, min_matrix, max_matrix)
    history=[]
    islands = []
    for _ in range(num_islands):
        pop = [template.copy()]
        for _ in range(1, POP_SIZE):
            pop.append(template.randomize())
        islands.append(pop)
    
    best_global = None
    best_global_fitness = float('-inf')
    
    for gen in range(GEN):
        for island_idx in range(num_islands):
            pop = islands[island_idx]
            fitness_list = [ind.fitness(tester) for ind in pop]
            
            current_best_idx = np.argmax(fitness_list)
            current_best = pop[current_best_idx]
            current_fitness = fitness_list[current_best_idx]
            
            if current_fitness > best_global_fitness:
                best_global_fitness = current_fitness
                best_global = current_best.copy()
            
            new_pop = []
            while len(new_pop) < POP_SIZE:
                parents = select_parent_torneio(pop, fitness_list, 3)
                child = parents[0].cross(parents[1])
                child = child.mutate(MU_TAX)
                new_pop.append(child)
            islands[island_idx] = new_pop
        
        if gen % migration_interval == 0 and gen != 0 and num_islands > 1:
            migrants = []
            for island in islands:
                island.sort(key=lambda ind: ind.fitness(tester), reverse=True)
                migrants.append(island[:migration_size])
            
            for i in range(num_islands):
                next_idx = (i + 1) % num_islands
                islands[next_idx].sort(key=lambda ind: ind.fitness(tester))
                for j in range(migration_size):
                    islands[next_idx][j] = migrants[i][j].copy()
        history.append(best_global_fitness)
    
    return best_global, history
def best_realocate_island(initial_positions, min_matrix, max_matrix, b, num_islands, migration_interval, migration_size):
    return run_ag_island(initial_positions, min_matrix, max_matrix, b, num_islands, migration_interval, migration_size)
POP_SIZE = 200
GEN = 200
MU_TAX = 0.1

NUM_ISLAND = 4
MIGRATION_INT = 5
MIGRATION_SIZE = 2

initial_positions = [
    [1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1]
]
min_matrix = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
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
n = len(initial_positions)
tester = [[0.0] * n for _ in range(n)]
for i in range(n):
    for j in range(n):
        if initial_positions[i][j] != 0:
            val = initial_positions[i][j] * random.random() * (max_matrix[i][j] - min_matrix[i][j])
            tester[i][j] = val
            tester[j][i] = val
try:
    A = np.array(tester)
    b_np = np.array(b)
    solution = np.linalg.solve(A, b_np)
    print("Solução:")
    for s in solution:
        print(f"== {s}")
except np.linalg.LinAlgError as e:
    print("Erro ao resolver:", e)

ind_initial = Individual(initial_positions, min_matrix, max_matrix, b)
print("Fitness:", ind_initial.fitness(tester))

best, history = best_realocate(initial_positions, min_matrix, max_matrix, b)
print("Melhor indivíduo encontrado:", best)

best_island, history = best_realocate_island(
    initial_positions,
    min_matrix,
    max_matrix,
    b,
    NUM_ISLAND,
    MIGRATION_INT,
    MIGRATION_SIZE,
)
try:
    A = np.array(best_island._initialize_graph())
    b_np = np.array(b)
    solution = np.linalg.solve(A, b_np)
    print("Solução:")
    for s in solution:
        print(f"== {s}")
except np.linalg.LinAlgError as e:
    print("Erro ao resolver:", e)
print("Fitness:", best_island.fitness(tester))
print("Melhor indivíduo com estratégia de ilhas:")
print(best_island)