
import random
import copy
from typing import List
import numpy as np

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

    def fitness(self) -> float:
        tester = generate_tester(self.initial_positions, self.min_matrix, self.max_matrix)
        n = self.n
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                A[i][j] = self.initial_positions[i][j] * tester[i][j]
        
        try:
            solution = np.linalg.solve(A, self.b_vector.copy())
            total_abs = sum(abs(x) for x in solution) 
            if total_abs == 0:
                fitness_value = 0.0
            else:
                fitness_value = 10.0 * n / total_abs
        except Exception as e:
            fitness_value = 0.0
        
        return fitness_value


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


def round_to_nearest_5(value: float) -> float:
    return round(value / 10) * 10

def generate_tester(initial_positions: List[List[float]],
                    min_matrix: List[List[float]],
                    max_matrix: List[List[float]]) -> List[List[float]]:
    n = len(initial_positions)
    tester = [[0.0] * n for _ in range(n)]

    value_pool = []
    for i in range(n):
        for j in range(i, n):
            if max_matrix[i][j] > 0:
                min_val = round_to_nearest_5(min_matrix[i][j])
                max_val = round_to_nearest_5(max_matrix[i][j])
                possible_vals = list(range(int(min_val), int(max_val) + 1, 10))
                value_pool.extend(possible_vals)

    for i in range(n):
        for j in range(i, n):
            if max_matrix[i][j] > 0:
                while True:
                    val = random.choice(value_pool)
                    if min_matrix[i][j] <= val <= max_matrix[i][j]:
                        break
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