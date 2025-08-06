import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from typing import List
from dotenv import load_dotenv
import os

from methods import generate_tester, select_parent_torneio, Individual

load_dotenv()

# Dados fixos do grafo e vetor b
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
max_matrix = min_matrix
b_vector = [200, 75, 175, 90, 100]

class AGApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aplicação AG com Tkinter")
        self.geometry("900x450")
        self._create_main_layout()

    def _create_main_layout(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        input_frame = ttk.LabelFrame(main_frame, text="Parâmetros de Treinamento")
        input_frame.grid(row=0, column=0, sticky="nsew", padx=(0,10))
        graph_frame = ttk.LabelFrame(main_frame, text="Cadastro do Grafo")
        graph_frame.grid(row=0, column=1, sticky="nsew")

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        labels = ["População (POP_SIZE)", "Gerações (GEN)", "Tamanho do Torneio", "Taxa de Mutação"]
        self.entries = {}
        defaults = [os.getenv('POP_SIZE', 10), os.getenv('GEN', 1000), 2, os.getenv('MU_TAX', 0.05)]
        for i, text in enumerate(labels):
            ttk.Label(input_frame, text=text).grid(row=i, column=0, sticky="w", pady=5)
            ent = ttk.Entry(input_frame)
            ent.insert(0, str(defaults[i]))
            ent.grid(row=i, column=1, pady=5)
            self.entries[text] = ent

        ttk.Button(input_frame, text="Realizar Previsão", command=self._on_run)\
            .grid(row=len(labels), column=0, columnspan=2, pady=(20,0))

        ttk.Label(graph_frame, text="[Inputs do grafo a implementar]").pack(padx=10, pady=10)

    def _on_run(self):
        try:
            POP_SIZE = int(self.entries["População (POP_SIZE)"].get())
            GEN = int(self.entries["Gerações (GEN)"].get())
            TOURNAMENT_SIZE = int(self.entries["Tamanho do Torneio"].get())
            MU_TAX = float(self.entries["Taxa de Mutação"].get())
        except Exception:
            messagebox.showerror("Erro", "Verifique os valores de entrada.")
            return

        best, history = self.run_ag(POP_SIZE, GEN, TOURNAMENT_SIZE, MU_TAX)

        win = tk.Toplevel(self)
        win.title("Resultado da Previsão")
        win.geometry("900x450")

        left = ttk.Frame(win)
        left.grid(row=0, column=0, sticky="nsew", padx=(10,5), pady=10)
        right = ttk.Frame(win)
        right.grid(row=0, column=1, sticky="nsew", padx=(5,10), pady=10)
        win.columnconfigure(0, weight=1)
        win.columnconfigure(1, weight=1)

        info = f"Melhor Fitness: {best.fitness(generate_tester(initial_positions, min_matrix, max_matrix)):.4f}"
        ttk.Label(left, text=info).pack(pady=10)

        fig = Figure(figsize=(5,4))
        ax = fig.add_subplot(111)
        ax.set_title("Evolução do Fitness")
        ax.set_xlabel("Geração")
        ax.set_ylabel("Fitness")
        ax.plot(history)

        canvas = FigureCanvasTkAgg(fig, master=right)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_ag(self, POP_SIZE: int, GEN: int, TOURNAMENT_SIZE: int, MU_TAX: float):
        template = Individual(initial_positions, min_matrix, max_matrix, b_vector)
        tester = generate_tester(initial_positions, min_matrix, max_matrix)

        population = [template.copy()] + [template.randomize() for _ in range(POP_SIZE-1)]
        best_ind, best_fit = None, float('-inf')
        history = []
        for _ in range(GEN):
            fits = [ind.fitness(tester) for ind in population]
            idx = int(np.argmax(fits))
            curr_fit = fits[idx]
            if curr_fit > best_fit:
                best_fit = curr_fit
                best_ind = population[idx].copy()
            history.append(curr_fit)

            new_pop, tries = [], 0
            while len(new_pop) < POP_SIZE and tries < 10*POP_SIZE:
                p1, p2 = select_parent_torneio(population, fits, TOURNAMENT_SIZE)
                child = p1.cross(p2).mutate(MU_TAX)
                if child.fitness(tester) > max(p1.fitness(tester), p2.fitness(tester)):
                    new_pop.append(child)
                tries += 1
            new_pop += [best_ind.copy()] * (POP_SIZE - len(new_pop))
            population = new_pop

        return best_ind, history

if __name__ == "__main__":
    app = AGApp()
    app.mainloop()
