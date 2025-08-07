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

class AGApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aplicação AG com Tkinter")
        self.geometry("700x700")
        self._create_main_layout()

    def _create_main_layout(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Parâmetros de Treinamento
        input_frame = ttk.LabelFrame(main_frame, text="Parâmetros de Treinamento")
        input_frame.grid(row=0, column=0, sticky="nsew", padx=(0,10), pady=10)

        # Cadastro do Grafo
        graph_frame = ttk.LabelFrame(main_frame, text="Cadastro do Grafo")
        graph_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=10)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Inputs de AG
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

        # Inputs de Grafo
        ttk.Label(graph_frame, text="Número de nós (n):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.n_entry = ttk.Entry(graph_frame)
        self.n_entry.insert(0, '5')
        self.n_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        ttk.Button(graph_frame, text="Gerar Estruturas", command=self._generate_graph_inputs)\
            .grid(row=0, column=2, padx=5, pady=5)

        # Frames onde as matrizes serão inseridas
        canvas = tk.Canvas(graph_frame, height=500, width=500)
        canvas.grid(row=1, column=0, columnspan=3, sticky="nsew")

        scrollbar_y = ttk.Scrollbar(graph_frame, orient="vertical", command=canvas.yview)
        scrollbar_y.grid(row=1, column=4, sticky="ns")

        scrollbar_x = ttk.Scrollbar(graph_frame, orient="horizontal", command=canvas.xview)
        scrollbar_x.grid(row=2, column=0, columnspan=4, sticky="ew")

        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        # Frame rolável dentro do canvas
        self.struct_frame = ttk.Frame(canvas)
        self.struct_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.struct_frame, anchor="nw")


    def _generate_graph_inputs(self):
        # Limpa qualquer conteúdo anterior
        for widget in self.struct_frame.winfo_children():
            widget.destroy()

        try:
            n = int(self.n_entry.get())
            if n <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Erro", "Informe um valor inteiro positivo para n.")
            return

        for widget in self.struct_frame.winfo_children():
            widget.destroy()

        self.matrix_entries = {
            'initial': [], 'min': [], 'max': [], 'b': []
        }

        # Criação de frame para os títulos (labels)
        header_frame = ttk.Frame(self.struct_frame)
        header_frame.grid(row=0, column=0, sticky='w', pady=(0, 5))

        # Títulos dos grupos
        titles = ['Matriz Inicial', 'Min Matrix', 'Max Matrix', 'Vetor b']
        padx_values = [5,5,5,5]  # mesmo espaçamento lateral usado nos packs
        lat = 35
        group_widths = [n * lat, n * lat, n * lat, lat*2]  # largura estimada para alinhamento

        for title, padx, width in zip(titles, padx_values, group_widths):
            lbl = ttk.Label(header_frame, text=title, anchor='center', width=width//6)
            lbl.pack(side='left', padx=padx)

        # Geração das entradas
        for i in range(n):
            row_frame = ttk.Frame(self.struct_frame)
            row_frame.grid(row=1 + i, column=0, sticky='w', pady=2)

            # Frames para cada grupo de entradas
            frame_init = ttk.Frame(row_frame)
            frame_init.pack(side='left', padx=20)

            frame_min = ttk.Frame(row_frame)
            frame_min.pack(side='left', padx=20)

            frame_max = ttk.Frame(row_frame)
            frame_max.pack(side='left', padx=20)

            frame_b = ttk.Frame(row_frame)
            frame_b.pack(side='left', padx=20)

            row_init = []
            row_min = []
            row_max = []

            for j in range(n):
                ent_init = ttk.Entry(frame_init, width=4)
                ent_init.insert(0, '0')
                ent_init.grid(row=0, column=j, padx=2)
                row_init.append(ent_init)

                ent_min = ttk.Entry(frame_min, width=4)
                ent_min.insert(0, '0')
                ent_min.grid(row=0, column=j, padx=2)
                row_min.append(ent_min)

                ent_max = ttk.Entry(frame_max, width=4)
                ent_max.insert(0, '0')
                ent_max.grid(row=0, column=j, padx=2)
                row_max.append(ent_max)

            ent_b = ttk.Entry(frame_b, width=6)
            ent_b.insert(0, '0')
            ent_b.grid(row=0, column=0, padx=2)

            self.matrix_entries['initial'].append(row_init)
            self.matrix_entries['min'].append(row_min)
            self.matrix_entries['max'].append(row_max)
            self.matrix_entries['b'].append(ent_b)




    def _collect_graph_structures(self, n: int):
        initial = []
        min_m = []
        max_m = []
        b_vec = []

        for i in range(n):
            row_init = []
            row_min = []
            row_max = []
            for j in range(n):
                row_init.append(int(self.matrix_entries['initial'][i][j].get()))
                row_min.append(int(self.matrix_entries['min'][i][j].get()))
                row_max.append(int(self.matrix_entries['max'][i][j].get()))
            initial.append(row_init)
            min_m.append(row_min)
            max_m.append(row_max)
            b_vec.append(int(self.matrix_entries['b'][i].get()))

        return initial, min_m, max_m, b_vec

    def _on_run(self):
        # Coleta parâmetros de AG
        try:
            POP_SIZE = int(self.entries["População (POP_SIZE)"].get())
            GEN = int(self.entries["Gerações (GEN)"].get())
            TOURNAMENT_SIZE = int(self.entries["Tamanho do Torneio"].get())
            MU_TAX = float(self.entries["Taxa de Mutação"].get())
            n = int(self.n_entry.get())
        except Exception:
            messagebox.showerror("Erro", "Verifique os valores de entrada.")
            return

        # Coleta estruturas do grafo
        try:
            initial_positions, min_matrix, max_matrix, b_vector = self._collect_graph_structures(n)
        except Exception:
            messagebox.showerror("Erro", "Verifique as entradas do grafo.")
            return

        self.run_ag(POP_SIZE, GEN, TOURNAMENT_SIZE, MU_TAX,
                                     initial_positions, min_matrix, max_matrix, b_vector)

    def run_ag(self, POP_SIZE: int, GEN: int, TOURNAMENT_SIZE: int, MU_TAX: float,
           initial_positions: List[List[int]], min_matrix: List[List[int]],
           max_matrix: List[List[int]], b_vector: List[int]):
    
        template = Individual(initial_positions, min_matrix, max_matrix, b_vector)
        tester = generate_tester(initial_positions, min_matrix, max_matrix)
        population = [template.copy()] + [template.randomize() for _ in range(POP_SIZE - 1)]

        best_individual = None
        best_fitness = float('-inf')
        history = []
        melhoria_indices = []
        melhoria_valores = []

        # Janela de resultado
        win = tk.Toplevel(self)
        win.title("Resultado da Previsão Interativa")
        win.geometry("900x500")
        left = ttk.Frame(win)
        left.grid(row=0, column=0, sticky="nsew", padx=(10,5), pady=10)
        right = ttk.Frame(win)
        right.grid(row=0, column=1, sticky="nsew", padx=(5,10), pady=10)
        win.columnconfigure(0, weight=1)
        win.columnconfigure(1, weight=1)

        # Gráfico dinâmico
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.set_title("Evolução do Fitness ao Longo das Gerações")
        ax.set_xlabel("Geração")
        ax.set_ylabel("Fitness")
        linha, = ax.plot([], [], label="Melhor Fitness")
        pontos, = ax.plot([], [], 'ro', label="Pontos de Inflexão")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=right)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        for gen in range(GEN):
            fitness_list = [ind.fitness() for ind in population]
            idx = np.argmax(fitness_list)
            curr_fit = fitness_list[idx]
            best_candidate = population[idx]

            if curr_fit > best_fitness:
                best_fitness = curr_fit
                best_individual = best_candidate.copy()
                melhoria_indices.append(gen)
                melhoria_valores.append(curr_fit)

            history.append(curr_fit)

            # Atualiza gráfico a cada 20 gerações (ou nas bordas)
            if gen % 20 == 0 or gen == 0 or gen == GEN - 1:
                linha.set_xdata(range(len(history)))
                linha.set_ydata(history)
                pontos.set_xdata(melhoria_indices)
                pontos.set_ydata(melhoria_valores)
                ax.relim()
                ax.autoscale_view()
                canvas.draw()
                canvas.flush_events()

            new_pop = []
            tentativas = 0
            while len(new_pop) < POP_SIZE and tentativas < 10 * POP_SIZE:
                p1, p2 = select_parent_torneio(population, fitness_list, TOURNAMENT_SIZE)
                child = p1.cross(p2).mutate(MU_TAX)
                if child.fitness() > max(p1.fitness(), p2.fitness()):
                    new_pop.append(child)
                tentativas += 1

            while len(new_pop) < POP_SIZE:
                new_pop.append(best_individual.copy())

            population = new_pop

        info = f"Melhor Fitness: {best_individual.fitness():.4f}"
        ttk.Label(left, text=info).pack(pady=10)

        return best_individual, history


if __name__ == "__main__":
    app = AGApp()
    app.mainloop()
