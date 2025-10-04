from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Dict, List, Tuple
import csv

NETWORK_CONFIG = {
    "terminals": ["Термінал 1", "Термінал 2"],
    "warehouses": ["Склад 1", "Склад 2", "Склад 3", "Склад 4"],
    "stores": [f"Магазин {i}" for i in range(1, 15)],
    "edges": {
        "terminal_to_warehouse": [
            {"from": "Термінал 1", "to": "Склад 1", "cap": 25},
            {"from": "Термінал 1", "to": "Склад 2", "cap": 20},
            {"from": "Термінал 1", "to": "Склад 3", "cap": 15},
            {"from": "Термінал 2", "to": "Склад 3", "cap": 15},
            {"from": "Термінал 2", "to": "Склад 4", "cap": 30},
            {"from": "Термінал 2", "to": "Склад 2", "cap": 10},
        ],
        "warehouse_to_store": [
            {"from": "Склад 1", "to": "Магазин 1", "cap": 15},
            {"from": "Склад 1", "to": "Магазин 2", "cap": 10},
            {"from": "Склад 1", "to": "Магазин 3", "cap": 20},
            {"from": "Склад 2", "to": "Магазин 4", "cap": 15},
            {"from": "Склад 2", "to": "Магазин 5", "cap": 10},
            {"from": "Склад 2", "to": "Магазин 6", "cap": 25},
            {"from": "Склад 3", "to": "Магазин 7", "cap": 20},
            {"from": "Склад 3", "to": "Магазин 8", "cap": 15},
            {"from": "Склад 3", "to": "Магазин 9", "cap": 10},
            {"from": "Склад 4", "to": "Магазин 10", "cap": 20},
            {"from": "Склад 4", "to": "Магазин 11", "cap": 10},
            {"from": "Склад 4", "to": "Магазин 12", "cap": 15},
            {"from": "Склад 4", "to": "Магазин 13", "cap": 5},
            {"from": "Склад 4", "to": "Магазин 14", "cap": 10},
        ]
    }
}

@dataclass
class Graph:
    vertices: List[str]
    vertex_map: Dict[str, int]
    capacity: List[List[int]]
    terminal_nodes: List[str]
    warehouse_nodes: List[str]
    store_nodes: List[str]

def add_capacity_edge(capacity_matrix: List[List[int]], vertex_map: Dict[str, int], source: str, target: str, capacity: int) -> None:
    capacity_matrix[vertex_map[source]][vertex_map[target]] += capacity

def construct_graph(config: dict) -> Graph:
    """Створює граф на основі конфігурації."""
    required = ["terminals", "warehouses", "stores", "edges"]
    edge_types = ["terminal_to_warehouse", "warehouse_to_store"]
    for key in required:
        if key not in config:
            raise ValueError(f"Відсутній ключ у конфігурації: {key}")
    for edge_type in edge_types:
        if edge_type not in config["edges"]:
            raise ValueError(f"Відсутній тип ребер: {edge_type}")

    terminals = config["terminals"].copy()
    warehouses = config["warehouses"].copy()
    stores = config["stores"].copy()
    vertices = ["Source"] + terminals + warehouses + stores + ["Sink"]
    vertex_map = {v: i for i, v in enumerate(vertices)}
    num_vertices = len(vertices)
    capacity_matrix = [[0] * num_vertices for _ in range(num_vertices)]

    total_capacity = sum(edge["cap"] for edge in config["edges"]["warehouse_to_store"])
    large_capacity = max(1, total_capacity)

    for term in terminals:
        add_capacity_edge(capacity_matrix, vertex_map, "Source", term, large_capacity)
    for store in stores:
        add_capacity_edge(capacity_matrix, vertex_map, store, "Sink", large_capacity)

    for edge in config["edges"]["terminal_to_warehouse"]:
        source, target, cap = edge["from"], edge["to"], edge["cap"]
        if source not in terminals:
            raise ValueError(f"Термінал {source} не знайдено")
        if target not in warehouses:
            raise ValueError(f"Склад {target} не знайдено")
        add_capacity_edge(capacity_matrix, vertex_map, source, target, cap)

    for edge in config["edges"]["warehouse_to_store"]:
        source, target, cap = edge["from"], edge["to"], edge["cap"]
        if source not in warehouses:
            raise ValueError(f"Склад {source} не знайдено")
        if target not in stores:
            raise ValueError(f"Магазин {target} не знайдено")
        add_capacity_edge(capacity_matrix, vertex_map, source, target, cap)

    return Graph(vertices, vertex_map, capacity_matrix, terminals, warehouses, stores)

def edmonds_karp_algorithm(capacity: List[List[int]], source_idx: int, sink_idx: int, vertex_names: List[str] = None, verbose: bool = False) -> Tuple[int, List[List[int]]]:
    """Реалізація алгоритму Едмондса-Карпа для максимального потоку."""
    num_vertices = len(capacity)
    flow = [[0] * num_vertices for _ in range(num_vertices)]
    total_flow = 0
    iteration = 0

    while True:
        iteration += 1
        parents = [-1] * num_vertices
        parents[source_idx] = source_idx
        flow_limits = [0] * num_vertices
        flow_limits[source_idx] = float("inf")
        queue = deque([source_idx])

        while queue:
            current = queue.popleft()
            for next_vertex in range(num_vertices):
                residual = capacity[current][next_vertex] - flow[current][next_vertex]
                if residual > 0 and parents[next_vertex] == -1:
                    parents[next_vertex] = current
                    flow_limits[next_vertex] = min(flow_limits[current], residual)
                    if next_vertex == sink_idx:
                        queue.clear()
                        break
                    queue.append(next_vertex)

        if parents[sink_idx] == -1:
            iteration -= 1
            break

        increment = flow_limits[sink_idx]
        total_flow += increment

        if verbose and vertex_names:
            path = []
            vertex = sink_idx
            while vertex != source_idx:
                path.append(vertex)
                vertex = parents[vertex]
            path.append(source_idx)
            path.reverse()
            path_str = " → ".join(vertex_names[i] for i in path)
            print(f"[Ітерація {iteration:02d}] Шлях: {path_str} | Приріст: {increment}")

        vertex = sink_idx
        while vertex != source_idx:
            prev_vertex = parents[vertex]
            flow[prev_vertex][vertex] += increment
            flow[vertex][prev_vertex] -= increment
            vertex = prev_vertex

    if verbose:
        print(f"Загальна кількість ітерацій: {iteration}")
    return total_flow, flow

def compute_residual(capacity: List[List[int]], flow: List[List[int]]) -> List[List[int]]:
    """Обчислює залишкову мережу."""
    num_vertices = len(capacity)
    residual = [[0] * num_vertices for _ in range(num_vertices)]
    for i in range(num_vertices):
        for j in range(num_vertices):
            residual[i][j] = capacity[i][j] - flow[i][j]
    return residual

def find_min_cut(graph: Graph, flow: List[List[int]]) -> List[Tuple[str, str, int]]:
    """Знаходить ребра мінімального розрізу."""
    residual = compute_residual(graph.capacity, flow)
    visited = [False] * len(graph.vertices)
    queue = deque([graph.vertex_map["Source"]])
    visited[graph.vertex_map["Source"]] = True

    while queue:
        current = queue.popleft()
        for next_vertex in range(len(graph.vertices)):
            if residual[current][next_vertex] > 0 and not visited[next_vertex]:
                visited[next_vertex] = True
                queue.append(next_vertex)

    source_side = {graph.vertices[i] for i, v in enumerate(visited) if v}
    sink_side = set(graph.vertices) - source_side
    cut_edges = []
    for u in source_side:
        for v in sink_side:
            cap = graph.capacity[graph.vertex_map[u]][graph.vertex_map[v]]
            if cap > 0:
                cut_edges.append((u, v, cap))
    return cut_edges

def extract_flows(graph: Graph, flow: List[List[int]]) -> Dict[Tuple[str, str], int]:
    """Витягує потоки по ребрах."""
    flows = {}
    for u_name, u_idx in graph.vertex_map.items():
        for v_name, v_idx in graph.vertex_map.items():
            if graph.capacity[u_idx][v_idx] > 0 and flow[u_idx][v_idx] > 0:
                flows[(u_name, v_name)] = flow[u_idx][v_idx]
    return flows

def decompose_flows(flows: Dict[Tuple[str, str], int], terminals: List[str], warehouses: List[str], stores: List[str]) -> Dict[Tuple[str, str], int]:
    """Розкладає потоки від терміналів до магазинів."""
    term_to_ware = defaultdict(int)
    ware_to_store = defaultdict(int)
    store_to_sink = defaultdict(int)

    for (u, v), f in flows.items():
        if u in terminals and v in warehouses:
            term_to_ware[(u, v)] += f
        elif u in warehouses and v in stores:
            ware_to_store[(u, v)] += f
        elif u in stores and v == "Sink":
            store_to_sink[u] += f

    ware_in = defaultdict(lambda: defaultdict(int))
    ware_out = defaultdict(lambda: defaultdict(int))
    for (t, w), f in term_to_ware.items():
        ware_in[w][t] += f
    for (w, s), f in ware_to_store.items():
        ware_out[w][s] += f

    result = defaultdict(int)
    for warehouse in warehouses:
        inflows = dict(ware_in[warehouse])
        outflows = dict(ware_out[warehouse])
        if not inflows or not outflows:
            continue
        term_keys = list(inflows.keys())
        store_keys = list(outflows.keys())
        t_idx = s_idx = 0
        while sum(inflows.values()) > 0 and sum(outflows.values()) > 0:
            term = term_keys[t_idx % len(term_keys)]
            store = store_keys[s_idx % len(store_keys)]
            if inflows[term] == 0:
                t_idx += 1
                continue
            if outflows[store] == 0:
                s_idx += 1
                continue
            delta = min(inflows[term], outflows[store])
            inflows[term] -= delta
            outflows[store] -= delta
            result[(term, store)] += delta
            if outflows[store] == 0:
                s_idx += 1
            if inflows[term] == 0:
                t_idx += 1
    return result

def display_flow_table(terminal_to_store: Dict[Tuple[str, str], int]) -> None:
    """Виводить таблицю потоків."""
    print("\nТаблиця потоків між терміналами та магазинами")
    header = ["Термінал", "Магазин", "Потік (од.)"]
    separator = "-" * 35
    print(separator)
    print(f"{header[0]:<12} | {header[1]:<12} | {header[2]:<10}")
    print(separator)
    for (term, store), value in sorted(terminal_to_store.items()):
        print(f"{term:<12} | {store:<12} | {value:<10}")
    print(separator)

def save_flows_to_csv(terminal_to_store: Dict[Tuple[str, str], int], filename: str = "flows_output.csv") -> None:
    """Зберігає потоки у CSV-файл."""
    try:
        with open(filename, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Термінал", "Магазин", "Потік (од.)"])
            for (term, store), value in sorted(terminal_to_store.items()):
                writer.writerow([term, store, value])
        print(f"\n💾 Збережено у CSV: {filename}")
    except Exception as e:
        print(f"❌ Помилка при записі CSV: {e}")

def main():
    """Основна функція для обчислення максимального потоку з дефолтною конфігурацією."""
    # Використовуємо дефолтну конфігурацію
    try:
        graph = construct_graph(NETWORK_CONFIG)
    except ValueError as e:
        print(f"❌ Помилка конфігурації: {e}")
        return

    source_idx = graph.vertex_map["Source"]
    sink_idx = graph.vertex_map["Sink"]
    max_flow, flow_matrix = edmonds_karp_algorithm(graph.capacity, source_idx, sink_idx, graph.vertices, verbose=True)

    edge_flows = extract_flows(graph, flow_matrix)
    terminal_to_store = decompose_flows(edge_flows, graph.terminal_nodes, graph.warehouse_nodes, graph.store_nodes)

    print("\n=== Результати обчислень ===")
    print(f"Максимальний потік: {max_flow} одиниць")

    min_cut = find_min_cut(graph, flow_matrix)
    print("\nМінімальний розріз (вузькі місця):")
    for u, v, cap in min_cut:
        critical = " (критичне Т→Ск)" if u in graph.terminal_nodes and v in graph.warehouse_nodes else ""
        print(f"  {u} → {v}: {cap}{critical}")

    display_flow_table(terminal_to_store)

    # Обчислення агрегованих даних
    by_terminal = defaultdict(int)
    by_store = defaultdict(int)
    for (term, store), value in terminal_to_store.items():
        by_terminal[term] += value
        by_store[store] += value

    print("\nПоставки з терміналів:")
    for term in graph.terminal_nodes:
        print(f"  {term}: {by_terminal[term]} одиниць")

    print("\nНадходження до магазинів:")
    for store in graph.store_nodes:
        print(f"  {store}: {by_store[store]} одиниць")

    print("\n=== Аналітичні висновки ===")
    if by_terminal:
        top_terminal = max(by_terminal.items(), key=lambda x: x[1])[0]
        print(f"1) Термінал з найбільшим потоком: {top_terminal}")
    print("2) Критичні маршрути: ребра Т→Ск у мінрозрізі, що обмежують потік.")
    nonzero_stores = [(s, v) for s, v in by_store.items() if v > 0]
    zero_stores = [s for s in graph.store_nodes if by_store[s] == 0]
    min_store = min(nonzero_stores, key=lambda x: x[1])[0] if nonzero_stores else None
    print(f"3) Найменші поставки: {min_store if min_store else '—'}, нульові: {', '.join(zero_stores) if zero_stores else 'немає'}.")
    print("   Збільшення можливе за рахунок розширення місткостей ребер Т→Ск.")
    print("4) Вузькі місця: виходи з терміналів. Їх розширення підвищить max-flow.")

    save_flows_to_csv(terminal_to_store)

if __name__ == "__main__":
    main()