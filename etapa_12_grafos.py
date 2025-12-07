import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math

def load_data(filepath="resultado_final_promethee.csv"):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        try:
            df = pd.read_csv("tabela_promethee_ii.csv")
        except:
            print("Erro: Nenhum arquivo CSV encontrado.")
            return None

    if 'Phi_Plus' in df.columns:
        df = df.rename(columns={'Phi_Plus': 'phi_plus', 'Phi_Minus': 'phi_minus', 'Phi_Net': 'phi_net'})

    df = df.sort_values(['metodo', 'phi_net'], ascending=[True, False])
    df['label_id'] = df.groupby('metodo').cumcount() + 1
    df['label'] = df['metodo'] + "_" + df['label_id'].astype(str)
    
    return df

def get_dominance_edges(df_sub):
    edges = []
    nodes = df_sub.to_dict('records')
    
    for a in nodes:
        for b in nodes:
            if a['label'] == b['label']: continue
            
            better_plus = a['phi_plus'] >= b['phi_plus'] - 1e-6
            better_minus = a['phi_minus'] <= b['phi_minus'] + 1e-6
            strict = (a['phi_plus'] > b['phi_plus'] + 1e-6) or \
                     (a['phi_minus'] < b['phi_minus'] - 1e-6)
            
            if better_plus and better_minus and strict:
                edges.append((a['label'], b['label']))
    
    return edges

def get_kernel(nodes_list, edges):
    targets = {v for u, v in edges}
    all_nodes = {n['label'] for n in nodes_list}
    return list(all_nodes - targets)

def plot_circular_graph(df, top_n=10, filename="grafo_promethee_circular.png"):
    df_top = df.sort_values('phi_net', ascending=False).head(top_n).copy()
    
    G = nx.DiGraph()
    
    node_records = df_top.to_dict('records')
    for row in node_records:
        G.add_node(row['label'], **row)
        
    edges = get_dominance_edges(df_top)
    G.add_edges_from(edges)

    kernel_nodes = get_kernel(node_records, edges)
    best_node = df_top.iloc[0]['label']
    
    plt.figure(figsize=(10, 10))
    
    ordered_nodes = df_top['label'].tolist()
    pos = nx.circular_layout(G)
    
    angle_step = 2 * math.pi / len(ordered_nodes)
    for i, node in enumerate(ordered_nodes):
        theta = math.pi/2 - i * angle_step
        pos[node] = np.array([math.cos(theta), math.sin(theta)])
        
    node_colors = []
    edge_colors = []
    sizes = []
    labels = {}
    
    for node in G.nodes():
        if node in kernel_nodes:
            c = '#90EE90' 
        elif G.nodes[node]['metodo'] == 'Pw':
            c = '#FFDAB9' 
        else:
            c = '#ADD8E6' 
            
        if node == best_node:
            c = '#FFD700'
            s = 2500
        else:
            s = 1500
            
        node_colors.append(c)
        sizes.append(s)
        
        rank = df_top[df_top['label'] == node].index[0] + 1
        labels[node] = f"{node}\n#{rank}"

    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors, 
                          node_size=sizes, 
                          node_shape='o', 
                          edgecolors='gray', 
                          linewidths=1.5)
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold')
    
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray', 
                          arrowstyle='-|>', 
                          arrowsize=20, 
                          connectionstyle="arc3,rad=0.15",
                          width=1.2,
                          alpha=0.6)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700', markersize=15, label='Melhor Solução (Promethee II)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#90EE90', markersize=15, label='Kernel (Não Dominadas)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFDAB9', markersize=15, label='Pw (Dominada)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ADD8E6', markersize=15, label='Pe (Dominada)')
    ]
    
    plt.legend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))
    
    plt.title(f"Grafo de Sobreclassificação PROMETHEE I (Top {top_n})\n(Setas indicam dominância. Disposição por Ranking PROMETHEE II)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Grafo salvo: {filename}")

if __name__ == "__main__":
    df = load_data("resultado_final_promethee.csv")
    
    if df is not None:
        plot_circular_graph(df, top_n=10, filename="grafo_promethee_circular.png")