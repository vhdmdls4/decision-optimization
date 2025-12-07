import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math

# ============================================================
# 1. Carrega dados e prepara
# ============================================================

def load_data(filepath="resultado_final_promethee.csv"):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        # Tenta o nome alternativo gerado anteriormente
        try:
            df = pd.read_csv("tabela_promethee_ii.csv")
        except:
            print("Erro: Nenhum arquivo CSV encontrado.")
            return None

    # Garante colunas de fluxo (caso venha do script manual)
    if 'Phi_Plus' in df.columns: # Do script manual anterior
        df = df.rename(columns={'Phi_Plus': 'phi_plus', 'Phi_Minus': 'phi_minus', 'Phi_Net': 'phi_net'})
    
    # Cria labels curtos (Ex: Pw_1, Pe_1)
    # Ordena primeiro por metodo e depois por score para numerar
    df = df.sort_values(['metodo', 'phi_net'], ascending=[True, False])
    df['label_id'] = df.groupby('metodo').cumcount() + 1
    df['label'] = df['metodo'] + "_" + df['label_id'].astype(str)
    
    return df

# ============================================================
# 2. Lógica PROMETHEE I (Arestas de Dominância)
# ============================================================

def get_dominance_edges(df_sub):
    """
    Gera arestas A -> B se A domina B (Promethee I)
    Domínio: Phi+(A) >= Phi+(B) AND Phi-(A) <= Phi-(B)
    Com pelo menos uma desigualdade estrita.
    """
    edges = []
    nodes = df_sub.to_dict('records')
    
    for a in nodes:
        for b in nodes:
            if a['label'] == b['label']: continue
            
            # Checa condições
            # Maior Phi+ é bom
            better_plus = a['phi_plus'] >= b['phi_plus'] - 1e-6
            # Menor Phi- é bom
            better_minus = a['phi_minus'] <= b['phi_minus'] + 1e-6
            
            # Estrito
            strict = (a['phi_plus'] > b['phi_plus'] + 1e-6) or \
                     (a['phi_minus'] < b['phi_minus'] - 1e-6)
            
            if better_plus and better_minus and strict:
                edges.append((a['label'], b['label']))
    
    return edges

def get_kernel(nodes_list, edges):
    """Retorna nós que não possuem arestas de entrada (não são dominados)"""
    targets = {v for u, v in edges}
    all_nodes = {n['label'] for n in nodes_list}
    return list(all_nodes - targets)

# ============================================================
# 3. Plot Circular Bonito
# ============================================================

def plot_circular_graph(df, top_n=10, filename="grafo_promethee_circular.png"):
    # Seleciona Top N pelo PROMETHEE II (Phi Net)
    df_top = df.sort_values('phi_net', ascending=False).head(top_n).copy()
    
    # Cria grafo
    G = nx.DiGraph()
    
    # Adiciona nós com atributos
    node_records = df_top.to_dict('records')
    for row in node_records:
        G.add_node(row['label'], **row)
        
    # Adiciona arestas (Promethee I)
    edges = get_dominance_edges(df_top)
    G.add_edges_from(edges)
    
    # Identifica Kernel (dentro do subconjunto)
    kernel_nodes = get_kernel(node_records, edges)
    best_node = df_top.iloc[0]['label'] # O melhor do Promethee II
    
    plt.figure(figsize=(10, 10))
    
    # --- LAYOUT CIRCULAR INTELIGENTE ---
    # Ordenamos os nós pela lista ordenada do DataFrame (Melhor -> Pior)
    # Isso coloca o #1 no topo e os outros em sequência no círculo
    ordered_nodes = df_top['label'].tolist()
    pos = nx.circular_layout(G)
    
    # Reajusta posições para seguir a ordem do relógio (ou anti-horário)
    # A circular_layout padrão pode não seguir a ordem da lista. Vamos forçar:
    angle_step = 2 * math.pi / len(ordered_nodes)
    for i, node in enumerate(ordered_nodes):
        # Começa em pi/2 (90 graus, topo) e gira horário
        theta = math.pi/2 - i * angle_step
        pos[node] = np.array([math.cos(theta), math.sin(theta)])
        
    # --- ESTILOS ---
    node_colors = []
    edge_colors = []
    sizes = []
    labels = {}
    
    for node in G.nodes():
        # Cor baseada no Kernel vs Normal
        if node in kernel_nodes:
            c = '#90EE90' # Light Green (Kernel)
        elif G.nodes[node]['metodo'] == 'Pw':
            c = '#FFDAB9' # Peach Puff (Pw dominado)
        else:
            c = '#ADD8E6' # Light Blue (Pe dominado)
            
        # Destaque Vencedor Absoluto
        if node == best_node:
            c = '#FFD700' # Gold
            s = 2500
        else:
            s = 1500
            
        node_colors.append(c)
        sizes.append(s)
        
        # Label com Rank
        rank = df_top[df_top['label'] == node].index[0] + 1 # Indice no top N
        labels[node] = f"{node}\n#{rank}"

    # Desenhar Nós
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors, 
                          node_size=sizes, 
                          node_shape='o', 
                          edgecolors='gray', 
                          linewidths=1.5)
    
    # Desenhar Labels
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold')
    
    # Desenhar Arestas (Curvas para não sobrepor no meio)
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray', 
                          arrowstyle='-|>', 
                          arrowsize=20, 
                          connectionstyle="arc3,rad=0.15",
                          width=1.2,
                          alpha=0.6)
    
    # Legenda Personalizada
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

# ============================================================
# 4. EXECUÇÃO
# ============================================================
if __name__ == "__main__":
    # Tenta carregar o CSV final gerado no passo anterior
    df = load_data("resultado_final_promethee.csv")
    
    if df is not None:
        # Gera para Top 10 (mais informativo)
        plot_circular_graph(df, top_n=10, filename="grafo_promethee_circular.png")