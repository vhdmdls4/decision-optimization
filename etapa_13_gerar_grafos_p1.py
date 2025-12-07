import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
import io

# Dados do CSV (Resultados PROMETHEE)
csv_data = """Custo Total,Equilibrio,Robustez (Norm 0-100),Estabilidade (Norm 0-100),Score_Global,origem,phi,phi_plus,phi_minus
930.0,2.0,83.8199908470547,1.7791689736818013,0.7549465702096202,Pw,0.27428125298974537,0.5380684210526315,0.41187894736842107
951.0,2.0,100.0,1.1461223138897183,0.7518084209643172,Pw,0.2676746229996338,0.5414421052631578,0.4085052631578948
966.0,0.0,88.55066623905907,0.0,0.7463546819294713,Pw,0.25619306713680007,0.528257894736842,0.4287526315789473
909.0,3.0,68.67840772118832,2.6165743855429544,0.7414762029858418,Pw,0.23129186602870816,0.516557894736842,0.41372631578947366
958.0,1.0,89.80529983796334,1.7656338006277123,0.7132938883207399,Pw,0.22272727272727267,0.5191315789473684,0.4191315789473684
921.0,5.0,66.65498610400266,11.947592691405106,0.6947375603790441,Pe,0.11703131796433232,0.4578157894736842,0.4593473684210526
980.0,2.0,90.45206681121077,1.5791172676408456,0.69408165079343,Pe,0.11526315789473681,0.4636842105263158,0.4636842105263158
935.0,5.0,68.02629973581658,10.244778361248857,0.6748572301135959,Pe,0.08272335798173099,0.4406263157894737,0.4775315789473684
903.0,7.0,52.62210808921415,22.661562413718514,0.6722082342471006,Pe,0.06315789473684209,0.4284210526315789,0.49157894736842104
902.0,8.0,46.30022403161329,26.57945503438207,0.6488635979534846,Pe,0.05094867981893377,0.505021052631579,0.4949789473684211"""

def main():
    df = pd.read_csv(io.StringIO(csv_data))
    df = df.rename(columns={'origem': 'metodo', 'phi': 'phi_net'})
    
    # 1. Filtra Top 10
    df_top = df.sort_values('phi_net', ascending=False).head(10).copy()
    
    # Labels Curtos
    df_top['label_id'] = df_top.groupby('metodo').cumcount() + 1
    df_top['label'] = df_top['metodo'] + "_" + df_top['label_id'].astype(str)
    
    # 2. Cria Grafo e Arestas (PROMETHEE I)
    G = nx.DiGraph()
    node_data = df_top.to_dict('records')
    for r in node_data: G.add_node(r['label'], **r)
    
    edges = []
    for a in node_data:
        for b in node_data:
            if a['label'] == b['label']: continue
            # Dominância Estrita: Phi+(A) >= Phi+(B) AND Phi-(A) <= Phi-(B)
            # Com pelo menos um > ou <
            better_plus = a['phi_plus'] >= b['phi_plus'] - 1e-6
            better_minus = a['phi_minus'] <= b['phi_minus'] + 1e-6
            strict = (a['phi_plus'] > b['phi_plus'] + 1e-6) or \
                     (a['phi_minus'] < b['phi_minus'] - 1e-6)
            
            if better_plus and better_minus and strict:
                edges.append((a['label'], b['label']))
    
    G.add_edges_from(edges)
    
    # 3. Identifica Kernel (Nós sem arestas de entrada no subgrafo)
    # Kernel = Não dominados por ninguém do grupo
    dominated_nodes = {v for u, v in edges}
    kernel_nodes = [n for n in G.nodes() if n not in dominated_nodes]
    
    # 4. Layout Circular (Sem ordem de phi_net, apenas visual)
    pos = nx.circular_layout(G)
    
    # 5. Estilo
    node_colors = []
    for n in G.nodes():
        if n in kernel_nodes:
            c = '#90EE90' # Light Green (KERNEL)
        else:
            c = '#E0E0E0' # Cinza claro (Dominado)
        node_colors.append(c)
        
    plt.figure(figsize=(10, 10))
    
    # Nós
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2200, 
                          edgecolors='gray', linewidths=2.0)
    
    # Arestas (Curvas e com seta visível)
    nx.draw_networkx_edges(G, pos, 
                          arrowstyle='-|>', arrowsize=30, 
                          edge_color='#555555', width=1.5,
                          connectionstyle="arc3,rad=0.1",
                          node_size=2200)
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Legenda
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#90EE90', markersize=15, label='Kernel (Não Dominada)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E0E0E0', markersize=15, label='Solução Dominada')
    ]
    plt.legend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))
    
    plt.title("Grafo PROMETHEE I (Top 10)\n(Setas: Dominância; Verde: Kernel)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("grafo_promethee1_kernel.png", dpi=300)
    print("Grafo 1 salvo: grafo_promethee1_kernel.png")

if __name__ == "__main__":
    main()