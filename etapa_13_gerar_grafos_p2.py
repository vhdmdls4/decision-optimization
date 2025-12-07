import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
import io

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
    
    df_top = df.sort_values('phi_net', ascending=False).head(10).copy()
    df_top['label_id'] = df_top.groupby('metodo').cumcount() + 1
    df_top['label'] = df_top['metodo'] + "_" + df_top['label_id'].astype(str)
    
    G = nx.DiGraph()
    records = df_top.to_dict('records')
    for r in records: G.add_node(r['label'], **r)
    
    edges = []
    for i in range(len(records)-1):
        source = records[i]['label']
        target = records[i+1]['label']
        edges.append((source, target))
    G.add_edges_from(edges)
    
    ordered_nodes = df_top['label'].tolist()
    pos = {}
    angle_step = 2 * math.pi / len(ordered_nodes)
    for i, node in enumerate(ordered_nodes):
        theta = math.pi/2 - i * angle_step
        pos[node] = np.array([math.cos(theta), math.sin(theta)])
    
    plt.figure(figsize=(10, 10))
    
    node_colors = []
    for n in ordered_nodes:
        metodo = G.nodes[n]['metodo']
        if n == ordered_nodes[0]: c = '#FFD700'
        elif metodo == 'Pw': c = '#FFDAB9'
        else: c = '#ADD8E6'
        node_colors.append(c)
        
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, 
                          edgecolors='gray', linewidths=1.5)
    
    nx.draw_networkx_edges(G, pos, 
                          arrowstyle='-|>', arrowsize=30, 
                          edge_color='gray', width=2.0,
                          connectionstyle="arc3,rad=0.1",
                          node_size=2000)
    
    labels = {}
    for i, n in enumerate(ordered_nodes):
        phi = G.nodes[n]['phi_net']
        labels[n] = f"#{i+1}\n{n}\n$\phi$={phi:.2f}"
        
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold')
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700', markersize=15, label='Vencedora (#1)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFDAB9', markersize=15, label='Método Pw'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ADD8E6', markersize=15, label='Método Pe')
    ]
    plt.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    
    plt.title("Ranking PROMETHEE II (Top 10)\n(Disposição Horária por Ordem de Fluxo Líquido)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("grafo_promethee2_ranking.png", dpi=300)
    print("Grafo 2 salvo: grafo_promethee2_ranking.png")

if __name__ == "__main__":
    main()