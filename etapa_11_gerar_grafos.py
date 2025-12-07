# etapa_8_grafos_promethee_topN.py
# Geração de grafos PROMETHEE I e II para as TOP-N alternativas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CONFIGURAÇÃO PRINCIPAL: escolha TOP-N
# ---------------------------------------------------------
TOP_N = 10              # <<< troque aqui para 5 se quiser TOP-5
CSV_INPUT = "tabela_promethee_II.csv"


# ---------------------------------------------------------
# 1. Carregar resultados completos e filtrar TOP-N
# ---------------------------------------------------------
def load_and_filter_topN(path=CSV_INPUT, top_n=10):
    df = pd.read_csv(path)

    required = ["phi", "phi_plus", "phi_minus"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Coluna obrigatória '{c}' não encontrada em {path}")

    # Cria alt id
    if "origem" in df.columns:
        df["alt"] = [f"{row['origem']}_{i+1}" for i, row in df.reset_index(drop=True).iterrows()]
    else:
        df["alt"] = [f"A{i+1}" for i in range(len(df))]

    # Ordenar e pegar TOP-N
    df_top = df.sort_values("phi", ascending=False).head(top_n).reset_index(drop=True)
    return df_top


# ---------------------------------------------------------
# 2. Relações PROMETHEE I (parcial)
# ---------------------------------------------------------
def build_promethee_I_relations(df, tol=1e-6):
    phi_plus = df["phi_plus"].values
    phi_minus = df["phi_minus"].values
    n = len(df)

    P = np.zeros((n, n), dtype=bool)
    I = np.zeros((n, n), dtype=bool)
    J = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for k in range(n):
            if i == k:
                continue

            # φ+ (maior é melhor)
            if phi_plus[i] > phi_plus[k] + tol:
                rel_plus = "P"
            elif abs(phi_plus[i] - phi_plus[k]) <= tol:
                rel_plus = "I"
            else:
                rel_plus = "R"

            # φ- (menor é melhor)
            if phi_minus[i] + tol < phi_minus[k]:
                rel_minus = "P"
            elif abs(phi_minus[i] - phi_minus[k]) <= tol:
                rel_minus = "I"
            else:
                rel_minus = "R"

            # PROMETHEE I final
            if ((rel_plus == "P" and rel_minus == "P") or
                (rel_plus == "P" and rel_minus == "I") or
                (rel_plus == "I" and rel_minus == "P")):
                P[i, k] = True

            elif rel_plus == "I" and rel_minus == "I":
                I[i, k] = True

            else:
                J[i, k] = True

    return P, I, J


# ---------------------------------------------------------
# 3. Kernel = sem predecessores em P
# ---------------------------------------------------------
def compute_kernel(P):
    incoming = P.any(axis=0)
    return [i for i in range(len(P)) if not incoming[i]]


# ---------------------------------------------------------
# 4. Plot: PROMETHEE I – grafo parcial
# ---------------------------------------------------------
def plot_promethee_I(df, P, filename):
    n = len(df)
    labels = df["alt"].tolist()

    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = np.cos(ang)
    ys = np.sin(ang)

    plt.figure(figsize=(9, 9))
    ax = plt.gca()
    ax.axis("off")
    ax.set_aspect("equal")

    # nós
    for i in range(n):
        ax.scatter(xs[i], ys[i], s=600, color="white", edgecolors="black", zorder=3)
        ax.text(xs[i], ys[i], labels[i], ha="center", va="center", fontsize=9)

    # arestas P
    for i in range(n):
        for k in range(n):
            if P[i, k]:
                dx = xs[k] - xs[i]
                dy = ys[k] - ys[i]
                ax.annotate("",
                            xy=(xs[k]-0.08*dx, ys[k]-0.08*dy),
                            xytext=(xs[i]+0.08*dx, ys[i]+0.08*dy),
                            arrowprops=dict(arrowstyle="->", color="0.3", lw=1.3))

    plt.title(f"PROMETHEE I – Grafo de Sobreclassificação (TOP-{n})", fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[OK] PROMETHEE I salvo em {filename}")


# ---------------------------------------------------------
# 5. Plot PROMETHEE I com kernel
# ---------------------------------------------------------
def plot_promethee_I_kernel(df, P, kernel_idx, filename):
    n = len(df)
    labels = df["alt"].tolist()

    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = np.cos(ang)
    ys = np.sin(ang)

    plt.figure(figsize=(9, 9))
    ax = plt.gca()
    ax.axis("off")
    ax.set_aspect("equal")

    for i in range(n):
        if i in kernel_idx:
            fc = "gold"
            s = 750
        else:
            fc = "white"
            s = 600

        ax.scatter(xs[i], ys[i], s=s, color=fc, edgecolors="black")
        ax.text(xs[i], ys[i], labels[i], ha="center", va="center", fontsize=9)

    # Arestas P
    for i in range(n):
        for k in range(n):
            if P[i, k]:
                dx = xs[k] - xs[i]
                dy = ys[k] - ys[i]
                ax.annotate("",
                            xy=(xs[k]-0.08*dx, ys[k]-0.08*dy),
                            xytext=(xs[i]+0.08*dx, ys[i]+0.08*dy),
                            arrowprops=dict(arrowstyle="->", color="0.3", lw=1.3))

    ker_names = [labels[i] for i in kernel_idx]
    plt.title(f"PROMETHEE I – Kernel Destacado (TOP-{n})\nKernel = {{{', '.join(ker_names)}}}", fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[OK] PROMETHEE I + kernel salvo em {filename}")


# ---------------------------------------------------------
# 6. PROMETHEE II: grafo da pré-ordem total
# ---------------------------------------------------------
def plot_promethee_II_chain(df, filename):
    df_ord = df.sort_values("phi", ascending=False).reset_index(drop=True)
    n = len(df_ord)

    xs = np.zeros(n)
    ys = np.arange(n)[::-1]
    labels = df_ord["alt"]
    phis = df_ord["phi"]

    plt.figure(figsize=(6, 9))
    ax = plt.gca()
    ax.axis("off")

    # nós
    for i in range(n):
        ax.scatter(xs[i], ys[i], s=550, color="white", edgecolors="black")
        ax.text(xs[i]+0.05, ys[i], f"{labels[i]}  (φ={phis[i]:.3f})",
                ha="left", va="center", fontsize=9)

    # setas entre vizinhos
    for i in range(n-1):
        ax.annotate("",
                    xy=(xs[i+1], ys[i+1]+0.15),
                    xytext=(xs[i], ys[i]-0.15),
                    arrowprops=dict(arrowstyle="->", color="0.3", lw=1.5))

    plt.title(f"PROMETHEE II – Pré-ordem Total (TOP-{n})", fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[OK] PROMETHEE II salvo em {filename}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"Carregando TOP-{TOP_N} soluções…")
    df_top = load_and_filter_topN(CSV_INPUT, TOP_N)

    print("Construindo relações PROMETHEE I…")
    P, I, J = build_promethee_I_relations(df_top)

    kernel_idx = compute_kernel(P)
    print("Kernel (índices):", kernel_idx)
    print("Kernel (alternativas):", [df_top['alt'][i] for i in kernel_idx])

    # Gráficos PROMETHEE I
    plot_promethee_I(df_top, P, f"promethee_I_top{TOP_N}.png")
    plot_promethee_I_kernel(df_top, P, kernel_idx, f"promethee_I_kernel_top{TOP_N}.png")

    # Gráfico PROMETHEE II
    plot_promethee_II_chain(df_top, f"promethee_II_preordem_top{TOP_N}.png")
