import csv

def salvar_fronteiras_csv(fronteiras, nome_arquivo, tipo="Pw"):
    linhas = []
    for run_idx, fronteira in enumerate(fronteiras, start=1):
        for alt_idx, p in enumerate(fronteira, start=1):
            linha = {
                "run": run_idx,
                "alt": alt_idx,
                "f1": p["f1"],
                "f2": p["f2"],
                "pen_cap": p["pen_cap"],
            }

            if tipo == "Pw":
                linha["w1"] = p["w1"]
                linha["w2"] = p["w2"]
            else: 
                linha["epsilon"] = p["epsilon"]

            # salva também a solução como string
            linha["solucao"] = ",".join(str(int(a)) for a in p["solucao"])

            linhas.append(linha)

    with open(nome_arquivo, "w", newline="") as f:
        campos = list(linhas[0].keys())
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(linhas)

    print(f"Fronteiras salvas em: {nome_arquivo}")