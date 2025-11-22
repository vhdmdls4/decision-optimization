import numpy as np
import copy
import random

def heuristica_construtiva_gulosa_randomizada(probdata, alpha=0.2):
    """
    Constrói uma solução inicial usando uma estratégia gulosa randomizada baseada em Regret.
    
    Args:
        probdata: Estrutura com dados do problema.
        alpha: Fator de randomização (0.0 a 1.0). 
               0.0 = Guloso puro (sempre pega a tarefa mais crítica).
               1.0 = Totalmente aleatório.
    """
    m = probdata.m  # Número de agentes
    n = probdata.n  # Número de tarefas
    
    # Inicializa estrutura da solução
    x = type('Struct', (object,), {})() # Cria struct genérico
    x.solution = [[] for _ in range(m)]
    
    # Controle de capacidade disponível por agente
    # probdata.b é a capacidade total, vamos subtraindo conforme usamos
    capacidade_restante = probdata.b.copy()
    
    # Lista de tarefas que precisam ser alocadas
    tarefas_pendentes = list(range(n))
    
    while tarefas_pendentes:
        candidatos = []
        
        # 1. Para cada tarefa pendente, analisa as opções de agentes
        for tarefa in tarefas_pendentes:
            opcoes_agentes = []
            
            for agente in range(m):
                custo = probdata.custo[agente][tarefa]
                consumo = probdata.recurso[agente][tarefa]
                
                # Verifica se o agente tem capacidade (Soft constraint)
                # Se não tiver capacidade, aplicamos uma penalidade virtual alta no custo 
                # apenas para ordenação, para tentar evitar esse agente.
                penalidade_saturacao = 0
                if consumo > capacidade_restante[agente]:
                    penalidade_saturacao = 10000 # Valor alto para desencorajar
                
                custo_virtual = custo + penalidade_saturacao
                opcoes_agentes.append({'agente': agente, 'custo_virtual': custo_virtual, 'consumo': consumo})
            
            # Ordena os agentes do melhor para o pior custo para esta tarefa
            opcoes_agentes.sort(key=lambda k: k['custo_virtual'])
            
            melhor_opcao = opcoes_agentes[0]
            segunda_melhor = opcoes_agentes[1] if len(opcoes_agentes) > 1 else opcoes_agentes[0]
            
            # 2. Calcula o REGRET (Arrependimento): 
            # Quanto eu perco se não escolher o melhor agente agora?
            regret = segunda_melhor['custo_virtual'] - melhor_opcao['custo_virtual']
            
            candidatos.append({
                'tarefa': tarefa,
                'regret': regret,
                'melhor_agente': melhor_opcao['agente'],
                'consumo': melhor_opcao['consumo']
            })
        
        # 3. Ordena tarefas por arrependimento (decrescente)
        # Queremos alocar logo as tarefas que são muito caras se perderem seu melhor agente
        candidatos.sort(key=lambda k: k['regret'], reverse=True)
        
        # 4. Cria a Lista Restrita de Candidatos (RCL)
        # Pega uma porção das tarefas mais críticas baseada no alpha
        # Garante pelo menos 1 candidato, ou top 3 se a lista for grande
        tamanho_rcl = max(1, int(len(candidatos) * alpha))
        # Força um mínimo de aleatoriedade (ex: top 3) se alpha for muito pequeno
        if tamanho_rcl < 3 and len(candidatos) >= 3:
            tamanho_rcl = 3
            
        rcl = candidatos[:tamanho_rcl]
        
        # 5. Seleciona aleatoriamente uma tarefa da RCL
        escolhido = random.choice(rcl)
        
        tarefa_selecionada = escolhido['tarefa']
        agente_selecionado = escolhido['melhor_agente']
        
        # Realiza a atribuição
        x.solution[agente_selecionado].append(tarefa_selecionada)
        capacidade_restante[agente_selecionado] -= escolhido['consumo']
        
        # Remove da lista de pendentes
        tarefas_pendentes.remove(tarefa_selecionada)
        
    return x
```

### Como integrar no seu código atual

No bloco principal do seu código (`while` loop ou setup), substitua a chamada de `sol_inicial` por esta nova função:

```python
# ... (código anterior)

# No lugar de:
# x = sol_inicial(probdata, apply_constructive_heuristic=True)

# Use:
x = heuristica_construtiva_gulosa_randomizada(probdata, alpha=0.3)

# Avalia a solução gerada pela heurística
x = fobj(x, probdata)

print(f"Fitness Inicial Heurística: {x.fitness}")
print(f"Penalidade Inicial: {x.penalidade}")

# ... (continua para o loop do RVNS)