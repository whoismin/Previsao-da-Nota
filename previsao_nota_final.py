# -*- coding: utf-8 -*-
"""
Atividade P3 - Machine Learning
Previsão da Nota Final de Alunos

Objetivo:
    Criar uma base de dados fake de alunos e treinar um modelo de Machine
    Learning capaz de prever a nota final com base em:
        - horas de estudo semanais
        - frequência (%)
        - atividades entregues (%)
        - nota da avaliação anterior

    Ao final, calcular média, desvio padrão e intervalo de erro das
    previsões, e avaliar o desempenho do modelo.

Autores: <NOME_INTEGRANTE_1>, <NOME_INTEGRANTE_2>
Disciplina: Machine Learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# 1. GERAÇÃO DA BASE DE DADOS FAKE
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_ALUNOS = 300

rng = np.random.default_rng(RANDOM_STATE)

# Variáveis explicativas (features)
horas_estudo = np.clip(rng.normal(loc=8, scale=3.5, size=N_ALUNOS), 0, 20)          # horas/semana
frequencia = np.clip(rng.normal(loc=82, scale=12, size=N_ALUNOS), 40, 100)          # %
atividades_entregues = np.clip(rng.normal(loc=75, scale=18, size=N_ALUNOS), 0, 100) # %
nota_anterior = np.clip(rng.normal(loc=6.5, scale=1.8, size=N_ALUNOS), 0, 10)       # 0-10

# Relação "real" (não-linear + ruído) usada para simular a nota final.
# A nota final depende principalmente da nota anterior e da frequência,
# com contribuição menor de horas de estudo e atividades entregues.
ruido = rng.normal(loc=0, scale=0.6, size=N_ALUNOS)

nota_final = (
    0.35 * nota_anterior
    + 0.030 * frequencia
    + 0.15 * (horas_estudo / 20 * 10)      # normalizado para escala 0-10
    + 0.020 * atividades_entregues
    + ruido
)
nota_final = np.clip(nota_final, 0, 10)

df = pd.DataFrame({
    "horas_estudo": horas_estudo.round(1),
    "frequencia": frequencia.round(1),
    "atividades_entregues": atividades_entregues.round(1),
    "nota_anterior": nota_anterior.round(2),
    "nota_final": nota_final.round(2),
})

df.to_csv("base_alunos_fake.csv", index=False, encoding="utf-8-sig")
print(f"Base de dados fake criada com {len(df)} alunos -> base_alunos_fake.csv")
print(df.head(), "\n")

# ---------------------------------------------------------------------------
# 2. SEPARAÇÃO TREINO/TESTE
# ---------------------------------------------------------------------------
X = df[["horas_estudo", "frequencia", "atividades_entregues", "nota_anterior"]]
y = df["nota_final"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 3. TREINAMENTO DOS MODELOS
# ---------------------------------------------------------------------------
modelos = {
    "Regressão Linear": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
}

resultados = {}

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    erros = y_test.values - y_pred          # erro = valor real - valor previsto
    erro_medio = np.mean(erros)
    desvio_padrao_erro = np.std(erros, ddof=1)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Intervalo de erro (95% de confiança), assumindo distribuição
    # aproximadamente normal dos erros: erro_medio +- 1.96 * desvio_padrao
    margem = 1.96 * desvio_padrao_erro
    intervalo_erro = (erro_medio - margem, erro_medio + margem)

    resultados[nome] = {
        "modelo": modelo,
        "y_pred": y_pred,
        "erros": erros,
        "erro_medio": erro_medio,
        "desvio_padrao_erro": desvio_padrao_erro,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "intervalo_erro": intervalo_erro,
    }

    print(f"=== {nome} ===")
    print(f"MAE  (erro absoluto médio): {mae:.3f}")
    print(f"RMSE (raiz do erro quadrático médio): {rmse:.3f}")
    print(f"R²   (coeficiente de determinação): {r2:.3f}")
    print(f"Erro médio (real - previsto): {erro_medio:.3f}")
    print(f"Desvio padrão do erro: {desvio_padrao_erro:.3f}")
    print(f"Intervalo de erro (95%): [{intervalo_erro[0]:.3f}, {intervalo_erro[1]:.3f}]")
    print()

# Escolhe o melhor modelo pelo menor RMSE para os gráficos "principais"
melhor_nome = min(resultados, key=lambda k: resultados[k]["rmse"])
melhor = resultados[melhor_nome]
print(f">>> Melhor modelo: {melhor_nome}\n")

# ---------------------------------------------------------------------------
# 4. GRÁFICOS
# ---------------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# 4.1 Real vs Previsto (melhor modelo)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, melhor["y_pred"], alpha=0.6, edgecolor="k", linewidth=0.3)
lims = [0, 10]
plt.plot(lims, lims, "r--", label="Previsão perfeita")
plt.xlabel("Nota Final Real")
plt.ylabel("Nota Final Prevista")
plt.title(f"Real vs. Previsto — {melhor_nome}")
plt.legend()
plt.tight_layout()
plt.savefig("grafico_real_vs_previsto.png")
plt.close()

# 4.2 Distribuição dos erros (histograma)
plt.figure(figsize=(6, 4))
plt.hist(melhor["erros"], bins=15, color="#4C72B0", edgecolor="black", alpha=0.8)
plt.axvline(melhor["erro_medio"], color="red", linestyle="--",
            label=f"Erro médio = {melhor['erro_medio']:.2f}")
plt.xlabel("Erro (Real - Previsto)")
plt.ylabel("Frequência")
plt.title(f"Distribuição dos Erros — {melhor_nome}")
plt.legend()
plt.tight_layout()
plt.savefig("grafico_distribuicao_erros.png")
plt.close()

# 4.3 Comparação MAE/RMSE entre modelos
plt.figure(figsize=(6, 4))
nomes = list(resultados.keys())
maes = [resultados[n]["mae"] for n in nomes]
rmses = [resultados[n]["rmse"] for n in nomes]
x_pos = np.arange(len(nomes))
largura = 0.35
plt.bar(x_pos - largura/2, maes, largura, label="MAE")
plt.bar(x_pos + largura/2, rmses, largura, label="RMSE")
plt.xticks(x_pos, nomes)
plt.ylabel("Erro (pontos na escala 0-10)")
plt.title("Comparação de Desempenho entre Modelos")
plt.legend()
plt.tight_layout()
plt.savefig("grafico_comparacao_modelos.png")
plt.close()

# 4.4 Importância das variáveis (Random Forest)
rf = resultados["Random Forest"]["modelo"]
importancias = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(6, 4))
importancias.plot(kind="barh", color="#55A868")
plt.xlabel("Importância relativa")
plt.title("Importância das Variáveis — Random Forest")
plt.tight_layout()
plt.savefig("grafico_importancia_variaveis.png")
plt.close()

print("Gráficos salvos: grafico_real_vs_previsto.png, grafico_distribuicao_erros.png, "
      "grafico_comparacao_modelos.png, grafico_importancia_variaveis.png")

# ---------------------------------------------------------------------------
# 5. RESUMO ESTATÍSTICO FINAL (para o relatório)
# ---------------------------------------------------------------------------
resumo = pd.DataFrame({
    nome: {
        "MAE": resultados[nome]["mae"],
        "RMSE": resultados[nome]["rmse"],
        "R2": resultados[nome]["r2"],
        "Erro médio": resultados[nome]["erro_medio"],
        "Desvio padrão do erro": resultados[nome]["desvio_padrao_erro"],
        "IC 95% inferior": resultados[nome]["intervalo_erro"][0],
        "IC 95% superior": resultados[nome]["intervalo_erro"][1],
    }
    for nome in resultados
}).T

resumo.to_csv("resumo_estatistico.csv", encoding="utf-8-sig")
print("\nResumo estatístico salvo em resumo_estatistico.csv")
print(resumo.round(3))
