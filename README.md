# Atividade P3 — Previsão da Nota Final de Alunos (Machine Learning)

Projeto desenvolvido para a Atividade P3 da disciplina de Machine Learning.

**Integrantes:** Yasmin Oliveira, Leticia Borges

## Objetivo

Criar uma base de dados fake de alunos e um modelo de Machine Learning capaz
de prever a **nota final** com base em:

- Horas de estudo semanais
- Frequência (%)
- Atividades entregues (%)
- Nota da avaliação anterior

O projeto calcula média, desvio padrão e intervalo de erro (95%) das
previsões, e interpreta o desempenho do modelo.

## Estrutura

```
.
├── previsao_nota_final.py     # script principal (gera dados, treina e avalia os modelos)
├── base_alunos_fake.csv       # base de dados fake gerada (300 alunos)
├── resumo_estatistico.csv     # métricas de erro por modelo
├── graficos/                  # gráficos gerados (real x previsto, erros, comparação, importância)
├── requirements.txt
```

## Como executar

```bash
pip install -r requirements.txt
python previsao_nota_final.py
```

## Resultados (resumo)

| Modelo            | MAE   | RMSE  | R²    | Erro médio | IC 95% do erro     |
|-------------------|-------|-------|-------|------------|---------------------|
| Regressão Linear  | 0,485 | 0,617 | 0,547 | -0,073     | [-1,28 ; 1,14]      |
| Random Forest     | 0,576 | 0,717 | 0,388 | -0,021     | [-1,44 ; 1,39]      |

A Regressão Linear foi escolhida como modelo final, com erro médio absoluto
de menos de meio ponto (escala 0–10) e sem viés sistemático relevante.
Detalhes completos da análise estão no relatório em Word.
