# Atividade P3 — Previsão da Nota Final de Alunos com Machine Learning

Projeto acadêmico desenvolvido para a Atividade P3 da disciplina de Machine
Learning. O trabalho consiste em criar uma base de dados fictícia (fake) de
alunos e treinar um modelo de Machine Learning capaz de prever a **nota
final** de cada aluno a partir de indicadores de desempenho ao longo do
período letivo.

**Integrantes:** Yasmin Oliveira e Leticia Borges.

---

## Sumário

- [Objetivo](#objetivo)
- [Visão geral da solução](#visão-geral-da-solução)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Base de dados](#base-de-dados)
- [Metodologia](#metodologia)
- [Modelos utilizados](#modelos-utilizados)
- [Métricas de avaliação](#métricas-de-avaliação)
- [Resultados](#resultados)
- [Gráficos gerados](#gráficos-gerados)
- [Interpretação dos resultados](#interpretação-dos-resultados)
- [Limitações e melhorias futuras](#limitações-e-melhorias-futuras)
- [Como executar o projeto](#como-executar-o-projeto)
- [Requisitos](#requisitos)
- [Licença](#licença)

---

## Objetivo

Construir uma base de dados fake de alunos e um modelo de Machine Learning
capaz de prever a **nota final** com base em quatro variáveis de desempenho:

- Horas de estudo semanais
- Frequência (%)
- Percentual de atividades entregues
- Nota da avaliação anterior

Além do treinamento do modelo, o projeto calcula estatísticas de erro
(média, desvio padrão e intervalo de erro das previsões) e apresenta uma
interpretação crítica sobre a qualidade e a confiabilidade do modelo obtido.

## Visão geral da solução

O pipeline implementado em `previsao_nota_final.py` executa, em sequência:

1. Geração de uma base de dados sintética (fake) de 300 alunos.
2. Divisão dos dados em conjuntos de treino (75%) e teste (25%).
3. Treinamento de dois modelos de regressão: Regressão Linear e Random
   Forest.
4. Avaliação de ambos os modelos no conjunto de teste (dados nunca vistos
   durante o treinamento).
5. Cálculo de métricas estatísticas de erro (MAE, RMSE, R², erro médio,
   desvio padrão e intervalo de confiança de 95%).
6. Geração de gráficos para visualização e comunicação dos resultados.
7. Exportação da base, dos resultados e dos gráficos em arquivos CSV/PNG.

## Estrutura do repositório

```
.
├── previsao_nota_final.py                  # script principal (fonte de tudo)
├── base_alunos_fake.csv                    # base de dados fake gerada (300 alunos)
├── resumo_estatistico.csv                  # métricas de erro por modelo
├── resumo_atividade.txt                    # resumo textual da atividade
├── graficos/
│   ├── grafico_real_vs_previsto.png        # dispersão nota real x nota prevista
│   ├── grafico_distribuicao_erros.png      # histograma dos erros de previsão
│   ├── grafico_comparacao_modelos.png      # MAE/RMSE dos dois modelos lado a lado
│   └── grafico_importancia_variaveis.png   # importância das variáveis (Random Forest)
├── Atividade_P3_Previsao_Nota_Final.docx   # relatório completo (código + gráficos + análise)
├── requirements.txt                        # dependências Python
└── README.md                               # este arquivo
```

## Base de dados

A base é 100% sintética (fake), gerada com `numpy.random` a partir de
distribuições normais truncadas, de forma a simular valores plausíveis de
uma turma real. São 300 registros (alunos) com as seguintes colunas:

| Coluna                  | Descrição                                    | Faixa      | Média simulada |
|-------------------------|-----------------------------------------------|------------|-----------------|
| `horas_estudo`          | Horas de estudo semanais                      | 0 – 20 h   | 8 h             |
| `frequencia`            | Frequência às aulas                           | 40 – 100 % | 82 %            |
| `atividades_entregues`  | Percentual de atividades entregues            | 0 – 100 %  | 75 %            |
| `nota_anterior`         | Nota da avaliação anterior                    | 0 – 10     | 6,5             |
| `nota_final`            | **Variável-alvo** — nota final do aluno       | 0 – 10     | —               |

A `nota_final` é calculada como uma combinação ponderada das quatro
variáveis explicativas — com maior peso para `nota_anterior` e
`frequencia` — somada a um ruído aleatório gaussiano. Esse ruído simula
fatores não observáveis do desempenho real de um aluno (motivação, saúde,
imprevistos pessoais etc.) e evita que a relação entre variáveis e nota
seja perfeitamente determinística, o que tornaria o problema artificial
demais para fins de aprendizado de máquina.

## Metodologia

1. **Geração dos dados**: `numpy.random.default_rng` com semente fixa
   (`random_state = 42`) garante que a base gerada seja sempre a mesma a
   cada execução (reprodutibilidade).
2. **Divisão treino/teste**: `train_test_split` do scikit-learn, com 75%
   dos dados para treino (225 alunos) e 25% para teste (75 alunos).
3. **Treinamento**: os modelos são treinados apenas com o conjunto de
   treino; o conjunto de teste é usado exclusivamente para avaliação,
   simulando o comportamento do modelo diante de alunos "novos".
4. **Avaliação**: cálculo de métricas de erro e geração de gráficos a
   partir das previsões no conjunto de teste.

## Modelos utilizados

| Modelo               | Biblioteca               | Características                                                      |
|-----------------------|---------------------------|------------------------------------------------------------------------|
| Regressão Linear      | `sklearn.linear_model`    | Modelo simples e interpretável; assume relação linear entre variáveis. |
| Random Forest Regressor | `sklearn.ensemble` (300 árvores) | Modelo não-linear, baseado em ensemble de árvores de decisão; captura interações mais complexas, mas exige mais dados para não sofrer overfitting. |

## Métricas de avaliação

- **MAE (Erro Absoluto Médio)** — média do valor absoluto dos erros;
  indica, em média, quantos pontos a previsão erra em relação à nota real.
- **RMSE (Raiz do Erro Quadrático Médio)** — penaliza mais fortemente
  erros grandes; útil para identificar previsões muito distantes do valor
  real.
- **R² (Coeficiente de Determinação)** — indica a proporção da variação da
  nota final explicada pelas variáveis usadas (0 = nenhuma explicação,
  1 = explicação perfeita).
- **Erro médio** (real − previsto) — valores próximos de zero indicam
  ausência de viés sistemático (o modelo não superestima nem subestima de
  forma consistente).
- **Desvio padrão do erro** — mede a dispersão dos erros em torno da
  média; quanto menor, mais consistentes são as previsões.
- **Intervalo de erro (IC 95%)** — faixa em que se espera que o erro de
  uma nova previsão esteja contido em 95% dos casos, calculada como
  `erro_médio ± 1,96 × desvio_padrão`.

## Resultados

Métricas calculadas sobre o conjunto de teste (75 alunos, nunca vistos
pelo modelo durante o treinamento):

| Modelo            | MAE   | RMSE  | R²    | Erro médio | Desvio padrão do erro | IC 95% do erro     |
|-------------------|-------|-------|-------|------------|-------------------------|----------------------|
| Regressão Linear  | 0,485 | 0,617 | 0,547 | -0,073     | 0,617                   | [-1,28 ; 1,14]       |
| Random Forest     | 0,576 | 0,717 | 0,388 | -0,021     | 0,722                   | [-1,44 ; 1,39]       |

**Modelo escolhido: Regressão Linear**, por apresentar o menor erro (MAE e
RMSE) e melhor R² neste cenário.

## Gráficos gerados

Todos os gráficos estão na pasta `graficos/` e também reproduzidos, com
comentários, no relatório em Word:

- `grafico_real_vs_previsto.png` — dispersão entre nota real e nota
  prevista, com linha de referência de previsão perfeita.
- `grafico_distribuicao_erros.png` — histograma dos erros de previsão,
  usado para verificar simetria e ausência de viés.
- `grafico_comparacao_modelos.png` — comparação de MAE e RMSE entre os
  dois modelos treinados.
- `grafico_importancia_variaveis.png` — importância relativa de cada
  variável segundo o Random Forest.

## Interpretação dos resultados

Com MAE ≈ 0,49 e RMSE ≈ 0,62 (escala de 0 a 10), o modelo de Regressão
Linear erra, em média, menos de meio ponto na nota final prevista — um
resultado bom para fins didáticos e para uso como ferramenta de apoio. O
R² de 0,55 indica que cerca de 55% da variação da nota final é explicada
pelas quatro variáveis utilizadas; o restante corresponde ao ruído
aleatório inserido propositalmente na simulação (representando fatores não
observáveis do desempenho real de um aluno).

O erro médio próximo de zero mostra que o modelo não tende a superestimar
nem subestimar as notas de forma sistemática, e o histograma de erros com
formato aproximadamente simétrico reforça essa conclusão. O intervalo de
erro de 95% ([-1,28 ; 1,14]) representa uma margem aceitável para uso como
apoio pedagógico (por exemplo, sinalizar alunos em risco de reprovação),
mas não deveria ser a única base para decisões acadêmicas.

**Conclusão geral:** o modelo apresenta desempenho aceitável para os
objetivos da atividade.

## Limitações e melhorias futuras

- A base de dados é sintética; um modelo real deveria ser treinado e
  validado com dados reais de alunos, respeitando LGPD e princípios de
  privacy by design.
- O tamanho da amostra (300 alunos) é relativamente pequeno; mais dados
  tenderiam a melhorar a generalização, especialmente do Random Forest.
- Recomenda-se aplicar validação cruzada (k-fold) para obter uma
  estimativa de erro mais robusta e menos dependente de uma única divisão
  treino/teste.
- Variáveis adicionais (ex.: participação em fóruns, número de faltas
  consecutivas, engajamento em atividades extracurriculares) poderiam
  aumentar o poder explicativo do modelo (R²).
- Testar outros algoritmos (ex.: Gradient Boosting, XGBoost) e ajuste de
  hiperparâmetros (GridSearch/RandomizedSearch) para tentar reduzir ainda
  mais o erro.

## Como executar o projeto

```bash
# 1. Clonar o repositório
git clone <link-do-repositorio>
cd <pasta-do-repositorio>

# 2. Instalar as dependências
pip install -r requirements.txt

# 3. Executar o script principal
python previsao_nota_final.py
```

Ao final da execução, os seguintes arquivos serão (re)gerados na raiz do
projeto:

- `base_alunos_fake.csv`
- `resumo_estatistico.csv`
- `grafico_real_vs_previsto.png`
- `grafico_distribuicao_erros.png`
- `grafico_comparacao_modelos.png`
- `grafico_importancia_variaveis.png`

## Requisitos

- Python 3.9+
- Bibliotecas listadas em `requirements.txt`:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`


## Grafico Comparação

<img width="720" height="480" alt="grafico_comparacao_modelos" src="https://github.com/user-attachments/assets/a80ec295-7f12-4d64-81ed-eb4a45ad2559" />

## Grafico Distribuição

<img width="720" height="480" alt="grafico_distribuicao_erros" src="https://github.com/user-attachments/assets/f729a9a8-71ee-4954-96d8-e472d3f95072" />

## Grafico Importancia 

<img width="720" height="480" alt="grafico_importancia_variaveis" src="https://github.com/user-attachments/assets/21618261-3a4a-421e-b47b-14465e5e8c78" />

## Grafico Real vs Previsto

<img width="720" height="720" alt="grafico_real_vs_previsto" src="https://github.com/user-attachments/assets/7afcd006-54b2-4132-9e71-eca420e9b34a" />

## Licença

Projeto acadêmico, sem fins comerciais, desenvolvido para fins de
aprendizado na disciplina de Machine Learning.



