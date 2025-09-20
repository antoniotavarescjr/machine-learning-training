# Desafio: Calculadora de Métricas de Classificação

Este script serve como uma ferramenta para analisar a performance de um modelo de classificação binária a partir dos resultados de sua **matriz de confusão**.

A matriz de confusão é a base para entender onde um modelo está acertando e errando. Os valores (Verdadeiro Positivo, Falso Positivo, Verdadeiro Negativo, Falso Negativo) são usados para calcular métricas mais avançadas.

## Métricas Calculadas

-   **Acurácia:** A porcentagem geral de previsões corretas.
-   **Sensibilidade (Recall):** A capacidade do modelo de encontrar todos os casos positivos.
-   **Especificidade:** A capacidade do modelo de identificar corretamente os casos negativos.
-   **Precisão:** Das previsões positivas feitas, quantas estavam corretas.
-   **F-score:** Uma média harmônica entre precisão e sensibilidade, útil para datasets desbalanceados.

## Como Usar

1.  Altere os valores no dicionário `matriz_confusao` no script com os resultados do seu modelo.
2.  Execute o script:
    ```bash
    python matriz_de_confusao.py
    ```
3.  Os valores calculados para cada métrica serão impressos no terminal e uma visualização da matriz de confusão será exibida.