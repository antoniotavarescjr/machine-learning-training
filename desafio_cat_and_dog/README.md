# Desafio: Classificação de Cães e Gatos com Transfer Learning

Este projeto implementa um classificador de imagens para distinguir entre cães e gatos, utilizando a técnica de **Transfer Learning** com a biblioteca TensorFlow e Keras.

O modelo base utilizado é o **MobileNetV2**, pré-treinado no massivo dataset ImageNet. Isso nos permite aproveitar o conhecimento de extração de características que o modelo já possui e adaptá-lo para a nossa tarefa específica com muito menos dados e tempo de treinamento.

## Pipeline do Projeto

1.  **Download e Extração:** O script baixa e extrai automaticamente o dataset "Cats and Dogs".
2.  **Pré-processamento:** As imagens são carregadas e redimensionadas para o formato esperado pelo MobileNetV2.
3.  **Data Augmentation:** Para aumentar a variedade do nosso conjunto de dados e evitar overfitting, aplicamos transformações aleatórias nas imagens de treino (flips horizontais, rotações e zooms).
4.  **Construção do Modelo:**
    * O modelo base MobileNetV2 é carregado com seus pesos congelados (não treináveis).
    * Uma nova cabeça de classificação (camadas `GlobalAveragePooling2D` e `Dense`) é adicionada no topo.
5.  **Treinamento Inicial:** Apenas a nova cabeça de classificação é treinada por algumas épocas.
6.  **Fine-Tuning:** Algumas das últimas camadas do modelo base são descongeladas e o modelo inteiro é treinado novamente com uma taxa de aprendizado muito baixa para ajustar finamente os pesos à nossa tarefa.
7.  **Avaliação:** O modelo final é avaliado no conjunto de validação para medir sua acurácia.

## Como Executar

Execute o script diretamente. Ele cuidará de todo o processo de download, extração e treinamento.

```bash
python cat_and_dogs.py
```
O modelo treinado será salvo como `cats_dogs_classifier.h5`.