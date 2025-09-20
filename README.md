# Bootcamp Machine Learning & LLMs - BairesDev & DIO

Este repositório contém todos os projetos e desafios de código desenvolvidos durante o Bootcamp de Machine Learning e LLMs, uma parceria entre a [BairesDev](https://www.bairesdev.com/) e a [Digital Innovation One (DIO)](https://www.dio.me/).

O objetivo deste bootcamp é fornecer uma imersão completa no universo do aprendizado de máquina, desde os conceitos fundamentais de processamento de imagem até o treinamento e fine-tuning de modelos complexos como o YOLOv8 para detecção de objetos.

## 🚀 Projetos Desenvolvidos

Aqui você encontrará uma coleção de desafios práticos que demonstram diferentes técnicas e ferramentas do Machine Learning.

### 1. [Desafio de Binarização de Imagens](./desafio_binarizacao/README.md)
Um script em Python puro que manipula os bytes de uma imagem BMP para realizar a conversão para tons de cinza e, em seguida, para uma imagem binarizada (preto e branco) com base em um limiar. Um ótimo exercício para entender o processamento de imagens em baixo nível.

### 2. [Desafio de Matriz de Confusão](./desafio_matriz_de_confusao/README.md)
Este projeto foca na avaliação de modelos de classificação. O script calcula e exibe as principais métricas de performance (Acurácia, Precisão, Sensibilidade, Especificidade e F-score) a partir de valores de uma matriz de confusão e plota a matriz visualmente usando Seaborn.

### 3. [Desafio de Classificação de Cães e Gatos](./desafio_cat_and_dog/README.md)
Um projeto completo de classificação de imagens usando Transfer Learning. O script baixa um dataset, aplica data augmentation e utiliza o modelo pré-treinado MobileNetV2 para treinar e fazer o fine-tuning de um classificador capaz de distinguir entre imagens de cães e gatos com alta acurácia.

### 4. [Desafio de Treinamento da Rede YOLO](./desafio_treinamento_rede_yolo/README.md)
O projeto mais complexo do bootcamp. Este desafio é dividido em duas partes:
* **Preparação de Dados:** Um script robusto que baixa e processa um subconjunto customizado do dataset COCO, focado em 'pessoas' e 'carros', com otimizações para lidar com arquivos JSON massivos sem esgotar a memória RAM.
* **Treinamento YOLO:** Um segundo script que utiliza o dataset preparado para treinar um modelo de detecção de objetos YOLOv8, aplicando as melhores práticas para treinamento em GPU local, mesmo em ambientes com recursos limitados como o WSL.

## 🛠️ Ambiente e Ferramentas

* **Linguagem:** Python
* **Bibliotecas Principais:** TensorFlow, Keras, PyTorch (via Ultralytics), Seaborn, Matplotlib, ijson.
* **Ambiente:** O projeto foi desenvolvido e depurado para rodar em WSL 2 (Ubuntu) com aceleração via GPU NVIDIA (CUDA).

### Como replicar o ambiente
Para instalar todas as dependências necessárias, crie um ambiente virtual e execute:
```bash
pip install -r requirements.txt
```