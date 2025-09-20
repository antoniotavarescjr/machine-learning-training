# Desafio: Treinamento de uma Rede YOLOv8 para Detecção de Objetos

Este é o projeto final do bootcamp, demonstrando um pipeline completo para treinar um modelo de detecção de objetos **YOLOv8** em um dataset customizado. O projeto é dividido em duas etapas principais.

## Etapa 1: Preparação do Dataset (`download_coco.py`)

O primeiro script (`download_coco.py`) é responsável por criar um subconjunto de dados a partir do gigantesco dataset COCO.

**Objetivo:** Criar um dataset pequeno e focado contendo apenas as classes **'pessoa'** e **'carro'**, no formato exigido pelo YOLO.

**Otimizações Implementadas:**
* Para evitar o esgotamento de memória RAM que ocorre ao carregar os arquivos de anotação do COCO (que podem ter mais de 600MB), o script utiliza a biblioteca `ijson` para ler os arquivos em modo *streaming*.
* O processo filtra e coleta apenas as anotações e informações das imagens de interesse, mantendo o uso de memória baixo e estável, o que é crucial para rodar em máquinas locais e no WSL.

## Etapa 2: Treinamento do Modelo (`treinamento_rede_yolo.py`)

O segundo script (`treinamento_rede_yolo.py`) utiliza o dataset gerado na Etapa 1 para treinar um modelo **YOLOv8n** usando a biblioteca `ultralytics`.

**Otimizações Implementadas:**
* **Estabilidade no WSL:** O script é configurado com `workers=0` e `amp=False` para garantir a estabilidade do treinamento no ambiente WSL, evitando travamentos comuns relacionados a multiprocessamento e mixed precision no PyTorch.
* **Gerenciamento de VRAM:** O tamanho do lote (`batch=8`) foi ajustado para ser compatível com GPUs com VRAM limitada (ex: NVIDIA RTX 3050 4GB).

## Como Executar o Projeto Completo

1.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    *Nota: Certifique-se de que o `requirements.txt` na raiz do projeto inclui `ultralytics` e `ijson`.*

2.  **Execute o script de preparação de dados primeiro:**
    ```bash
    python download_coco.py
    ```
    Este passo pode levar um tempo considerável para baixar e processar os dados. Ao final, ele criará a pasta `data/coco_limited` com tudo pronto.

3.  **Execute o script de treinamento YOLO:**
    ```bash
    python treinamento_rede_yolo.py
    ```

## Verificando os Resultados

Após o treinamento, uma pasta `runs/detect/train/` será criada. Dentro dela, você encontrará:
* `weights/best.pt`: O seu modelo treinado e pronto para uso.
* `results.png`: Gráficos com as métricas de treinamento (perda, precisão, etc.).
* `val_batch0_pred.jpg`: Uma imagem de exemplo do lote de validação com as detecções do seu modelo.

Além disso, o script de treinamento executa predições em 3 imagens aleatórias e salva os resultados em uma pasta `runs/detect/predict/`, que é a prova final de que seu detector de objetos está funcionando.