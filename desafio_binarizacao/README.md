# Desafio: Binarização de Imagem com Python Puro

Este script demonstra como manipular uma imagem no formato BMP (24-bit, sem compressão) utilizando apenas as bibliotecas padrão do Python, sem depender de bibliotecas de imagem como Pillow ou OpenCV.

O processo é dividido em três etapas principais:
1.  **Leitura do BMP:** A função `read_bmp` abre o arquivo `imagem.bmp`, lê o cabeçalho de 54 bytes para extrair as dimensões e, em seguida, lê os dados dos pixels, tratando o padding de bytes para reconstruir a matriz da imagem.
2.  **Conversão para Tons de Cinza:** A função `rgb_to_gray` aplica a fórmula de luminância (`0.299*R + 0.587*G + 0.114*B`) a cada pixel para converter a imagem colorida em uma imagem em tons de cinza. O resultado é salvo como `resultado_cinza.bmp`.
3.  **Binarização:** A função `gray_to_binary` recebe a imagem em tons de cinza e aplica um limiar (threshold). Pixels com valor acima do limiar se tornam brancos (255) e os abaixo se tornam pretos (0). O resultado final é salvo como `resultado_binario.bmp`.

## Como Executar

1.  Certifique-se de que a imagem de entrada `imagem.bmp` está na pasta `assets`.
2.  Execute o script Python:
    ```bash
    python binarizacao.py
    ```
3.  As imagens processadas serão salvas na pasta `resultados`.

### Entrada
![Imagem Original](./assets/imagem.bmp)

### Saídas
**Tons de Cinza:**
![Resultado em Tons de Cinza](./resultados/resultado_cinza.bmp)

**Binarizada:**
![Resultado Binarizado](./resultados/resultado_binario.bmp)