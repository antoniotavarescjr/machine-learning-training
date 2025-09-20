import os


def read_bmp(filename):
    """
    Lê um arquivo BMP 24-bit sem compressão.
    Retorna como lista de listas de tuplas (R, G, B).
    """
    with open(filename, 'rb') as f:
        # Lê o cabeçalho (54 bytes)
        header = f.read(54)

        # Extrai dimensões (little-endian)
        width = int.from_bytes(header[18:22], 'little')
        height = int.from_bytes(header[22:26], 'little')

        # Calcula padding (cada linha deve ser múltipla de 4 bytes)
        padding = (4 - (width * 3) % 4) % 4

        # Lê os pixels (BGR, de baixo para cima)
        pixels = []
        for _ in range(height):
            row = []
            for _ in range(width):
                b, g, r = f.read(3)
                row.append((r, g, b))
            # Pula o padding
            f.read(padding)
            pixels.insert(0, row)  # Inverte a ordem das linhas
        return pixels

def rgb_to_gray(image):
    """
    Converte uma imagem RGB para níveis de cinza (0-255).
    :param image: Lista de listas de tuplas (R, G, B)
    :return: Lista de listas de inteiros (tons de cinza)
    """
    gray_image = []
    for row in image:
        gray_row = []
        for pixel in row:
            r, g, b = pixel
            # Fórmula de luminância padrão (percepção humana)
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray_row.append(gray_value)
        gray_image.append(gray_row)
    return gray_image

def gray_to_binary(gray_image, threshold=128):
    """
    Converte uma imagem em tons de cinza para binarizada (0 e 255).
    :param gray_image: Lista de listas de inteiros (tons de cinza)
    :param threshold: Limiar para binarização (padrão: 128)
    :return: Lista de listas de inteiros (0 ou 255)
    """
    binary_image = []
    for row in gray_image:
        binary_row = []
        for pixel in row:
            # Binarização: acima do limiar = branco (255), abaixo = preto (0)
            binary_value = 255 if pixel >= threshold else 0
            binary_row.append(binary_value)
        binary_image.append(binary_row)
    return binary_image

def save_bmp_grayscale(filename, gray_image):
    """
    Salva uma imagem em tons de cinza como BMP 24-bit.
    :param filename: Nome do arquivo de saída
    :param gray_image: Lista de listas de inteiros (tons de cinza)
    """
    height = len(gray_image)
    width = len(gray_image[0]) if height > 0 else 0

    # Calcula padding
    padding = (4 - (width * 3) % 4) % 4

    with open(filename, 'wb') as f:
        # Cabeçalho BMP (54 bytes)
        # Tipo de arquivo (2 bytes)
        f.write(b'BM')
        # Tamanho do arquivo (4 bytes)
        file_size = 54 + (width * 3 + padding) * height
        f.write(file_size.to_bytes(4, 'little'))
        # Reservado (4 bytes)
        f.write(b'\x00\x00\x00\x00')
        # Offset dos dados (4 bytes)
        f.write(b'\x36\x00\x00\x00')
        # Tamanho do cabeçalho de informações (4 bytes)
        f.write(b'\x28\x00\x00\x00')
        # Largura (4 bytes)
        f.write(width.to_bytes(4, 'little'))
        # Altura (4 bytes)
        f.write(height.to_bytes(4, 'little'))
        # Planos de cor (2 bytes)
        f.write(b'\x01\x00')
        # Bits por pixel (2 bytes)
        f.write(b'\x18\x00')
        # Compressão (4 bytes)
        f.write(b'\x00\x00\x00\x00')
        # Tamanho da imagem (4 bytes)
        f.write(b'\x00\x00\x00\x00')
        # Resolução horizontal (4 bytes)
        f.write(b'\x13\x0B\x00\x00')
        # Resolução vertical (4 bytes)
        f.write(b'\x13\x0B\x00\x00')
        # Cores na paleta (4 bytes)
        f.write(b'\x00\x00\x00\x00')
        # Cores importantes (4 bytes)
        f.write(b'\x00\x00\x00\x00')

        # Dados dos pixels (de baixo para cima)
        for y in range(height-1, -1, -1):
            for x in range(width):
                gray = gray_image[y][x]
                # Escreve BGR (mesmo valor para todos os canais)
                f.write(bytes([gray, gray, gray]))
            # Escreve padding
            f.write(b'\x00' * padding)

def save_bmp_binary(filename, binary_image):
    """
    Salva uma imagem binarizada como BMP 24-bit.
    :param filename: Nome do arquivo de saída
    :param binary_image: Lista de listas de inteiros (0 ou 255)
    """
    height = len(binary_image)
    width = len(binary_image[0]) if height > 0 else 0

    # Calcula padding
    padding = (4 - (width * 3) % 4) % 4

    with open(filename, 'wb') as f:
        # Cabeçalho BMP (54 bytes) - mesmo que grayscale
        f.write(b'BM')
        file_size = 54 + (width * 3 + padding) * height
        f.write(file_size.to_bytes(4, 'little'))
        f.write(b'\x00\x00\x00\x00')
        f.write(b'\x36\x00\x00\x00')
        f.write(b'\x28\x00\x00\x00')
        f.write(width.to_bytes(4, 'little'))
        f.write(height.to_bytes(4, 'little'))
        f.write(b'\x01\x00')
        f.write(b'\x18\x00')
        f.write(b'\x00\x00\x00\x00')
        f.write(b'\x00\x00\x00\x00')
        f.write(b'\x13\x0B\x00\x00')
        f.write(b'\x13\x0B\x00\x00')
        f.write(b'\x00\x00\x00\x00')
        f.write(b'\x00\x00\x00\x00')

        # Dados dos pixels (de baixo para cima)
        for y in range(height-1, -1, -1):
            for x in range(width):
                binary = binary_image[y][x]
                # Escreve BGR (mesmo valor para todos os canais)
                f.write(bytes([binary, binary, binary]))
            # Escreve padding
            f.write(b'\x00' * padding)

def print_image_info(image):
    """
    Imprime informações básicas sobre a imagem.
    """
    height = len(image)
    width = len(image[0]) if height > 0 else 0
    print(f"Dimensões: {width}x{height}")
    print(f"Total de pixels: {width * height}")
    print("\nPrimeiros 5 pixels (canto superior esquerdo):")
    for i in range(min(5, height)):
        for j in range(min(5, width)):
            print(f"Pixel [{i}][{j}]: {image[i][j]}", end="  ")
        print()

# Exemplo de uso
if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))


    input_file = os.path.join(script_dir, "assets", "imagem.bmp")
    output_dir = os.path.join(script_dir, "resultados")
    gray_output = os.path.join(output_dir, "resultado_cinza.bmp")
    binary_output = os.path.join(output_dir, "resultado_binario.bmp")


    os.makedirs(output_dir, exist_ok=True)
    
  
    threshold = 128

    try:
        print("Lendo imagem BMP...")
        image_rgb = read_bmp(input_file)

        print("\nInformações da imagem original:")
        print_image_info(image_rgb)

        print("\nConvertendo para tons de cinza...")
        gray_image = rgb_to_gray(image_rgb)

        print("\nInformações da imagem em tons de cinza:")
        print_image_info(gray_image)

        print(f"\nSalvando imagem em tons de cinza como '{gray_output}'...")
        save_bmp_grayscale(gray_output, gray_image)

        print(f"\nConvertendo para imagem binarizada (limiar={threshold})...")
        binary_image = gray_to_binary(gray_image, threshold)

        print("\nInformações da imagem binarizada:")
        print_image_info(binary_image)

        print(f"\nSalvando imagem binarizada como '{binary_output}'...")
        save_bmp_binary(binary_output, binary_image)

        print("\nProcessamento concluído com sucesso!")
        print(f"Arquivos gerados:")
        print(f"  - {gray_output} (tons de cinza)")
        print(f"  - {binary_output} (binarizada)")

    except FileNotFoundError:
        print(f"Erro: Arquivo '{input_file}' não encontrado!")
        print("Certifique-se de que a imagem BMP está na mesma pasta do script.")
    except Exception as e:
        print(f"Erro durante o processamento: {e}")