# ==============================================================================
# SCRIPT FINAL: TREINAMENTO E VERIFICAÇÃO DO MODELO YOLO (COM CORREÇÃO PARA AMP+WSL)
# ==============================================================================

from ultralytics import YOLO
import os
import glob
import random
import torch

def main():
    """
    Função principal para treinar o modelo YOLO e verificar os resultados.
    """
    print("=" * 70)
    print("INICIANDO ETAPA FINAL: TREINAMENTO DO MODELO YOLO")
    print("=" * 70)

    # --- PASSO 1: VERIFICAR SE O DATASET EXISTE ---
    
    data_config_path = os.path.join('desafio_treinamento_rede_yolo', 'data', 'coco_limited', 'data.yaml')

    if not os.path.exists(data_config_path):
        print(f"❌ ERRO: Arquivo de configuração não encontrado em '{data_config_path}'")
        print("Por favor, execute o script de preparação de dados primeiro.")
        return

    print(f"✅ Arquivo de configuração do dataset encontrado em: {data_config_path}")

    # --- PASSO 2: TREINAR O MODELO YOLO ---
    
    model = YOLO('yolov8n.pt')
    
    if torch.cuda.is_available():
        print(f"✅ GPU detectada pelo PyTorch: {torch.cuda.get_device_name(0)}")
        device_to_use = 0
    else:
        print("⚠️ AVISO: Nenhuma GPU detectada pelo PyTorch. O treinamento será na CPU e muito lento.")
        device_to_use = 'cpu'

    print("\nIniciando o treinamento do YOLO... Isso pode levar algum tempo.")
    print("Acompanhe o progresso no terminal. Os resultados serão salvos na pasta 'runs/detect/'.")

    # Inicia o treinamento com a correção final para o AMP no WSL
    results = model.train(
        data=data_config_path,
        epochs=50,
        imgsz=640,
        device=device_to_use,
        patience=5,
        workers=0,
        batch=8,
        amp=False  # <-- ✨ CORREÇÃO FINAL E DEFINITIVA: Desabilita o Mixed Precision para evitar o crash no WSL.
    )

    print("\n" + "=" * 70)
    print("✅ TREINAMENTO YOLO CONCLUÍDO!")
    print("=" * 70)

    # --- PASSO 3: VERIFICAR A SAÍDA COM IMAGENS DE VALIDAÇÃO ---

    path_to_best_weights = results.save_dir / 'weights' / 'best.pt'
    print(f"\nCarregando o melhor modelo treinado de: {path_to_best_weights}")
    
    trained_model = YOLO(path_to_best_weights)

    validation_images_path = os.path.join('desafio_treinamento_rede_yolo', 'data', 'coco_limited', 'val', 'images', '*.jpg')
    list_of_validation_images = glob.glob(validation_images_path)
    
    if not list_of_validation_images:
        print("❌ Nenhuma imagem de validação encontrada para teste.")
        return
        
    images_to_test = random.sample(list_of_validation_images, min(3, len(list_of_validation_images)))
    
    print(f"\nExecutando predições em {len(images_to_test)} imagens de validação para verificar a saída...")
    
    predict_results = trained_model.predict(
        source=images_to_test,
        save=True,
        conf=0.5
    )
    
    save_directory = predict_results[0].save_dir if predict_results else "uma subpasta em 'runs/detect'"

    print("\n" + "=" * 70)
    print("VERIFICAÇÃO CONCLUÍDA!")
    print(f"As imagens com as detecções foram salvas na pasta '{save_directory}'")
    print("Abra essa pasta para ver o resultado final do seu detector de objetos!")
    print("=" * 70)


if __name__ == "__main__":
    main()