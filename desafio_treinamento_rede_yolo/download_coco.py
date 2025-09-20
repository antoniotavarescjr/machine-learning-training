# ==============================================================================
# SCRIPT FINAL OTIMIZADO (GPU + RAM + TF.DATA)
# ==============================================================================

import os
import sys
import requests
import zipfile
import shutil
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import ijson  # <-- Import para streaming de JSON, crucial para evitar alto uso de RAM

# ==============================================================================
# 1. CONFIGURAÇÃO OTIMIZADA DA GPU
# ==============================================================================
print("=" * 70)
print("CONFIGURANDO GPU PARA MÁXIMA PERFORMANCE")
print("=" * 70)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✅ GPU configurada com memória dinâmica e Mixed Precision.")
    except RuntimeError as e:
        print(f"⚠️ Erro ao configurar GPU: {e}")
else:
    print("❌ Nenhuma GPU encontrada. O script rodará na CPU.")

# Configurar pasta do desafio
DESAFIO_FOLDER = "desafio_treinamento_rede_yolo"
os.makedirs(DESAFIO_FOLDER, exist_ok=True)
os.chdir(DESAFIO_FOLDER)

print(f"\nTrabalhando na pasta: {os.getcwd()}")


# Substitua a sua função antiga por esta versão completa e corrigida
def download_coco_directly(target_size_mb=500):
    """
    Baixa e prepara um subconjunto do COCO de forma eficiente,
    usando ijson para evitar alto consumo de RAM em TODAS as etapas.
    """
    print("\n" + "=" * 70)
    print("DOWNLOAD E PREPARAÇÃO DO DATASET COCO (VERSÃO EFICIENTE EM RAM)")
    print("=" * 70)
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/coco_direct', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    annotations_path = 'data/coco_direct/annotations.zip'
    
    print("Passo 1: Verificando anotações...")
    if not os.path.exists('data/coco_direct/annotations'):
        if not os.path.exists(annotations_path):
            print("Baixando anotações (~241MB)...")
            response = requests.get(annotations_url, stream=True)
            with open(annotations_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
        print("Extraindo anotações...")
        with zipfile.ZipFile(annotations_path, 'r') as zip_ref:
            zip_ref.extractall('data/coco_direct')
    print("✓ Anotações prontas!")
    
    print("Passo 2: Filtrando imagens com baixo uso de RAM via streaming...")
    person_cat_id, car_cat_id = 1, 3
    
    val_ann_file = 'data/coco_direct/annotations/instances_val2017.json'
    train_ann_file = 'data/coco_direct/annotations/instances_train2017.json'

    # --- Processamento de Validação (já estava correto) ---
    val_images_with_targets = set()
    with open(val_ann_file, 'rb') as f:
        for ann in ijson.items(f, 'annotations.item'):
            if ann.get('category_id') in [person_cat_id, car_cat_id]:
                val_images_with_targets.add(ann['image_id'])
    val_image_id_to_name = {img['id']: img['file_name'] for img in ijson.items(open(val_ann_file, 'rb'), 'images.item')}
    selected_val_images = [val_image_id_to_name[img_id] for img_id in list(val_images_with_targets)[:200] if img_id in val_image_id_to_name]

    # --- CORREÇÃO: Lógica de Treinamento ---
    # 1. Identificar IDs de imagens de treino de interesse
    train_images_with_targets = set()
    with open(train_ann_file, 'rb') as f:
        for ann in tqdm(ijson.items(f, 'annotations.item'), desc="1/3 Filtrando IDs de treino"):
            if ann.get('category_id') in [person_cat_id, car_cat_id]:
                train_images_with_targets.add(ann['image_id'])
    
    # 2. Criar mapas de ID->Nome e ID->Info (largura/altura)
    train_image_id_to_name = {}
    train_image_id_to_info = {}
    with open(train_ann_file, 'rb') as f:
        for img in tqdm(ijson.items(f, 'images.item'), desc="2/3 Mapeando infos de treino"):
            if img['id'] in train_images_with_targets:
                train_image_id_to_name[img['id']] = img['file_name']
                train_image_id_to_info[img['id']] = {'width': img['width'], 'height': img['height']}
    
    # 3. Selecionar 300 imagens aleatórias e criar um pequeno dicionário APENAS com suas anotações
    train_image_ids = list(train_images_with_targets)
    random.shuffle(train_image_ids)
    selected_train_ids = set(train_image_ids[:300])
    
    selected_train_annotations = {}
    with open(train_ann_file, 'rb') as f:
        for ann in tqdm(ijson.items(f, 'annotations.item'), desc="3/3 Coletando anotações"):
            if ann.get('image_id') in selected_train_ids:
                img_id = ann['image_id']
                if img_id not in selected_train_annotations:
                    selected_train_annotations[img_id] = []
                selected_train_annotations[img_id].append(ann)

    print(f"Selecionadas {len(selected_train_ids)} de treino e {len(selected_val_images)} de validação.")

    # --- Processo de Download e Extração (agora usando os dados eficientes) ---
    print("Passo 3: Baixando e extraindo imagens de validação...")
    # (O código de validação não precisa mudar)
    with open(val_ann_file, 'r') as f: val_annotations_full = json.load(f) # Validação é pequeno, OK
    # ... (Restante do Passo 3 omitido para brevidade, continua igual)
    val_zip_url = 'http://images.cocodataset.org/zips/val2017.zip'
    val_zip_path = 'data/coco_direct/val2017.zip'
    if not os.path.exists(val_zip_path):
        response = requests.get(val_zip_url, stream=True)
        with open(val_zip_path, 'wb') as f:
            total_length = int(response.headers.get('content-length'))
            with tqdm(total=total_length, unit='B', unit_scale=True, desc="Baixando val2017.zip") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk); pbar.update(len(chunk))
    os.makedirs('data/coco_limited/val/images', exist_ok=True)
    os.makedirs('data/coco_limited/val/labels', exist_ok=True)
    with zipfile.ZipFile(val_zip_path, 'r') as zip_ref:
        for file_name in tqdm(selected_val_images, desc="Extraindo e criando labels de val"):
            try:
                source_path = f"val2017/{file_name}"; target_path = f"data/coco_limited/val/images/{file_name}"
                with zip_ref.open(source_path) as source, open(target_path, 'wb') as target: shutil.copyfileobj(source, target)
                label_path = f"data/coco_limited/val/labels/{file_name.replace('.jpg', '.txt')}"
                img_id = next(img['id'] for img in val_annotations_full['images'] if img['file_name'] == file_name)
                with open(label_path, 'w') as f:
                    for ann in val_annotations_full['annotations']:
                        if ann['image_id'] == img_id and ann['category_id'] in [person_cat_id, car_cat_id]:
                            bbox = ann['bbox']; img_info = next(img for img in val_annotations_full['images'] if img['id'] == img_id)
                            x_center, y_center = (bbox[0] + bbox[2]/2) / img_info['width'], (bbox[1] + bbox[3]/2) / img_info['height']
                            width, height = bbox[2] / img_info['width'], bbox[3] / img_info['height']
                            class_id = 0 if ann['category_id'] == person_cat_id else 1
                            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            except Exception: continue

    print("Passo 4: Baixando imagens de treinamento...")
    os.makedirs('data/coco_limited/train/images', exist_ok=True)
    os.makedirs('data/coco_limited/train/labels', exist_ok=True)
    base_url = "http://images.cocodataset.org/train2017/"
    
    # CORREÇÃO: O loop agora usa os dados pré-filtrados e eficientes
    for img_id in tqdm(selected_train_ids, desc="Baixando e criando labels de treino"):
        try:
            file_name = train_image_id_to_name[img_id]
            img_path = f"data/coco_limited/train/images/{file_name}"
            if not os.path.exists(img_path):
                response = requests.get(f"{base_url}{file_name}")
                if response.status_code == 200:
                    with open(img_path, 'wb') as f: f.write(response.content)

            label_path = f"data/coco_limited/train/labels/{file_name.replace('.jpg', '.txt')}"
            img_info = train_image_id_to_info[img_id]
            
            with open(label_path, 'w') as f:
                if img_id in selected_train_annotations:
                    for ann in selected_train_annotations[img_id]:
                        bbox = ann['bbox']
                        x_center = (bbox[0] + bbox[2]/2) / img_info['width']
                        y_center = (bbox[1] + bbox[3]/2) / img_info['height']
                        width, height = bbox[2] / img_info['width'], bbox[3] / img_info['height']
                        class_id = 0 if ann['category_id'] == person_cat_id else 1
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        except Exception as e:
            print(f"Erro processando img_id {img_id}: {e}")
            continue

    print("Passo 5: Criando arquivo data.yaml...")
    data_yaml = f"""path: {os.path.abspath('data/coco_limited')}\ntrain: train/images\nval: val/images\n\nnames:\n  0: person\n  1: car\n"""
    with open('data/coco_limited/data.yaml', 'w') as f: f.write(data_yaml)

    print("\n✅ Dataset COCO limitado criado com sucesso!")
    return 'data/coco_limited'

def load_image_and_label_tf(image_path, label_path):
    """Função de carregamento 100% TensorFlow para evitar vazamento de memória."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    
    label_content = tf.io.read_file(label_path)
    lines = tf.strings.split(label_content, '\n')
    lines = tf.cond(tf.equal(tf.strings.length(lines[-1]), 0), lambda: lines[:-1], lambda: lines)

    def get_class_id(line):
        parts = tf.strings.split(line, ' ')
        return tf.strings.to_number(parts[0], out_type=tf.int32)
    
    class_ids = tf.cond(tf.size(lines) > 0, lambda: tf.map_fn(get_class_id, lines, dtype=tf.int32), lambda: tf.constant([], dtype=tf.int32))
    
    has_person = tf.cast(tf.reduce_any(tf.equal(class_ids, 0)), tf.float32)
    has_car = tf.cast(tf.reduce_any(tf.equal(class_ids, 1)), tf.float32)
    label = tf.stack([has_person, has_car])
    
    image.set_shape([224, 224, 3])
    label.set_shape([2])
    return image, label


def create_tf_dataset(dataset_dir):
    """Cria dataset TensorFlow usando o pipeline otimizado e estável."""
    train_img_dir = os.path.join(dataset_dir, 'train', 'images')
    val_img_dir = os.path.join(dataset_dir, 'val', 'images')
    
    train_images = [os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
    train_labels = [f.replace('images', 'labels').replace('.jpg', '.txt') for f in train_images]
    val_images = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.endswith('.jpg')]
    val_labels = [f.replace('images', 'labels').replace('.jpg', '.txt') for f in val_images]
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.map(load_image_and_label_tf, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=1000).batch(16).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_ds = val_ds.map(load_image_and_label_tf, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(16).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds


def create_model():
    """Cria o modelo otimizado com BatchNormalization e GlobalAveragePooling2D."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        
        tf.keras.layers.GlobalAveragePooling2D(), # Mais eficiente que Flatten
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(2, activation='sigmoid', dtype='float32') 
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate(train_ds, val_ds):
    """Treina e avalia o modelo."""
    print("\nCriando modelo otimizado...")
    model = create_model()
    model.summary()
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('output/best_model.keras', save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)
    ]
    
    print("\nIniciando treinamento...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks, verbose=1)
    
    print("\nAvaliando modelo...")
    loss, accuracy = model.evaluate(val_ds)
    print(f"Acurácia final no conjunto de validação: {accuracy:.4f}")
    
    # Pega uma amostra para visualização
    sample_ds = val_ds.unbatch().take(6).batch(6)
    visualize_results(model, sample_ds, history)
    return model, history


def visualize_results(model, sample_ds, history):
    """Visualiza os resultados do treinamento e previsões."""
    print("\nGerando gráficos de resultados...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Acurácia de Treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.legend()
    plt.title('Acurácia')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Perda de Treino')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.legend()
    plt.title('Perda')
    plt.savefig('results/training_history.png')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    for images, labels in sample_ds:
        predictions = model.predict(images)
        plt.figure(figsize=(15, 10))
        for i in range(min(6, len(images))):
            plt.subplot(2, 3, i + 1)
            plt.imshow(images[i])
            real_person = "Sim" if labels[i][0] > 0.5 else "Não"
            real_car = "Sim" if labels[i][1] > 0.5 else "Não"
            pred_person, pred_car = predictions[i][0], predictions[i][1]
            plt.title(f"Real: P={real_person}, C={real_car}\nPrev: P={pred_person:.2f}, C={pred_car:.2f}")
            plt.axis('off')
        plt.savefig('results/predictions.png')
        plt.show(block=False)
        plt.pause(2)
        plt.close()


def main():
    """Função principal que orquestra todo o processo."""
    dataset_dir = download_coco_directly()
    if dataset_dir:
        train_ds, val_ds = create_tf_dataset(dataset_dir)
        train_and_evaluate(train_ds, val_ds)
        print("\n" + "=" * 70)
        print("DESAFIO CONCLUÍDO COM SUCESSO!")
        print("=" * 70)

if __name__ == "__main__":
    main()