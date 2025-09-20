# ==============================================================================
# SCRIPT OTIMIZADO PARA GPU RTX 3050 (VERSÃO CORRIGIDA)
# ==============================================================================

import tensorflow as tf
import sys
import time
import numpy as np

# IMPORTANTE: Configurar GPU ANTES de qualquer operação do TensorFlow
print("=" * 60)
print("CONFIGURANDO GPU PARA MÁXIMA PERFORMANCE")
print("=" * 60)

# 1. Configurar GPU ANTES de qualquer operação
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"✅ Encontradas {len(gpus)} GPUs")
    
    try:
        # Configurar crescimento de memória
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Memória dinâmica configurada para: {gpu}")
    except RuntimeError as e:
        print(f"❌ Erro ao configurar memória: {e}")
    
    # Configurar threading para GPU
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        print("✅ Threading otimizado para GPU")
    except RuntimeError as e:
        print(f"⚠️  Threading já inicializado: {e}")
    
    # Habilitar mixed precision (IMPORTANTE para RTX 3050)
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✅ Mixed precision (float16) habilitado")
    except Exception as e:
        print(f"⚠️  Mixed precision não disponível: {e}")
else:
    print("❌ Nenhuma GPU encontrada")

# Agora podemos continuar com as verificações
print("\n" + "=" * 60)
print("VERIFICAÇÃO DE GPU OTIMIZADA")
print("=" * 60)

print(f"TensorFlow Version: {tf.__version__}")
print(f"Python Version: {sys.version}")

# Verificar dispositivos
devices = tf.config.list_physical_devices()
print(f"\nDispositivos disponíveis: {len(devices)}")
for device in devices:
    print(f"  - {device}")

# Teste de performance otimizado
print("\n" + "=" * 60)
print("TESTE DE PERFORMANCE OTIMIZADO")
print("=" * 60)

def create_optimized_model():
    """Modelo otimizado para GPU RTX 3050"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1000,)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Otimizador otimizado para GPU
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

# Criar dados de teste
print("Criando dados de teste...")
x_train = np.random.random((2000, 1000)).astype(np.float32)  # Mais dados
y_train = np.random.random((2000, 10)).astype(np.float32)

# Teste com CPU
print("\nTestando com CPU...")
with tf.device('/CPU:0'):
    cpu_model = create_optimized_model()
    
    start_time = time.time()
    # NOTA: O teste com epochs=1 pode ser enganoso. A GPU parece mais lenta
    # devido ao tempo de compilação inicial. Com mais épocas, a GPU seria muito mais rápida.
    cpu_model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=0)
    cpu_time = time.time() - start_time
    print(f"⏱️  Tempo CPU: {cpu_time:.2f} segundos")

# Limpar modelo
del cpu_model
tf.keras.backend.clear_session()

# Teste com GPU
if gpus:
    print("\nTestando com GPU...")
    with tf.device('/GPU:0'):
        gpu_model = create_optimized_model()
        
        start_time = time.time()
        gpu_model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=0)
        gpu_time = time.time() - start_time
        print(f"⏱️  Tempo GPU: {gpu_time:.2f} segundos")
        
        # Calcular speedup
        if cpu_time > 0 and gpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"🚀 Speedup GPU: {speedup:.2f}x mais rápido")
    
    # Limpar modelo
    del gpu_model
    tf.keras.backend.clear_session()

# Teste com modelo real (seu caso)
print("\n" + "=" * 60)
print("TESTE COM MODELO REAL (SEU CASO)")
print("=" * 60)

def create_real_optimized_model():
    """Modelo real otimizado para GPU"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        
        # Camadas convolucionais otimizadas
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        
        # Global pooling em vez de Flatten (melhor para GPU)
        tf.keras.layers.GlobalAveragePooling2D(),
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])
    
    # Otimizador com learning rate ajustado
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Criar dados realistas
print("Criando dados realistas...")
x_real = np.random.random((128, 224, 224, 3)).astype(np.float32)
y_real = np.random.random((128, 2)).astype(np.float32)

if gpus:
    print("Testando modelo real com GPU...")
    with tf.device('/GPU:0'):
        real_model = create_real_optimized_model()
        
        start_time = time.time()
        
        # =================================================================
        # === CORREÇÃO PRINCIPAL PARA EVITAR ERRO DE FALTA DE MEMÓRIA ===
        # O batch_size foi reduzido de 32 para 16. A RTX 3050 (4GB) não
        # tinha VRAM suficiente para um batch de 32 com imagens 224x224.
        # Se 16 ainda causar erro, tente 8.
        # =================================================================
        history = real_model.fit(x_real, y_real, epochs=3, batch_size=16, verbose=1)
        
        gpu_real_time = time.time() - start_time
        print(f"⏱️  Tempo modelo real (GPU): {gpu_real_time:.2f} segundos")
    
    # Limpar modelo
    del real_model
    tf.keras.backend.clear_session()

# Verificar memória GPU
print("\n" + "=" * 60)
print("VERIFICANDO MEMÓRIA GPU")
print("=" * 60)

if gpus:
    try:
        for i, gpu in enumerate(gpus):
            # A função get_memory_info pode não estar disponível ou funcionar de forma diferente
            # dependendo da versão do TF e drivers. Usamos um try-except para segurança.
            try:
                gpu_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                print(f"GPU {i}:")
                print(f"   Memória atual: {gpu_info['current'] / 1024**3:.2f} GB")
                print(f"   Memória pico: {gpu_info['peak'] / 1024**3:.2f} GB")
            except:
                print(f"Não foi possível obter informações de memória para a GPU {i}.")

    except Exception as e:
        print(f"❌ Erro ao verificar memória: {e}")

print("\n" + "=" * 60)
print("RECOMENDAÇÕES PARA SEU SCRIPT PRINCIPAL")
print("=" * 60)

print("""
1. ADICIONE ESTAS CONFIGURAÇÕES NO INÍCIO DO SEU SCRIPT:

import tensorflow as tf

# Configurar GPU antes de qualquer operação
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✅ GPU configurada para máxima performance")
    except RuntimeError as e:
        print(f"⚠️  Erro ao configurar GPU: {e}")

2. USE BATCH SIZE MENOR NO TREINAMENTO (AJUSTE À SUA VRAM):
   - Para a RTX 3050, comece com batch_size=16 ou 8 para imagens 224x224.

3. USE BATCH NORMALIZATION:
   - Adicione BatchNormalization após camadas Dense/Conv2D (você já fez isso, ótimo!).

4. USE GLOBAL AVERAGE POOLING:
   - Em vez de Flatten(), use GlobalAveragePooling2D() (você já fez isso, ótimo!).

5. AJUSTE O LEARNING RATE:
   - Diminua para ~0.0001 quando usar mixed precision (você já fez isso, ótimo!).
""")

print("\n" + "=" * 60)
print("RESUMO FINAL")
print("=" * 60)

# tf.test.is_gpu_available() está obsoleto, a verificação inicial com list_physical_devices é a forma correta.
print(f"GPU disponível: {'✅ Sim' if gpus else '❌ Não'}")
print(f"CUDA disponível: {tf.test.is_built_with_cuda()}")


if gpus:
    print("\n🎉 SUA GPU RTX 3050 ESTÁ FUNCIONANDO CORRETAMENTE!")
    print("   Com as otimizações e o batch_size ajustado, o script deve rodar sem erros.")
else:
    print("\n⚠️  GPU não detectada!")