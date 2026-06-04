import os
import numpy as np
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# Importa as nossas funções dos módulos
from modules.cnn_model import build_v10_cnn
from modules.sp_callback import SPMaxCheckpoint

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    'input_x': 'data/X_rings_normalized.npy',
    'input_y': 'data/y_target.npy',
    'base_data_dir': 'data',
    'base_model_dir': 'models',
    
    # NOVO: Proporção de anéis a serem extraídos DE CADA CAMADA (ex: 0.5 = 50%)
    'ring_proportion': 0.125, 
    
    'k_folds': 10,             
    'epochs': 100,             
    'batch_size': 256,         
    'learning_rate': 0.001,
    'patience': 3             
}

# Definição oficial dos 100 anéis concatenados por camada (Tabela 5.1 da Tese do Joao)
ATLAS_LAYERS = {
    'PS': 8,
    'EM1': 64,
    'EM2': 8,
    'EM3': 8,
    'HAD1': 4,
    'HAD2': 4,
    'HAD3': 4
}
# ==============================================================================

def get_layer_indices(proportion):
    """
    Calcula os índices exatos para fatiar a proporção desejada de anéis 
    preservando a amostragem do núcleo do chuveiro em cada camada do calorímetro.
    """
    indices_to_keep = []
    current_offset = 0
    
    print("\n[*] Mapeamento de Anéis por Camada:")
    for layer_name, total_rings in ATLAS_LAYERS.items():
        # Calcula quantos anéis manter nesta camada (arredondando para o inteiro mais próximo)
        keep_count = int(round(total_rings * proportion))
        
        # Garante que pelo menos 1 anel seja mantido se a proporção for muito baixa
        if keep_count == 0 and proportion > 0:
            keep_count = 1 
            
        layer_indices = list(range(current_offset, current_offset + keep_count))
        indices_to_keep.extend(layer_indices)
        
        print(f"    -> {layer_name}: {keep_count}/{total_rings} anéis mantidos (Índices {current_offset} a {current_offset + keep_count - 1})")
        
        current_offset += total_rings
        
    return indices_to_keep

def main():
    # 1. Mapeamento da nova janela baseada na proporção
    proportion = CONFIG['ring_proportion']
    selected_indices = get_layer_indices(proportion)
    num_rings = len(selected_indices)
    
    print("==================================================")
    print(f" SCRIPT 02: HYPO TRAINING - CNN 1D ({num_rings} RINGS) ")
    print(f" PROPORÇÃO SELECIONADA: {proportion*100}% DE CADA CAMADA ")
    print("==================================================")

    current_model_dir = os.path.join(CONFIG['base_model_dir'], f'{num_rings}_rings')
    current_data_dir = os.path.join(CONFIG['base_data_dir'], f'{num_rings}_rings')
    
    os.makedirs(current_model_dir, exist_ok=True)
    os.makedirs(current_data_dir, exist_ok=True)
    output_pred_file = os.path.join(current_data_dir, 'validation_predictions.npy')

    # 2. Carregamento e Fatiamento Físico da Matriz
    print(f"[*] Carregando e fatiando matrizes FEX...")
    X_full = np.load(CONFIG['input_x'])
    
    # AGORA SIM! Fatiamos pegando os índices espaciais corretos de cada camada
    X = X_full[:, selected_indices] 
    y = np.load(CONFIG['input_y'])
    X = np.expand_dims(X, axis=-1)

    skf = StratifiedKFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=42)
    validation_predictions = {}

    # 3. Treinamento Orientado pelo K-Fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Iniciando Treinamento do Fold {fold}/{CONFIG['k_folds']} ---")
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
        
        # A rede neural é construída com o número total de anéis extraídos
        model = build_v10_cnn(input_length=num_rings, learning_rate=CONFIG['learning_rate'])
        model_path = os.path.join(current_model_dir, f'cnn_fold_{fold}.h5')
        
        sp_checkpoint = SPMaxCheckpoint(
            X_val=X_val, 
            y_val=y_val, 
            filepath=model_path, 
            patience=CONFIG['patience']
        )
        
        model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  epochs=CONFIG['epochs'],
                  batch_size=CONFIG['batch_size'],
                  class_weight=class_weight_dict,
                  callbacks=[sp_checkpoint], 
                  verbose=1)
        
        print(f"[*] Extraindo predições do pico de SP_max...")
        y_pred = model.predict(X_val).flatten()
        
        validation_predictions[f'fold_{fold}'] = {
            'y_true': y_val,
            'y_pred': y_pred
        }

    np.save(output_pred_file, validation_predictions)
    print(f"\n[SUCESSO] Treinamento concluído e orientado pelo SP_max!")

if __name__ == "__main__":
    main()
