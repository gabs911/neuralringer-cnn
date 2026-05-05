import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from modules.cnn_model import NeuralRingerCNN

# ==============================================================================
# HYPERPARAMETER CONFIGURATION
# Modify these values to tune the network without altering the core logic
# ==============================================================================
CONFIG = {
    'data_dir': 'data',
    'output_dir': 'models',
    'k_folds': 10,
    'epochs': 50,
    'batch_size': 256,
    'learning_rate': 0.001,
    'filters_1': 4,
    'filters_2': 8,
    'kernel_size': 2,
    'hidden_units': 16,
    'input_steps': 50,      #  First 50 concentric rings
    'input_channels': 1
}
# ==============================================================================

def main():
    print("==================================================")
    print(" STARTING TRAINING - NEURAL RINGER FPGA           ")
    print("==================================================")

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # 1. Load Pre-processed Numpy Arrays
    print("[Training] Loading dataset matrices...")
    x_path = os.path.join(CONFIG['data_dir'], 'X_norm.npy')
    y_path = os.path.join(CONFIG['data_dir'], 'y.npy')

    X_norm = np.load(x_path)
    y = np.load(y_path)
    print(f"[Training] Data successfully loaded! X: {X_norm.shape} | y: {y.shape}")

    # 2. Setup Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=42)
    
    fold_no = 1
    val_predictions = {}

    for train_index, val_index in skf.split(X_norm, y):
        print(f"\n--- Starting Fold {fold_no} / {CONFIG['k_folds']} ---")

        X_train, X_val = X_norm[train_index], X_norm[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # 3. Handle Class Imbalance via Class Weights
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        total = neg_count + pos_count
        
        weight_for_0 = (1 / neg_count) * (total / 2.0)
        weight_for_1 = (1 / pos_count) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        print(f"[Training] Class weights applied: Jet(0)={weight_for_0:.2f} | Electron(1)={weight_for_1:.2f}")

        # 4. Instantiate the configurable model
        cnn_builder = NeuralRingerCNN(
            input_shape=(CONFIG['input_steps'], CONFIG['input_channels']),
            filters_1=CONFIG['filters_1'],
            filters_2=CONFIG['filters_2'],
            kernel_size=CONFIG['kernel_size'],
            hidden_units=CONFIG['hidden_units'],
            learning_rate=CONFIG['learning_rate']
        )
        model = cnn_builder.get_model()

        # 5. Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            class_weight=class_weight,
            verbose=1
        )

        # 6. Save the Fold Weights
        model_path = os.path.join(CONFIG['output_dir'], f"cnn1d_ringer_fold_{fold_no}.h5")
        model.save(model_path)
        print(f"[Training] Model weights saved at: {model_path}")

        # 7. Save raw predictions for the validation set (Needed for Script 03 - SP Index)
        preds = model.predict(X_val, verbose=0)
        val_predictions[f"fold_{fold_no}"] = {
            'y_true': y_val,
            'y_pred': preds.flatten()
        }

        fold_no += 1

    # 8. Export all predictions for the evaluation script
    preds_path = os.path.join(CONFIG['output_dir'], 'validation_predictions.npy')
    np.save(preds_path, val_predictions, allow_pickle=True)
    print(f"\n[Training] K-Fold Cross-validation completed! Predictions dictionary saved at: {preds_path}")

if __name__ == "__main__":
    main()
