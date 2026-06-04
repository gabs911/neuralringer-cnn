import numpy as np
from tensorflow.keras.callbacks import Callback

class SPMaxCheckpoint(Callback):
    """
    Callback customizado para monitorar o SP_max (Índice Soma-Produto máximo) 
    do ATLAS no final de cada época. Salva o melhor modelo e aplica Early Stopping.
    """
    def __init__(self, X_val, y_val, filepath, patience=10):
        super(SPMaxCheckpoint, self).__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.filepath = filepath
        self.patience = patience
        
        self.best_sp = -1.0
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        # 1. Realiza a predição contínua (0 a 1) para o conjunto de validação
        y_pred_prob = self.model.predict(self.X_val, verbose=0).flatten()
        
        # 2. Varredura de 100 limiares (thresholds) para encontrar o SP_max da época
        thresholds = np.linspace(0, 1, 101)
        sp_max_epoch = -1.0
        
        for t in thresholds:
            y_pred_bin = (y_pred_prob >= t).astype(int)
            
            # Matriz de Confusão Vetorizada (Rápida na CPU)
            TP = np.sum((self.y_val == 1) & (y_pred_bin == 1))
            TN = np.sum((self.y_val == 0) & (y_pred_bin == 0))
            FP = np.sum((self.y_val == 0) & (y_pred_bin == 1))
            FN = np.sum((self.y_val == 1) & (y_pred_bin == 0))
            
            # Eficiência de Sinal (P_D) e Probabilidade de Falso Alarme (F_A)
            P_D = TP / (TP + FN + 1e-10)
            F_A = FP / (FP + TN + 1e-10)
            
            # Fórmula Oficial do Índice SP do NeuralRinger
            # SP = sqrt( sqrt(P_D * (1 - F_A)) * (P_D + (1 - F_A)) / 2 )
            sp = np.sqrt(np.sqrt(P_D * (1.0 - F_A)) * (P_D + (1.0 - F_A)) / 2.0)
            
            if sp > sp_max_epoch:
                sp_max_epoch = sp
        
        print(f" - val_SPmax: {sp_max_epoch:.4f}")
        
        # 3. Lógica de Salvamento e Early Stopping baseada no SP_max
        if sp_max_epoch > self.best_sp:
            print(f"\n[*] SP_max bateu recorde! Subiu de {self.best_sp:.4f} para {sp_max_epoch:.4f}. Salvando pesos...")
            self.best_sp = sp_max_epoch
            self.wait = 0
            self.best_weights = self.model.get_weights()
            self.model.save(self.filepath)
        else:
            self.wait += 1
            print(f"[*] SP_max não melhorou ({self.wait}/{self.patience})")
            if self.wait >= self.patience:
                print(f"\n[!] Early Stopping acionado! O SP_max parou de subir.")
                self.model.stop_training = True
                # Restaura a rede para o estado em que ela alcançou o maior SP_max
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)