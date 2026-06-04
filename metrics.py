
import os
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    'num_rings': 25, 
    'base_data_dir': 'data',
    'base_output_dir': 'results'
}
# ==============================================================================

def calculate_metrics_joao_thesis(y_true, y_pred_prob, thresholds):
    """
    Calcula as figuras de mérito exatas descritas na Tabela 5.4 da Tese do João.
    """
    pd_list, fa_list, sp_list = [], [], []
    
    # Total real de elétrons (Theta_e) e total real de jatos/ruído (Theta_b)
    theta_e = np.sum(y_true == 1)
    theta_b = np.sum(y_true == 0)
    
    for t in thresholds:
        # Decisão do classificador baseada no limiar
        y_pred_bin = (y_pred_prob >= t).astype(int)
        
        # Elétrons classificados corretamente (Theta_{e|e})
        theta_e_e = np.sum((y_true == 1) & (y_pred_bin == 1))
        # Jatos classificados erroneamente como elétrons (Theta_{e|b})
        theta_e_b = np.sum((y_true == 0) & (y_pred_bin == 1))
        
        # 1. Probabilidade de Detecção (P_D)
        P_D = theta_e_e / (theta_e + 1e-10)
        
        # 2. Probabilidade de Falso Alarme (F_A)
        F_A = theta_e_b / (theta_b + 1e-10)
        
        # 3. Índice Soma-Produto (SP)
        sp = np.sqrt(np.sqrt(P_D * (1.0 - F_A)) * (P_D + (1.0 - F_A)) / 2.0)
        
        pd_list.append(P_D)
        fa_list.append(F_A)
        sp_list.append(sp)
        
    return np.array(pd_list), np.array(fa_list), np.array(sp_list)

def main():
    print("==================================================")
    print(f" SCRIPT 03: AVALIAÇÃO DE MÉTRICAS DA TESE ({CONFIG['num_rings']} ANÉIS) ")
    print("==================================================")

    # 1. Configuração Inteligente de Pastas (Salva identificando os anéis)
    num_rings = CONFIG['num_rings']
    input_file = os.path.join(CONFIG['base_data_dir'], f'{num_rings}_rings', 'validation_predictions.npy')
    
    # Criação da pasta de resultados nomeada pela quantidade de anéis
    output_dir = os.path.join(CONFIG['base_output_dir'], f'{num_rings}_rings')
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_file):
        print(f"[ERRO] Arquivo de predições não encontrado em: {input_file}")
        return

    # 2. Carregamento dos resultados das 10 partições (k-folds)
    print(f"[*] Carregando predições de validação: {input_file}")
    cv_predictions = np.load(input_file, allow_pickle=True).item()

    thresholds = np.linspace(0, 1, 1000)
    
    best_sp_folds, best_pd_folds, best_fa_folds = [], [], []
    plt.figure(figsize=(12, 5))
    
    print("[*] Extraindo curvas ROC e SP_max por partição de validação...")
    
    # 3. Varredura por Limiares para cada Fold
    for fold_name, data in cv_predictions.items():
        y_true = data['y_true']
        y_pred = data['y_pred']
        
        # Calcula P_D, F_A e SP conforme a tese
        pd_arr, fa_arr, sp_arr = calculate_metrics_joao_thesis(y_true, y_pred, thresholds)
        
        # Encontra o ponto de SP_max
        max_idx = np.argmax(sp_arr)
        best_sp_folds.append(sp_arr[max_idx])
        best_pd_folds.append(pd_arr[max_idx])
        best_fa_folds.append(fa_arr[max_idx])
        
        plt.subplot(1, 2, 1)
        plt.plot(pd_arr, 1.0 - fa_arr, alpha=0.3, color='blue') # ROC: P_D vs (1 - F_A)
        plt.subplot(1, 2, 2)
        plt.plot(thresholds, sp_arr, alpha=0.3, color='green')

    # 4. Cálculo da Flutuação Estatística (Médias)
    mean_sp, std_sp = np.mean(best_sp_folds), np.std(best_sp_folds)
    mean_pd, std_pd = np.mean(best_pd_folds), np.std(best_pd_folds)
    mean_fa, std_fa = np.mean(best_fa_folds), np.std(best_fa_folds)

    # 5. Formatação dos gráficos
    plt.subplot(1, 2, 1)
    plt.title(f'Curva ROC - {num_rings} Anéis')
    plt.xlabel('Probabilidade de Detecção (P_D)')
    plt.ylabel('Rejeição de Ruído (1 - F_A)')
    plt.grid(True, linestyle='--')

    plt.subplot(1, 2, 2)
    plt.title(f'Curva do Índice SP - {num_rings} Anéis')
    plt.xlabel('Limiar de Decisão')
    plt.ylabel('Índice SP')
    plt.grid(True, linestyle='--')

    plot_path = os.path.join(output_dir, f'roc_sp_curves_{num_rings}rings.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    
    # 6. Relatório Final Fiel à Tese
    print("\n==================================================")
    print(" RESULTADOS DO CONJUNTO DE VALIDAÇÃO CRUZADA      ")
    print("==================================================")
    print(f" -> Resultados armazenados em: {output_dir}/")
    print("--------------------------------------------------")
    print(f" * Probabilidade de Detecção (P_D): {mean_pd:.4f} ± {std_pd:.4f}")
    print(f" * Probabilidade de Falso Alarme (F_A): {mean_fa:.4f} ± {std_fa:.4f}")
    print(f" * Índice Soma-Produto (SP_max): {mean_sp:.4f} ± {std_sp:.4f}")
    print("==================================================")

if __name__ == "__main__":
    main()