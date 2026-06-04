import os
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    'input_x': '../data/X_rings_normalized.npy',
    'input_y': '../data/y_target.npy',
    'num_rings': 100, # Pode alterar para 100 se estiver usando a matriz completa
    'output_dir': 'results'
}

def main():
    # 1. Carregamento dos Dados
    if not os.path.exists(CONFIG['input_x']) or not os.path.exists(CONFIG['input_y']):
        print("[ERRO] Matrizes de dados não encontradas na pasta 'data/'.")
        return

    print(f"[*] Carregando matrizes de dados...")
    X = np.load(CONFIG['input_x'])[:, :CONFIG['num_rings']]
    y = np.load(CONFIG['input_y'])

    # 2. Separação das Classes
    # y == 1 (Sinal / Elétrons) | y == 0 (Ruído / Jatos Hadrônicos)
    X_electrons = X[y == 1]
    X_jets = X[y == 0]

    print(f"[*] Amostras de Elétrons encontradas: {len(X_electrons)}")
    print(f"[*] Amostras de Jatos encontradas: {len(X_jets)}")

    # 3. Cálculo do Perfil Médio de Energia
    # Calcula a média de cada uma das colunas (anéis) ao longo de todas as amostras
    mean_electrons = np.mean(X_electrons, axis=0)
    mean_jets = np.mean(X_jets, axis=0)
    
    # 4. Construção do Gráfico
    print("[*] Gerando o gráfico de perfil...")
    plt.figure(figsize=(10, 6))
    
    rings_axis = np.arange(1, CONFIG['num_rings'] + 1)
    
    # Plotando as linhas com marcadores para evidenciar cada anel
    plt.plot(rings_axis, mean_electrons, marker='o', markersize=4, linestyle='-', 
             color='blue', label=r'Sinal (Elétrons $Z \rightarrow ee$)')
    plt.plot(rings_axis, mean_jets, marker='v', markersize=4, linestyle='-', 
             color='red', label='Ruído (Jatos Hadrônicos)')

    # 5. Formatação baseada na literatura do ATLAS
    plt.title(f"Perfil Médio de Deposição de Energia ({CONFIG['num_rings']} Anéis)")
    plt.xlabel("Número do Anel")
    plt.ylabel("Energia Normalizada Média")
    
    # Opcional: Adiciona escala logarítmica no eixo Y se a diferença for muito extrema
    # plt.yscale('log') 
    
    plt.grid(True, which="both", linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', frameon=True)
    
    # Destaca as divisões longitudinais se estiver usando o modelo padrão de 50 anéis
    if CONFIG['num_rings'] == 50:
        # Posição aproximada da transição das camadas (PS, EM1, EM2, EM3, HAD1, HAD2, HAD3)
        camadas_indices = [4-9]
        camadas_nomes = ['PS', 'EM1', 'EM2', 'EM3', 'HAD1', 'HAD2', 'HAD3']
        
        for idx in camadas_indices:
            plt.axvline(x=idx + 0.5, color='black', linestyle=':', alpha=0.5)
            
    # 6 Salva gráfico
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    output_path = os.path.join(CONFIG['output_dir'], f"perfil_energia_{CONFIG['num_rings']}rings.png")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[SUCESSO] Gráfico salvo em: {output_path}")
    print("==================================================")

if __name__ == "__main__":
    main()