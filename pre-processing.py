import os
import glob
import numpy as np
import polars as pl
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    # Caminho exato que você usou no terminal
    'input_files': '/home/gabriel.guimaraes/data/mc21_isabela_qt_2sigma_restriction/data.parquet/*.parquet', 
    'output_x': 'data/X_rings_normalized.npy',
    'output_y': 'data/y_target.npy'
}
# ==============================================================================

def main():
    print("==================================================")
    print(" SCRIPT 01: PRE-PROCESSING & NORMALIZATION (BATCH)")
    print("==================================================")
    
    # 1. Mapeia todos os arquivos Parquet disponíveis
    files = glob.glob(CONFIG['input_files'])
    if not files:
        print(f"[ERRO] Nenhum arquivo encontrado em: {CONFIG['input_files']}")
        return
        
    print(f"[*] Encontrados {len(files)} arquivos Parquet. Iniciando extração...")
    
    X_list = []
    y_list = []
    
    # 2. Leitura iterativa arquivo por arquivo com Barra de Progresso (tqdm)
    # O tqdm vai calcular o tempo médio por arquivo e te dar a estimativa de duração!
    for f in tqdm(files, desc="Processando arquivos", unit="arq"):
        # Lê apenas o arquivo da iteração atual
        df = pl.read_parquet(f).select(['trig_L2_calo_rings', 'target'])
        
        # Descompacta o lote atual (não satura a RAM inteira)
        X_chunk = np.vstack(df.get_column('trig_L2_calo_rings').to_list())
        y_chunk = df.get_column('target').to_numpy().flatten()
        
        # Salva as matrizes Numpy nativas na lista
        X_list.append(X_chunk)
        y_list.append(y_chunk)
        
        # Deleta o dataframe do Polars para liberar RAM imediatamente
        del df
        
    # 3. Concatenação final das matrizes Numpy
    print("\n[*] Todos os arquivos lidos! Empilhando as matrizes...")
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    # Libera a lista original da memória
    del X_list, y_list
    
    # 4. Normalização Física do NeuralRinger
    print("[*] Aplicando normalização física (X / |sum(X)|)...")
    abs_sum = np.abs(np.sum(X, axis=1, keepdims=True)) + 1e-10
    X_normalized = X / abs_sum
    
    # 5. Exportação
    print("[*] Salvando matriz X e vetor y no disco...")
    os.makedirs('data', exist_ok=True)
    np.save(CONFIG['output_x'], X_normalized)
    np.save(CONFIG['output_y'], y)
    
    print(f"\n[SUCESSO] Pré-processamento concluído!")
    print(f" -> Formato da Matriz X: {X_normalized.shape}")
    print(f" -> Formato do Vetor y: {y.shape}")

if __name__ == "__main__":
    main()