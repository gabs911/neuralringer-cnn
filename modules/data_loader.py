import polars as pl
import numpy as np
import os

class NeuralRingerDataLoader:
    def __init__(self, file_list, max_rings=50):
        """
        Optimized loader module using Polars (Lazy Evaluation).
        :param file_list: List of paths to the .parquet files
        :param max_rings: Spatial limit of rings (Default: 50 for FPGA)
        """
        self.file_list = file_list
        self.max_rings = max_rings
        self.X_norm = None
        self.y = None

    def load_and_process_batch(self):
        """Executes parallel reading and mathematical data processing."""
        print(f"[Loader] Creating execution plan for {len(self.file_list)} file(s)...")
        
        # Lazy reading and strict parallel extraction of target columns
        lazy_df = pl.scan_parquet(self.file_list)
        df = lazy_df.select(['trig_L2_calo_rings', 'target']).collect()
        
        print("[Loader] Data in memory. Converting tensors...")
        self.y = df['target'].to_numpy().astype(np.float32)
        
        # Stacking the rings list into a 2D matrix (N_events x N_rings)
        X_raw = np.stack(df['trig_L2_calo_rings'].to_list()).astype(np.float32)
        
        # v12/v14 Strategy: Spatial cut to mitigate collimated (boosted) events
        if X_raw.shape[1] > self.max_rings:
            X_raw = X_raw[:, :self.max_rings]
            
        print("[Loader] Applying official NeuralRinger normalization...")
        absolute_sum = np.abs(X_raw).sum(axis=1, keepdims=True)
        absolute_sum[absolute_sum == 0] = 1.0 # Prevents division by zero
        
        X_norm_2d = X_raw / absolute_sum
        
        # 3D reshaping for Keras Conv1D layer: (batch_size, steps, channels)
        self.X_norm = np.expand_dims(X_norm_2d, axis=-1)
        
        print(f"[Loader] Done! X matrix: {self.X_norm.shape} | y vector: {self.y.shape}")

    def save_numpy(self, output_dir='data'):
        """Saves the processed matrices into Numpy binary files."""
        os.makedirs(output_dir, exist_ok=True)
        x_path = os.path.join(output_dir, 'X_norm.npy')
        y_path = os.path.join(output_dir, 'y.npy')
        
        np.save(x_path, self.X_norm)
        np.save(y_path, self.y)
        print(f"[Loader] Matrices successfully exported to '{output_dir}/'.")

    def process_pipeline(self):
        """Orchestrates the complete module flow."""
        self.load_and_process_batch()
        self.save_numpy()