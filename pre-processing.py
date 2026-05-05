import os
import glob
from modules.data_loader import NeuralRingerDataLoader

def main():
    print("==================================================")
    print(" STARTING PRE-PROCESSING - NEURAL RINGER FPGA     ")
    print("==================================================")
    
    # 1. Map all .parquet files inside the 'data' folder
    data_dir = '/mnt/shared/storage03/projects/cern/homes/gabriel.lisboa/data/ringer-datasets/mc21_isabela_qt_2sigma_restriction/data.parquet'
    parquet_files = glob.glob(os.path.join(data_dir, '*.parquet'))
    
    if not parquet_files:
        print(f"ERROR: No .parquet files found in the '{data_dir}' folder.")
        return
        
    # 2. Instantiate the DataLoader module with 50 rings defined for the FPGA
    dataloader = NeuralRingerDataLoader(file_list=parquet_files, max_rings=50)
    
    # 3. Execute the pre-processing pipeline
    dataloader.process_pipeline()
    
    print("==================================================")
    print(" PRE-PROCESSING SUCCESSFULLY COMPLETED!           ")
    print("==================================================")

if __name__ == "__main__":
    main()