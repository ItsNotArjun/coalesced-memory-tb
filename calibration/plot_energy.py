import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(filename):
    """Helper function to load CSV data gracefully."""
    if not os.path.exists(filename):
        print(f"Warning: Could not find {filename}")
        return None
    
    # We let pandas infer the column names directly from the CSV header
    # This ensures it works flawlessly with the newly added 'Memory' column!
    return pd.read_csv(filename, header=0)

def generate_graphs():
    # 1. Load the data
    df_opt = load_data('results_optimizer.csv')
    df_rand = load_data('results_random.csv')

    if df_opt is None and df_rand is None:
        print("Error: Neither CSV file was found. Run your Makefile sweeps first!")
        return

    # 2. Create the target directory structure
    base_dir = "energy_graphs"
    dirs = {
        "opt": os.path.join(base_dir, "optimized"),
        "rand": os.path.join(base_dir, "randomized"),
        "comb": os.path.join(base_dir, "combined")
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 3. Determine grouping logic (Handle presence of the new 'Memory' column)
    sample_df = df_opt if df_opt is not None else df_rand
    group_cols = [col for col in ['Memory', 'Stride'] if col in sample_df.columns]

    # Extract all unique combinations of Memory and Stride
    combinations = sample_df[group_cols].drop_duplicates().to_dict('records')
    print(f"Found {len(combinations)} unique parameter combinations. Generating graphs...")

    target_ticks = [1, 32, 128, 512, 1024]

    # 4. Generate the 3 sets of graphs for each combination
    for combo in combinations:
        # Filter the dataframes for the current Memory/Stride combination
        mask_opt = pd.Series(True, index=df_opt.index) if df_opt is not None else None
        mask_rand = pd.Series(True, index=df_rand.index) if df_rand is not None else None
        
        title_parts = []
        filename_parts = []
        
        for col, val in combo.items():
            if df_opt is not None: mask_opt &= (df_opt[col] == val)
            if df_rand is not None: mask_rand &= (df_rand[col] == val)
            title_parts.append(f"{col}: {val}")
            filename_parts.append(f"{str(col).lower()}_{val}")

        # Formatting strings for titles and filenames
        combo_title_suffix = " (" + " | ".join(title_parts) + ")"
        file_suffix = "_".join(filename_parts) + ".png"

        # Extract and sort the data for plotting
        data_opt = df_opt[mask_opt].sort_values(by='ThreadsPerBlock') if df_opt is not None else pd.DataFrame()
        data_rand = df_rand[mask_rand].sort_values(by='ThreadsPerBlock') if df_rand is not None else pd.DataFrame()

        # ==========================================
        # GRAPH 1: OPTIMIZED ONLY
        # ==========================================
        if not data_opt.empty:
            plt.figure(figsize=(8, 5))
            plt.plot(data_opt['ThreadsPerBlock'], data_opt['EnergyPerAccess'], 
                     marker='o', linestyle='-', color='blue', label='Optimized (Uniform Stride)')
            
            plt.title('Energy per Access vs Threads' + combo_title_suffix)
            plt.xscale('log', base=2)
            plt.xticks(target_ticks, labels=[str(t) for t in target_ticks])
            plt.xlabel('Threads per Block')
            plt.ylabel('Energy per Access (pJ)')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend()
            
            plt.savefig(os.path.join(dirs["opt"], file_suffix), dpi=300, bbox_inches='tight')
            plt.close()

        # ==========================================
        # GRAPH 2: RANDOMIZED ONLY
        # ==========================================
        if not data_rand.empty:
            plt.figure(figsize=(8, 5))
            plt.plot(data_rand['ThreadsPerBlock'], data_rand['EnergyPerAccess'], 
                     marker='s', linestyle='-', color='red', label='Randomized (Page Hopping)')
            
            plt.title('Energy per Access vs Threads' + combo_title_suffix)
            plt.xscale('log', base=2)
            plt.xticks(target_ticks, labels=[str(t) for t in target_ticks])
            plt.xlabel('Threads per Block')
            plt.ylabel('Energy per Access (pJ)')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend()
            
            plt.savefig(os.path.join(dirs["rand"], file_suffix), dpi=300, bbox_inches='tight')
            plt.close()

        # ==========================================
        # GRAPH 3: COMBINED OVERLAY
        # ==========================================
        if not data_opt.empty and not data_rand.empty:
            plt.figure(figsize=(8, 5))
            
            # Plot both lines on the same axes
            plt.plot(data_opt['ThreadsPerBlock'], data_opt['EnergyPerAccess'], 
                     marker='o', linestyle='-', color='blue', label='Optimized (Uniform Stride)')
            plt.plot(data_rand['ThreadsPerBlock'], data_rand['EnergyPerAccess'], 
                     marker='s', linestyle='--', color='red', label='Randomized (Page Hopping)')
            
            plt.title('Combined Energy Trace' + combo_title_suffix)
            plt.xscale('log', base=2)
            plt.xticks(target_ticks, labels=[str(t) for t in target_ticks])
            plt.xlabel('Threads per Block')
            plt.ylabel('Energy per Access (pJ)')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend()
            
            plt.savefig(os.path.join(dirs["comb"], file_suffix), dpi=300, bbox_inches='tight')
            plt.close()

    print(f"Successfully generated all plots in the '{base_dir}' directory.")

if __name__ == "__main__":
    generate_graphs()