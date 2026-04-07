import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_graphs(csv_filename):
    # Check if file exists
    if not os.path.exists(csv_filename):
        print(f"Error: Could not find {csv_filename}")
        return

    # Load the data
    df = pd.read_csv(csv_filename, names=['Stride', 'ThreadsPerBlock', 'EnergyPerAccess'], header=0)

    # UNCOMMENT THIS LINE if you want to flip your current negative data right-side up:
    # df['EnergyPerAccess'] = df['EnergyPerAccess'].abs()

    # Get all unique stride lengths to create individual plots
    unique_strides = df['Stride'].unique()
    print(f"Found data for {len(unique_strides)} different stride lengths: {unique_strides}")

    # Create a directory to save the plots
    output_dir = "energy_graphs"
    os.makedirs(output_dir, exist_ok=True)

    # Group the data by stride length and plot each one
    for stride in unique_strides:
        # Filter data for the current stride
        stride_data = df[df['Stride'] == stride]
        
        # Sort by ThreadsPerBlock to ensure the lines connect properly
        stride_data = stride_data.sort_values(by='ThreadsPerBlock')

        # Create a new figure for this specific stride
        plt.figure(figsize=(8, 5))
        
        # Plot Threads per Block (X) vs Energy per Access (Y)
        plt.plot(stride_data['ThreadsPerBlock'], stride_data['EnergyPerAccess'], 
                 marker='o', linestyle='-', color='b', label=f'Stride {stride}')
        
        # Formatting the graph
        plt.title(f'Energy per Access vs Threads per Block (Stride Length: {stride})')
        
        # 1. Update X-Axis to a log scale with exact specified markers
        plt.xscale('log', base=2)
        target_ticks = [1, 32, 128, 512, 1024]
        plt.xticks(target_ticks, labels=[str(t) for t in target_ticks])
        plt.xlabel('Threads per Block')
        
        # 2. Update Y-Axis to include the exact units (pJ)
        plt.ylabel('Energy per Access (pJ)')
        
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()

        # Save the graph as a PNG image
        output_path = os.path.join(output_dir, f'stride_{stride}_plot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close() # Close the figure to free up memory
        
        print(f"Saved graph: {output_path}")

if __name__ == "__main__":
    generate_graphs('results.csv')
