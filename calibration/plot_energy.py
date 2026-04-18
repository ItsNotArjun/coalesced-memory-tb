import pandas as pd
import matplotlib.pyplot as plt
import os


MODE_LABELS = {
    0: 'Coalesced',
    1: 'Strided',
    2: 'Random',
}


def plot_mode_results(df, output_dir):
    baseline_rows = df[df['Mode'] == 0].sort_values(by=['Stride', 'ThreadsPerBlock'])
    if baseline_rows.empty:
        print('Warning: no MODE=0 baseline found; skipping mode amplification plot.')
        return

    baseline_e_useful = baseline_rows.iloc[0]['EnergyPerUsefulBytePJ']

    mode_df = df[['Mode', 'EnergyPerUsefulBytePJ']].copy()
    mode_df['Mode'] = mode_df['Mode'].astype(int)
    mode_df = mode_df.groupby('Mode', as_index=False)['EnergyPerUsefulBytePJ'].mean()
    mode_df = mode_df.sort_values(by='Mode')
    mode_df['Amplification'] = mode_df['EnergyPerUsefulBytePJ'] / baseline_e_useful

    labels = [MODE_LABELS.get(m, f'Mode {m}') for m in mode_df['Mode']]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(labels, mode_df['EnergyPerUsefulBytePJ'], color=['#4e79a7', '#f28e2b', '#e15759'])
    axes[0].set_title('Energy per Useful Byte by Mode')
    axes[0].set_xlabel('Mode')
    axes[0].set_ylabel('Energy per Useful Byte (pJ/byte)')
    axes[0].grid(True, axis='y', linestyle='--', linewidth=0.5)

    axes[1].bar(labels, mode_df['Amplification'], color=['#4e79a7', '#f28e2b', '#e15759'])
    axes[1].set_title('Energy Amplification vs Coalesced')
    axes[1].set_xlabel('Mode')
    axes[1].set_ylabel('Amplification (x)')
    axes[1].axhline(1.0, color='black', linestyle=':', linewidth=1.0)
    axes[1].grid(True, axis='y', linestyle='--', linewidth=0.5)

    fig.tight_layout()
    output_path = os.path.join(output_dir, 'mode_energy_amplification.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved graph: {output_path}')


def plot_disorder_sweep(df, output_dir):
    strided = df[df['Mode'] == 1].copy()
    if strided.empty:
        print('Warning: no MODE=1 rows found; skipping disorder sweep plot.')
        return

    baseline_rows = df[df['Mode'] == 0].sort_values(by=['Stride', 'ThreadsPerBlock'])
    baseline_e_useful = baseline_rows.iloc[0]['EnergyPerUsefulBytePJ'] if not baseline_rows.empty else None

    strided = strided.sort_values(by='Stride')
    plt.figure(figsize=(8, 5))
    plt.plot(
        strided['Stride'],
        strided['EnergyPerUsefulBytePJ'],
        marker='o',
        linestyle='-',
        color='#f28e2b',
        label='Strided mode',
    )
    plt.xscale('log', base=2)
    plt.xlabel('Spatial Stride')
    plt.ylabel('Energy per Useful Byte (pJ/byte)')
    plt.title('Disorder Sweep: Stride vs Energy')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if baseline_e_useful and baseline_e_useful != 0:
        plt.axhline(baseline_e_useful, color='#4e79a7', linestyle=':', linewidth=1.5, label='Coalesced baseline')

    plt.legend()
    output_path = os.path.join(output_dir, 'disorder_stride_sweep.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved graph: {output_path}')


def plot_stride_results(df, output_dir):
    # Get all unique stride lengths to create individual plots
    unique_strides = df['Stride'].unique()
    print(f'Found data for {len(unique_strides)} different stride lengths: {unique_strides}')

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

        # Update X-axis to a log scale with exact specified markers
        plt.xscale('log', base=2)
        target_ticks = [1, 32, 128, 512, 1024]
        plt.xticks(target_ticks, labels=[str(t) for t in target_ticks])
        plt.xlabel('Threads per Block')

        # Update Y-axis to include the exact units (pJ)
        plt.ylabel('Energy per Access (pJ)')

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()

        # Save the graph as a PNG image
        output_path = os.path.join(output_dir, f'stride_{stride}_plot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free up memory

        print(f'Saved graph: {output_path}')

def generate_graphs(csv_filename):
    # Check if file exists
    if not os.path.exists(csv_filename):
        print(f"Error: Could not find {csv_filename}")
        return

    # Load the data
    df = pd.read_csv(csv_filename)
    df.columns = [c.strip() for c in df.columns]

    if 'EnergyPerAccessPJ' in df.columns and 'EnergyPerAccess' not in df.columns:
        df['EnergyPerAccess'] = df['EnergyPerAccessPJ']
    if 'EnergyPerUsefulBytePJ' in df.columns and 'EnergyPerUsefulByte' not in df.columns:
        df['EnergyPerUsefulByte'] = df['EnergyPerUsefulBytePJ']

    # UNCOMMENT THIS LINE if you want to flip your current negative data right-side up:
    # df['EnergyPerAccess'] = df['EnergyPerAccess'].abs()

    # Create a directory to save the plots
    output_dir = "energy_graphs"
    os.makedirs(output_dir, exist_ok=True)

    if {'Mode', 'Stride', 'ThreadsPerBlock', 'EnergyPerUsefulBytePJ'}.issubset(df.columns):
        plot_mode_results(df, output_dir)
        plot_disorder_sweep(df, output_dir)
        return

    if {'Mode', 'Stride', 'ThreadsPerBlock', 'EnergyPerAccess'}.issubset(df.columns):
        plot_mode_results(df.assign(EnergyPerUsefulBytePJ=df['EnergyPerAccess']), output_dir)
        plot_disorder_sweep(df.assign(EnergyPerUsefulBytePJ=df['EnergyPerAccess']), output_dir)
        return

    if {'Stride', 'ThreadsPerBlock', 'EnergyPerAccess'}.issubset(df.columns):
        plot_stride_results(df, output_dir)
        return

    print('Error: CSV schema not recognized. Expected either:')
    print('  Mode,Stride,ThreadsPerBlock,...,EnergyPerUsefulBytePJ')
    print('or')
    print('  Mode,Stride,ThreadsPerBlock,EnergyPerAccess')
    print('or')
    print('  Stride,ThreadsPerBlock,EnergyPerAccess')

if __name__ == "__main__":
    generate_graphs('results.csv')
