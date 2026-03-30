"""
Compare HMM vs K-means Regime Assignment
Generate comparison tables for manuscript
"""

import pandas as pd
import numpy as np

def load_results():
    """Load both K-means and HMM ablation results."""
    kmeans_results = pd.read_csv("ablation_results.csv")
    hmm_results = pd.read_csv("ablation_results_hmm.csv")

    # Standardize column names
    kmeans_results = kmeans_results.rename(columns={
        'Dataset': 'dataset',
        'n_regimes': 'R',
        'NRMSE': 'nrmse',
        'max_spectral_radius': 'rho_max'
    })

    # Add method column
    kmeans_results['method'] = 'K-means'
    hmm_results['method'] = 'HMM'

    # Combine
    all_results = pd.concat([kmeans_results, hmm_results], ignore_index=True)

    return all_results, kmeans_results, hmm_results


def generate_comparison_table():
    """Generate LaTeX comparison table."""

    all_results, kmeans_df, hmm_df = load_results()

    print("\n" + "="*80)
    print("HMM vs K-means: Best Performance by Dataset")
    print("="*80)

    # For each dataset and horizon, find best method
    datasets = kmeans_df['dataset'].unique()
    horizons = [1, 5, 10]

    comparison = []

    for dataset in datasets:
        for h in horizons:
            # Get results for this dataset/horizon
            kmeans_best = kmeans_df[
                (kmeans_df['dataset'] == dataset) &
                (kmeans_df['horizon'] == h)
            ].nsmallest(1, 'nrmse')

            hmm_best = hmm_df[
                (hmm_df['dataset'] == dataset) &
                (hmm_df['horizon'] == h)
            ].nsmallest(1, 'nrmse')

            if len(kmeans_best) > 0 and len(hmm_best) > 0:
                km_nrmse = kmeans_best.iloc[0]['nrmse']
                km_R = kmeans_best.iloc[0]['R']
                km_rho = kmeans_best.iloc[0]['rho_max']

                hm_nrmse = hmm_best.iloc[0]['nrmse']
                hm_R = hmm_best.iloc[0]['R']
                hm_rho = hmm_best.iloc[0]['rho_max']

                better = 'HMM' if hm_nrmse < km_nrmse else 'K-means'
                improvement = abs(km_nrmse - hm_nrmse) / km_nrmse * 100

                comparison.append({
                    'Dataset': dataset,
                    'Horizon': h,
                    'K-means R': km_R,
                    'K-means NRMSE': km_nrmse,
                    'K-means ρ': km_rho,
                    'HMM R': hm_R,
                    'HMM NRMSE': hm_nrmse,
                    'HMM ρ': hm_rho,
                    'Better': better,
                    'Improvement %': improvement
                })

    comp_df = pd.DataFrame(comparison)

    # Print summary
    print("\nSummary:")
    print(comp_df.to_string(index=False))

    # Count wins
    kmeans_wins = (comp_df['Better'] == 'K-means').sum()
    hmm_wins = (comp_df['Better'] == 'HMM').sum()

    print(f"\n\nOverall Wins:")
    print(f"  K-means: {kmeans_wins}/{len(comp_df)}")
    print(f"  HMM:     {hmm_wins}/{len(comp_df)}")

    # Generate LaTeX table (compact version)
    print("\n" + "="*80)
    print("LaTeX Comparison Table (Compact)")
    print("="*80)

    latex_lines = []
    latex_lines.append(r"\begin{table}[h!]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{HMM vs K-means: Best Performance by Dataset and Horizon}")
    latex_lines.append(r"\label{tab:hmm-kmeans-comparison}")
    latex_lines.append(r"\begin{tabular}{lc|ccc|ccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"\textbf{Dataset} & $\bm{h}$ & \multicolumn{3}{c|}{\textbf{K-means (Hard)}} & \multicolumn{3}{c}{\textbf{HMM (Soft)}} \\")
    latex_lines.append(r" & & $R$ & NRMSE & $\rho_{\max}$ & $R$ & NRMSE & $\rho_{\max}$ \\")
    latex_lines.append(r"\midrule")

    for dataset in datasets:
        first_row = True
        for h in [1, 5, 10]:
            row_data = comp_df[(comp_df['Dataset'] == dataset) & (comp_df['Horizon'] == h)]
            if len(row_data) > 0:
                r = row_data.iloc[0]

                # Format NRMSE with bold for winner
                km_nrmse_str = f"{r['K-means NRMSE']:.4f}"
                hm_nrmse_str = f"{r['HMM NRMSE']:.4f}"

                if r['Better'] == 'K-means':
                    km_nrmse_str = r"\textbf{" + km_nrmse_str + "}"
                else:
                    hm_nrmse_str = r"\textbf{" + hm_nrmse_str + "}"

                # Add instability markers
                km_rho_str = f"{r['K-means ρ']:.3f}"
                if r['K-means ρ'] > 1.0:
                    km_rho_str += r"$^*$"

                hm_rho_str = f"{r['HMM ρ']:.3f}"
                if r['HMM ρ'] > 1.0:
                    hm_rho_str += r"$^*$"

                if first_row:
                    dataset_str = dataset
                    first_row = False
                else:
                    dataset_str = ""

                line = f"{dataset_str} & {h} & {int(r['K-means R'])} & {km_nrmse_str} & {km_rho_str} & {int(r['HMM R'])} & {hm_nrmse_str} & {hm_rho_str} \\\\"
                latex_lines.append(line)

        if dataset != datasets[-1]:
            latex_lines.append(r"\midrule")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\vspace{0.1cm}")
    latex_lines.append(r"\footnotesize{$^*$ Indicates $\rho_{\max} > 1$ (unstable). Bold indicates better NRMSE.}")
    latex_lines.append(r"\end{table}")

    latex_str = "\n".join(latex_lines)
    print(latex_str)

    # Save to file
    with open("tables/hmm_kmeans_comparison.tex", "w") as f:
        f.write(latex_str)

    print("\nLaTeX table saved to: tables/hmm_kmeans_comparison.tex")

    # Generate summary statistics table
    print("\n" + "="*80)
    print("Key Findings")
    print("="*80)

    # Stability comparison
    kmeans_unstable = kmeans_df[kmeans_df['rho_max'] > 1.0].shape[0]
    hmm_unstable = hmm_df[hmm_df['rho_max'] > 1.0].shape[0]

    print(f"\nInstability (ρ > 1):")
    print(f"  K-means: {kmeans_unstable}/{len(kmeans_df)} configs ({kmeans_unstable/len(kmeans_df)*100:.1f}%)")
    print(f"  HMM:     {hmm_unstable}/{len(hmm_df)} configs ({hmm_unstable/len(hmm_df)*100:.1f}%)")

    # Average performance by regime count
    print(f"\nAverage NRMSE by Regime Count:")
    print("\nK-means:")
    print(kmeans_df.groupby('R')['nrmse'].mean())
    print("\nHMM:")
    print(hmm_df.groupby('R')['nrmse'].mean())

    return comp_df


if __name__ == "__main__":
    comp_df = generate_comparison_table()
