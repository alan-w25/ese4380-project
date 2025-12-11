"""
Generate publication-ready LaTeX tables from ablation study results.
"""

import pandas as pd
import numpy as np

def generate_regime_ablation_table(results_path='ablation_results.csv'):
    """
    Generate LaTeX table comparing regime counts across datasets.
    """
    df = pd.read_csv(results_path)

    # Create table for h=1 results (primary comparison)
    df_h1 = df[df['horizon'] == 1].copy()

    latex_lines = []
    latex_lines.append(r"\begin{table}[h!]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Regime-Switching Ablation Study: Multi-Horizon Forecast Performance}")
    latex_lines.append(r"\label{tab:regime-ablation}")
    latex_lines.append(r"\begin{tabular}{llcccccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Dataset & $R$ & Horizon & MSE & RMSE & NRMSE & $\rho_{\max}$ & Stable \\")
    latex_lines.append(r"\midrule")

    datasets = df['Dataset'].unique()

    for dataset in datasets:
        df_dataset = df[df['Dataset'] == dataset]

        for i, (_, row) in enumerate(df_dataset.iterrows()):
            n_reg = int(row['n_regimes'])
            h = int(row['horizon'])

            # Only include h=1,5,10 and R=1,2,3,4
            if h not in [1, 5, 10] or n_reg not in [1, 2, 3, 4]:
                continue

            # Dataset name only on first row
            ds_str = dataset if (i == 0) else ""

            # Format metrics
            mse = row['MSE']
            rmse = row['RMSE']
            nrmse = row['NRMSE']
            rho = row['max_spectral_radius']
            n_stable = int(row['n_stable_regimes'])
            n_total = n_reg

            # Scientific notation for very small values
            if mse < 0.001:
                mse_str = f"{mse:.2e}"
            else:
                mse_str = f"{mse:.4f}"

            if rmse < 0.01:
                rmse_str = f"{rmse:.3e}"
            else:
                rmse_str = f"{rmse:.4f}"

            # Highlight best NRMSE for each dataset+horizon combination
            best_nrmse = df_dataset[df_dataset['horizon'] == h]['NRMSE'].min()
            if abs(nrmse - best_nrmse) < 1e-6:
                nrmse_str = rf"\textbf{{{nrmse:.4f}}}"
            else:
                nrmse_str = f"{nrmse:.4f}"

            latex_lines.append(
                f"{ds_str} & {n_reg} & {h} & {mse_str} & {rmse_str} & {nrmse_str} & "
                f"{rho:.3f} & {n_stable}/{n_total} \\\\"
            )

        latex_lines.append(r"\midrule")

    latex_lines[-1] = r"\bottomrule"  # Replace last midrule with bottomrule
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    return "\n".join(latex_lines)


def generate_compact_comparison_table(results_path='ablation_results.csv'):
    """
    Generate compact table showing best performing regime count per dataset/horizon.
    """
    df = pd.read_csv(results_path)

    latex_lines = []
    latex_lines.append(r"\begin{table}[h!]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Best Performing Regime Count by Dataset and Horizon}")
    latex_lines.append(r"\label{tab:regime-best}")
    latex_lines.append(r"\begin{tabular}{lccccccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Dataset & \multicolumn{2}{c}{$h=1$} & \multicolumn{2}{c}{$h=5$} & \multicolumn{2}{c}{$h=10$} \\")
    latex_lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    latex_lines.append(r" & Best $R$ & NRMSE & Best $R$ & NRMSE & Best $R$ & NRMSE \\")
    latex_lines.append(r"\midrule")

    for dataset in df['Dataset'].unique():
        df_ds = df[df['Dataset'] == dataset]

        row_parts = [dataset]

        for h in [1, 5, 10]:
            df_h = df_ds[df_ds['horizon'] == h]
            if len(df_h) > 0:
                best_idx = df_h['NRMSE'].idxmin()
                best_row = df_h.loc[best_idx]
                best_r = int(best_row['n_regimes'])
                best_nrmse = best_row['NRMSE']
                row_parts.append(f"{best_r}")
                row_parts.append(f"{best_nrmse:.4f}")
            else:
                row_parts.extend(["--", "--"])

        latex_lines.append(" & ".join(row_parts) + r" \\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    return "\n".join(latex_lines)


if __name__ == "__main__":
    print("="*60)
    print("GENERATING LATEX TABLES")
    print("="*60)

    # Full ablation table
    print("\n--- Full Ablation Table ---\n")
    full_table = generate_regime_ablation_table()
    print(full_table)

    with open('tables/regime_ablation_full.tex', 'w') as f:
        f.write(full_table)
    print("\nSaved to: tables/regime_ablation_full.tex")

    # Compact comparison table
    print("\n\n--- Compact Comparison Table ---\n")
    compact_table = generate_compact_comparison_table()
    print(compact_table)

    with open('tables/regime_ablation_compact.tex', 'w') as f:
        f.write(compact_table)
    print("\nSaved to: tables/regime_ablation_compact.tex")

    print("\n" + "="*60)
    print("LaTeX tables generated successfully!")
    print("="*60)
