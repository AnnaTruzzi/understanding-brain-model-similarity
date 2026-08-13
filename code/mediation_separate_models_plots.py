import numpy as np
import pylab as py
import scipy.io
from PIL import Image
import os
import pandas as pd
import glob
import re
import pickle
from scipy.spatial import distance
import matplotlib
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
from itertools import combinations
import pandas as pd
import seaborn as sns
import collections
from scipy.stats import sem
from matplotlib.patches import Patch
from config import FIGURES_ROOT, RESULTS_ROOT


DC_RANDOM_COLOR = '#0072B2'
DC_TRAINED_COLOR = '#E69F00'
ALEXNET_TRAINED_COLOR = '#009E73'

def calculate_ci_and_significance(data, alpha=0.05):
    """
    Calculate confidence intervals for bootstrap distribution.
    Returns lower CI, upper CI, and whether it's significant (CI doesn't include 0).
    """
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_ci = np.percentile(data, lower_percentile)
    upper_ci = np.percentile(data, upper_percentile)
    
    # Significant if CI doesn't include 0
    significant = not (lower_ci <= 0 <= upper_ci)
    
    return lower_ci, upper_ci, significant


def get_robust_ylim(values):
    """
    Compute robust y-axis limits that reduce outlier-driven compression.
    """
    values = np.asarray(values)
    if values.size == 0:
        return -0.1, 0.1

    lower = np.percentile(values, 1)
    upper = np.percentile(values, 99)

    # Include zero for interpretability and add proportional padding
    lower = min(lower, 0)
    upper = max(upper, 0)

    span = upper - lower
    if span <= 0:
        span = 0.1
    pad = max(0.03, 0.15 * span)

    return lower - pad, upper + pad


def plot_feature_panel(ax, data_subset, net_layer_order, custom_palette, alpha_corrected,
                       x_tick_labels=None, show_ylabel=True, title_text=None,
                       title_fontsize=16, record_ci_rows=None):
    """
    Draw one feature panel on a provided axis.
    Returns True if data were plotted, False otherwise.
    """
    if len(data_subset) == 0:
        return False

    grouped_values = [
        data_subset[data_subset['net_layer'] == nl]['proportion'].values
        for nl in net_layer_order
    ]

    means = [np.mean(vals) if vals.size > 0 else np.nan for vals in grouped_values]
    errors = [sem(vals) if vals.size > 1 else 0 for vals in grouped_values]
    bar_colors = [custom_palette[nl.split('_')[0]] for nl in net_layer_order]

    ax.bar(
        range(len(net_layer_order)),
        means,
        yerr=errors,
        color=bar_colors,
        alpha=0.75,
        width=0.65,
        edgecolor='black',
        linewidth=0.8,
        capsize=4,
        zorder=2
    )

    rng = np.random.default_rng(42)
    for i, vals in enumerate(grouped_values):
        if vals.size == 0:
            continue
        x_jitter = rng.normal(loc=i, scale=0.06, size=vals.size)
        ax.scatter(
            x_jitter,
            vals,
            s=12,
            color='black',
            alpha=0.25,
            linewidths=0,
            zorder=3
        )

    y_min, y_max = get_robust_ylim(data_subset['proportion'].values)
    star_offset = 0.03 * (y_max - y_min)
    star_positions = []

    for i, nl in enumerate(net_layer_order):
        subset_data = data_subset[data_subset['net_layer'] == nl]['proportion'].values
        if subset_data.size == 0:
            continue

        lower_ci, upper_ci, significant = calculate_ci_and_significance(
            subset_data, alpha=alpha_corrected
        )

        if record_ci_rows is not None:
            net_name, layer_name = nl.split('_', 1)
            record_ci_rows.append({
                'ROI': data_subset['ROI'].iloc[0],
                'feature': data_subset['feature'].iloc[0],
                'net': net_name,
                'layer': layer_name,
                'alpha': alpha_corrected,
                'ci_lower': lower_ci,
                'ci_upper': upper_ci,
                'significant': significant
            })

        if significant:
            max_val = np.percentile(subset_data, 99)
            star_positions.append((i, max_val + star_offset))

    if star_positions:
        current_span = y_max - y_min
        if current_span <= 0:
            current_span = 0.1
        y_max = max(y_max, max(pos[1] for pos in star_positions) + 0.08 * current_span)

    ax.set_ylim([y_min, y_max])

    final_span = y_max - y_min
    if final_span <= 0:
        final_span = 0.1
    star_ceiling = y_max - 0.04 * final_span
    for x_pos, y_pos in star_positions:
        y_plot = min(y_pos, star_ceiling)
        ax.text(x_pos, y_plot, '*', ha='center', va='bottom',
                fontsize=20, fontweight='bold', clip_on=True)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    if show_ylabel:
        ax.set_ylabel('Proportion of Total Path\nExplained by Indirect Path', fontsize=13)
    else:
        ax.set_ylabel('')

    if title_text is not None:
        ax.set_title(title_text, fontsize=title_fontsize, fontweight='bold')

    ax.set_xticks(range(len(net_layer_order)))
    if x_tick_labels is None:
        ax.set_xticklabels(net_layer_order, rotation=30, ha='right', fontsize=11)
    else:
        ax.set_xticklabels(x_tick_labels, rotation=0, ha='center', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    return True


def main():
    df = pd.read_csv(RESULTS_ROOT / 'mediation_separate_models_bootstrap.csv')
    ci_summary_rows = []

    required_cols = {'ROI', 'feature', 'net', 'layer', 'proportion'}
    if df.empty:
        print('No rows found in mediation_separate_models_bootstrap.csv. Nothing to plot.')
        return
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols.difference(set(df.columns))
        print(f'Missing required columns in bootstrap file: {sorted(missing)}')
        return

    # Keep only finite numeric proportions for plotting
    df = df[np.isfinite(df['proportion'])].copy()
    if df.empty:
        print('No finite proportion values found after filtering. Nothing to plot.')
        return

    # Define custom palette
    costum_palette = {'dcrandom': DC_RANDOM_COLOR,
                      'dctrained': DC_TRAINED_COLOR,
                      'alexnettrained': ALEXNET_TRAINED_COLOR}
    
    sns.set()
    sns.set_style("white")
    
    # Features to plot
    all_features = ['semantic_category', 'semantic_animacy',
                   'size', 'contrast', 'hue', 'lurid', 
                   'thinness', 'radiansoffhorizontal', 'silhouette']
    
    # Bonferroni correction: we have 54 models per brain area (3 networks x 2 layers x 9 features), so we divide alpha by 54
    n_comparisons = 54
    alpha_corrected = 0.05 / n_comparisons
    
    print(f"Using Bonferroni-corrected alpha = {alpha_corrected:.6f}")
    
    # Create a plot for each feature
    for feature in all_features:
        print(f"Creating plot for {feature}...")

        for ROI in ['IT', 'EVC']:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
            
            # Filter data for this ROI and feature
            data_subset = df[(df['ROI'] == ROI) & (df['feature'] == feature)].copy()
            
            if len(data_subset) == 0:
                print(f"  Warning: No data for {feature} in {ROI}")
                continue
            
            # Create a combined label for net and layer
            data_subset['net_layer'] = data_subset['net'] + '_' + data_subset['layer']
            
            # Define order for x-axis
            net_layer_order = []
            for net in networks:
                for layer in layers:
                    net_layer_order.append(f'{net}_{layer}')
            
            # Filter to only include combinations that exist in data
            net_layer_order = [nl for nl in net_layer_order if nl in data_subset['net_layer'].values]

            if len(net_layer_order) == 0:
                print(f"  Warning: No network/layer combinations for {feature} in {ROI}")
                continue
            
            plot_feature_panel(
                ax=ax,
                data_subset=data_subset,
                net_layer_order=net_layer_order,
                custom_palette=costum_palette,
                alpha_corrected=alpha_corrected,
                x_tick_labels=None,
                show_ylabel=True,
                title_text=f'{ROI}',
                title_fontsize=16,
                record_ci_rows=ci_summary_rows
            )
        
            # Overall title
            feature_display = feature.replace('_', ' ').title()
            plt.suptitle(f'{feature_display} - {ROI} - Proportion of Total Path Explained',
                fontsize=18, fontweight='bold', y=0.98)

                # Create custom legend for networks
            legend_elements = [Patch(facecolor=costum_palette[net], alpha=0.7, label=net)
                    for net in networks]
            fig.legend(handles=legend_elements, loc='upper center',
                bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False, fontsize=11)

            plt.xlabel('Network and Layer', fontsize=13)
            plt.tight_layout()
            plt.subplots_adjust(top=0.86)

            # Save figure
            filename = f'mediation_separate_{feature}_{ROI}_barplot'
            plt.savefig(FIGURES_ROOT / f'{filename}.png',
                bbox_inches='tight', dpi=300)
            plt.savefig(FIGURES_ROOT / f'{filename}.pdf',
                bbox_inches='tight')
            plt.close()

            print(f"  Saved: {filename}")

    # Create additional 4x2 summary figures (one per ROI)
    summary_features = [
        'hue',
        'lurid',
        'radiansoffhorizontal',
        'silhouette',
        'size',
        'thinness',
        'semantic_animacy',
        'semantic_category'
    ]
    summary_feature_titles = {
        'hue': 'Hue',
        'lurid': 'Lurid',
        'radiansoffhorizontal': 'Radiansoffhorizontal',
        'silhouette': 'Silhouette',
        'size': 'Size',
        'thinness': 'Thinness',
        'semantic_animacy': 'Animacy',
        'semantic_category': 'Category'
    }
    summary_xlabels = ['ReLu2', 'ReLu7', 'ReLu2', 'ReLu7', 'ReLu2', 'ReLu7']
    full_net_layer_order = [f'{net}_{layer}' for net in networks for layer in layers]

    for ROI in ['IT', 'EVC']:
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 24), constrained_layout=False)
        axes = axes.flatten()

        for idx, feature in enumerate(summary_features):
            ax = axes[idx]
            data_subset = df[(df['ROI'] == ROI) & (df['feature'] == feature)].copy()

            if len(data_subset) == 0:
                ax.set_title(summary_feature_titles[feature], fontsize=11, fontweight='bold')
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10)
                ax.set_xticks(range(len(full_net_layer_order)))
                ax.set_xticklabels(summary_xlabels, rotation=0, ha='center', fontsize=10)
                if idx % 2 != 0:
                    ax.set_ylabel('')
                continue

            data_subset['net_layer'] = data_subset['net'] + '_' + data_subset['layer']
            plot_feature_panel(
                ax=ax,
                data_subset=data_subset,
                net_layer_order=full_net_layer_order,
                custom_palette=costum_palette,
                alpha_corrected=alpha_corrected,
                x_tick_labels=summary_xlabels,
                show_ylabel=(idx % 2 == 0),
                title_text=summary_feature_titles[feature],
                title_fontsize=11,
                record_ci_rows=None
            )

        legend_elements = [Patch(facecolor=costum_palette[net], alpha=0.7, label=net)
                           for net in networks]
        fig.legend(handles=legend_elements, loc='upper center',
                   bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False, fontsize=12)

        fig.suptitle(f'{ROI}', fontsize=16, fontweight='bold', y=0.999)
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.975])

        summary_filename = f'mediation_separate_summary_{ROI}_4x2'
        fig.savefig(FIGURES_ROOT / f'{summary_filename}.png',
                    bbox_inches='tight', dpi=300)
        fig.savefig(FIGURES_ROOT / f'{summary_filename}.pdf',
                    bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {summary_filename}')

    # Save CI bounds and significance summary
    if ci_summary_rows:
        ci_df = pd.DataFrame(ci_summary_rows)
        ci_df = ci_df.sort_values(['ROI', 'feature', 'net', 'layer']).reset_index(drop=True)
        ci_output_path = RESULTS_ROOT / 'mediation_separate_models_ci_significance.txt'
        ci_level_pct = (1 - alpha_corrected) * 100
        lower_pct = (alpha_corrected / 2) * 100
        upper_pct = (1 - alpha_corrected / 2) * 100

        with open(ci_output_path, 'w') as f:
            f.write('Bootstrap CI and significance summary for mediation separate models\n')
            f.write(f'Bonferroni-corrected alpha: {alpha_corrected:.8f}\n')
            f.write(f'Approximate CI level: {ci_level_pct:.4f}%\n')
            f.write(f'Percentile bounds: [{lower_pct:.4f}, {upper_pct:.4f}]\n\n')
            f.write(ci_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
            f.write('\n')

        print(f'Saved CI/significance summary: {ci_output_path}')
    else:
        print('No CI/significance rows were generated; summary file not written.')


if __name__ == '__main__':
    networks = ['dcrandom', 'dctrained', 'alexnettrained']
    layers = ['ReLu2', 'ReLu7']
    main()
