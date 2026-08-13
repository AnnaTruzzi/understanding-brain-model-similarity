import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURES_ROOT, RESULTS_ROOT, load_bootstrap_values


RESULTS_PATH = RESULTS_ROOT
FIGURES_PATH = FIGURES_ROOT

DC_RANDOM_COLOR = '#0072B2'
DC_TRAINED_COLOR = '#E69F00'
ALEXNET_TRAINED_COLOR = '#009E73'

MODEL_COLOR_MAP = {
    'dcrandomstate': DC_RANDOM_COLOR,
    'dc100epochs': DC_TRAINED_COLOR,
    'alexnetpretrained': ALEXNET_TRAINED_COLOR,
}

NET_LABEL_MAP = {
    'dcrandomstate': 'DeepCluster random',
    'dc100epochs': 'DeepCluster trained',
    'alexnetpretrained': 'AlexNet trained',
}

LAYER_ORDER_7 = ['ReLu1', 'ReLu2', 'ReLu3', 'ReLu4', 'ReLu5', 'ReLu6', 'ReLu7']
ROI_ORDER = ['EVC', 'IT']


def set_theme():
    sns.set_theme(style='whitegrid', context='paper')
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False


def save_figure(fig, stem):
    fig.savefig(f'{FIGURES_PATH}/{stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{FIGURES_PATH}/{stem}.pdf', bbox_inches='tight')
    plt.close(fig)


def prep_corr_df(csv_path):
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'layer' in df.columns:
        df['layer'] = pd.Categorical(df['layer'], categories=LAYER_ORDER_7, ordered=True)
    return df


def add_participant_variability_ceiling(ax, roi):
    ceiling_bounds = {'EVC': (0.31, 0.38), 'IT': (0.28, 0.42)}
    lower, upper = ceiling_bounds[roi]
    ax.axhline(upper, color='gray', lw=2, alpha=0.4)
    ax.axhline(lower, color='gray', lw=2, alpha=0.4)
    ax.axhspan(lower, upper, facecolor='gray', alpha=0.4)


def add_instance_variability_ceiling(ax):
    ceiling_values = [
        ((0.971, 0.976), (0.9977, 0.9981)),
        ((0.972, 0.977), (0.993, 0.995)),
        ((0.971, 0.977), (0.986, 0.989)),
        ((0.964, 0.971), (0.943, 0.954)),
        ((0.949, 0.958), (0.917, 0.933)),
        ((0.987, 0.989), (0.952, 0.961)),
        ((0.982, 0.985), (0.947, 0.957)),
    ]
    for idx, (random_bounds, trained_bounds) in enumerate(ceiling_values):
        xmin = idx / len(LAYER_ORDER_7)
        xmax = (idx + 1) / len(LAYER_ORDER_7)
        for lower, upper, color in [
            (*random_bounds, DC_RANDOM_COLOR),
            (*trained_bounds, DC_TRAINED_COLOR),
        ]:
            ax.axhline(upper, xmin=xmin, xmax=xmax, color=color, lw=2, alpha=0.4)
            ax.axhline(lower, xmin=xmin, xmax=xmax, color=color, lw=2, alpha=0.4)
            ax.axhspan(lower, upper, xmin=xmin, xmax=xmax, facecolor=color, alpha=0.4)


def add_significance_stars(ax, stars, y_values):
    for idx, (star, y_value) in enumerate(zip(stars, y_values)):
        if star:
            ax.text(idx, y_value, star, ha='center', va='bottom', fontsize=10)


def mediation_ylim(values):
    values = np.asarray(values)
    lower = min(np.percentile(values, 1), 0)
    upper = max(np.percentile(values, 99), 0)
    span = max(upper - lower, 0.1)
    padding = max(0.03, 0.15 * span)
    return lower - padding, upper + padding


def mediation_significant(values, alpha):
    lower = np.percentile(values, (alpha / 2) * 100)
    upper = np.percentile(values, (1 - alpha / 2) * 100)
    return not (lower <= 0 <= upper)


def plot_dc_brain_lines(corr_df, title, out_stem):
    palette = {'randomstate': DC_RANDOM_COLOR, '100epochs': DC_TRAINED_COLOR}
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.5), sharey=True)

    for idx, roi in enumerate(ROI_ORDER):
        ax = axes[idx]
        subset = corr_df[corr_df['ROI'] == roi].copy()
        is_participant_variability = 'participant' in title.lower()
        ax.set_ylim((-0.15, 0.45) if is_participant_variability else (-0.05, 1.05))
        if is_participant_variability:
            add_participant_variability_ceiling(ax, roi)
        else:
            add_instance_variability_ceiling(ax)

        sns.lineplot(
            data=subset,
            x='layer',
            y='corr',
            hue='state',
            hue_order=['randomstate', '100epochs'],
            estimator=np.mean,
            errorbar=('ci', 95),
            markers=True,
            dashes=False,
            linewidth=2.2,
            marker='o',
            palette=palette,
            ax=ax,
        )

        ax.set_title(roi)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Correlation with brain')
        ax.tick_params(axis='x', rotation=20)
        if is_participant_variability:
            add_significance_stars(
                ax,
                ['***', '***', '**', '***', '***', '**', '*'] if idx == 0
                else ['***', '***', '**', '*', '**', '', ''],
                [0.15, 0.20, 0.18, 0.18, 0.18, 0.18, 0.18],
            )
        else:
            add_significance_stars(ax, ['', '', '', '**', '', '', ''], [0] * 7)

        handles, labels = ax.get_legend_handles_labels()
        if idx == 1:
            ax.legend(handles=handles,
                      labels=['DeepCluster random', 'DeepCluster trained'],
                      frameon=False,
                      title='Model')
        else:
            ax.get_legend().remove()

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, out_stem)


def build_alexnet_comparison_df():
    corr_dc = prep_corr_df(f'{RESULTS_PATH}/corr_dc_mri_variability.csv')
    corr_alex = prep_corr_df(f'{RESULTS_PATH}/corr_alexnet_mri_variability.csv')
    corr = pd.concat([corr_dc, corr_alex], ignore_index=True)
    corr['net_type'] = corr['net'] + corr['state']
    corr = corr[corr['net_type'] != 'alexnetrandomstate'].copy()
    return corr


def draw_one_corr_panel(ax, df, roi, net_order, subtitle):
    subset = df[(df['ROI'] == roi) & (df['net_type'].isin(net_order))].copy()
    sns.lineplot(
        data=subset,
        x='layer',
        y='corr',
        hue='net_type',
        hue_order=net_order,
        estimator=np.mean,
        errorbar=('ci', 95),
        markers=True,
        dashes=False,
        linewidth=2.2,
        marker='o',
        palette=MODEL_COLOR_MAP,
        ax=ax,
    )
    ax.set_title(f'{subtitle} - {roi}', fontsize=11)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Correlation with brain')
    ax.tick_params(axis='x', rotation=20)


def plot_alexnet_vs_dc_figure(corr_df):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharey='row')

    top_order = ['dcrandomstate', 'alexnetpretrained']
    bottom_order = ['dcrandomstate', 'dc100epochs', 'alexnetpretrained']

    draw_one_corr_panel(axes[0, 0], corr_df, 'EVC', top_order, 'DC random vs AlexNet trained')
    draw_one_corr_panel(axes[0, 1], corr_df, 'IT', top_order, 'DC random vs AlexNet trained')
    draw_one_corr_panel(axes[1, 0], corr_df, 'EVC', bottom_order, 'All three models')
    draw_one_corr_panel(axes[1, 1], corr_df, 'IT', bottom_order, 'All three models')

    for r in range(2):
        for c in range(2):
            if axes[r, c].get_legend() is not None:
                axes[r, c].get_legend().remove()

    handles = [plt.Line2D([0], [0], color=MODEL_COLOR_MAP[k], marker='o', lw=2) for k in bottom_order]
    labels = [NET_LABEL_MAP[k] for k in bottom_order]
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, title='Model')

    for ax, roi in zip(axes.flat, ['EVC', 'IT', 'EVC', 'IT']):
        ax.set_ylim((-0.15, 0.45))
        add_participant_variability_ceiling(ax, roi)

    add_significance_stars(axes[0, 0], ['***', '***', '***', '***', '***', '**', '**'],
                           [0.15, 0.20, 0.18, 0.18, 0.18, 0.18, 0.18])
    add_significance_stars(axes[0, 1], ['***', '***', '**', '*', '**', '', ''],
                           [0.15, 0.20, 0.18, 0.18, 0.18, 0.18, 0.18])

    fig.suptitle('Participant Variability: Brain Correlation Model Comparisons', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    save_figure(fig, 'paper_fig3_dc_vs_alexnet_lines')


def plot_linear_models_figure():
    corr_values = load_bootstrap_values()

    all_values = np.array(list(corr_values.values()))
    grand_average = np.mean(all_values)
    across_model_average = np.mean(all_values, axis=0)
    corrected_values = {
        key: np.mean(values - across_model_average + grand_average, axis=1)
        for key, values in corr_values.items()
    }

    model_order = [
        'all-alexnettrained',
        'all-dctrained',
        'all-dcrandom',
        'all-alexnettrained-with-dcrandom2',
        'all-dctrained-with-dcrandom2',
    ]
    model_labels = [
        'AlexNet trained',
        'DeepCluster trained',
        'DeepCluster random',
        'AlexNet + DC random L2',
        'DC trained + DC random L2',
    ]
    model_colors = [
        ALEXNET_TRAINED_COLOR,
        DC_TRAINED_COLOR,
        DC_RANDOM_COLOR,
        '#4D9B9B',
        '#B07A4A',
    ]
    plot_df = pd.DataFrame({
        'model': np.concatenate([corrected_values[key] for key in model_order]),
        'model_label': np.repeat(model_labels, [len(corrected_values[key]) for key in model_order]),
    })
    plot_df['model_label'] = pd.Categorical(plot_df['model_label'], categories=model_labels, ordered=True)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.violinplot(
        data=plot_df,
        x='model_label',
        y='model',
        hue='model_label',
        order=model_labels,
        hue_order=model_labels,
        palette=model_colors,
        legend=False,
        inner='box',
        cut=0,
        linewidth=1.1,
        saturation=0.9,
        ax=ax,
    )
    swarm_df = pd.concat(
        [
            group.sample(n=min(450, len(group)), random_state=123)
            for _, group in plot_df.groupby('model_label', observed=True)
        ],
        ignore_index=True,
    )
    swarm_df['model_label'] = pd.Categorical(swarm_df['model_label'], categories=model_labels, ordered=True)
    sns.swarmplot(
        data=swarm_df,
        x='model_label',
        y='model',
        order=model_labels,
        color='black',
        size=1.5,
        alpha=0.22,
        linewidth=0,
        ax=ax,
    )

    label_y = plot_df.groupby('model_label', observed=True)['model'].max().reindex(model_labels).to_numpy()
    y_top = np.nanmax(label_y)

    def add_significance_bracket(left, right, y_value, stars):
        bracket_height = 0.006
        ax.plot(
            [left, left, right, right],
            [y_value, y_value + bracket_height, y_value + bracket_height, y_value],
            color='black', lw=1.1, clip_on=False,
        )
        ax.text(
            (left + right) / 2,
            y_value + bracket_height + 0.003,
            stars,
            ha='center', va='bottom', fontsize=11,
        )

    bracket_base = y_top + 0.018
    add_significance_bracket(0, 3, bracket_base, '***')
    add_significance_bracket(1, 4, bracket_base + 0.045, '***')

    ax.set_xlabel('Regression model')
    ax.set_ylabel('Cross-validated Kendall correlation')
    ax.set_ylim(0, max(0.4, bracket_base + 0.09))
    plt.setp(ax.get_xticklabels(), rotation=18, ha='right')
    ax.set_title('Bootstrap performance of linear regression models', pad=14)
    ax.grid(axis='y', alpha=0.25)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    save_figure(fig, 'paper_fig4_linear_regression_bootstrap')


def _plot_mediation_roi_summary(df, roi, out_stem):
    summary_features = [
        'hue',
        'lurid',
        'radiansoffhorizontal',
        'silhouette',
        'size',
        'thinness',
        'semantic_animacy',
        'semantic_category',
    ]
    summary_feature_titles = {
        'hue': 'Hue',
        'lurid': 'Lurid',
        'radiansoffhorizontal': 'Radians off horizontal',
        'silhouette': 'Silhouette',
        'size': 'Size',
        'thinness': 'Thinness',
        'semantic_animacy': 'Animacy',
        'semantic_category': 'Category',
    }

    net_layer_order = [
        'dcrandom_ReLu2', 'dcrandom_ReLu7',
        'dctrained_ReLu2', 'dctrained_ReLu7',
        'alexnettrained_ReLu2', 'alexnettrained_ReLu7'
    ]
    color_map = {
        'dcrandom': DC_RANDOM_COLOR,
        'dctrained': DC_TRAINED_COLOR,
        'alexnettrained': ALEXNET_TRAINED_COLOR,
    }
    alpha_corrected = 0.05 / 54

    x_labels = ['R-L2', 'R-L7', 'T-L2', 'T-L7', 'A-L2', 'A-L7']

    fig, axes = plt.subplots(4, 2, figsize=(15, 16), constrained_layout=False)
    axes = axes.flatten()

    for idx, feature in enumerate(summary_features):
        ax = axes[idx]
        subset = df[(df['ROI'] == roi) & (df['feature'] == feature)].copy()
        if subset.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(summary_feature_titles[feature], fontsize=11)
            ax.set_xticks(range(len(net_layer_order)))
            ax.set_xticklabels(x_labels)
            continue

        subset['net_layer'] = subset['net'] + '_' + subset['layer']

        means = []
        sems = []
        bar_colors = []
        grouped_values = []
        for nl in net_layer_order:
            vals = subset[subset['net_layer'] == nl]['proportion'].values
            grouped_values.append(vals)
            means.append(np.mean(vals) if vals.size > 0 else np.nan)
            sems.append((np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)
            bar_colors.append(color_map[nl.split('_')[0]])

        x = np.arange(len(net_layer_order))
        ax.bar(x, means, yerr=sems, color=bar_colors, alpha=0.8,
               edgecolor='black', linewidth=0.7, capsize=3)

        rng = np.random.default_rng(42)
        for i, vals in enumerate(grouped_values):
            if vals.size == 0:
                continue
            jitter = rng.normal(i, 0.06, size=vals.size)
            ax.scatter(jitter, vals, s=10, color='black', alpha=0.22, linewidths=0)

        y_min, y_max = mediation_ylim(subset['proportion'].values)
        star_offset = 0.03 * (y_max - y_min)
        star_positions = []
        for i, vals in enumerate(grouped_values):
            if vals.size and mediation_significant(vals, alpha_corrected):
                star_positions.append((i, np.percentile(vals, 99) + star_offset))

        if star_positions:
            y_max = max(y_max, max(y for _, y in star_positions) + 0.08 * (y_max - y_min))
        ax.set_ylim(y_min, y_max)

        final_span = y_max - y_min
        star_ceiling = y_max - 0.04 * final_span
        for x_pos, y_pos in star_positions:
            ax.text(x_pos, min(y_pos, star_ceiling), '*', ha='center', va='bottom',
                    fontsize=16, fontweight='bold')

        ax.axhline(0, color='black', lw=0.9, ls='--', alpha=0.6)
        ax.set_title(summary_feature_titles[feature], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel('Model-layer group')
        if idx % 2 == 0:
            ax.set_ylabel('Proportion of total effect explained')
        else:
            ax.set_ylabel('')

    legend_handles = [
        plt.Line2D([0], [0], color=DC_RANDOM_COLOR, lw=8),
        plt.Line2D([0], [0], color=DC_TRAINED_COLOR, lw=8),
        plt.Line2D([0], [0], color=ALEXNET_TRAINED_COLOR, lw=8),
    ]
    legend_labels = ['DeepCluster random', 'DeepCluster trained', 'AlexNet trained']
    fig.legend(
        legend_handles,
        legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
    )

    fig.suptitle(f'Mediation Model Results - {roi}', fontsize=14, y=1.01)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_figure(fig, out_stem)


def plot_mediation_figures():
    df = pd.read_csv(f'{RESULTS_PATH}/mediation_separate_models_bootstrap.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = df[np.isfinite(df['proportion'])].copy()

    _plot_mediation_roi_summary(df, 'EVC', 'paper_fig5_mediation_evc')
    _plot_mediation_roi_summary(df, 'IT', 'paper_fig6_mediation_it')


def main():
    set_theme()

    dc_mri = prep_corr_df(f'{RESULTS_PATH}/corr_dc_mri_variability.csv')
    dc_layer = prep_corr_df(f'{RESULTS_PATH}/corr_dc_layer_variability.csv')

    plot_dc_brain_lines(dc_mri, 'Participant Variability', 'paper_fig1_participant_variability_lines')
    plot_dc_brain_lines(dc_layer, 'Instance Variability', 'paper_fig2_instance_variability_lines')

    corr_comp = build_alexnet_comparison_df()
    plot_alexnet_vs_dc_figure(corr_comp)

    plot_linear_models_figure()
    plot_mediation_figures()

    print('Saved all paper figures (1 to 6) to the figures directory.')


if __name__ == '__main__':
    main()