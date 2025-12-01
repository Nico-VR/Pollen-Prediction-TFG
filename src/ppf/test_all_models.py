import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os

matplotlib.use('Agg')


# === UTILS ===

def model_to_initials(model_name: str, uses_covariates: bool = False):
    """Genera iniciales del modelo, añadiendo '-CV' si usa covariables."""
    if 'train' not in model_name:
        parts = model_name.split('.')[1:]
    else:
        parts = model_name.split('.')

    initials = ""
    if parts:
        if any("train" in p for p in parts):
            initials += parts[0][:2].upper()
            initials += ''.join(part[0].upper() for part in parts[1:] if part)
        else:
            initials = ''.join(part[0].upper() for part in parts if part)
    
    if uses_covariates:
        initials += "-CV"

    return initials


def get_boxplot_data(df, model_names, model_type, col):
    data, labels, positions, abbrev_map = [], [], [], {}
    pos = 1
    for model in model_names:
        for cv_val in [False, True]:
            rows = df[(df['model_name'] == model) & (df['uses_covariates'] == cv_val)]
            if rows.empty:
                continue
            abbrev = model_to_initials(model, uses_covariates=cv_val)
            abbrev_map[abbrev] = (model, cv_val)
            data.append(rows[col].values)
            labels.append(abbrev)
            positions.append(pos)
            pos += 1
        pos += 1  # separación entre modelos
    return data, labels, positions, abbrev_map


def annotate_groups(ax, positions, model_names, start_label='Start Deviation', end_label='End Deviation'):
    n = len(positions) // 2
    group1_center = (positions[0] + positions[n - 1]) / 2
    group2_center = (positions[n] + positions[-1]) / 2
    y_min, y_max = ax.get_ylim()
    label_y = y_min - (y_max - y_min) * 0.15
    ax.text(group1_center, label_y, start_label, ha='center', va='top', fontweight='bold')
    ax.text(group2_center, label_y, end_label, ha='center', va='top', fontweight='bold')


# === PLOTTING FUNCTIONS ===

def generate_individual_plot(pred_df, model_names, model_type, pred_n, pastel_colors, output_path, uses_covariates, y_max=None, plot_title=None):
    abbrev_colors = {}
    color_cycle = iter(pastel_colors)

    data_start, labels_start, positions_start, abbrev_map_start = get_boxplot_data(pred_df, model_names, model_type, 'start_dev')
    data_end, labels_end, positions_end, abbrev_map_end = get_boxplot_data(pred_df, model_names, model_type, 'end_dev')
    abbrev_map = {**abbrev_map_start, **abbrev_map_end}

    if positions_start:
        offset = max(positions_start) + 2
        positions_end = [p + offset for p in positions_end]

    data = data_start + data_end
    labels = labels_start + labels_end
    positions = positions_start + positions_end

    for abbrev in sorted({l.replace("-CV","") for l in labels}):
        abbrev_colors[abbrev] = next(color_cycle)

    fig_width = max(12, len(positions) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    box = ax.boxplot(data, positions=positions, widths=0.6, showfliers=True, patch_artist=True)
    for i, lbl in enumerate(labels):
        base_abbrev = lbl.replace("-CV", "")
        color = abbrev_colors[base_abbrev]
        box['boxes'][i].set_facecolor(color)
        if lbl.endswith("-CV"):
            box['boxes'][i].set_edgecolor("black")
            box['boxes'][i].set_linewidth(1.5)

    ax.set_xticks(positions)
    rotation = 90 if len(labels) > 10 else 45
    ha = 'right' if len(labels) > 10 else 'center'
    ax.set_xticklabels(labels, rotation=rotation, ha=ha)

    ax.set_ylabel("Days")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    if y_max is not None:
        ax.set_ylim(0, y_max)

    if plot_title:
        ax.set_title(plot_title, fontweight='bold')
    else:
        ax.set_title(f"{model_type.capitalize()} - pred_n = {pred_n}", fontweight='bold')

    if positions_start and positions_end:
        annotate_groups(ax, positions, model_names)

    patches = []
    for abbrev, (model_full, cv_val) in sorted(abbrev_map.items()):
        label = f"{abbrev}: {model_full}" + (" (with covariates)" if cv_val else "")
        color = abbrev_colors[abbrev.replace("-CV", "")]
        patches.append(mpatches.Patch(facecolor=color, edgecolor="black" if cv_val else color, linewidth=1.5 if cv_val else 1, label=label))

    ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, title="Models", frameon=True, edgecolor='black', borderaxespad=0)

    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_combined_plot(model_df, model_names, model_type, pred_n_values, pastel_colors, output_path, uses_covariates, y_max=None):
    abbrev_colors = {model_to_initials(m, uses_covariates=uses_covariates): pastel_colors[i % len(pastel_colors)] for i, m in enumerate(model_names)}
    subplot_height = 4
    fig, axes = plt.subplots(len(pred_n_values), 1, figsize=(12, subplot_height * len(pred_n_values)), sharex=True)
    if len(pred_n_values) == 1:
        axes = [axes]

    legend_map = {}

    for idx, pred_n in enumerate(pred_n_values):
        ax = axes[idx]
        pred_df = model_df[model_df['pred_n'] == pred_n]
        data_start, labels_start, positions_start, abbrev_map_start = get_boxplot_data(pred_df, model_names, model_type, 'start_dev')
        data_end, labels_end, positions_end, abbrev_map_end = get_boxplot_data(pred_df, model_names, model_type, 'end_dev')
        abbrev_map = {**abbrev_map_start, **abbrev_map_end}

        if positions_start:
            offset = max(positions_start) + 2
            positions_end = [p + offset for p in positions_end]

        data = data_start + data_end
        labels = labels_start + labels_end
        positions = positions_start + positions_end

        box = ax.boxplot(data, positions=positions, widths=0.6, showfliers=True, patch_artist=True)
        for i, lbl in enumerate(labels):
            base_abbrev = lbl.replace("-CV", "")
            color = abbrev_colors[base_abbrev]
            box['boxes'][i].set_facecolor(color)
            if lbl.endswith("-CV"):
                box['boxes'][i].set_edgecolor("black")
                box['boxes'][i].set_linewidth(1.5)
            legend_map[lbl] = abbrev_map[lbl]

        ax.set_ylabel("Days")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        if y_max is not None:
            ax.set_ylim(0, y_max)

        if idx == len(pred_n_values) - 1:
            ax.set_xticks(positions)
            rotation = 90 if len(labels) > 10 else 45
            ha = 'right' if len(labels) > 10 else 'center'
            ax.set_xticklabels(labels, rotation=rotation, ha=ha)
            if positions_start and positions_end:
                annotate_groups(ax, positions, model_names)
        else:
            ax.set_xticks([])

        if idx == 0:
            ax.set_title(f"{model_type.capitalize()} - All Prediction Horizons", fontsize=16, fontweight='bold', pad=10)

    patches = []
    for abbrev, (model_full, cv_val) in sorted(legend_map.items()):
        label = f"{abbrev}: {model_full}" + (" (with covariates)" if cv_val else "")
        color = abbrev_colors[abbrev.replace("-CV", "")]
        patches.append(mpatches.Patch(facecolor=color, edgecolor="black" if cv_val else color, linewidth=1.5 if cv_val else 1, label=label))

    fig.legend(handles=patches, loc='center left', bbox_to_anchor=(0.87, 0.5), fontsize=max(6, 16 - len(model_names) // 2), title="Models", frameon=True, edgecolor='black', borderaxespad=0)

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot: {output_path}")


# === MAIN GENERATION ===

def generate_boxplots(evaluation_path: str, output_dir: str, uses_covariates=None):
    df = pd.read_csv(evaluation_path)

    pastel_colors = [
        '#AEC6CF', '#FFB347', '#B39EB5', '#77DD77', '#FF6961', '#FDFD96', '#CFCFC4',
        '#836953', '#F49AC2', '#B0E0E6', '#FFD1DC', '#C23B22', '#E6E6FA', '#B284BE',
        '#03C03C', '#779ECB', '#966FD6', '#F7CAC9', '#92A8D1', '#F7786B', '#DEB887',
        '#B6D7A8', '#FFB7B2', '#B5EAD7', '#FFDAC1', '#E2F0CB', '#C7CEEA', '#FFFACD'
    ]

    print(f"Loaded data: {len(df)} total rows")

    if uses_covariates is None:
        filtered_df = df.copy()
        covariates_label = "all_models"
    else:
        filtered_df = df[df['uses_covariates'] == uses_covariates].copy()
        covariates_label = "with_covariates" if uses_covariates else "without_covariates"

    filtered_df['model_type'] = filtered_df['model_name'].str.split('.').str[0]
    model_types = sorted(filtered_df['model_type'].unique())

    print(f"Rows after filtering: {len(filtered_df)}")
    print(f"Found model types: {model_types}")

    base_dir = os.path.join(output_dir, covariates_label)
    os.makedirs(base_dir, exist_ok=True)

    for model_type in model_types:
        print(f"\n=== Processing {model_type.upper()} ===")
        model_df = filtered_df[filtered_df['model_type'] == model_type]
        if model_df.empty:
            print(f"No data found for model type: {model_type}")
            continue

        start_vals = model_df['start_dev'].dropna()
        end_vals = model_df['end_dev'].dropna()
        y_max = max(start_vals.max() if not start_vals.empty else 0,
                    end_vals.max() if not end_vals.empty else 0)
        print(f"Max value for model_type '{model_type}': {y_max}")

        model_names = sorted(model_df['model_name'].unique())
        pred_n_values = sorted(model_df['pred_n'].unique())
        model_dir = os.path.join(base_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)

        # Plots individuales
        for pred_n in pred_n_values:
            pred_df = model_df[model_df['pred_n'] == pred_n]
            output_path = os.path.join(model_dir, f'boxplot_{model_type}_pred_n_{pred_n}.png')
            generate_individual_plot(pred_df, model_names, model_type, pred_n, pastel_colors, output_path, uses_covariates, y_max)

        # Plot combinado
        combined_path = os.path.join(model_dir, f'boxplot_{model_type}_all_pred_n_combined.png')
        generate_combined_plot(model_df, model_names, model_type, pred_n_values, pastel_colors, combined_path, uses_covariates, y_max)

    print("\n" + "=" * 80)
    print("PROCESS COMPLETED")
    print("All model plots have been generated.")
    print(f"Output directory structure under: {os.path.abspath(output_dir)}")
    print("=" * 80)