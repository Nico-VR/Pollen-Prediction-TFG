import pandas as pd
import numpy as np
import itertools, os, re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests

def kruskal_wallis_by_models(
    evaluation_path: str,
    model_names_include: list[str],
    columnas_a_comparar: list,
    uses_covariates: bool = None
):
    """
    Realiza un test de Kruskal–Wallis para comparar distribuciones
    entre modelos. Permite filtrar por uso de covariables.
    
    Parameters
    ----------
    evaluation_path : str
        Ruta al CSV con los resultados.
    model_names_include : list[str]
        Lista de patrones regex para filtrar nombres de modelos.
    columnas_a_comparar : list[str]
        Columnas del dataset a comparar (ej. ['start_dev', 'end_dev']).
    uses_covariates : bool, optional
        Si se indica, filtra los modelos que usan o no covariables.
        
    Returns
    -------
    dict
        Resultados del test para cada columna en formato:
        {
            "columna": {
                "H_statistic": float,
                "p_value": float,
                "modelos_incluidos": list[str]
            }
        }
    """
    # Leer datos
    df = pd.read_csv(evaluation_path)

    # Filtrar opcionalmente por uses_covariates
    if uses_covariates is not None:
        df = df[df['uses_covariates'] == uses_covariates]

    resultados = {}

    for col in columnas_a_comparar:
        # Selección de modelos según regex
        modelos = [
            m for m in df['model_name'].unique()
            if any(re.search(pattern, m) for pattern in model_names_include)
        ]

        if len(modelos) < 2:
            resultados[col] = {
                "H_statistic": None,
                "p_value": None,
                "modelos_incluidos": modelos,
                "mensaje": "No hay suficientes modelos para comparar"
            }
            continue

        # Agrupar valores por modelo
        grupos = [df.loc[df['model_name'] == m, col].dropna() for m in modelos]

        # Test Kruskal–Wallis
        H, p = kruskal(*grupos)

        resultados[col] = {
            "H_statistic": H,
            "p_value": p,
            "modelos_incluidos": modelos
        }

    return resultados

def model_to_initials(model_name: str, uses_covariates: bool = False, add_cv_suffix: bool = False):
    """
    Genera iniciales del modelo.
    - Si contiene "train" en alguna parte, la primera parte aporta sus 2 primeras letras.
    - El resto de partes aportan solo la primera letra.
    - Si add_cv_suffix=True y uses_covariates=True, añade '-CV'.
    """
    # Partes del nombre
    if 'train' not in model_name:
        parts = model_name.split('.')[1:]
    else:
        parts = model_name.split('.')

    initials = ""
    if parts:
        if any("train" in p for p in parts):  
            # primeras 2 letras de la primera parte
            initials += parts[0][:2].upper()
            # más 1 letra de las demás partes
            initials += ''.join(part[0].upper() for part in parts[1:] if part)
        else:
            # comportamiento normal
            initials = ''.join(part[0].upper() for part in parts if part)

    if add_cv_suffix and uses_covariates:
        initials += "-CV"

    return initials

def mann_whitney_with_holm_bonferroni(
    evaluation_path: str,
    model_names_include: list[str],
    columnas_a_comparar: list,
    uses_covariates: bool = None
):
    # Leer datos
    df = pd.read_csv(evaluation_path)

    # Filtrar modelos según covariables si se indica
    filtered_df = df.copy()
    if uses_covariates is not None:
        filtered_df = filtered_df[filtered_df['uses_covariates'] == uses_covariates]

    # Filtrar modelos por patrones
    filtered_df = filtered_df[filtered_df['model_name'].apply(lambda x: any(re.search(pat, x) for pat in model_names_include))]

    # Crear identificador de modelo (abreviado) y nombre de despliegue (completo)
    filtered_df['model_id'] = filtered_df.apply(
        lambda row: model_to_initials(row['model_name'], row['uses_covariates'], add_cv_suffix=(uses_covariates is None)),
        axis=1
    )
    filtered_df['model_display'] = filtered_df.apply(
        lambda row: f"{row['model_name']} (with covariates)" if row['uses_covariates'] else row['model_name'],
        axis=1
    )

    # Diccionario abreviado -> nombre completo
    model_map = dict(zip(filtered_df['model_id'], filtered_df['model_display']))

    resultados_por_columna = {col: [] for col in columnas_a_comparar}

    for col in columnas_a_comparar:
        modelos = filtered_df['model_id'].unique()
        if len(modelos) < 2:
            continue  # No hay suficientes modelos para comparar

        # Comparaciones por pares
        pares = list(itertools.permutations(modelos, 2))
        p_values = []
        comparaciones = []

        for m1, m2 in pares:
            grupo1 = filtered_df.loc[filtered_df['model_id'] == m1, col].dropna()
            grupo2 = filtered_df.loc[filtered_df['model_id'] == m2, col].dropna()
            if len(grupo1) == 0 or len(grupo2) == 0:
                continue
            _, p = mannwhitneyu(x=grupo1, y=grupo2, alternative='less')
            p_values.append(p)
            comparaciones.append((m1, m2))

        if len(p_values) > 0:
            _, pvals_corr, _, _ = multipletests(p_values, alpha=0.05, method='holm')
        else:
            pvals_corr = []

        # Guardar resultados
        for i, (m1, m2) in enumerate(comparaciones):
            resultados_por_columna[col].append({
                'model_1': m1,
                'model_2': m2,
                'q_value': pvals_corr[i],
            })

    # Convertir a DataFrames
    resultados_start_dev = pd.DataFrame(resultados_por_columna.get('start_dev', []), columns=['model_1','model_2','q_value'])
    resultados_end_dev   = pd.DataFrame(resultados_por_columna.get('end_dev', []),   columns=['model_1','model_2','q_value'])

    return resultados_start_dev, resultados_end_dev, model_map


def results_to_matrix(resultados_df: pd.DataFrame, model_order: list[str] = None, col_metric: str = "q_value"):
    if model_order is None:
        modelos = sorted(set(resultados_df["model_1"]).union(resultados_df["model_2"]))
    else:
        modelos = [m for m in model_order if m in set(resultados_df["model_1"]).union(resultados_df["model_2"])]

    matriz = pd.DataFrame(np.nan, index=modelos, columns=modelos)
    
    for _, row in resultados_df.iterrows():
        m1 = row["model_1"]
        m2 = row["model_2"]
        matriz.loc[m1, m2] = row[col_metric]
    
    return matriz

def compare_independent_mw(
    evaluation_path: str,
    model_names_include: list[str],
    output_path: str,
    uses_covariates: bool = None,
    columnas_a_comparar: list = ['start_dev', 'end_dev']
):

    
    # Generar título automáticamente
    if len(model_names_include) > 1:
        plot_title = f"Comparación del rendimiento de todos los modelos"
    else:
        plot_title = f"Comparación del rendimiento de las variantes de {model_names_include[0]}"
        
    # Resultados de Mann–Whitney + corrección Holm-Bonferroni
    resultados_start_dev, resultados_end_dev, model_map = mann_whitney_with_holm_bonferroni(
        evaluation_path, model_names_include, columnas_a_comparar, uses_covariates
    )

    # Todos los modelos que aparecen en los resultados
    all_model_ids = set(resultados_start_dev['model_1']).union(
        resultados_start_dev['model_2'],
        resultados_end_dev['model_1'],
        resultados_end_dev['model_2']
    )

    # Construir orden por patrones: primero los modelos de cada patrón en orden, luego los restantes
    models_ordered = []
    for pattern in model_names_include:
        matched = sorted([m_id for m_id in all_model_ids if re.search(pattern, m_id)])
        models_ordered.extend(m for m in matched if m not in models_ordered)
    for m_id in sorted(all_model_ids):
        if m_id not in models_ordered:
            models_ordered.append(m_id)

    # Convertir resultados a matrices usando el orden correcto
    matrix_start_dev = results_to_matrix(resultados_start_dev, model_order=models_ordered)
    matrix_end_dev   = results_to_matrix(resultados_end_dev, model_order=models_ordered)
    models, n = matrix_start_dev.index, len(matrix_start_dev)

    # Inicializar tabla y contadores
    table_data = np.full((n, n + 6), '', dtype=object)
    up_counts_1 = np.zeros(n, dtype=int)
    down_counts_1 = np.zeros(n, dtype=int)
    up_counts_2 = np.zeros(n, dtype=int)
    down_counts_2 = np.zeros(n, dtype=int)

    for i, j in itertools.combinations(range(n), 2):
        # === start_dev ===
        p_ij = matrix_start_dev.values[i, j]
        p_ji = matrix_start_dev.values[j, i]
        
        if p_ij < 0.05:
            table_data[i, j] = '↑'
            table_data[j, i] = '↓'
            up_counts_1[i] += 1
            down_counts_1[j] += 1
        elif p_ji < 0.05:
            table_data[i, j] = '↓'
            table_data[j, i] = '↑'
            down_counts_1[i] += 1
            up_counts_1[j] += 1
        else:
            table_data[i, j] = '-'
            table_data[j, i] = '-'

        # === end_dev ===
        p_ij = matrix_end_dev.values[i, j]
        p_ji = matrix_end_dev.values[j, i]

        if p_ij < 0.05:
            table_data[i, j] += '↑'
            table_data[j, i] += '↓'
            up_counts_2[i] += 1
            down_counts_2[j] += 1
        elif p_ji < 0.05:
            table_data[i, j] += '↓'
            table_data[j, i] += '↑'
            down_counts_2[i] += 1
            up_counts_2[j] += 1
        else:
            table_data[i, j] += '-'
            table_data[j, i] += '-'

    # Totales por fila
    for i in range(n):
        table_data[i, n:n+6] = [
            str(up_counts_1[i]), str(down_counts_1[i]),
            str(up_counts_2[i]), str(down_counts_2[i]),
            str(up_counts_1[i] + up_counts_2[i]),
            str(down_counts_1[i] + down_counts_2[i])
        ]


    # Colores para cada modelo
    pastel_colors = [
        '#AEC6CF', '#FFB347', '#B39EB5', '#77DD77', '#FF6961', '#FDFD96', '#CFCFC4',
        '#836953', '#F49AC2', '#B0E0E6', '#FFD1DC', '#C23B22', '#E6E6FA', '#B284BE',
        '#03C03C', '#779ECB', '#966FD6', '#F7CAC9', '#92A8D1', '#F7786B', '#DEB887',
        '#B6D7A8', '#FFB7B2', '#B5EAD7', '#FFDAC1', '#E2F0CB', '#C7CEEA', '#FFFACD'
    ]
    color_map = {m: pastel_colors[i % len(pastel_colors)] for i, m in enumerate(models_ordered)}

    # Crear figura
    fig, ax = plt.subplots(figsize=(max(12, n * 0.8), max(6, n * 0.5)))
    ax.axis('off')
    base_fontsize = max(8, int(18 - 0.3 * n))

    col_labels = list(models) + [
        ''.join(p[0].upper() for p in columnas_a_comparar[0].split('_')) + ' ↑',
        ''.join(p[0].upper() for p in columnas_a_comparar[0].split('_')) + ' ↓',
        ''.join(p[0].upper() for p in columnas_a_comparar[1].split('_')) + ' ↑',
        ''.join(p[0].upper() for p in columnas_a_comparar[1].split('_')) + ' ↓',
        '↑', '↓'
    ]

    table = plt.table(
        cellText=table_data,
        rowLabels=models,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 0.85, 1]
    )

    total_cols = len(col_labels) + 1
    for (row, col), cell in table.get_celld().items():
        cell.set_width(1 / total_cols)
        cell.set_height(1 / (n + 1))
        cell.get_text().set_fontsize(base_fontsize)
        cell.get_text().set_color('black')

        if row == 0 and col < n:
            model_id = col_labels[col]
            if model_id in color_map:
                cell.set_facecolor(color_map[model_id])
        if col == -1 and row > 0:
            model_id = models[row - 1]
            if model_id in color_map:
                cell.set_facecolor(color_map[model_id])

    for i in range(n):
        table[i + 1, i].set_facecolor('#cccccc')

    patches = []
    for m in models:
        full_name = model_map.get(m, m)
        patch = mpatches.Patch(
            facecolor=color_map[m],
            edgecolor='black',
            linewidth=1.5,
            label=f"{m}: {full_name}"
        )
        patches.append(patch)

    ax.legend(
        handles=patches,
        loc='center left',
        bbox_to_anchor=(0.87, 0.5),
        fontsize=max(6, 16 - n // 2),
        title="Models",
        frameon=True,
        framealpha=1.0,
        edgecolor='black',
        facecolor='white',
        borderaxespad=0
    )

    ax.set_title(plot_title, fontweight='bold', fontsize=14)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=900)
    plt.close()
