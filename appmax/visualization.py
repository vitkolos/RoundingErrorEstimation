from pathlib import Path
import typing
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import appmax.experiment
import appmax.logger


def plot_results(experiment_path: Path | str, run_id: str):
    experiment_path = Path(experiment_path)
    df_results = pd.read_csv(experiment_path / f'{run_id}_results.csv')
    target_dir = experiment_path / f'{run_id}_plots'
    target_dir.mkdir(parents=True, exist_ok=True)

    for col in ['error_sample', 'error_nearby', 'polytope_width', 'integral']:
        plt.hist(df_results[col], 50)
        plt.title(col)
        plt.savefig(target_dir / f'{col}_hist.png')
        plt.close()


def compare_results(experiment_path: Path | str, run_ids: list[str], aliases: dict[str, str]) -> str:
    experiment_path = Path(experiment_path)
    dfs = {run_id: pd.read_csv(experiment_path / f'{run_id}_described_unscaled.csv', index_col=0) for run_id in run_ids}

    def extract_metrics(name: str, df: pd.DataFrame):
        return {
            'run': f'{experiment_path.name}: {aliases[name] if name in aliases else name}',
            'sample_max': df.loc['max', 'error_sample'],
            'sample_mean': df.loc['mean', 'error_sample'],
            'nearby_max': df.loc['max', 'error_nearby'],
            'nearby_mean': df.loc['mean', 'error_nearby'],
            'nearby_weighted_sum': df.loc['weighted', 'error_nearby'],
            'integral_divided_sum': df.loc['weighted', 'integral'],
        }

    df = pd.DataFrame(extract_metrics(*item) for item in dfs.items())
    df = df.set_index('run')
    df.index.name = None
    styled_df = df.style.highlight_min(color='lightgreen', axis=0)
    return styled_df.to_html()


def list_points(experiment_path: Path | str, run_id: str, error_scaling: float, indices: list[int], aliases: dict[str, str]) -> str:
    experiment_path = Path(experiment_path)
    df_results = pd.read_csv(experiment_path / f'{run_id}_results.csv')
    df_results.loc[:, appmax.experiment.UNSCALED_COLS] *= error_scaling
    weights_sum = df_results.get('polytope_width').sum()

    def row(item: pd.Series):
        return {
            'index': int(item['sample_index']),
            'error_sample': item['error_sample'],
            'error_nearby': item['error_nearby'],
            'polytope_width': item['polytope_width'],
            'weight': item['polytope_width'] / weights_sum,
            'nearby_weighted': (item['polytope_width'] / weights_sum) * item['error_nearby'],
            'integral_width': item['integral'],
            'integral_divided': item['integral'] / weights_sum,
        }

    df = pd.DataFrame(row(df_results.loc[index]) for index in indices)
    df = df.set_index('index')
    df.index.name = None
    hl_args = {'axis': 0, 'props': 'font-weight:bold'}
    styled_df = df.style.highlight_min(**hl_args).highlight_max(**hl_args)

    run_name = aliases[run_id] if run_id in aliases else run_id
    s_tex = r'\sum_{x\in T} \tilde d_n(\Xi_x)'
    unscaled_text = '' if error_scaling == 1.0 else f'(unscaled = multiplied by {error_scaling:.6f} to get the original units)'
    header = f'<p><b>{experiment_path.name}: {run_name}</b> {unscaled_text}</p><p>\\( S = {s_tex} = \\) {weights_sum:.6f}</p>'
    return header + styled_df.to_html()


TEX_ALIASES = {
    'sample_max': r'E_T',
    'sample_mean': r'\overline{E_T}',
    'nearby_max': r'E_{\Xi_T}',
    'nearby_mean': r'\overline{E_{\Xi_T}}',
    'nearby_weighted_sum': r'\overline E^{\tilde d}_{\Xi_T}',
    'integral_divided_sum': r'\overline E^{\tilde d}_{\Xi_T^E}',
    'error_sample': r'E(x)',
    'error_nearby': r'E_{\Xi_x}',
    'polytope_width': r'\tilde d_n(\Xi_x)',
    'weight': r'\frac{\tilde d_n(\Xi_x)}{S}',
    'nearby_weighted': r'\frac{\tilde d_n(\Xi_x)}{S} E_{\Xi_x}',
    'integral_width': r'\tilde d_{n+1}(\Xi_x^E)',
    'integral_divided': r'\tilde d_{n+1}(\Xi_x^E)\over S',
    'union_mean': r'\overline E_{\overline \Xi_T}',
    'union_weighted_sum': r'\overline E^{\tilde d}_{\overline \Xi_T}',
}


def tables_to_html(tables, into_one=True):
    html = ''.join(tables)

    if into_one:
        html = '<table>' + re.sub(r'</?table.*?>', '', html) + '</table>'

    for column, alias in TEX_ALIASES.items():
        html = html.replace(f'>{column}</th>', f'>\\( {alias} \\)<small>{column.replace('_', ' ')}</small></th>')

    style = 'body{font-family:sans-serif} table{border-collapse: collapse;} td,th{padding:0.5rem 1rem;} th{text-align:right} th:not(:first-child){vertical-align:bottom; text-align:left} small{display:block; margin-top:0.5rem}'
    katex = """
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/katex.min.css" integrity="sha384-nH0MfJ44wi1dd7w6jinlyBgljjS8EJAh2JBoRad8a3VDw2K69vfaaqm4WnR+gXtA" crossorigin="anonymous">
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/katex.min.js" integrity="sha384-CwjPRVHTvLiMBFjEoij+QZViMV5rhTOIp7CJzl24JEqpRDA1sJFHVXXLURktbYYp" crossorigin="anonymous"></script>
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/contrib/auto-render.min.js" integrity="sha384-bjyGPfbij8/NDKJhSGZNP/khQVgtHUE5exjm4Ydllo42FwIgYsdLO2lXGmRBf5Mz" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
    """
    return f'<!doctype html><html><head>{katex}<style>{style}</style></head><body>{html}</body></html>'


def plot_tracked_widths(experiments: dict[str, str]):
    experiment_paths = {e: Path(p) for e, p in experiments.items()}
    data, grouped = {}, {}
    types = ['polytope', 'integral']
    first_k = 10

    for e, p in experiment_paths.items():
        data[e] = pd.read_csv(p / 'data.csv', index_col=0)
        grouped[e] = data[e].groupby(['sample', 'type'])

    def s(data):
        return data[25:]

    def plot_chart(category, name, identifiers):
        for experiment, key, label in identifiers:
            group_data = grouped[experiment].get_group(key)
            plt.plot(s(group_data['directions']), s(group_data['width']), label=label)

        line = {'c': 'black', 'ls': 'dotted'}
        plt.axvline(50, **line)
        plt.axvline(100, **line, lw=2)
        plt.axvline(150, **line)
        plt.axvline(200, **line)

        if any(x[2] for x in identifiers):
            plt.legend()

        experiment_first = identifiers[0][0]
        category_path = experiment_paths[experiment_first] / category
        category_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(category_path / f'{name}.png')
        plt.close()

    def plot_charts(category, name, identifiers):
        num = len(identifiers)
        fig, axes = plt.subplots(num, figsize=(6.4, 3*num))

        for ax, (experiment, key, label) in zip(axes, identifiers):
            group_data = grouped[experiment].get_group(key)
            ax.plot(s(group_data['directions']), s(group_data['width']), label=label)
            line = {'c': 'black', 'ls': 'dotted'}
            ax.axvline(50, **line)
            ax.axvline(100, **line, lw=2)
            ax.axvline(150, **line)
            ax.axvline(200, **line)
            ax.legend()

        experiment_first = identifiers[0][0]
        category_path = experiment_paths[experiment_first] / category
        category_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(category_path / f'{name}.png')
        plt.close()

    # one chart per polytope
    for experiment in experiments.keys():
        for key in grouped[experiment].groups.keys():
            sample, type_ = typing.cast(tuple[int, str], key)
            plot_chart('single', f'{type_}_{sample+1:02d}', [(experiment, key, None)])

    # polytope and integral in the same chart
    for experiment in experiments.keys():
        for sample in range(first_k):
            plot_chart('both', f'{sample+1:02d}', [(experiment, (sample, t), t) for t in types])

    # several polytopes in one chart
    for experiment in experiments.keys():
        for type_ in types:
            plot_chart('combined', type_, [(experiment, (i, type_), None) for i in range(first_k)])

    # several datasets in one chart
    for type_ in types:
        for sample in range(first_k):
            plot_charts('different', f'{type_}_{sample+1:02d}', [(e, (sample, type_), e) for e in experiments.keys()])


COL_SIZE = ('size', 'exact')


def evaluate_subsets(experiment_path: Path | str, run_id: str, error_scaling: float):
    SEED = 42
    NUM_SUBSETS = 100
    STEP = 50
    START = STEP
    experiment_path = Path(experiment_path)
    df_results = pd.read_csv(experiment_path / f'{run_id}_results.csv', index_col=0)
    df_results.loc[:, appmax.experiment.UNSCALED_COLS] *= error_scaling
    rng = np.random.default_rng(SEED)
    stats_for_sizes = []

    for size in appmax.logger.progress(range(START, len(df_results), STEP)):
        subsets_same_size = []

        for _ in range(NUM_SUBSETS):
            indices = rng.choice(len(df_results), size, replace=False)
            described = appmax.experiment.describe(df_results.loc[indices])
            subsets_same_size.append({
                'sample_max': described.loc['max', 'error_sample'],
                'sample_mean': described.loc['mean', 'error_sample'],
                'nearby_max': described.loc['max', 'error_nearby'],
                'nearby_mean': described.loc['mean', 'error_nearby'],
                'nearby_weighted_sum': described.loc['weighted', 'error_nearby'],
                'integral_divided_sum': described.loc['weighted', 'integral'],
                'union_mean': described.loc['mean', 'union_error'],
                'union_weighted_sum': described.loc['weighted', 'union_error'],
            })

        stats_same_size = pd.DataFrame(subsets_same_size).describe()
        stats_compact = stats_same_size.loc[['mean', 'std']].unstack()
        stats_compact.loc[COL_SIZE] = size
        stats_for_sizes.append(stats_compact)

    pd.DataFrame(stats_for_sizes).to_csv(experiment_path / f'{run_id}_subsets.csv')


def plot_subsets(experiment_path: Path | str, run_id: str):
    experiment_path = Path(experiment_path)
    df = pd.read_csv(experiment_path / f'{run_id}_subsets.csv', header=[0, 1], index_col=0)
    columns = df.columns.get_level_values(0).unique().drop('size')

    with PdfPages(experiment_path / f'{run_id}_subsets.pdf') as pdf:
        plt.rcParams['text.usetex'] = True

        for column in columns:
            size = df.loc[:, COL_SIZE]
            mean = df.loc[:, (column, 'mean')]
            std = df.loc[:, (column, 'std')]
            fig, ax = plt.subplots()
            plt.plot(size, mean, '.-')
            plt.fill_between(size, mean-std, mean+std, alpha=0.2)
            title = column.replace('_', ' ')

            if tex := TEX_ALIASES.get(column):
                title = f'${tex}$ {title}'

            plt.title(title)
            plt.grid(True, linestyle='--', alpha=0.5)
            ax.set_xlabel('cardinality')
            ax.set_ylabel(r'metric ($\mu\pm\sigma$)')
            pdf.savefig()
            plt.close()
