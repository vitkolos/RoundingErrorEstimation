from pathlib import Path
import typing
import re
import collections

import click
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import appmax.experiment
import appmax.logger
import appmax.applications


SEED = 42
EXPERIMENTS_DIR = Path('experiments')
LOAD_MODE = 'batch'

rng = np.random.default_rng(SEED)


@click.command()
@click.argument('visualization')
@click.argument('dataset')
@click.argument('run-ids', default=['run'], nargs=-1)
@click.option('--plot-only', is_flag=True)
def main(visualization, dataset, run_ids, plot_only):
    bundle = appmax.applications.DataBundle(dataset)
    error_scaling = bundle.data_split.metadata.error_scaling
    dataset_path = EXPERIMENTS_DIR / dataset

    match visualization:
        case 'check-2000':
            for run_id in run_ids:
                check_len(dataset_path, run_id, desired_len=2000)

        case 'comparison':
            compare_results(dataset_path, run_ids, error_scaling)

        case 'cardinalities':
            for run_id in run_ids:
                if not plot_only:
                    evaluate_subsets(dataset_path, run_id, error_scaling)

                plot_subsets(dataset_path, run_id)

        case 'input-face':
            for run_id in run_ids:
                show_input_faces(dataset_path, run_id)

        case 'histograms':
            for run_id in run_ids:
                plot_histograms(dataset_path, run_id)

        case 'union-combined':
            for run_id in run_ids:
                plot_union_combined(dataset_path, run_id, error_scaling)

        # ---

        case 'points':
            print_points()

        case 'widths':
            plot_tracked_widths({'california': EXPERIMENTS_DIR / 'california' / 'widths',
                                 'year': EXPERIMENTS_DIR / 'year' / 'widths'})

        case 'union':
            plot_tracked_union(dataset_path / 'union')

        case _:
            raise NotImplementedError(f'{visualization} not implemented')


TEX_ALIASES = {
    'sample_max': r'E_T',
    'sample_mean': r'\overline{E_T}',
    'nearby_max': r'E_{\Xi_T}',
    'nearby_mean': r'\overline{E_{\Xi_T}}',
    'nearby_weighted_sum': r'\overline{E}^{\tilde d}_{\Xi_T}',
    'integral_divided_sum': r'\overline{E}^{\tilde d}_{\Xi_T^E}',
    'error_sample': r'E(x)',
    'error_nearby': r'E_{\Xi_x}',
    'polytope_width': r'\tilde d_n(\Xi_x)',
    'weight': r'\frac{\tilde d_n(\Xi_x)}{S}',
    'nearby_weighted': r'\frac{\tilde d_n(\Xi_x)}{S} E_{\Xi_x}',
    'integral_width': r'\tilde d_{n+1}(\Xi_x^E)',
    'integral_divided': r'\tilde d_{n+1}(\Xi_x^E)\over S',
    'union_mean': r'\overline{E}_{\overline{\Xi}_T}',
    'union_weighted_sum': r'\overline{E}^{\tilde d}_{\overline{\Xi}_T}',
}


def to_display_label(label: str) -> str:
    return label.replace('_', ' ')


def load_df_results(experiment_path: Path, run_id: str) -> pd.DataFrame:
    """loads the plain results (experiment output) into a DataFrame"""
    match LOAD_MODE:
        case 'csv':
            df = pd.read_csv(experiment_path / f'{run_id}_results.csv', index_col=0)
        case 'batch':
            results = appmax.experiment.load_batch_results(experiment_path, run_id)
            df = pd.DataFrame(appmax.experiment.dict2flat(r) for r in results)
            df = df.set_index('sample_index').sort_index()
        case _:
            raise NotImplementedError

    return df


def extract_metrics(df_results: pd.DataFrame):
    """extracts our metrics from a DataFrame containing plain results"""
    described = appmax.experiment.describe(df_results)
    return {
        'sample_max': described.loc['max', 'error_sample'],
        'sample_mean': described.loc['mean', 'error_sample'],
        'nearby_max': described.loc['max', 'error_nearby'],
        'nearby_mean': described.loc['mean', 'error_nearby'],
        'nearby_weighted_sum': described.loc['weighted', 'error_nearby'],
        'integral_divided_sum': described.loc['weighted', 'integral'],
        'union_mean': described.loc['mean', 'union_error'],
        'union_weighted_sum': described.loc['weighted', 'union_error'],
    }


def wrap_html_tables(tables, into_one=True):
    """wraps HTML tables so that the output looks as a nice webpage (which supports KaTeX)"""
    html = ''.join(tables)

    if into_one:
        html = '<table>' + re.sub(r'</?table.*?>', '', html) + '</table>'

    for column, alias in TEX_ALIASES.items():
        html = html.replace(f'>{column}</th>', f'>\\( {alias} \\)<small>{to_display_label(column)}</small></th>')

    style = 'body{font-family:sans-serif} table{border-collapse: collapse;} td,th{padding:0.5rem 1rem;} th{text-align:right} th:not(:first-child){vertical-align:bottom; text-align:left} small{display:block; margin-top:0.5rem}'
    katex = """
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/katex.min.css" integrity="sha384-nH0MfJ44wi1dd7w6jinlyBgljjS8EJAh2JBoRad8a3VDw2K69vfaaqm4WnR+gXtA" crossorigin="anonymous">
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/katex.min.js" integrity="sha384-CwjPRVHTvLiMBFjEoij+QZViMV5rhTOIp7CJzl24JEqpRDA1sJFHVXXLURktbYYp" crossorigin="anonymous"></script>
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/contrib/auto-render.min.js" integrity="sha384-bjyGPfbij8/NDKJhSGZNP/khQVgtHUE5exjm4Ydllo42FwIgYsdLO2lXGmRBf5Mz" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
    """
    return f'<!doctype html><html><head>{katex}<style>{style}</style></head><body>{html}</body></html>'


def compare_results(experiment_path: Path, run_ids: list[str], error_scaling: float, aliases: dict[str, str] = {}):
    """writes nice tables containing the metrics and comparing them between runs"""
    dfs = {run_id: load_df_results(experiment_path, run_id) for run_id in run_ids}

    def analyze(name: str, df_results: pd.DataFrame):
        df_results.loc[:, appmax.experiment.UNSCALED_COLS] *= error_scaling
        return {
            'run': f'{experiment_path.name}: {aliases[name] if name in aliases else name}',
            **extract_metrics(df_results),
        }

    df = pd.DataFrame(analyze(*item) for item in dfs.items())
    df = df.set_index('run')
    df.index.name = None

    target_dir = experiment_path / 'common_outputs'
    target_dir.mkdir(parents=True, exist_ok=True)

    with open(target_dir / 'comparison.tex', 'w') as f:
        f.write(df.to_latex())

    with open(target_dir / 'comparison.html', 'w') as f:
        f.write(wrap_html_tables([df.to_html()]))


COL_SIZE = ('size', 'exact')


def evaluate_subsets(experiment_path: Path, run_id: str, error_scaling: float):
    """
    1. iterates over different cardinalities,
    2. chooses NUM_SUBSETS random subsets of a given cardinality,
    3. computes our metrics,
    4. finds the mean and std,
    5. stores the results in a csv file
    """
    NUM_SUBSETS = 100
    STEP = 50
    START = STEP
    df_results = load_df_results(experiment_path, run_id)
    df_results.loc[:, appmax.experiment.UNSCALED_COLS] *= error_scaling
    stats_for_sizes = []

    for size in appmax.logger.progress(range(START, len(df_results), STEP)):
        subsets_same_size = []

        for _ in range(NUM_SUBSETS):
            indices = rng.choice(len(df_results), size, replace=False)
            metrics = extract_metrics(df_results.loc[indices])
            subsets_same_size.append(metrics)

        stats_same_size = pd.DataFrame(subsets_same_size).describe()
        stats_compact = stats_same_size.loc[['mean', 'std']].unstack()
        stats_compact.loc[COL_SIZE] = size
        stats_for_sizes.append(stats_compact)

    subsets_dir = experiment_path / f'{run_id}_outputs' / 'subsets'
    subsets_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stats_for_sizes).to_csv(subsets_dir / 'subsets.csv')


def plot_subsets(experiment_path: Path, run_id: str):
    """plots the mean and std for different cardinalities (as computed by evaluate_subsets), groups metrics with similar properties"""
    subsets_dir = experiment_path / f'{run_id}_outputs' / 'subsets'
    df = pd.read_csv(subsets_dir / 'subsets.csv', header=[0, 1], index_col=0)

    columns = df.columns.get_level_values(0).unique().drop('size')
    grouped_columns = collections.defaultdict(list)

    for column in columns:
        match column.split('_'):
            case [_, 'max']:
                grouped_columns['max'].append(column)
            case ['integral', *_]:
                grouped_columns['integral'].append(column)
            case _:
                grouped_columns['weighted, mean'].append(column)

    with PdfPages(subsets_dir / 'bundle.pdf') as pdf:
        # plt.rcParams['text.usetex'] = True

        for title, group in grouped_columns.items():
            fig, ax = plt.subplots()
            legend = []

            for column in group:
                size = df.loc[:, COL_SIZE]
                mean = df.loc[:, (column, 'mean')]
                std = df.loc[:, (column, 'std')]
                label = to_display_label(column)

                if tex := TEX_ALIASES.get(column):
                    label = f'${tex}$ {label}'

                handle, = ax.plot(size, mean, '.-')
                ax.fill_between(size, mean-std, mean+std, alpha=0.2)
                legend.append({'label': label, 'handle': handle, 'last_value': mean.iloc[-1]})

            legend.sort(key=lambda item: item['last_value'], reverse=True)

            def print_legend(legend, loc='best'):
                return ax.legend([item['handle'] for item in legend], [item['label'] for item in legend], loc=loc)

            if len(legend) > 3:
                ax.add_artist(print_legend(legend[:2], 'upper left'))
                print_legend(legend[2:])
            else:
                print_legend(legend)

            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xlabel('cardinality')
            ax.set_ylabel(r'metric ($\mu\pm\sigma$)')
            fig.savefig(subsets_dir / f'{title}.svg', bbox_inches='tight')
            pdf.savefig(fig)
            plt.close(fig)


def show_input_faces(experiment_path: Path, run_id: str):
    NUM_FACES = 30
    results = appmax.experiment.load_batch_results(experiment_path, run_id)

    target_dir = experiment_path / f'{run_id}_outputs' / 'faces'
    target_dir.mkdir(parents=True, exist_ok=True)

    def x_to_img(x: torch.Tensor):
        return (x.movedim(0, -1) + 1) / 2

    for i, item in enumerate(results[:NUM_FACES]):
        xs: dict[str, torch.Tensor] = {}
        xs['original'] = item['result_sample']['x']
        xs['nearby'] = item['result_nearby']['x']
        xs['union'] = item['result_nearby']['union']['x']

        for name, x in xs.items():
            plt.imsave(target_dir / f'face_{i:04d}_{name}.png', x_to_img(x))


def plot_histograms(experiment_path: Path, run_id: str):
    df_results = load_df_results(experiment_path, run_id)

    target_dir = experiment_path / f'{run_id}_outputs' / 'histograms'
    target_dir.mkdir(parents=True, exist_ok=True)

    for col in df_results.columns:
        fig, ax = plt.subplots()
        ax.set_xlabel(to_display_label(col))
        ax.set_ylabel('frequency')
        data = df_results[col]

        # we remove leading and trailing bins with counts 0 or 1
        counts, bin_edges = np.histogram(df_results[col], bins='auto')
        bins_gt_one = np.flatnonzero(counts > 1)
        first_gt_one = bins_gt_one.min()
        last_gt_one = bins_gt_one.max()
        limit_lower = bin_edges[first_gt_one]
        limit_upper = bin_edges[last_gt_one+1]
        data = np.clip(data, limit_lower, limit_upper)

        ax.hist(data, bins='auto', histtype='stepfilled')
        _, _, patches = ax.hist(data, bins='auto', color='none')

        if first_gt_one > 0:
            patches[0].set_facecolor('red')

        if last_gt_one+2 < len(counts):
            patches[-1].set_facecolor('red')

        fig.savefig(target_dir / f'{col}.svg', bbox_inches='tight')
        plt.close(fig)


def plot_union_combined(experiment_path: Path, run_id: str, error_scaling: float):
    results = appmax.experiment.load_batch_results(experiment_path, run_id)

    target_dir = experiment_path / f'{run_id}_outputs' / 'union'
    target_dir.mkdir(parents=True, exist_ok=True)

    widths: list[float] = []
    maxima_jagged: list[list[float]] = []
    max_len = 0

    for item in results:
        progress = item['result_nearby']['union']['progress']
        widths.append(item['result_nearby']['union']['width'])
        maxima = []
        last_n = 0
        maximum = progress[0][1]

        for n, fun in progress:
            maximum = max(maximum, fun)

            if n > last_n:
                # new polytope found
                maxima.append(maximum)

            last_n = n

        maxima_jagged.append(maxima)
        max_len = max(max_len, len(maxima))

    for maxima in maxima_jagged:
        maxima.extend([maxima[-1]] * (max_len - len(maxima)))

    maxima_unscaled = np.array(maxima_jagged) * error_scaling
    means = np.mean(maxima_unscaled, axis=0)
    weighted = np.average(maxima_unscaled, axis=0, weights=widths)
    ns = range(1, len(means)+1)

    fig, ax = plt.subplots()
    ax.set_xlabel('discovered subpolytopes')
    ax.set_ylabel('mean maximum error')
    ax.plot(ns, weighted, '.-', label=f'${TEX_ALIASES['union_weighted_sum']}$ weighted mean')
    ax.plot(ns, means, '.-', label=f'${TEX_ALIASES['union_mean']}$ arithmetic mean')
    ax.legend()
    fig.savefig(target_dir / f'combined.svg', bbox_inches='tight')
    plt.close(fig)


def check_len(experiment_path: Path, run_id: str, desired_len: int):
    results = appmax.experiment.load_batch_results(experiment_path, run_id)

    if len(results) != desired_len:
        raise ValueError(f'run {experiment_path.name}/{run_id} does not contain {desired_len} items')


# ---


def print_points():
    indices = sorted(rng.permutation(1000)[:20].tolist())
    datasets = [
        (EXPERIMENTS_DIR / 'california', appmax.applications.california_housing.CaliforniaHousingSplit().metadata.error_scaling),
        (EXPERIMENTS_DIR / 'year', appmax.applications.year_prediction.YearPredictionSplit().metadata.error_scaling),
    ]
    runs = ('run', 'sym8', 'second', 'sym4')

    with open(EXPERIMENTS_DIR / 'points.html', 'w') as f:
        tables = [list_points(d, r, 1.0, indices) for d, _ in datasets for r in runs]
        f.write(wrap_html_tables(tables, into_one=False))

    with open(EXPERIMENTS_DIR / 'points_unscaled.html', 'w') as f:
        tables = [list_points(d, r, s, indices) for d, s in datasets for r in runs]
        f.write(wrap_html_tables(tables, into_one=False))


def list_points(experiment_path: Path, run_id: str, error_scaling: float, indices: list[int], aliases: dict[str, str] = {}) -> str:
    df_results = load_df_results(experiment_path, run_id)
    df_results.loc[:, appmax.experiment.UNSCALED_COLS] *= error_scaling
    weights = df_results.get('polytope_width')
    assert weights is not None
    weights_sum = weights.sum()

    def row(item):
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
    styled_df = df.style.highlight_min(**hl_args).highlight_max(**hl_args)  # type: ignore[arg-type]

    run_name = aliases[run_id] if run_id in aliases else run_id
    s_tex = r'\sum_{x\in T} \tilde d_n(\Xi_x)'
    unscaled_text = '' if error_scaling == 1.0 else f'(unscaled = multiplied by {error_scaling:.6f} to get the original units)'
    header = f'<p><b>{experiment_path.name}: {run_name}</b> {unscaled_text}</p><p>\\( S = {s_tex} = \\) {weights_sum:.6f}</p>'
    return header + styled_df.to_html()


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


def plot_tracked_union(experiment_path: Path):
    data = pd.read_csv(experiment_path / 'data.csv', index_col=0)
    grouped = data.groupby('sample')

    def plot_chart(category, sample):
        group_data = grouped.get_group(sample)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(group_data['point'], group_data['fun'], '.', label='function value')
        ax1.plot(group_data['point'], group_data['max'], label='maximum')
        ax2.plot(group_data['point'], group_data['polytopes'], color='red', label='found polytopes')
        ax2.set_ylim(0, 50)

        line = {'c': 'black', 'ls': 'dotted'}
        ax1.axvline(25, **line)
        ax1.axvline(50, **line, lw=2)
        ax1.axvline(75, **line)
        ax1.axvline(100, **line)

        category_path = experiment_path / category
        category_path.mkdir(parents=True, exist_ok=True)
        ax1.legend(loc='center left')
        ax2.legend(loc='center right')
        plt.savefig(category_path / f'{sample+1:02d}.png')
        plt.close()

    for sample in grouped.groups.keys():
        plot_chart('single', sample)


if __name__ == '__main__':
    main()
