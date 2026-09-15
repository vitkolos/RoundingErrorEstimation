from pathlib import Path
import re
import json

import PIL
import click
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors

import appmax.experiment
import appmax.logger
import appmax.applications


SEED = 42
EXPERIMENTS_DIR = Path('experiments')
LOAD_MODE = 'batch'

ZOOM = 1.5
FIGSIZE = (6.4/ZOOM, 4.8/ZOOM)

PRINT_COLORS = False

rng = np.random.default_rng(SEED)


@click.command()
@click.argument('visualization')
@click.argument('dataset')
@click.argument('run-ids', default=['run'], nargs=-1)
@click.option('--plot-only', is_flag=True)
@click.option('--fresh-metadata', is_flag=True)
@click.option('--with-legend', is_flag=True)
def main(visualization, dataset, run_ids, plot_only, fresh_metadata, with_legend):
    dataset_path = EXPERIMENTS_DIR / dataset
    error_scaling = load_scaling(dataset, dataset_path, fresh_metadata)

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

                plot_subsets(dataset_path, run_id, with_legend)

        case 'union-combined':
            for run_id in run_ids:
                plot_union_combined(dataset_path, run_id, error_scaling, with_legend)

        case 'input-face':
            for run_id in run_ids:
                show_input_faces(dataset_path, run_id, error_scaling)

        case 'histograms':
            for run_id in run_ids:
                plot_histograms(dataset_path, run_id)

        case 'palette':
            print_palette()

        case _:
            raise NotImplementedError(f'{visualization} not implemented')


TEX_ALIASES = {
    'sample_max': r'E_T',
    'sample_mean': r'\overline{E}_T',
    'nearby_max': r'E_{\Xi_T}',
    'nearby_mean': r'\overline{E}_{\Xi_T}',
    'nearby_weighted_sum': r'\overline{E}^{\widetilde d}_{\Xi_T}',
    'integral_divided_sum': r'\overline{E}^{\widetilde d}_{\Xi_T^E}',
    'error_sample': r'E(x)',
    'error_nearby': r'E_{\Xi_x}',
    'polytope_width': r'\tilde d_n(\Xi_x)',
    'weight': r'\frac{\tilde d_n(\Xi_x)}{S}',
    'nearby_weighted': r'\frac{\tilde d_n(\Xi_x)}{S} E_{\Xi_x}',
    'integral_width': r'\tilde d_{n+1}(\Xi_x^E)',
    'integral_divided': r'\tilde d_{n+1}(\Xi_x^E)\over S',
    'union_max': r'E_{\overline{\Xi}_T}',
    'union_mean': r'\overline{E}_{\overline{\Xi}_T}',
    'union_weighted_sum': r'\overline{E}^{\widetilde d}_{\overline{\Xi}_T}',
}

NETS = {
    'california': 1,
    'year': 2,
    'utkface': 3,
}

LABEL_ALIASES = {
    'union_width': 'extended polytope width',
}


def to_display_label(label: str) -> str:
    label = LABEL_ALIASES.get(label, label)
    return label.replace('_', ' ')


def load_scaling(dataset: str, experiment_path: Path, fresh_metadata: bool) -> float:
    metadata_file = experiment_path / 'metadata.pt'

    if not fresh_metadata and metadata_file.is_file():
        metadata_dict = torch.load(metadata_file)
        return metadata_dict['error_scaling']

    bundle = appmax.applications.DataBundle(dataset)
    metadata_dict = bundle.data_split.metadata.to_dict()
    torch.save(metadata_dict, metadata_file)

    with open(experiment_path / f'metadata.json', 'w') as file_json:
        json.dump(metadata_dict, file_json)

    return metadata_dict['error_scaling']


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
        'union_max': described.loc['max', 'union_error'],
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


def compare_results(experiment_path: Path, run_ids: list[str], error_scaling: float):
    """writes nice tables containing the metrics and comparing them between runs"""
    dfs = {run_id: load_df_results(experiment_path, run_id) for run_id in run_ids}

    def analyze(name: str, df_results: pd.DataFrame):
        df_results.loc[:, appmax.experiment.UNSCALED_COLS] *= error_scaling
        net_index = NETS.get(experiment_path.name, experiment_path.name[0])
        bits = ''.join(char for char in name if char.isdigit())
        return {
            'run': r' \( \boldsymbol{\widetilde{\mathcal{N}}_' + str(net_index) + '^{' + bits + '}} \\)',
            **extract_metrics(df_results),
        }

    df = pd.DataFrame(analyze(*item) for item in dfs.items())
    df = df.set_index('run')
    df.index.name = None

    FMT = '{:.5f}'.format  # precision

    df['nearby_mw'] = df.apply(lambda row: f'{FMT(row['nearby_mean'])} ({FMT(row['nearby_weighted_sum'])})', axis=1)
    df['union_mw'] = df.apply(lambda row: f'{FMT(row['union_mean'])} ({FMT(row['union_weighted_sum'])})', axis=1)

    columns = [
        'sample_max', 'nearby_max', 'union_max',
        'sample_mean', 'nearby_mw', 'union_mw',
        'integral_divided_sum']

    target_dir = experiment_path / 'common_outputs'
    target_dir.mkdir(parents=True, exist_ok=True)

    with open(target_dir / 'comparison.tex', 'w') as f:
        f.write(latex_table(df[columns].to_latex(float_format=FMT)))

    with open(target_dir / 'comparison.html', 'w') as f:
        f.write(wrap_html_tables([df.to_html(columns=columns, float_format=FMT)]))


def latex_table(table):
    content_rows = []

    for row in table.split('\n'):
        if row and row[0] != '\\':
            content_rows.append(row[1:])

    header = r'''\begin{tabular}{c|| c | c | c || c | c | c || c}
        & \multicolumn{3}{c||}{} & \multicolumn{3}{c||}{}  & Weighted\\[2pt]
        & \multicolumn{3}{c||}{\textbf{Maximum} Error over}
        & \multicolumn{3}{c||}{\textbf{(Weighted) Average} Maximum Error over}
        & \textbf{Integral}-Based\\[3pt]
        & Dataset & Polytopes & Extended & Dataset & Polytopes & Extended & Error over\\
        & & & Polytopes & & & Polytopes & Polytopes\\[2pt]
        & $\bm{E_T}$ & $\bm{E_{\Xi_T}}$ & $\bm{E_{\overline{\Xi}_T}}$ & $\bm{\overline{E}_T}$
        & $\bm{\overline{E}_{\Xi_T}}$~~$\Big(\bm{\overline{E}_{\Xi_T}^{\,\widetilde{d}}}\Big)$
        & $\bm{\overline{E}_{\overline{\Xi}_T}}$~~$\Big(\bm{\overline{E}_{\overline{\Xi}_T}^{\,\widetilde{d}}}\Big)$
        & $\bm{\overline{E}_{\Xi_T^E}^{\,\widetilde{d}}}$\\[5pt]
        \hline
    '''
    content = '[3pt]\n\\hline\\rule{0pt}{3.5ex}'.join(content_rows)
    return header + '% ' + content + '\n\\end{tabular}\n'


COL_SIZE = ('size', 'exact')


def evaluate_subsets(experiment_path: Path, run_id: str, error_scaling: float):
    """
    1. iterates over different cardinalities,
    2. chooses NUM_SUBSETS random subsets of a given cardinality,
    3. computes our metrics,
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


def get_subset_label(column):
    match column:
        case 'sample_max' | 'sample_mean':
            return 'over dataset'
        case 'nearby_max' | 'nearby_mean':
            return 'over polytopes'
        case 'nearby_weighted_sum':
            return 'weighted over polytopes'
        case 'union_max' | 'union_mean':
            return 'over ext. polytopes'
        case 'union_weighted_sum':
            return 'weighted over ext. polyt.'
        case 'integral_divided_sum':
            return 'over polytopes'
        case _:
            return column


def plot_subsets(experiment_path: Path, run_id: str, with_legend: bool):
    """plots the mean and std for different cardinalities (as computed by evaluate_subsets), groups metrics with similar properties"""
    subsets_dir = experiment_path / f'{run_id}_outputs' / 'subsets'
    df = pd.read_csv(subsets_dir / 'subsets.csv', header=[0, 1], index_col=0)

    groups = [
        ('max', 'maximum errors', ['sample_max', 'nearby_max', 'union_max']),
        ('weighted, mean', 'average errors', ['sample_mean', 'nearby_mean',
         'union_mean', 'nearby_weighted_sum', 'union_weighted_sum']),
        ('integral', 'integral-based error', ['integral_divided_sum']),
    ]

    for file_name, x_label, columns in groups:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        legend = []

        for column in columns:
            size = df.loc[:, COL_SIZE]
            mean = df.loc[:, (column, 'mean')]
            std = df.loc[:, (column, 'std')]
            label = get_subset_label(column)

            if tex := TEX_ALIASES.get(column):
                label = f'${tex}$ {label}'

            handle, = ax.plot(size, mean, '.-', label=label)
            ax.fill_between(size, mean-std, mean+std, alpha=0.2)
            legend.append({'label': label, 'handle': handle, 'last_value': mean.iloc[-1]})

        legend.sort(key=lambda item: item['last_value'], reverse=True)

        if with_legend:
            ax.legend([item['handle'] for item in legend], [item['label'] for item in legend])
        elif PRINT_COLORS:
            print_legend_colors(ax)

        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel('dataset cardinality')
        ax.set_ylabel(x_label + r' ($\mu\pm\sigma$)')
        fig.savefig(subsets_dir / f'{file_name}.pdf', bbox_inches='tight')
        plt.close(fig)


def plot_union_combined(experiment_path: Path, run_id: str, error_scaling: float, with_legend: bool):
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

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlabel('number of subpolytopes')
    ax.set_ylabel('average maximum errors')
    offset = 1
    ax.plot(ns[offset:], weighted[offset:], '.-', label=f'${TEX_ALIASES['union_weighted_sum']}$ weighted average')
    ax.plot(ns[offset:], means[offset:], '.-', label=f'${TEX_ALIASES['union_mean']}$ arithmetic average')
    ax.grid(True, linestyle='--', alpha=0.5)

    if with_legend:
        ax.legend()
    elif PRINT_COLORS:
        print_legend_colors(ax)

    fig.savefig(target_dir / f'combined.pdf', bbox_inches='tight')
    plt.close(fig)


def show_input_faces(experiment_path: Path, run_id: str, error_scaling: float):
    bundle = appmax.applications.DataBundle('utkface')
    model = bundle.load_model()
    model_approx = bundle.load_model()
    bits = int(''.join(char for char in run_id if char.isdigit()))  # hack to extract bits from run_id
    model_approx.round(bits=bits)
    samples_test = model.subset(bundle.data_split.test)
    results = appmax.experiment.load_batch_results(experiment_path, run_id)

    selected = [1849, 1096, 1222, 1397, 1779, 561,
                1775, 1198]
    results = [results[idx] for idx in selected]

    target_dir = experiment_path / f'{run_id}_outputs' / 'faces'
    target_dir.mkdir(parents=True, exist_ok=True)

    def x_to_img(x: torch.Tensor):
        return (x.movedim(0, -1) + 1) / 2

    def to_age(value):
        return bundle.data_split.metadata.scaler.inverse_transform(np.array([[value]])).item()

    for i, item in enumerate(results):
        xs: dict[str, torch.Tensor] = {}
        xs['original'] = item['result_sample']['x']
        xs['nearby'] = item['result_nearby']['x']
        xs['union'] = item['result_nearby']['union']['x']

        for name, x in xs.items():
            idx = selected[i]
            age_gold = to_age(samples_test[idx][1].item())
            age_model = to_age(model(x.unsqueeze(0)).item())
            age_approx = to_age(model_approx(x.unsqueeze(0)).item())

            img_np = (x_to_img(x).numpy() * 255).astype(np.uint8)
            img_pil = PIL.Image.fromarray(img_np)
            FACE_SIZE = 512
            MARGIN = 8
            img_pil = img_pil.resize((FACE_SIZE, FACE_SIZE), resample=PIL.Image.Resampling.NEAREST)
            draw = PIL.ImageDraw.Draw(img_pil)
            kwargs = {'fill': "white", 'stroke_width': 4, 'stroke_fill': "black", 'font_size': FACE_SIZE/8}

            draw.text((MARGIN, FACE_SIZE-MARGIN), str(round(age_model)), anchor='lb', **kwargs)

            if name == 'original':
                draw.text((MARGIN, MARGIN), str(round(age_gold)), anchor='lt', **kwargs)
            else:
                draw.text((FACE_SIZE-MARGIN, FACE_SIZE-MARGIN), str(round(age_approx)), anchor='rb', **kwargs)

            img_pil.save(target_dir / f'face_{i:04d}_{name}.png')

    df = pd.DataFrame(appmax.experiment.dict2flat(r) for r in results)
    df.loc[:, appmax.experiment.UNSCALED_COLS] *= error_scaling
    columns = ['sample_index', 'error_sample', 'error_nearby', 'union_error',
               'polytope_width', 'union_width', 'integral']
    df.to_csv(target_dir / 'faces.csv', columns=columns)


def plot_histograms(experiment_path: Path, run_id: str):
    df_results = load_df_results(experiment_path, run_id)

    target_dir = experiment_path / f'{run_id}_outputs' / 'histograms'
    target_dir.mkdir(parents=True, exist_ok=True)

    for col in df_results.columns:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set_xlabel(to_display_label(col))
        ax.set_ylabel('frequency')
        data = df_results[col]

        # we remove leading and trailing bins with counts 0 or 1
        counts, bin_edges = np.histogram(data, bins='auto')
        bins_gt_one = np.flatnonzero(counts > 1)
        first_gt_one = 0  # bins_gt_one.min()
        last_gt_one = bins_gt_one.max()
        limit_lower = bin_edges[first_gt_one]
        limit_upper = bin_edges[last_gt_one+1]
        outliers_lower = counts[:first_gt_one].sum()
        outliers_upper = counts[last_gt_one+1:].sum()

        if outliers_lower > 0:
            text = f'{outliers_lower} outliers ∈ [{bin_edges[0]:.2f}, {limit_lower:.2f})'
            fig.text(0.15, 0.83, text, ha='left', va='top')

        if outliers_upper > 0:
            text = f'{outliers_upper} outliers ∈ ({limit_upper:.2f}, {bin_edges[-1]:.2f}]'
            fig.text(0.87, 0.83, text, ha='right', va='top')
            # bottom corner -> y=0.2

        ax.hist(data, bins='auto', range=(limit_lower, limit_upper), histtype='stepfilled')
        fig.savefig(target_dir / f'{col}.pdf', bbox_inches='tight')
        plt.close(fig)


def check_len(experiment_path: Path, run_id: str, desired_len: int):
    results = appmax.experiment.load_batch_results(experiment_path, run_id)

    if len(results) != desired_len:
        raise ValueError(f'run {experiment_path.name}/{run_id} does not contain {desired_len} items')


hex_to_tab = {
    matplotlib.colors.to_hex(c): name for name, c in matplotlib.colors.TABLEAU_COLORS.items()
}


def print_legend_colors(ax):
    for handle, label in zip(*ax.get_legend_handles_labels()):
        color_name = hex_to_tab.get(matplotlib.colors.to_hex(handle.get_color())).replace('tab:', 'my')
        pretty_label = re.sub(r'\\|widetilde |\{|\}|\$', '', label).replace('overline', '‾')
        print(f"{color_name: <10}{pretty_label}")


def print_palette():
    for name, hex_val in list(matplotlib.colors.TABLEAU_COLORS.items()):
        latex_name = name.replace('tab:', 'my')
        rgb_str = ",".join(str(round(c*255)) for c in matplotlib.colors.to_rgb(hex_val))
        print(f"\\definecolor{{{latex_name}}}{{RGB}}{{{rgb_str}}}")


if __name__ == '__main__':
    main()
