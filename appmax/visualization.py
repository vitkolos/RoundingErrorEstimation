from pathlib import Path
import typing

import pandas as pd
import matplotlib.pyplot as plt


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
