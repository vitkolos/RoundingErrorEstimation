#!/bin/bash

set -ueo pipefail

python="${1:-uv run}"
prefix="$python -m appmax.visualization"
echo "passing commands to '$prefix'"

california="california 4bit 6bit 8bit"
year="year 4bit 6bit 8bit"
utkface="utkface 8bit 12bit"

for dataset_runs in "$california" "$year" "$utkface"; do
    $prefix check-2000 $dataset_runs
    $prefix comparison $dataset_runs
    $prefix cardinalities $dataset_runs
    $prefix histograms $dataset_runs
    $prefix union-combined $dataset_runs
done

$prefix input-face $utkface

# copy files, create archive

temp_dir="$(mktemp -d)"
trap "rm -r $temp_dir" EXIT

for dataset in california year utkface; do
    dir="$temp_dir/$dataset"
    mkdir -p $dir
    cp -r experiments/$dataset/*_outputs $dir
done

outputs_zip="experiments/outputs.zip"
outputs_zip_full="$(pwd)/$outputs_zip"
cd $temp_dir
rm -f $outputs_zip_full
zip -rq $outputs_zip_full .
echo "outputs saved into $outputs_zip"
