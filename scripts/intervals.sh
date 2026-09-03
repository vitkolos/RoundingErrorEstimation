#!/bin/bash

set -ueo pipefail

python="${1:-uv run}"
prefix="$python -m appmax.intervals"
echo "passing commands to '$prefix'"

target="experiments/intervals.txt"
rm -f $target

# $prefix california -b 4 >> $target
# $prefix california -b 6 >> $target
$prefix california -b 8 >> $target

# $prefix year -b 4 >> $target
# $prefix year -b 6 >> $target
# $prefix year -b 8 >> $target

$prefix utkface -b 6 >> $target
# $prefix utkface -b 8 >> $target
# $prefix utkface -b 12 >> $target
