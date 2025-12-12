#!/usr/bin/env bash

declare -A days=(
    [01]="uv run python day01.py"
    [02]="uv run python day02.py"
    [03]="uv run python day03.py"
    [04]="uv run --with "cudf-cu13==25.10.*" python day04.py"
    [05]="uv run python day05.py"
    [06]="uv run python day06.py"
    [07]="./build/day07"
    [08]="./build/day08"
    [09]="./build/day09"
    [10]="./build/day10"
    [11]="./build/day11"
    [12]="./build/day12"
    [04bonus]="uv run --with "cudf-cu13==25.10.*" python day04.py"
    [12bonus]="uv run --with "cudf-cu13==25.10.*" python day12.py"
)

IFS=$'\n' readarray -t sorted_days < <(printf "%s\n" "${!days[@]}" | sort)

for day in "${sorted_days[@]}"; do
    echo "Day ${day}:"
    ${days[${day}]}
    echo "--------------------------------"
done
