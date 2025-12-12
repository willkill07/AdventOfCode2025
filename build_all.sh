#!/usr/bin/env bash

source /opt/nvidia/hpc_sdk/activate

cmake -B build
cmake --build build -j
