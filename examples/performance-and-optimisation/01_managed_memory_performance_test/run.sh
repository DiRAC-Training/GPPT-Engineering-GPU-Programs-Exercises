#!/usr/bin/env bash

nvcc -O2 -o managed_example main.cu && ./managed_example
