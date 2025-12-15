#!/bin/bash

# Convert Jupyter notebooks to Python scripts
conda activate et-precip-ppe 2>/dev/null || true
jupyter nbconvert --to python ../notebooks/*.ipynb --output-dir figures --PythonExporter.format=percent
