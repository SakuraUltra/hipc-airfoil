#!/bin/bash

# Load CUDA module
module load CUDA/12.1.1

# Create output directory if it doesn't exist
mkdir -p output
mkdir -p out

# Run the CUDA airfoil simulation
echo "Running CUDA airfoil simulation..."
./airfoil "$@"

echo ""
echo "Simulation complete!"
echo "Results:"
echo "  - Step data: output/cuda_steps.txt"
echo "  - VTK output: out/airfoil.vtr"
