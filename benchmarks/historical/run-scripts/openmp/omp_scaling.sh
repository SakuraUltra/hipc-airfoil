#!/bin/bash
#SBATCH --job-name=omp_scaling
#SBATCH --output=logs/omp_scaling_%j.out
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G

# Go to working directory
cd "${SLURM_SUBMIT_DIR:-.}"

# Thread counts to test
THREADS=(1 2 4 8 16)

# Run tests for each thread count
for NTHREADS in "${THREADS[@]}"
do
    echo "========================================="
    echo "--- Running with $NTHREADS OpenMP threads ---"
    echo "========================================="
    
    export OMP_NUM_THREADS=$NTHREADS
    
    echo "Running with $NTHREADS OpenMP threads"
    echo "Global domain: 1024 x 256"
    
    # Run the program with time measurement
    time ./airfoil_omp_optimized
    
    echo ""
    echo "--- Finished $NTHREADS threads ---"
    echo ""
done

echo "========================================="
echo "--- All Jobs Finished ---"
echo "========================================="
