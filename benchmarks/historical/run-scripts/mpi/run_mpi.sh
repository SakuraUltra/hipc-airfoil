#!/bin/bash

#SBATCH --job-name=airfoil_mpi03_scaling
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=mpi_scaling_%j.out

module load OpenMPI

cd "${SLURM_SUBMIT_DIR:-.}"

mkdir -p out
mkdir -p logs

# 循环运行不同数量的 MPI 进程
for np in 1 2 4 8 16
do
    echo "========================================="
    echo "--- Running with $np MPI processes ---"
    echo "========================================="
    
    # 运行并计时
    time mpirun -np $np ./airfoil_mpi -t 2 -f 100 -n 2>&1 | tee logs/mpi_${np}proc.log
    
    echo ""
    echo "--- Finished $np processes ---"
    echo ""
done

echo "========================================="
echo "--- All Jobs Finished ---"
echo "========================================="
