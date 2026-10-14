#!/bin/bash
#SBATCH --job-name=omp_vtk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --partition=nodes
#SBATCH --output=logs/vtk_%j.out

export OMP_NUM_THREADS=16

rm -rf out/airfoil_16
mkdir -p out/airfoil_16

echo "OMP 16线程 VTK输出 (每100步)"
./airfoil_omp_optimized -x 1024 -y 256 -o out/airfoil_16/airfoil -c -f 100

echo "完成！文件数: $(ls out/airfoil_16/ | wc -l)"
