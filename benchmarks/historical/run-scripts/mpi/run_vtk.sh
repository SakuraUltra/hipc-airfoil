#!/bin/bash
#SBATCH --job-name=mpi03_vtk
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=00:30:00
#SBATCH --partition=nodes
#SBATCH --output=logs/vtk_%j.out

module load OpenMPI

rm -rf out/airfoil_16
mkdir -p out/airfoil_16
echo "Cleaned output directory out/airfoil_16"

echo "MPI03 16进程 VTK输出 (每100步)"
mpirun -np 16 ./airfoil_mpi -x 1024 -y 256 -o out/airfoil_16/airfoil -c -f 100

echo "完成！文件数: $(ls out/airfoil_16/ | wc -l)"
