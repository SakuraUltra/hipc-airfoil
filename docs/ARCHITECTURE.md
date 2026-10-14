# Architecture

This project parallelises an existing serial CFD implementation. It advances a
two-dimensional incompressible-flow calculation around a four-digit NACA
airfoil. [Provenance](PROVENANCE.md) distinguishes the archived serial reference,
parallel implementations and subsequent maintenance.

## Solver stages and data

Each timestep performs the same stages in every implementation:

1. Select a CFL-limited timestep, unless a fixed `--del-t` was supplied.
2. Compute tentative velocity fields `f` and `g` from `u` and `v`.
3. Form the pressure equation's right-hand side from their divergence.
4. Solve pressure with red/black successive over-relaxation (SOR), checking the
   residual after every complete two-colour sweep.
5. Correct velocities using the pressure gradient, then apply inlet, outlet,
   wall and obstacle boundary conditions.
6. Optionally emit checkpoints; write the final state unless `--noio` is set.

The CPU arrays are contiguous allocations addressed as `field[i][j]`, with `j`
contiguous in memory. Two extra entries per dimension hold boundary/ghost data.
Obstacle flags select fluid cells and encode neighbouring-fluid directions.
The CUDA implementation uses the equivalent flat offset `i * (ny + 2) + j`.

The unusual velocity-correction bounds are deliberately inherited: `u` uses
`1 <= i < nx-2`, `1 <= j < ny-1`; `v` uses `1 <= i < nx-1`,
`1 <= j < ny-2`. The MPI version applies these bounds in **global** coordinates.
Keeping them avoids silently changing the reference discretisation.
Agreement between implementations does not establish physical accuracy,
mesh convergence, pressure convergence at every timestep, or suitability for
engineering design.

## Execution variants

| Variant | Parallel work and ordering |
| --- | --- |
| Serial | Executable numerical comparison baseline. The optimised version hoists repeated arithmetic and visits alternate SOR cells directly. |
| OpenMP | Work-sharing loops distribute independent cells. Sum reductions compute pressure norms/residuals; maximum reductions implement CFL checks. |
| Optimised OpenMP | Loop collapse and static scheduling distribute independent cells. Shared parallel regions combine the tentative-velocity, velocity-update and CFL loops, using `nowait` where the arrays are independent. Each SOR colour retains a separate work-sharing region; its completion barrier orders red, black and residual evaluation. |
| MPI | X-directed slabs divide global columns among ranks, with one ghost column on each side. Blocking `MPI_Sendrecv` exchanges neighbouring columns; no communication/computation overlap is claimed. |
| CUDA | Device-resident flat arrays feed custom stencil and reduction kernels. Consecutive lanes vary `j` for contiguous accesses. Separate kernel launches in one stream order the red and black sweeps. |

MPI distributes remainder columns to the first ranks. Its SOR colour uses
`global_i_offset + local_i + j`, so odd offsets retain the same global
checkerboard. Pressure halos are exchanged after each colour; velocity halos
are exchanged around boundary updates. Obstacle rules from an east ghost cell
also update the final owned horizontal-velocity face.

`MPI_Allreduce` combines fluid counts, pressure norms and residuals; the minimum
local CFL timestep becomes the global timestep. Narrow slabs, including a
single owned column, use the same global bounds. More ranks than X cells are
rejected. Output gathers owned columns and the two physical edge columns with
`MPI_Gatherv`; rank 0 writes one file and broadcasts output status. This also
works with one rank, but requires rank 0 to hold the complete output arrays and
is not a scalable parallel-I/O design.

## CUDA trade-offs and current limits

Block-local shared-memory reductions produce partial sums, followed by a second
GPU reduction; workspaces are allocated before the timed loop. Out-of-domain
lanes contribute neutral values and still participate in block barriers. Scalar
convergence results return to the host, so the pressure solve retains frequent
host/device synchronization.

Obstacle boundary rules can write overlapping locations. A compact obstacle
list is processed by one GPU lane in the original order to preserve the serial
semantics. This is a correctness-first bottleneck, not a fully parallelised
boundary algorithm. Transfers for initialisation and requested output remain
necessary; there is no multi-GPU path, communication overlap, or tuned
architecture-specific kernel claim.

The current CUDA source has passed compilation with the CUDA 12.8.1 NVCC toolchain in CI.
Numerical execution on an NVIDIA GPU remains unverified, so revised performance
cannot yet be assessed. Historical CUDA output demonstrates an earlier
execution, not runtime verification of the current kernels. See [validation](VALIDATION.md) and
[benchmark methodology](BENCHMARKS.md) for the recorded checks and timing scope.

## Output representation

The shared ASCII VTK writer and MPI gather writer export ghost-inclusive raw
arrays as `PointData` fields `uv` and `p`, with spacing `delx` and `dely`.
The final coordinate therefore extends one grid spacing past the physical
`xlength`/`ylength`. This is an index-aligned diagnostic view of the stored
staggered fields, not a collocated reconstruction of physical velocity and
pressure. Full-precision output supports automated field comparisons; changing
the VTK association and coordinates does not change the solver state.
