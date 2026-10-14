# Historical benchmark evidence

This directory preserves small text records from the original implementation,
including slower runs, profiling limitations, and speculative notes. These are
archived observations, not measurements of the maintained code in this repository.
No new cluster benchmark was run while assembling this archive.

## Four separate scaling campaigns

The values below are the shell `time` **real** durations, in seconds. Each cell
contains one historical observation. The records show a 1024 × 256 global grid
and a final step count of 2098, reaching simulated time 2.00080872. OpenMP workers
are threads; MPI workers are processes.

| Backend and campaign | 1 worker | 2 workers | 4 workers | 8 workers | 16 workers | Published log |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| OpenMP, primary | 332.667 | 175.806 | 89.735 | 55.545 | 34.389 | [omp_scaling_27579401.out](omp_scaling_27579401.out) |
| OpenMP, additional | 399.462 | 195.044 | 103.288 | 56.529 | 41.682 | [strong_scaling_27404398.out](additional/strong_scaling_27404398.out) |
| MPI, primary | 417.738 | 214.875 | 110.360 | 58.958 | 31.535 | [mpi_scaling_27437194.out](mpi_scaling_27437194.out) |
| MPI, additional | 416.107 | 215.570 | 111.666 | 57.632 | 30.173 | [mpi_test_27232003.out](additional/mpi_test_27232003.out) |

[strong_scaling.csv](strong_scaling.csv) contains the primary campaigns;
[additional_scaling.csv](additional_scaling.csv) contains the additional campaigns.
Speedup and efficiency in each CSV use that campaign's own one-worker duration.
The primary campaigns remain identifiable because they were the first records
summarised; the label does not mean they are more representative or better runs.

**Do not average campaigns together, treat them as controlled repeats, or combine
the fastest cells into a new scaling curve.** The archive does not establish that
the executable, allocation, compiler, placement, or machine state was identical
between campaigns. Repetitions, warm-ups, uncertainty, and complete hardware
metadata were not recorded. Retaining all four campaigns exposes this variation
instead of selecting a favourable result.

## Configuration and timing limits

The archived default configuration is a 1024 × 256 grid, domain 4 × 1, NACA 2412,
end time 2.0, Reynolds number 500, safety factor 0.5, and adaptive timestepping.
The pressure solver uses at most 100 SOR sweeps per step, tolerance 0.001, and
relaxation factor 1.7. The sampled progress lines show a timestep of approximately
0.000953674316. These defaults and progress records support the workload
description; they do not reconstruct missing executable provenance.

- The available OpenMP scaling script launches `airfoil_omp_optimized` without
  disabling output. The archived source's defaults enable a final VTK result and
  disable checkpoints. Its shell timing can therefore include final output,
  startup, allocation, and console logging. The script is useful configuration
  evidence, but it cannot prove the invocation of every separate campaign.
- The available MPI scaling script requests `-t 2 -f 100 -n` and times a launcher
  pipeline that also captures stdout. The archived MPI main routine has no final
  result-writing call. Historical executable hashes and a complete build record
  are missing, so the exact output behaviour of each logged executable is not
  independently established. Do not assume MPI and OpenMP timings have identical
  I/O overhead.
- The archived OpenMP Makefile builds `airfoil_omp`, whereas the scaling script
  launches `airfoil_omp_optimized`. An additional log reports that the optimized
  target was already up to date rather than showing a fresh compilation. The
  retained Makefiles are historical configuration evidence, not proof of the
  compiler version or flags used for each measured executable.
- Scheduler job wall times include the surrounding job and are distinct from
  the per-invocation shell timings above. Diagnostic output jobs are not extra
  compute-only scaling samples.

### CUDA timing

[cuda_steps.txt](cuda_steps.txt) records **46290.664 ms** under the original label
`Total GPU time`, with the same reported final step count and simulated time.
The original implementation places CUDA events around its simulation loop,
after setup and initial host-to-device upload, and before final result download
and VTK output. The interval spans host-driven convergence checks, transfers,
stream idle gaps, and in-loop logging; enabled checkpoints would also occur
inside that interval. It is not a sum of kernel execution times or a full-process
wall-clock duration.

Consequently, do not divide a CPU shell duration by this CUDA event duration to
claim a controlled CPU-to-GPU speedup. The retained CUDA launch script names a
toolkit module, and its Makefile specifies a compilation architecture. Neither
identifies the physical GPU, driver, or exact executable used for this record.

## Contents and evidence quality

- `profiles/serial/` preserves the original serial gprof reports. In
  `analysis.txt` and `analysis_opt.txt`, Poisson accounts for roughly 95% of
  sampled execution. Their cumulative sampled totals, 362.67 and 309.70 seconds,
  are **not wall-clock timings**. The accompanying serial build configurations
  also change optimization flags, so differences cannot be attributed solely to
  an algorithm change. `analysis_opt_pro.txt` reports a different trajectory
  ending at step 8389 and must not be pooled with the 2098-step workload.
- `profiles/openmp/` preserves reports that attribute almost all sampled time to
  `frame_dummy`. This attribution limits their usefulness for hotspot analysis;
  it does not establish that this symbol is the true numerical bottleneck.
- `diagnostics/openmp/` and `diagnostics/mpi/` retain progress, decomposition,
  output-generation, and scheduler records. Matching printed residuals are
  useful observations but do not replace a numerical comparison of full fields.
  Repeated or identical stdout files are preserved under their original aliases;
  duplicate content is not an independent performance measurement.
- `notes/affinity_results.txt` is an analysis note rather than raw timing output.
  Its reported 40.218/35.827/35.853/35.562-second affinity results lack supporting
  raw run logs in the available archive. Its **26.20-second, 13.84×, and 86.5%**
  combined result is explicitly a prediction, not a measurement. The causal
  explanations in that note were not independently established. None of these
  values is included in the measured scaling CSVs.
- `run-scripts/{openmp,mpi,cuda}/` and
  `build-config/{serial,openmp,mpi,cuda}/` preserve the available launch and build
  settings. They are archived evidence, not commands to execute blindly.

Large VTK datasets, executables, and object files are intentionally excluded.

## Privacy edits and provenance

[manifest.json](manifest.json) maps each archived file to its original
archive-relative path and records original and published SHA256 hashes, byte
counts, and privacy edits. Files with matching hashes are byte-identical;
different hashes identify documented redacted copies. The numerical data and
performance lines in the original logs are unchanged.

All 38 original-file hashes were checked directly against the supplied source
ZIP when assembling this archive. Twenty-five retained files are byte-identical;
13 contain only the listed privacy edits. The two CSV summaries are derived
files and have their own retained hashes.

Personal and environment-identifying metadata has been removed. Launch-script
account directives are omitted, personal working-directory changes use
`SLURM_SUBMIT_DIR`, and output destinations use relative paths. These edits make
the published scripts privacy-safe examples; they do not recover the historical
allocation or guarantee a matching rerun environment. The archive checker
verifies retained file hashes and the primary and additional timing tables.
Original hashes provide a reference for checking a separately held source
archive; that source archive is not required by the checker.

## A fair follow-up comparison

1. On the intended cluster, retain baseline and candidate source separately and
   record their revisions or hashes. Build both with the same compiler, flags,
   libraries, and numerical settings. If reconstructing a baseline from source,
   label it as a reconstruction rather than the unidentified historical binary.
2. Use the same allocation, CPU/GPU model, thread or process placement, global
   grid, timestep policy, end condition, and logging/output policy. Explicitly
   distinguish a compute-only run from a checkpoint/output workload. For a
   historical-style run, start with the documented 1024 × 256, end-time-2.0
   workload; do not compare it with a 50-step smoke test.
3. Validate the final fields and stopping behaviour before interpreting speed.
   Warm up each configuration, then run at least five interleaved measurements
   without competing benchmark jobs. Save every command, run log, environment
   record, and timing; report the median and spread. Use a common end-to-end wall
   timing boundary across backends and report CUDA-event timing separately.
4. Choose cluster optimizations using those **cluster measurements**. The separate
   Mac experiment only showed that the original Poisson structure beat a newly
   added persistent-region candidate on that Mac's short fixed-step workload.
   It did not compare Mac hardware with cluster hardware and does not determine
   which implementation is fastest on the cluster. The maintained OpenMP code
   retains the original core structure pending a controlled cluster comparison.
