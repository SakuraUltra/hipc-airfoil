# Historical presentation data

These records transcribe two tables from a separately supplied historical
presentation. They supplement the raw logs in the parent directory. No new
simulation or GPU performance measurement was made to produce these records.
The original presentation remains private because it contains identifying
material. Only data images without identifying material are included here.

## Retained sources

| Source slide | Retained image | Structured transcription |
| --- | --- | --- |
| 18 | [Original problem-size chart](problem-size-chart.png) | Use the more precise values in the slide 20 table |
| 19 | [Summary table](summary-table.png) | [summary.csv](summary.csv), 13 records |
| 20 | [Problem-size table](problem-size-table.png) | [problem_sizes.csv](problem_sizes.csv), 12 records |

The images are lossless copies of the presentation's embedded PNGs, without
content edits. The manifest records each image's media part, slide number,
SHA-256 and the original presentation's SHA-256. Both CSV transcriptions were
independently checked against the visible source tables.

## What the numbers mean

The summary table labels its durations `Real_Time_Seconds` and
`User_Time_Seconds`. It reports a serial baseline of 378.57 seconds, an optimised
serial duration of 309.81 seconds, and CUDA `real` time of **46.33 seconds**.
`reported_speedup` and `reported_efficiency_percent` preserve the presentation's
claims at their displayed precision; they are not newly validated controlled
comparisons. In particular, the reported CUDA 8.17× ratio is not used as a
measured CPU/GPU speedup for the maintained implementation.

The presentation's ratios use its common **378.57 s serial baseline**. For
example, it reports 11.01× for 16-thread OpenMP and 12.01× for 16-process MPI.
The raw-log CPU scaling CSVs instead divide each campaign's own one-worker
duration by its multi-worker duration. Different denominators explain these
different ratios; do not mix the two definitions.

The problem-size table reports separate total durations, including CUDA
**22.04 seconds** at 1024×256 and **3954.56 seconds** at 4096×1024. Its timing
boundaries and exact binaries, settings and per-size step counts are not
available. The companion chart labels its GPU **Tesla T4**. The summary table
labels its row `CUDA(Tesla 4)`; no independent hardware record is available.
This hardware description does not independently identify the device that
generated the separately retained raw `cuda_steps.txt` event log.

Keep these three CUDA observations distinct:

- **46.290664 seconds:** the raw log's `46290.664 ms` CUDA event interval.
- **46.33 seconds:** process `real` duration reported in the slide 19 summary.
- **22.04 seconds:** the slide 20 problem-size table's 1024×256 duration.

The presentation does not explain the difference between its two CUDA records.
There is no evidence here establishing a measured code-version speedup between
them. Missing raw shell timing logs and run metadata also prevent proving that
each presentation row uses an identical measurement boundary.

The serial baseline and CPU entries in the summary table remain presentation
records. Rounded or truncated values must not replace the millisecond-precision
CPU logs, and the two evidence sets must not be pooled as controlled repeats.
The generated figures label their source and keep these campaigns separate.

## Transcription conventions

- Preserve each displayed timing and ratio exactly; do not silently recompute or
  correct presentation values.
- `reported_user_seconds` is reported CPU user time, not elapsed time. It may
  exceed process `real` time in a parallel run.
- The summary's CUDA efficiency `N/A` becomes an empty CSV field. CPU worker
  units follow the table's core/thread/process descriptions; CUDA uses `gpu`.
- The size category `Extra_Large` is normalised to `ExtraLarge` in the CSV.
- `reported_time_per_step` preserves the source table's displayed per-step
  values. The source column does not state the unit. These values are not used
  in the performance figures or to reconstruct exact step counts.
- GPU worker count is left empty in the problem-size CSV because that table
  does not state the count; CPU configurations explicitly use 16 workers.

`scripts/check_historical.py` checks retained hashes, source-image references,
coverage and structural consistency. It does not extract or verify text from
PNG images. Manual image-to-CSV checks establish the transcription; matching
hashes keep those checked files identifiable.
