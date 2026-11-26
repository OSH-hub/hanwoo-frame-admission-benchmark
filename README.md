# Freshness-aware frame admission benchmark for cattle monitoring on a shared GPU

Code, recorded logs, aggregate tables and figures for the article
"Freshness-Aware Frame Admission for Hanwoo Monitoring on a Shared GPU".

The benchmark replays prerecorded cattle video frames through a detector that shares one GPU
with a co-tenant process, and compares four frame-admission policies (FIFO, latest-pending,
drop-newest and periodic admission) by delivered latency, information age and frame loss.

The repository holds computing timestamps, admission and discard accounting, aggregate tables,
figures and analysis scripts. It holds no video, no video frame, no cattle image and no trained
model weight. The farm recordings are not redistributed; Cows2021 and the AI-Hub dataset are
available from their providers under their own terms.

## Contents

| Path | Contents |
|---|---|
| `tables/` | Aggregate tables as CSV (one file per table; see the index below) |
| `figures/` | Article Figures 1-4 and three additional figures of the historical and GPU 1 measurements |
| `reproducibility/` | Runners, protocols, recorded logs and analysis scripts, one folder per experimental stage |
| `requirements.txt` | Package versions of the analysis environment |

Measurements are organized by hardware and stage: historical RTX 4070 Ti records; the initial
replication and the single-execution source extension on one RTX 5090 (GPU 1); the repeated
benchmark and the admission-fraction grid on a second RTX 5090 (GPU 2); and the multi-camera
follow-up on GPU 2 with a synthetic and a training co-tenant.

## Tables and the article items they support

| File in `tables/` | Article item | Contents |
|---|---|---|
| `repeated_summary_80.csv` | Table 1, single-source rows | 12 source-policy summaries at 80 arrivals/s on GPU 2 |
| `detection_count_summary.csv` | Table 1, box-count and mismatch columns; Results, detection counts | Mean, minimum and maximum over three executions of the detection-count quantities per source-rate-policy condition |
| `multicamera_cells.csv` | Table 1, multi-camera rows; Results, multi-camera follow-up | 40 co-tenant-detector-load-policy cells |
| `historical_stage_timing.csv` | Table 2 | Stage and whole-frame timing, 200 frames, RTX 4070 Ti |
| `historical_stress_runs.csv` | Table 3 | Five historical stress runs |
| `initial_replication_summary.csv` | Table 4 | Initial RTX 5090 replication, six repetitions per condition |
| `repeated_all_arms.csv` | Figure 2 | All 72 repeated arms: counts, latency, age, arrival jitter, drainage and rate definitions |
| `repeated_summary.csv` | Results, repeated comparisons | 24 condition summaries at both arrival rates |
| `repeated_run_intervals.csv` | Methods, statistical analysis | 144 metric summaries with standard deviations and t intervals (n = 3, df = 2) |
| `repeated_run_differences.csv` | Methods, statistical analysis | 54 within-repetition latest-minus-control summaries and retained-count flags |
| `repeated_gpu_telemetry.csv` | Methods, repeated evaluation | GPU 2 telemetry samples |
| `admission_grid_all_arms.csv` | Figure 3 | All 36 grid arms with retained counts |
| `admission_grid_summary.csv` | Results, fixed-admission sensitivity | 12 fraction-policy summaries |
| `admission_grid_run_intervals.csv` | Methods, statistical analysis | 72 metric summaries with t intervals |
| `admission_grid_run_differences.csv` | Methods, statistical analysis | 24 latest-minus-control summaries and count-match flags |
| `admission_grid_gpu_telemetry.csv` | Methods, fixed-admission grid | GPU 2 telemetry samples of the grid |
| `detection_count_by_arm.csv` | Results, detection counts | One row per repeated arm |
| `multicamera_arms.csv` | Results, multi-camera follow-up | All 120 measured arms |
| `multicamera_calibration.csv` | Methods, multi-camera follow-up | 120-call service calibration per camera and detector |
| `historical_csv_json_consistency.csv` | Methods, historical live experiments | Delivered counts recomputed from frame records against the run summaries |
| `historical_moderate_load.csv` | Results, historical contention | Two arms at moderate load |
| `energy_short_runs.csv` | Results, historical processing and energy | Power and energy per frame for two numerical precisions |
| `chronological_simulation.csv` | Results, chronological simulation | FIFO and latest-pending on the seeded trace |
| `initial_replication_all_arms.csv` | Figure 4a; Table 4 | All 36 arms of the initial replication |
| `gpu1_extension_all_arms.csv` | Results, earlier source-block observations | 25 arms of the GPU 1 source extension (24 one-minute arms and the ten-minute arm) |
| `gpu1_ten_minute_arm.csv` | Results, ten-minute run | The ten-minute arm alone; do not add it to the previous file |
| `gpu1_retained_window_diagnostic.csv` | Results, temporal-coverage diagnostic | Retained-window fractions for windows of 1, 2, 4, 8 and 16 frames, with the equal-count control |
| `gpu1_extension_gpu_telemetry.csv` | Figure 4c | GPU 1 telemetry samples |
| `gpu1_extension_paired_differences.csv` | Results, earlier source-block observations | Source-matched policy contrasts |
| `gpu1_ten_minute_one_second_bins.csv` | Figure 4b | One-second series of the ten-minute run |

## Figures

| File in `figures/` | Article item | Source of plotted values |
|---|---|---|
| `Figure1` | Figure 1 | Schematic drawn from the Methods text (no data) |
| `Figure2` | Figure 2 | `tables/repeated_all_arms.csv` |
| `Figure3` | Figure 3 | `tables/admission_grid_all_arms.csv` |
| `Figure4` | Figure 4 | `tables/initial_replication_all_arms.csv`, `tables/gpu1_ten_minute_one_second_bins.csv`, `tables/gpu1_extension_gpu_telemetry.csv` |
| `historical_whole_frame_latency` | not in the article | `reproducibility/reanalyse.py` |
| `historical_stress_p99` | not in the article | `reproducibility/reanalyse.py` |
| `gpu1_extension_80` | not in the article | `reproducibility/analyse_extended.py` |

Figures 1-4 are drawn by one script, which reads the tables and recomputes nothing:

```text
python reproducibility/render_display_figures.py --tables tables --out new_figures
```

The analysis scripts below also write their own diagnostic figures under other file names;
these show the same values with embedded titles and are not the article figures.

## Requirements

```text
pip install -r requirements.txt
```

The repeated-benchmark and grid analyses were run with Python 3.11.9 and the pinned versions.
The multi-camera analysis needs only Python and NumPy.

## Recomputing the GPU 2 results from the stored logs

Copy `reproducibility/repeated_gpu2_study/` to a new scratch folder and run there; the output
folders must not exist beforehand.

```text
python analyze_repeated.py --input repeated_gpu2 --output fresh_repeated_analysis
python analyze_repeated.py --input frontier_gpu2 --output fresh_grid_analysis
python render_grid_distinct_markers.py --metrics fresh_grid_analysis/all_arm_metrics.csv --output fresh_grid_analysis/loss_latency_sensitivity
python analyse_detection_counts.py
```

These commands need only the stored timing, discard and telemetry logs. They verify completion
markers, the design records, configuration hashes, model and decoded-frame hashes, the accounting
of every frame, timestamp identities, serial service, the periodic admission sequence and the
absence of inference errors, and then recompute delivered-latency percentiles, common-window
information age and run-level means, ranges, standard deviations and t intervals. Output bytes
such as PDF metadata or the last decimal digit can differ between environments.

The repeated benchmark has 72 arms: three sources x two arrival rates (80 and 30 arrivals/s) x
four policies x three executions. All 80-arrival/s arms preceded all 30-arrival/s arms; the 12
source-policy cells were randomized within each rate and repetition with seed 20261026.
Totals: 237,600 arrivals = 220,515 deliveries + 17,085 discards. The grid has 36 arms: June NVR
source, 80 arrivals/s, offered fractions 0.50, 0.65, 0.80 and 0.95, three policies, three
executions: 172,800 arrivals = 116,811 deliveries + 55,989 discards. Both stages together:
410,400 arrivals = 337,326 deliveries + 73,074 discards, with no inference errors.

Rate definitions: `delivered_rate_to_last_completion_fps` divides the delivered count by the
time to the final completion; `completion_rate_within_offer_window_fps` counts completions up to
the last scheduled arrival; `bookkeeping_inclusive_run_rate_fps` includes file output and monitor
shutdown and is not an inference throughput. Latency percentiles include queue drainage;
information age is integrated over the initialized window common to all runs of a source-rate
block. The t intervals use the sample standard deviation and the two-sided 95% Student t
quantile with two degrees of freedom.

`analyse_detection_counts.py` first checks that every replay index returned one identical
detection count wherever it was delivered, then recovers the counts of discarded frames from that
reference. It writes into `fresh_detection_count_analysis/`; `tables/detection_count_by_arm.csv`
and `tables/detection_count_summary.csv` are copies of that output.

## Recomputing the multi-camera follow-up

From a scratch copy of `reproducibility/realistic_load_study/`:

```text
python analyze_realistic.py stage1_gpu2 stage1b_gpu2 stage2_gpu2 --out fresh_multicamera_analysis
```

The folder holds the protocol written before the measured arms (`protocol.md`), the runner
(`realistic_load.py`), the analysis script, the three run folders with every arrival, delivery,
discard and telemetry record, and the analysis output (`analysis_out`). The 120 measured arms
account for 442,800 arrivals = 370,471 deliveries + 72,329 discards. Stage 1 was interrupted by
an unrelated GPU process during design order 52; that arm is kept in `stage1_gpu2` and excluded
by the analysis script, and orders 52-96 were run in `stage1b_gpu2`. The smoke arms, console logs
and the weights of the training co-tenant are not included.

## Recomputing the earlier measurements

Copy the repository to a scratch folder. In its `reproducibility/` folder,
`python analyse_extended.py` regenerates the tables and diagnostic figures of the 25-arm GPU 1
source extension (127,200 arrivals = 115,014 deliveries + 12,186 discards), and
`python reanalyse.py` recomputes the historical tables, the chronological simulation and the
counterexample to the expression S + max(S - T, 0). Both scripts write into the copy's `tables/`
and `figures/` folders. `p2_live_v2/` and `p2_live_80fps/` hold the 36 arms (6,480 arrivals) of
the initial replication at 30 and 80 arrivals/s.

## Re-running inference

`repeated_gpu2.py`, `frontier_gpu2.py`, `extended_live.py`, `live_replication.py` and
`realistic_load.py` record how the measurements were run. Re-running them requires authorized
access to the source videos and the detector checkpoints, the recorded PyTorch and Ultralytics
versions, an unused GPU, and explicitly configured device, input and output paths. The clips
(158, 600 and 600 decoded frames) were replayed from RAM, and the unit of replication is a
computing run. `postflight_verification.json` records that the model and decoded-frame hashes
and the source sizes matched their recorded values after execution.

## Paths in this repository

Site-specific directories in logs, metadata and scripts are replaced by the placeholders
`<HOME>`, `<DATA_ROOT>` and `<PROJECT_ROOT>`; recording file names are kept for provenance.
In the two runners `repeated_gpu2_study/repeated_gpu2.py` and `repeated_gpu2_study/frontier_gpu2.py`
the two directory strings were replaced as well, so their SHA-256 differs from the value frozen in
`config.json` before the runs; `analyze_repeated.py` prints a note instead of stopping. No other
line of the runners was changed. The numerical tables and per-frame timing records keep their
recorded values.

## Scope

The NVR recordings were provided by one Korean Hanwoo farm for research use with the farm
owner's consent; the Cows2021 clip is public. The endpoints are delivered latency, information
age, frame loss, temporal coverage and changes in detector box counts; the multi-camera
follow-up additionally measures per-camera delivery.

## License

MIT (see `LICENSE`). The license covers the code, logs and tables of this repository and does
not extend to the source videos, images or model weights, which are not part of it.
