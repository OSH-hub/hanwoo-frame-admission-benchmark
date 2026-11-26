# Protocol: multi-camera native-rate load, heavier detector and training co-tenant (P2)

Written on 15 November 2026 before any measured arm of this experiment. The runner records the SHA-256 of this file. Earlier results (26 October 2026) are known, so this is a follow-up design recorded before its measured arms, not a preregistration. All measured arms are retained and reported, including unfavourable ones. Earlier results are not changed.

## Why

In the earlier benchmark, overload occurred only at 80 artificial arrivals/s from one source, a rate that no single camera produces. The farm recorder stores five channels; the November recording has a container frame rate of about 15 frames/s. This experiment asks whether overload arises under loads a farm installation can actually present:

1. several cameras, each at its native rate (15 arrivals/s) or at 30 arrivals/s, feeding one detector;
2. one camera with a heavier cattle-trained detector at 1280-pixel input;
3. a real detector-training job as the GPU co-tenant instead of synthetic matrix multiplication.

It also runs on a different day from the earlier benchmark and randomises load levels within each repetition, so load level is no longer confounded with block order.

## Fixed settings

- Host and GPU: the same Linux host and the same physical RTX 5090 (GPU 2) as the earlier repeated evaluation. The GPU must have no unrelated compute process at start; compute processes are sampled about every two seconds, and a run stops (data retained) if an unrelated process appears.
- Software: the environment of the earlier repeated evaluation (PyTorch 2.11.0+cu128, Ultralytics 8.4.87); versions are recorded.
- Cameras: channels ch02, ch01, ch03, ch04, ch05 (in this order) of the 1 November 2025 recording. The first 600 decoded frames of each channel are resized to 640 × 360 and replayed from RAM in a loop. K cameras means the first K of this order. The ch02 block is the "November source" of the earlier benchmark; its decoded-frame hash is checked against the earlier record.
- Detector A (as before): the YOLOv8n cattle checkpoint, input size 640, batch one, FP32, confidence 0.25, IoU 0.7, at most 300 detections; checkpoint hash checked against the earlier record.
- Detector B (heavier): the cattle-trained YOLOv8x checkpoint trained at 1280-pixel input (`x1280_best.pt`), input size 1280, same other settings, on ch02 frames resized to 1280 × 720.
- Arrival schedule: camera c of K, frame i, is scheduled at (i + c/K)/r seconds, r being the per-camera rate. Arrival times are artificial replay times, not camera timestamps.
- One inference worker. Policies, generalised to several cameras:
  - `fifo`: one global unbounded first-in-first-out queue.
  - `latest`: one pending slot per camera; a new arrival replaces that camera's pending frame. The worker serves the pending frame with the earliest arrival time.
  - `drop_newest`: one pending slot per camera; an arrival is discarded if that camera's slot is occupied. Same service order.
  - `periodic`: deterministic per-camera admission with fraction min(1, 0.95/(K · r · mean service seconds)), then the global FIFO queue. The mean service time comes from a 120-call calibration per camera (per detector) under the co-tenant at the start of each stage and stays fixed.
- Each arm: 60 seconds of arrivals; pending work drains before the arm ends. No per-arm warm-up. Every scheduled frame gets a producer timestamp and either a completion record or a discard reason.
- Seed 20261115 for randomisation.

## Stages and arms

Stage 1, synthetic co-tenant (continuous 16384 × 16384 FP32 matrix multiplication, as before), 96 arms:

- Detector A: (K, r) ∈ {(1, 15), (3, 15), (5, 15), (1, 30), (3, 30), (5, 30)} × 4 policies × 3 repetitions = 72 arms (aggregate 15, 45, 75, 30, 90 and 150 arrivals/s).
- Detector B: K = 1, r ∈ {15, 30} × 4 policies × 3 repetitions = 24 arms.
- Within each repetition all 32 cells are run in one random order.

Stage 2, training co-tenant, 24 arms:

- The co-tenant is a separate process that trains a YOLOv8n detector from random initialisation on the local cattle training images (input 640, batch 16, eight data-loader workers, automatic mixed precision, no validation, no plots) on the same GPU. Measurement starts after the training process has completed at least one epoch. Its weights are discarded; its epoch log is kept as evidence that training was running.
- Detector A: (K, r) ∈ {(5, 15), (3, 30)} × 4 policies × 3 repetitions, random order within repetition.

Before the measured stages, a short smoke run (five-second arms) checks accounting, timestamps and error handling. Smoke arms are not results and tune nothing. A dry mode with a simulated service time is used only to test queue logic.

## Outcomes

Per arm: delivered and discarded counts by reason, delivered-latency percentiles (p50, p95, p99, maximum), mean service time, percentage of delivered frames above 33 ms and above the per-camera frame interval, maximum queue length, and the time-average information age per camera (time since the arrival of that camera's most recently completed frame) over the offer window after each camera's first completion. Summaries are the mean and range over the three executions of a cell. No hypothesis test is planned. The experimental unit is one execution.

## Interpretation limits

Replayed frames from one recording day; one host and one GPU; artificial arrival times; no annotated events, so no detection-accuracy claim; the training co-tenant is one configuration of one training job. Results describe this implementation under these loads.
