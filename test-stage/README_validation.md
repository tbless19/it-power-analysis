# IT Power — Validation overlay

Overlays **measured** M100 cluster power against **modeled** power computed by
feeding the dataset's own job variables through the `server.py` formula:

    P(t) = Σk Nk·Pk_max [ ηk(t)·Σm(um·αm(t)) + (1−ηk(t))·ρk ]

## Files
- `exadata.py` — reads the Hive-partitioned cleaned bundle (jobs + measured power).
- `validate_server.py` — Flask app (port 5001). Imports the model engine from
  `server.py` unchanged; adds `/api/windows` and `/api/validate`.
- `validate.html` — overlay UI (measured vs modeled + residual + MAPE/energy).
- `config.json` — gained a `validate` block (cluster constants, default source).

`server.py`, `config.json`, `exadata.py`, `validate_server.py`, `validate.html`
must all sit in the same folder.

## Run
```bash
conda activate <env>        # needs flask, pandas, pyarrow, numpy
# point at the data folder, the dataset dir, or set it in the UI:
export EXADATA_ROOT="/path/to/data"        # parent holding dataset=main_datasets
#   or  .../data/dataset=main_datasets
python validate_server.py
# open http://localhost:5001
```
Root resolution is tolerant: give it the `data` parent, the `dataset=<name>`
dir (any name — main_dataset / main_datasets), or a dir already holding
`year_month=*`. The resolved path is shown in the UI and is editable.

## How it works
1. **Jobs → model tasks.** `job_table` rows overlapping the chosen window are
   read; `num_nodes × gpus_per_node (4)` gives V100 device count; stage is
   assigned by the existing `replay.stage_thresholds`.
2. **Modeled curve.** The formula is evaluated on a wall-clock grid
   (`grid_seconds`, default 60 s) with full-cluster inventory (V100 = 3920).
   `draws > 1` averages independent rng passes to damp stochastic noise.
3. **Measured curve.** A cluster-total power signal, resampled onto the same
   grid. Which signals are available depends on the plugins present that month:
   - `logics_ict` — `Tot_ict` facility IT total (kW). **Default, and the only
     power signal in the current cleaned bundle** (months 22-03…22-09 ship
     job_table, logics_pub, nagios_pub, slurm_pub, vertiv_pub).
   - `ganglia_gpu` — Σ `Gpu*_power_usage` (W→kW). Only if `ganglia_pub` exists.
   - `ipmi_total` — Σ `total_power` per node (W→kW). Only if `ipmi_pub` exists.
   The UI offers only the sources whose plugin is present for the picked month.
4. **Metrics.** Power MAPE (target < 15 %) and energy error (target < 10 %),
   plus bias and a residual trace.

## Notes / next iterations
- **Scope mismatch to keep in mind:** `Tot_ict` is *facility IT total* — GPUs
  **plus** CPUs, memory, network — while the current formula models V100 GPU
  power only. Expect the modeled curve to sit systematically below `Tot_ict`
  (a roughly constant non-GPU offset). Options: add a CPU/base-power term to
  the model, fit/subtract a baseline, or compare against a GPU-only signal once
  `ganglia_pub` is in the bundle.
- `job_table` has no `num_gpus`; all allocated nodes are assumed to carry 4
  V100s at the modeled stage utilisation. A partition filter (GPU partitions
  only) would tighten this — easy to add in `exadata.load_jobs`.
- Parquet reads use timestamp/metric predicate pushdown; keep windows to hours,
  not whole months, for ganglia (~20 s cadence × ~980 nodes).