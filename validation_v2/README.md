# M100 Schedule Explorer (validation v2)

## What it shows

1. **Measured GPU power** — sum of `Gpu{0–3}_power_usage` from `ganglia_pub`
2. **Modeled fleet GPU power** from `job_table` via `server.py`
3. **Job schedule Gantt** time-aligned below
4. **Stage charts** (training / fine-tuning / inference)

Facility `Tot_ict` is not shown — this page validates GPU power only.

## Setup

```bash
cd "/Users/ptawia3/Desktop/Dr-Hu/it-power"
python validation_v2/prepare_slice.py
python validation_v2/app.py
# → http://localhost:5051/
```

Edit `validation_v2/slice_config.json` for the day window (`grid_seconds`, `substep_seconds`). Modeled power is averaged over `substep_seconds` (default 30) within each reporting bin so checkpoint/eval periods are not aliased to the 300 s grid. Cache: `validation_v2/slice_cache/day_slice.json`.
