# Validation v3

Compare **measured Tot_ict** vs **simulated GPU fleet power** (+ editable non-GPU baseline) on a **1-day** slice.

## Run

```bash
cd "/Users/ptawia3/Desktop/Dr. Hu/it-power"
python validation_v3/app.py
# → http://localhost:5052/
```

## Configure the day

Edit `validation_v3/slice_config.json` (`t0` / `t1`). Default: `2022-03-15` → `2022-03-16`.

Cache: `validation_v3/slice_cache/day_slice.json`
