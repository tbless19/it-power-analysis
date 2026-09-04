# Correlate config.json with Ganglia on your machine

This cloud VM cannot see your Downloads folder. Run this **locally**.

Your layout matches what the script expects:

```
dataset=main_datasets/
  year_month=22-03/plugin=ganglia_pub/
  year_month=22-03/plugin=job_table/
  year_month=22-04/ …
  year_month=22-09/ …
```

## Command

From the `it-power-analysis` repo (use the real path to `dataset=main_datasets`):

```bash
pip install -r validation/requirements.txt -r rl_tune/requirements.txt

python -m rl_tune.run_local_correlation \
    --root "/full/path/to/dataset=main_datasets"
```

That walks every `year_month=*`, loads **one day per month** (the 15th; next days if that date is empty), and writes:

- `rl_tune/results_local/config_variables_vs_ganglia.md`
- `rl_tune/results_local/config_variables_vs_ganglia.csv`
- plots (`config_vars_vs_ganglia.png`, …)

Optional:

```bash
# only some months, 20th of each month, rebuild caches
python -m rl_tune.run_local_correlation \
    --root "/full/path/to/dataset=main_datasets" \
    --months 22-03 22-04 22-05 22-06 22-07 22-09 \
    --day 20 \
    --force
```

A full month of raw Ganglia is huge (~20 s × ~980 nodes). One day per month is enough to rank config knobs; do not pass a whole month unless you have a lot of RAM/time.
