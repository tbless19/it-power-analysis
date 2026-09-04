# Local ExaData for Ganglia correlation

This cloud agent **cannot read files on your laptop**. Copy the bundle into
this repo’s `data/` folder (gitignored) or set `EXADATA_ROOT`.

## Layout

```
data/dataset=main_datasets/
  year_month=22-03/
    plugin=ganglia_pub/part-000.parquet   # or part-000-*.parquet
    plugin=job_table/part-000.parquet
  year_month=22-04/
    ...
```

`validation/exadata.py` also accepts `data/` already holding `year_month=*`.

## Run on your machine

```bash
export EXADATA_ROOT="/absolute/path/to/dataset=main_datasets"
python -m rl_tune.correlate_params
```

That writes `docs/config_variables_vs_ganglia.md` from **your** Ganglia, not the
cached 2022-03-20 slice.

Force a rebuild after replacing parquet files:

```bash
EXADATA_FORCE=1 python -m rl_tune.correlate_params
```

## Give the files to the cloud agent

Do **not** git-commit parquet (already gitignored). Any of:

1. Attach `ganglia_pub` + `job_table` parquet for at least one month in chat
2. Share the Drive folder as **Anyone with the link → Viewer**, then say to retry
3. Open this repo in Cursor Desktop on the machine that already has `data/`
