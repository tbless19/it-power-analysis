# M100 IT Power Validation Upload App

Run:

```bash
pip install flask pandas pyarrow numpy
python validation_server.py
```

Open:

```text
http://localhost:5050
```

Upload:

- `logics_pub/part-000.parquet` for measured `Tot_ict`
- `job_table/part-000.parquet` for model replay
- optional `slurm_pub/part-000.parquet`
- optional `ganglia_pub/part-000.parquet`

Fix included:

- Reads only required flat columns from Parquet files.
- Avoids nested/list columns that can trigger `Repetition level histogram size mismatch`.
- Uses overlapping job-window filtering.
