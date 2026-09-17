# Processed Stage-1.2 data

`records.parquet` is a local convenience aggregate and is intentionally excluded from Git because it exceeds GitHub's 100 MiB per-file limit. The committed per-model/per-seed raw Parquet files are the complete authoritative records; rerunning this analyzer reconstructs the aggregate exactly. `condition_summary.parquet` and `training_log.parquet` are committed.
