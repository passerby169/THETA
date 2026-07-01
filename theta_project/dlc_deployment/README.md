# THETA DLC Deployment

This directory is the cloud-training deployment package for THETA. It is not a copy of the THETA research codebase. The canonical model code remains in `theta_project/THETA/`; this package keeps only the container entrypoint and deployment notes needed by Alibaba Cloud PAI-DLC.

## Files

| File | Purpose |
|------|---------|
| `dlc_entry.py` | Entry script executed inside the DLC container: data conversion, preprocessing, training, visualization, logs, and backend callback. |
| `README_zh.md` | Chinese deployment guide for OSS layout, DLC mounts, environment variables, and release steps. |

## Runtime Layout

The backend syncs the deployment bundle to OSS before submitting a DLC job:

```
oss://<bucket>/code/dlc_entry.py
oss://<bucket>/code/src/models/       # from theta_project/THETA/src/models/
oss://<bucket>/code/scripts/          # from theta_project/THETA/scripts/
oss://<bucket>/code/services/         # backend compatibility helpers
oss://<bucket>/models/
oss://<bucket>/raw_data/
oss://<bucket>/results/
```

DLC mounts these OSS prefixes in the container:

```
/mnt/code
/mnt/models
/mnt/raw_data
/mnt/results
```

The DLC command is:

```bash
python /mnt/code/dlc_entry.py
```

## Backend Integration

The backend submits jobs through `theta_project/services/dlc_service.py` or the provider implementation in `theta_project/services/providers/aliyun.py`. The sync helper is `theta_project/utils/oss_util.py::sync_theta_project_to_oss`.

Before submitting a job, run the sync helper so `dlc_entry.py` and the current THETA model code are both present under `oss://<bucket>/code/`.
