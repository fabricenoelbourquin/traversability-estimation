# traversability-estimation

Pipeline for computing traversability metrics from rosbags and overlaying them on georeferenced maps.

## Layout
- `config/` – YAML configs (topics, paths, map)
- `src/` – scripts (extract, sync, metrics, plotting)
- `data/` – working data (ignored by git; symlinks to external SSD recommended)
- `reports/` – figures/artefacts (ignored by git)

Data lives on an external SSD. Code should read a DATA_ROOT (env var) or `config/paths.yaml`.
