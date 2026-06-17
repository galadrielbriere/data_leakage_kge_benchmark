# Radar plots — Reproducing the paper's figures

The original six notebooks (3 KGs × 2 figure sets) are replaced by
**2 notebooks + 2 modules**. All the duplicated code (radar factory, YAML
parsing, plotting functions) now lives in a single library, and all KG-specific
settings in a single config.

## Files

| File | Role |
|---|---|
| `radar_plots.py` | Shared library: `radar_factory`, `parse_model_metrics`, `plot_radar`, `plot_radar_3way`. |
| `kg_configs.py` | Per-KG config (`biokg` / `hetionet` / `shepkg`): paths, target-relation key, made-directed breakdown, `y_lim`, output directories. |
| `make_radar_plots.ipynb` | DL1 / DL2 / DL3 figures + made-directed relations. **Set `KG` at the top, Run All.** |
| `radar_random_vs_coldstart.ipynb` | Random-split vs cold-start figure. Same principle. |

## Usage

In each notebook, only one thing needs to change:

```python
KG = "shepkg"   # or "hetionet" / "biokg"
```

then *Run All*. The notebooks must sit in the same directory as `radar_plots.py`
and `kg_configs.py` (otherwise add that directory to `sys.path`).

> Adjust `DATA_ROOT` at the top of `kg_configs.py` if the directory tree differs
> on another machine. Figures are written under `Figures/<KG>` (e.g.
> `Figures/BioKG`, `Figures/HetionetKG`, `Figures/ShepherdKG`).

## Adding a model

As soon as a new model directory exists on disk
(with `evaluation_metrics.yaml` + `inference_metrics.yaml`), it appears
automatically on the radars.

To pin its position on the radar instead of leaving it at the end of the list,
add its name to `model_order` in `kg_configs.py`:

```python
model_order=[
    "ANALOGY", "ComplEx", "HolE", "RESCAL", "DistMult", "TransE", "TransD",
    "MY_MODEL_1", "MY_MODEL_2",     # <-- here
]
```

With 9 models, the bottom labels may overlap slightly (the original fine-tuning
was calibrated for 7). To adjust, pass `special_offsets` to `radar_factory`,
e.g. `radar_factory(9, special_offsets={6: -0.08, 7: -0.06})`.

