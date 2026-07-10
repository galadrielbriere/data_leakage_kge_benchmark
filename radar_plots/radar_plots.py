"""
radar_plots.py
==============
Shared library to generate the MRR radar plots of the drug-repurposing
benchmark (BioKG / Hetionet / ShepherdKG).

Everything that used to be duplicated across the per-KG notebooks lives here:
    - radar_factory       : matplotlib "radar" projection (regular polygon)
    - parse_model_metrics : reads the evaluation/inference YAML files -> MRR dict
    - plot_radar          : 1 or 2 curves on a radar (experiment / metric comparison)
    - plot_radar_3way     : N curves on a radar (random split vs cold-start)

All KG-specific details are described by a ``KGConfig`` object (see
``kg_configs.py``). This library contains no hard-coded paths.

Adding a model
--------------
Only the models listed in ``KGConfig.model_order`` are plotted (whitelist).
To add a model, put its name in ``model_order`` once its results are complete.
Model directories found on disk but absent from ``model_order`` are reported
(never dropped silently) and skipped, so a half-finished or partially-pulled
model does not sneak into the figures. Pass ``include_extra=True`` to the plot
functions to also show those unlisted models, appended at the end.

Label / legend spacing
----------------------
Model names are kept off the polygon edge by a uniform *radial* padding
(``label_pad``, in points), applied via ``tick_params`` for any ``num_vars``.
The legend position is controlled by ``legend_anchor`` and can be turned off
per-axis with ``show_legend=False`` (useful when assembling a multi-panel
figure with a single shared ``fig.legend``).
"""

from __future__ import annotations

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D, ScaledTranslation

# NOTE: there is intentionally no default model list here. The model order is
# KG-specific and lives in ``KGConfig.model_order`` (kg_configs.py), which is the
# single source of truth. Always pass ``cfg.model_order`` to the functions below.

# Base metrics always extracted from the YAML files.
BASE_METRICS = [
    "global_mrr", "made_directed_relations", "target_relations",
    "target_frequent", "target_infrequent", "inference_mrr", "validation_mrr",
]


def _as_float(value):
    """Return ``value`` as a float, or None if it isn't numeric.

    Guards the parsing against stray non-numeric entries (e.g. a metric stored
    as text in a YAML or CSV file), which would otherwise crash the max() over
    runs with a float-vs-str comparison error.
    """
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else x


# ---------------------------------------------------------------------------
# 1. Radar projection (regular polygon)
# ---------------------------------------------------------------------------
def radar_factory(num_vars, frame="polygon", special_offsets=None,
                  default_offset=0.06, label_pad=0, side_pad=0.0):
    """Register and return the 'radar' projection for ``num_vars`` axes.

    ``label_pad`` (points) is the *radial* padding applied to every model label
    via ``tick_params``. This is the primary spacing mechanism and keeps the
    labels off the polygon edge for any ``num_vars`` (no per-N hand tuning).

    Left/right labels are kept off the polygon primarily by *angle-based
    alignment*: each label is anchored so its text grows outward (left labels
    right-aligned, right labels left-aligned, top/bottom centred). This is
    automatic and needs no parameter, and -- unlike ``set_position()`` -- it is
    not reset when the axes is drawn.

    ``side_pad`` is an OPTIONAL extra continuous horizontal push (in points),
    weighted by how sideways the label is (0 at top/bottom, max on the sides).
    It is applied as a transform offset (so it also survives redraw). Raise it
    only if the alignment alone is not enough; typical range 4-12.

    ``special_offsets`` / ``default_offset`` only drive an optional fine vertical
    nudge of individual labels and are OFF by default (``default_offset=-0.001``,
    ``special_offsets=None`` -> {}). Set them only if you need to micro-adjust a
    specific label after ``label_pad`` has done the bulk of the work.
    ``special_offsets`` is a dict {label_index: vertical_offset}.
    """
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    if special_offsets is None:
        special_offsets = {}

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            angles_deg = np.degrees(theta)
            self.set_thetagrids(angles_deg, labels)
            # Base radial padding (survives redraw; this is the supported knob).
            self.tick_params(axis="x", pad=label_pad)
            fig = self.figure
            for idx, (angle, label) in enumerate(zip(angles_deg, self.get_xticklabels())):
                label.set_fontsize(14)
                # Direction of the label relative to the centre. Zero is at "N"
                # and theta increases counter-clockwise, so:
                #   x_dir = -sin(angle)  -> +1 on the right, -1 on the left
                #   y_dir =  cos(angle)  -> +1 at the top,   -1 at the bottom
                ang = np.deg2rad(angle)
                x_dir = -np.sin(ang)
                y_dir = np.cos(ang)
                eps = 0.1
                # Anchor each label so its text grows OUTWARD from the polygon.
                # This is what actually clears the left/right labels, and unlike
                # set_position() it is not reset when the axes is drawn.
                label.set_horizontalalignment(
                    "left" if x_dir > eps else "right" if x_dir < -eps else "center")
                label.set_verticalalignment(
                    "bottom" if y_dir > eps else "top" if y_dir < -eps else "center")
                # Optional extra continuous horizontal push, in points, weighted
                # by how "sideways" the label is (0 at top/bottom). Implemented as
                # a transform offset so it survives redraw.
                if side_pad:
                    label.set_transform(
                        label.get_transform()
                        + ScaledTranslation(side_pad * x_dir / 72.0, 0, fig.dpi_scale_trans))
                # Optional fine vertical nudge: inactive while default_offset /
                # special_offsets are empty.
                if default_offset or special_offsets:
                    # Leave the top / bottom labels (0 / 180 / 360 deg) untouched.
                    if abs(angle) < 1 or abs(angle - 180) < 10 or abs(angle - 360) < 1:
                        continue
                    offset = special_offsets.get(idx, default_offset)
                    pos = label.get_position()
                    label.set_position((pos[0], pos[1] + offset))

        def _gen_axes_patch(self):
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            raise ValueError(f"unknown value for 'frame': {frame}")

        def draw(self, renderer):
            if frame == "polygon":
                for gl in self.yaxis.get_gridlines():
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                spine = Spine(axes=self, spine_type="circle",
                              path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes)
                return {"polar": spine}
            raise ValueError(f"unknown value for 'frame': {frame}")

    register_projection(RadarAxes)
    return theta


# ---------------------------------------------------------------------------
# 2. Metric parsing
# ---------------------------------------------------------------------------
def _parse_one_model(eval_file, inference_file, cfg):
    """Read the MRR values of a single model (one directory) -> {metric: value}."""
    out = {}

    with open(eval_file, "r") as f:
        data = yaml.safe_load(f)

    out["global_mrr"] = data.get("Global_MRR", 0)
    out["made_directed_relations"] = data.get("made_directed_relations", {}).get("Global_MRR", 0)
    out["target_relations"] = data.get("target_relations", {}).get("Global_MRR", 0)

    freq = data.get("target_relations_by_frequency_10", {}).get(cfg.target_freq_key, {})
    out["target_frequent"] = freq.get("Frequent_MRR", 0)
    out["target_infrequent"] = freq.get("Infrequent_MRR", 0)

    # Breakdown of the "made-directed" relations (KG-specific).
    ind = data.get("made_directed_relations", {}).get("Individual_MRRs", {})
    for metric, spec in cfg.directed_relations.items():
        keys = spec["keys"]
        out[metric] = sum(ind.get(k, 0) for k in keys) / len(keys)

    with open(inference_file, "r") as f:
        out["inference_mrr"] = yaml.safe_load(f).get("Inference MRR", 0)

    csv_file = os.path.join(os.path.dirname(eval_file), "training_metrics.csv")
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if "Validation MRR" in df.columns:
            # Coerce to numeric so a stray text value doesn't turn the whole
            # column (and thus .max()) into a string.
            out["validation_mrr"] = pd.to_numeric(df["Validation MRR"], errors="coerce").max()
        else:
            out["validation_mrr"] = -1

    return out


def parse_model_metrics(experiments, cfg, runs=("run1", "run2")):
    """Parse every model of every experiment.

    ``experiments`` : dict {exp_key: path_to_experiment_directory}
    Returns : {exp_key: {run: {model: {metric: value}}, 'best': {model: {...}}}}
    'best' is the max over all runs.
    """
    all_metrics = BASE_METRICS + list(cfg.directed_relations)
    models_data = {}

    for key, exp_dir in experiments.items():
        models_data[key] = {}

        for run in runs:
            run_path = os.path.join(exp_dir, run)
            models_data[key][run] = {}
            for dirpath, _, filenames in os.walk(run_path):
                if "evaluation_metrics.yaml" in filenames and "inference_metrics.yaml" in filenames:
                    model = os.path.basename(dirpath)
                    eval_file = os.path.join(dirpath, "evaluation_metrics.yaml")
                    inf_file = os.path.join(dirpath, "inference_metrics.yaml")
                    models_data[key][run][model] = _parse_one_model(eval_file, inf_file, cfg)

        # Flag empty runs so a silently-missing run is visible: when one run has
        # no models, 'best' quietly collapses to the other run (or to nothing).
        empty_runs = [run for run in runs if not models_data[key][run]]
        if empty_runs:
            present = [run for run in runs if models_data[key][run]]
            if present:
                print(f"[radar_plots] '{key}': empty run(s) {empty_runs} -> "
                      f"'best' computed from {present} only.")
            else:
                print(f"[radar_plots] '{key}': all runs empty -> 'best' is empty.")

        models_data[key]["best"] = {}
        seen = set()
        for run in runs:
            seen |= set(models_data[key][run])
        for model in seen:
            best = {}
            for metric in all_metrics:
                vals = [models_data[key][run].get(model, {}).get(metric, 0) for run in runs]
                # Keep only values that are actually numeric: a non-numeric
                # entry (e.g. text in a YAML/CSV field) would otherwise break
                # the comparison done by max().
                nums = [x for x in (_as_float(v) for v in vals) if x is not None]
                best[metric] = max(nums) if nums else 0
            models_data[key]["best"][model] = best

    return models_data


def _ring_labels(rgrid, rgrid_labels):
    """Labels for the radial rings.

    All rings in ``rgrid`` are kept (the delimitations stay drawn); only the
    values listed in ``rgrid_labels`` get a number, the others get "".
    ``rgrid_labels=None`` -> every ring is labelled (default behaviour).
    """
    if rgrid_labels is None:
        return None
    wanted = [float(v) for v in rgrid_labels]
    return [f"{r:g}" if any(np.isclose(r, w) for w in wanted) else "" for r in rgrid]


def order_models(present, model_order=None, include_extra=False, verbose=True):
    """Order the models for display.

    By default only the models listed in ``model_order`` are kept (whitelist):
    this avoids pulling in half-finished model directories that happen to be on
    disk. Models discovered on disk but absent from ``model_order`` are reported
    (never dropped silently) so nothing disappears without notice.

    Set ``include_extra=True`` to also plot the discovered-but-unlisted models,
    appended at the end in alphabetical order.

    ``model_order`` is required: pass ``cfg.model_order`` from kg_configs.py
    (there is no built-in default — the config is the single source of truth).
    """
    if model_order is None:
        raise ValueError(
            "model_order is required — pass cfg.model_order from kg_configs.py."
        )
    present = list(present)
    ordered = [m for m in model_order if m in present]
    extra = sorted(m for m in present if m not in model_order)
    if extra and verbose:
        if include_extra:
            print(f"[radar_plots] models not in model_order, appended at the end: {extra}")
        else:
            print(f"[radar_plots] found on disk but NOT in model_order, skipped: {extra}")
    return ordered + (extra if include_extra else [])


# ---------------------------------------------------------------------------
# 3. Plot: 1 or 2 curves (experiment or metric comparison)
# ---------------------------------------------------------------------------
def plot_radar(results, experiment1, run, metric1,
               experiment2=None, metric2=None,
               y_lim=0.7, title="Radar Plot", legend_labels=None,
               colors=("red", "blue"), font_size=14,
               model_order=None, model_labels=None, rgrid=None, rgrid_labels=None,
               markers=False, label_pad=0, side_pad=0.0,
               show_legend=True, legend_anchor=(0.5, -0.18),
               savefile=None, savedir=None, saveformat="svg", show=True, ax=None,
               panel_label=None, legend_ncol=2, include_extra=False):
    """Plot one or two curves on a radar.

    - experiment comparison : pass ``experiment2`` (``metric2`` = ``metric1``)
    - metric comparison     : pass ``metric2`` (``experiment2`` = ``experiment1``)
    - single curve          : neither ``experiment2`` nor ``metric2``

    Spacing controls
    ----------------
    ``label_pad``     : radial padding (points) of the model labels.
    ``side_pad``      : optional extra horizontal push (in points) for the
                        left/right labels (see ``radar_factory``). The side
                        labels are already cleared by angle-based alignment;
                        raise this only if you want them pushed further out.
    ``show_legend``   : set False to draw no per-axis legend (e.g. when a single
                        shared ``fig.legend`` is added on a multi-panel figure).
    ``legend_anchor`` : ``bbox_to_anchor`` of the per-axis legend.
    """
    if experiment2 and experiment2 != experiment1 and metric2 is None:
        comparison_type, metric2 = "exp", metric1
    elif experiment2 is None and metric2 and metric2 != metric1:
        comparison_type, experiment2 = "metric", experiment1
    elif experiment2 is None and metric2 is None:
        comparison_type, experiment2, metric2 = "single", experiment1, None
    else:
        raise ValueError("Invalid combination of arguments (experiment2 / metric2).")

    for exp in {experiment1, experiment2}:
        if exp not in results:
            raise ValueError(f"Experiment '{exp}' not found in the results.")
        if run not in results[exp]:
            raise ValueError(f"Run '{run}' not found in experiment '{exp}'.")

    models = order_models(results[experiment1][run], model_order, include_extra=include_extra)
    N = len(models)
    theta = radar_factory(N, frame="polygon", label_pad=label_pad, side_pad=side_pad)

    values_1 = [results[experiment1][run][m][metric1] for m in models]
    values_2 = ([results[experiment2][run][m][metric2] for m in models]
                if comparison_type != "single" else None)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="radar"))
    else:
        fig = ax.figure
    ax.set_ylim(0, y_lim)

    if rgrid is None:
        rgrid = np.arange(0, y_lim, 0.1)
    ax.set_rgrids(rgrid, labels=_ring_labels(rgrid, rgrid_labels),
                  angle=np.degrees(theta[N - 1]), ha="center",
                  color="grey", fontsize=font_size)
    ax.set_title(title, ha="center", fontsize=font_size + 2, weight="bold", pad=15)

    if comparison_type == "exp":
        leg1, leg2 = (legend_labels or (experiment1, experiment2))
    elif comparison_type == "metric":
        leg1, leg2 = (legend_labels or (metric1, metric2))
    else:
        leg1 = legend_labels[0] if legend_labels else f"{experiment1} - {metric1}"
        leg2 = None

    ax.plot(theta, values_1, color=colors[0], label=leg1)
    if markers:
        ax.scatter(theta, values_1, color=colors[0], alpha=0.8, s=40, marker="v")
    if values_2 is not None:
        ax.plot(theta, values_2, color=colors[1], label=leg2)
        if markers:
            ax.scatter(theta, values_2, color=colors[1], alpha=0.8, s=40, marker="o")

    # Relabel for display only: data lookups above use the on-disk model names;
    # model_labels maps disk_name -> displayed_name (others shown unchanged).
    display_models = [model_labels.get(m, m) for m in models] if model_labels else models
    ax.set_varlabels(display_models)
    for label in ax.get_xticklabels():
        label.set_fontsize(font_size)

    lines, labels = ax.get_legend_handles_labels()
    if show_legend and labels:
        ax.legend(lines, labels, loc="lower center", bbox_to_anchor=legend_anchor,
                  ncol=legend_ncol, fontsize=font_size)

    if panel_label is not None:
        ax.text(-0.1, 1.05, panel_label, transform=ax.transAxes,
                fontsize=font_size + 4, fontweight="bold", va="top", ha="right")

    if savedir is not None:
        if savefile is None:
            if comparison_type == "exp":
                savefile = f"radar_{experiment1}_vs_{experiment2}_{metric1}_{run}"
            elif comparison_type == "metric":
                savefile = f"radar_{experiment1}_{metric1}_vs_{metric2}_{run}"
            else:
                savefile = f"radar_{experiment1}_{metric1}_{run}"
        os.makedirs(savedir, exist_ok=True)
        full_path = os.path.join(savedir, f"{savefile}.{saveformat}")
        fig.tight_layout()
        fig.savefig(full_path, format=saveformat, bbox_inches="tight")
        print(full_path)

    if show:
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# 4. Plot: N curves (random split vs cold-start)
# ---------------------------------------------------------------------------
def plot_radar_3way(results, experiments, run, metric,
                    legend_labels, colors, linestyles=None,
                    model_order=None, model_labels=None,
                    y_lim=0.3, rgrid=None, rgrid_labels=None, title="", font_size=14,
                    label_pad=0, side_pad=0.0, ax=None, panel_label=None, include_extra=False):
    """Plot N curves (typically 3) on a polygonal radar."""
    if linestyles is None:
        linestyles = ["-"] * len(experiments)

    models = order_models(results[experiments[0]][run], model_order, include_extra=include_extra)
    N = len(models)
    theta = radar_factory(N, frame="polygon", label_pad=label_pad, side_pad=side_pad)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="radar"))
    else:
        fig = ax.figure

    ax.set_ylim(0, y_lim)
    if rgrid is None:
        rgrid = np.arange(0, y_lim, 0.1)
    ax.set_rgrids(rgrid, labels=_ring_labels(rgrid, rgrid_labels),
                  angle=np.degrees(theta[N - 1]), ha="center",
                  color="grey", fontsize=font_size)
    ax.set_title(title, ha="center", fontsize=font_size + 2, weight="bold", pad=15)

    for exp, lab, col, ls in zip(experiments, legend_labels, colors, linestyles):
        values = [results[exp][run].get(m, {}).get(metric, 0) for m in models]
        ax.plot(theta, values, color=col, linestyle=ls, linewidth=2, label=lab)

    # Display-only relabel (disk_name -> displayed_name); data uses disk names.
    display_models = [model_labels.get(m, m) for m in models] if model_labels else models
    ax.set_varlabels(display_models)
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(font_size)

    if panel_label is not None:
        ax.text(-0.05, 1.05, panel_label, transform=ax.transAxes,
                fontsize=font_size + 6, fontweight="bold", va="top", ha="left")

    return fig, ax