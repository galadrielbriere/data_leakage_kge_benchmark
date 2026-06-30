"""
kg_configs.py
=============
KG-specific configuration (BioKG / Hetionet / ShepherdKG). This is the ONLY
place that holds the paths, relation names, model list/labels and axis settings
specific to a KG. To add a KG, add an entry to ``CONFIGS``.

Note: adjust ``DATA_ROOT`` if the directory tree differs on another machine.
"""

from __future__ import annotations
from dataclasses import dataclass, field

# Common root of all experiments. Change it once here if needed.
DATA_ROOT = "/home/galadriel/dr_benchmark"


@dataclass
class KGConfig:
    name: str                       # 'biokg', 'hetionet', 'shepkg'
    target_freq_key: str            # YAML key of the target relation (frequency breakdown)
    directed_relations: dict        # output_metric -> {'keys': [...], 'title': '...'}
    figures_dir: str                # output directory for the figures
    dl_experiments: dict            # exp_key -> path (DL1 / DL2)
    coldstart_experiments: dict     # exp_key -> path (random split vs cold-start)
    model_order: list               # models to plot, in display order — REQUIRED, KG-specific
    ylims: dict = field(default_factory=dict)        # y_lim per figure
    # Optional display relabelling: {disk_name: shown_name}. Keys are the on-disk
    # model names (same as in model_order); only the plotted label changes, the
    # data is still read from the original directory. Models not listed keep their
    # disk name. Example: {"GCN_RESCAL": "GCN_RESCAL_meanIntra"}.
    model_labels: dict = field(default_factory=dict)
    # Optional: which radial tick VALUES get a number label, per figure (same
    # keys as ylims). All the delimitation rings stay drawn (one every 0.1 up to
    # ylim); this only hides the numbers you don't list. Each value is:
    #   - a list  -> label ONLY these values (e.g. [0, 0.5]); other rings stay,
    #                just without a number
    #   - missing -> label every ring (default behaviour)
    rgrid_labels: dict = field(default_factory=dict)

    def rlabels_for(self, fig_key):
        """Tick values to label on figure ``fig_key`` (None -> label all rings)."""
        return self.rgrid_labels.get(fig_key)


CONFIGS = {

    "biokg": KGConfig(
        name="biokg",
        target_freq_key="DRUG_DISEASE_ASSOCIATION",
        directed_relations={
            "drug_drug_mrr":       {"keys": ["DDI", "DDI_inv"], "title": "Drug-Drug MRR"},
            "protein_protein_mrr": {"keys": ["PPI", "PPI_inv"], "title": "Protein-Protein MRR"},
            "dpi_mrr":             {"keys": ["DPI"],            "title": "DPI MRR"},
            "drug_target_mrr":     {"keys": ["DRUG_TARGET"],   "title": "Drug-Target MRR"},
        },
        figures_dir=f"{DATA_ROOT}/Figures/BioKG",
        dl_experiments={
            "with_dl1":    f"{DATA_ROOT}/DL1experiment/withDL1/models_biokg",
            "without_dl1": f"{DATA_ROOT}/DL1experiment/withoutDL1/models_biokg",
            "dl2":         f"{DATA_ROOT}/DL2experiment/models_biokg",
        },
        coldstart_experiments={
            "random_split":        f"{DATA_ROOT}/DL1experiment/withoutDL1/models_biokg",
            "cold_start_drugs":    f"{DATA_ROOT}/DL3experiment/cold_start/models_drugs_biokg",
            "cold_start_diseases": f"{DATA_ROOT}/DL3experiment/cold_start/models_diseases_biokg",
        },
        ylims={"fig1": 1, "fig2": 0.9, "fig3_test_vs_inf": 0.6,
               "fig3_freq": 0.6, "directed": 1, "coldstart": 0.6},
        # Models plotted for THIS KG, in display order. Trim/reorder freely:
        # only the names listed here are drawn; anything found on disk but not
        # listed is reported and skipped (see radar_plots.order_models).
        model_order=[
            "GAT_RESCAL", "GCN_RESCAL", "GCN_RESCAL_maxIntra",
            "ANALOGY", "ComplEx", "HolE", "RESCAL", "DistMult", "TransE", "TransD",
        ],
        # Rename on the plot only (data stays in the original dir):
        model_labels={"GAT_RESCAL": "GAT", "GCN_RESCAL_maxIntra" : "GCNmax", "GCN_RESCAL": "GCNmean"}, 
        # Keep every ring, but write the number only on 0 and 0.5 (ylim=1 figs).
        rgrid_labels={"fig1": [0, 0.2, 0.4, 0.6, 0.8], "fig2": [0, 0.2, 0.4, 0.6, 0.8],
                      "fig3_test_vs_inf": [0, 0.2, 0.4], "fig3_freq": [0, 0.2, 0.4],
                      "directed": [0, 0.2, 0.4, 0.6, 0.8],  "coldstart":[0, 0.2, 0.4]},
    ),

    "hetionetkg": KGConfig(
        name="hetionet",
        target_freq_key="CtD",
        directed_relations={
            "drug_drug_mrr":           {"keys": ["CrC", "CrC_inv"], "title": "Drug-Drug MRR"},
            "disease_disease_mrr":     {"keys": ["DrD", "DrD_inv"], "title": "Disease-Disease MRR"},
            "gene_interacts_gene_mrr": {"keys": ["GiG", "GiG_inv"], "title": "Gene interacts Gene MRR"},
            "gene_covaries_gene_mrr":  {"keys": ["GcG", "GcG_inv"], "title": "Gene covaries Gene MRR"},
        },
        figures_dir=f"{DATA_ROOT}/Figures/HetionetKG",
        dl_experiments={
            "with_dl1":    f"{DATA_ROOT}/DL1experiment/withDL1/models_hetionet",
            "without_dl1": f"{DATA_ROOT}/DL1experiment/withoutDL1/models_hetionet",
            "dl2":         f"{DATA_ROOT}/DL2experiment/models_hetionet",
        },
        coldstart_experiments={
            "random_split":        f"{DATA_ROOT}/DL1experiment/withoutDL1/models_hetionet",
            "cold_start_drugs":    f"{DATA_ROOT}/DL3experiment/cold_start/models_drugs_hetionet",
            "cold_start_diseases": f"{DATA_ROOT}/DL3experiment/cold_start/models_diseases_hetionet",
        },
        ylims={"fig1": 0.6, "fig2": 0.6, "fig3_test_vs_inf": 0.4,
               "fig3_freq": 0.4, "directed": 0.5, "coldstart": 0.4},
        # Models plotted for THIS KG, in display order (trim/reorder per KG).
        model_order=[
            "GAT_RESCAL", "GCN_RESCAL", "GCN_RESCAL_maxIntra",
            "ANALOGY", "ComplEx", "HolE", "RESCAL", "DistMult", "TransE", "TransD",
        ],
        # Rename on the plot only (data stays in the original dir):
        # e.g. {"GCN_RESCAL": "GCN_RESCAL_meanIntra"}.
        model_labels={"GAT_RESCAL": "GAT", "GCN_RESCAL_maxIntra" : "GCNmax", "GCN_RESCAL": "GCNmean"}, 
        # Label only some rings, e.g. {"fig1": [0, 0.5]}; empty -> label all.
        rgrid_labels={"fig1": [0, 0.2, 0.4], "directed": [0, 0.2, 0.4], "fig2": [0, 0.2, 0.4],
                      "fig3_test_vs_inf": [0, 0.2], "fig3_freq": [0, 0.2], "coldstart": [0, 0.2]},
    ),

    "shepherdkg": KGConfig(
        name="shepkg",
        target_freq_key="indication",
        directed_relations={
            "drug_drug_mrr":       {"keys": ["drug_drug", "drug_drug_inv"],         "title": "Drug-Drug MRR"},
            "disease_disease_mrr": {"keys": ["disease_disease", "disease_disease_inv"], "title": "Disease-Disease MRR"},
            "protein_protein_mrr": {"keys": ["protein_protein", "protein_protein_inv"], "title": "Protein-Protein MRR"},
        },
        figures_dir=f"{DATA_ROOT}/Figures/ShepherdKG",
        dl_experiments={
            "with_dl1":    f"{DATA_ROOT}/DL1experiment/withDL1/models",
            "without_dl1": f"{DATA_ROOT}/DL1experiment/withoutDL1/models",
            "dl2":         f"{DATA_ROOT}/DL2experiment/models",
        },
        coldstart_experiments={
            "random_split":        f"{DATA_ROOT}/DL1experiment/withoutDL1/models",
            "cold_start_drugs":    f"{DATA_ROOT}/DL3experiment/cold_start/models_drugs",
            "cold_start_diseases": f"{DATA_ROOT}/DL3experiment/cold_start/models_disease",
        },
        ylims={"fig1": 0.7, "fig2": 0.5, "fig3_test_vs_inf": 0.4,
               "fig3_freq": 0.4, "directed": 0.8, "coldstart": 0.3},
        # Models plotted for THIS KG, in display order (trim/reorder per KG).
        model_order=[
            "GAT_RESCAL", "GCN_RESCAL", "GCN_RESCAL_maxIntra",
            "ANALOGY", "ComplEx", "HolE", "RESCAL", "DistMult", "TransE", "TransD",
        ],
        # Rename on the plot only (data stays in the original dir):
        # e.g. {"GCN_RESCAL": "GCN_RESCAL_meanIntra"}.
        model_labels={"GAT_RESCAL": "GAT", "GCN_RESCAL_maxIntra" : "GCNmax", "GCN_RESCAL": "GCNmean"}, 
        # Label only some rings, e.g. {"fig1": [0, 0.5]}; empty -> label all.
        rgrid_labels={"fig1": [0, 0.2, 0.4, 0.6], "directed": [0, 0.2, 0.4, 0.6]},
    ),
}
