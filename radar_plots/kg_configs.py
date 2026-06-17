"""
kg_configs.py
=============
KG-specific configuration (BioKG / Hetionet / ShepherdKG). This is the ONLY
place that holds the paths and relation names specific to a KG. To add a KG,
add an entry to ``CONFIGS``.

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
    dl_experiments: dict            # exp_key -> path (DL1 / DL2 / zero-shot)
    coldstart_experiments: dict     # exp_key -> path (random split vs cold-start)
    ylims: dict = field(default_factory=dict)        # y_lim per figure
    model_order: list = field(default_factory=lambda: [
        "ANALOGY", "ComplEx", "HolE", "RESCAL", "DistMult", "TransE", "TransD",
    ])


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
            "zero_shot":   f"{DATA_ROOT}/DL3experiment/cold_start/models_biokg",
        },
        coldstart_experiments={
            "random_split":        f"{DATA_ROOT}/DL1experiment/withoutDL1/models_biokg",
            "cold_start_drugs":    f"{DATA_ROOT}/DL3experiment/cold_start/models_drugs_biokg",
            "cold_start_diseases": f"{DATA_ROOT}/DL3experiment/cold_start/models_diseases_biokg",
        },
        ylims={"fig1": 0.81, "fig2": 0.6, "fig3_test_vs_inf": 0.3,
               "fig3_freq": 0.2, "directed": 0.9, "coldstart": 0.3},
    ),

    "hetionet": KGConfig(
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
            "zero_shot":   f"{DATA_ROOT}/DL3experiment/cold_start/models_hetionet",
        },
        coldstart_experiments={
            "random_split":        f"{DATA_ROOT}/DL1experiment/withoutDL1/models_hetionet",
            "cold_start_drugs":    f"{DATA_ROOT}/DL3experiment/cold_start/models_drugs_hetionet",
            "cold_start_diseases": f"{DATA_ROOT}/DL3experiment/cold_start/models_diseases_hetionet",
        },
        ylims={"fig1": 0.4, "fig2": 0.4, "fig3_test_vs_inf": 0.5,
               "fig3_freq": 0.5, "directed": 0.6, "coldstart": 0.3},
    ),

    "shepkg": KGConfig(
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
            "zero_shot":   f"{DATA_ROOT}/DL3experiment/cold_start/models",
        },
        coldstart_experiments={
            "random_split":        f"{DATA_ROOT}/DL1experiment/withoutDL1/models",
            "cold_start_drugs":    f"{DATA_ROOT}/DL3experiment/cold_start/models_drugs",
            "cold_start_diseases": f"{DATA_ROOT}/DL3experiment/cold_start/models_disease",
        },
        ylims={"fig1": 0.81, "fig2": 0.6, "fig3_test_vs_inf": 0.4,
               "fig3_freq": 0.4, "directed": 0.8, "coldstart": 0.3},
    ),
}
