# -*- coding: utf-8 -*-
"""
Knowledge Graph Preparation and Cleaning Script
@author: Galadriel Brière <marie-galadriel.briere@univ-amu.fr>

This script is designed to prepare and clean a knowledge graph using various utility functions and configurations. It supports tasks such as parsing a YAML configuration, ensuring entity coverage, setting random seeds for reproducibility, cleaning duplicated triples, and saving/loading the knowledge graph.
"""

import sys
import os
import argparse
import pandas as pd
import pickle
import yaml
import random
import numpy as np
import torch
import logging
from torch import cat
from collections import defaultdict, Counter

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

sys.path.append('dev')
import my_data_redundancy
import my_knowledge_graph

def parse_yaml(config_path):
    """Load and parse the YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File {config_path} not found.")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config



def verify_entity_coverage(train_kg, full_kg):
    """
    Verify that all entities in the full knowledge graph are represented in the training set.

    Parameters
    ----------
    train_kg: KnowledgeGraph
        The training knowledge graph.
    full_kg: KnowledgeGraph
        The full knowledge graph.

    Returns
    -------
    tuple
        (bool, list)
        A tuple where the first element is True if all entities in the full knowledge graph are present in the training 
        knowledge graph, and the second element is a list of missing entities (names) if any are missing.
    """
    # Obtenir les identifiants d'entités pour le graphe d'entraînement et le graphe complet
    train_entities = set(cat((train_kg.head_idx, train_kg.tail_idx)).tolist())
    full_entities = set(cat((full_kg.head_idx, full_kg.tail_idx)).tolist())
    
    # Entités manquantes dans le graphe d'entraînement
    missing_entity_ids = full_entities - train_entities
    
    if missing_entity_ids:
        # Inverser le dictionnaire ent2ix pour obtenir idx: entity_name
        ix2ent = {v: k for k, v in full_kg.ent2ix.items()}
        
        # Récupérer les noms des entités manquantes à partir de leurs indices
        missing_entities = [ix2ent[idx] for idx in missing_entity_ids if idx in ix2ent]
        return False, missing_entities
    else:
        return True, []

def set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# def save_knowledge_graph(config, kg_train, kg_val, kg_test):
#     """Save the knowledge graph to files."""
#     pickle_filename = os.path.join(config['common']['out'], 'kg.pkl')
#     logging.info(f"Saving results to {pickle_filename}...")
#     with open(pickle_filename, 'wb') as file:
#         pickle.dump(kg_train, file)
#         pickle.dump(kg_val, file)
#         pickle.dump(kg_test, file)

def save_knowledge_graph(config, kg_train, kg_val, kg_test):
    """Save the knowledge graph to files."""
    out = config['common']['out']
    logging.info(f"Saving results to {out}...")
    torch.save(kg_train, os.path.join(out, 'kg_train.pt'))
    torch.save(kg_val,   os.path.join(out, 'kg_val.pt'))
    torch.save(kg_test,  os.path.join(out, 'kg_test.pt'))

def prepare_knowledge_graph(config):
    """Prepare and clean the knowledge graph."""
    # Load knowledge graph
    input_file = config["common"]['input_csv']
    kg_df = pd.read_csv(input_file, sep="\t")[["from", "to", "rel"]]

    if config.get("clean_kg", {}).get("smaller_kg", False):
        logging.info(f"Keeping only relations {config['clean_kg']['keep_relations']}")
        kg_df = kg_df[kg_df['rel'].isin(config["clean_kg"]['keep_relations'])]

    kg = my_knowledge_graph.KnowledgeGraph(df=kg_df)

    # Clean and process knowledge graph
    kg_train, kg_val, kg_test = clean_knowledge_graph(kg, config)

    # Save results
    save_knowledge_graph(config, kg_train, kg_val, kg_test)

    return kg_train, kg_val, kg_test

# def load_knowledge_graph(config):
#     """Load the knowledge graph from pickle files."""
#     pickle_filename = config["common"]['input_pkl']
#     logging.info(f'Will not run the preparation step. Using KG stored in: {pickle_filename}')
#     with open(pickle_filename, 'rb') as file:
#         kg_train = pickle.load(file)
#         kg_val = pickle.load(file)
#         kg_test = pickle.load(file)
#     return kg_train, kg_val, kg_test

def load_knowledge_graph(config):
    input_path = config["common"]['input_pkl']
    if input_path.endswith('.pkl'):
        # legacy
        with open(input_path, 'rb') as f:
            kg_train = pickle.load(f)
            kg_val   = pickle.load(f)
            kg_test  = pickle.load(f)
    else:
        # nouveau format .pt
        kg_train = torch.load(os.path.join(input_path, 'kg_train.pt'), weights_only=False)
        kg_val   = torch.load(os.path.join(input_path, 'kg_val.pt'),   weights_only=False)
        kg_test  = torch.load(os.path.join(input_path, 'kg_test.pt'),  weights_only=False)
    return kg_train, kg_val, kg_test

def clean_knowledge_graph(kg, config):
    """Clean and prepare the knowledge graph according to the configuration."""

    set_random_seeds(config["common"]["seed"])

    id_to_rel_name = {v: k for k, v in kg.rel2ix.items()}

    logging.info(f'KG with {len(kg.ent2ix)} entities')

    if config["clean_kg"]['remove_duplicates_triplets']:
        logging.info("Removing duplicated triplets...")
        kg = my_data_redundancy.remove_duplicates_triplets(kg, ix2rel=id_to_rel_name)

    duplicated_relations_list = []

    if config['clean_kg']['check_DL1']:
        logging.info("Checking for semantically redundant and Cartesian product relations...")
        theta1 = config['clean_kg']['check_DL1_params']['theta1']
        theta2 = config['clean_kg']['check_DL1_params']['theta2']

        duplicates_relations, rev_duplicates_relations = my_data_redundancy.duplicates(kg, theta1=theta1, theta2=theta2)
        if duplicates_relations:
            logging.info(f'Adding {len(duplicates_relations)} near-duplicate relations '
                 f'({[(id_to_rel_name[a], id_to_rel_name[b]) for a, b in duplicates_relations]}) '
                 f'to the list of known redundant relations.')
            # logging.info(f'Adding {len(duplicates_relations)} near-duplicate relations ({[id_to_rel_name[rel] for rel in duplicates_relations]}) to the list of known redundant relations.')
            duplicated_relations_list.extend(duplicates_relations)
        if rev_duplicates_relations:
            logging.info(f'Adding {len(rev_duplicates_relations)} near-reverse-duplicate relations '
                 f'({[(id_to_rel_name[a], id_to_rel_name[b]) for a, b in rev_duplicates_relations]}) '
                 f'to the list of known redundant relations.')
            # logging.info(f'Adding {len(rev_duplicates_relations)} near-reverse-duplicate relations ({[id_to_rel_name[rel] for rel in rev_duplicates_relations]}) to the list of known redundant relations.')
            duplicated_relations_list.extend(rev_duplicates_relations)

        theta = config.get("clean_kg", {}).get("check_DL1_params", {}).get("theta", 0.8)
        cartesian_rels = my_data_redundancy.cartesian_product_relations(kg, theta=theta)
        if cartesian_rels:
            logging.info(f'Adding {len(cartesian_rels)} Cartesian product relations ({[id_to_rel_name[rel] for rel in cartesian_rels]}) to the list of known Cartesian product relations.')

    if config['clean_kg']["permute_kg"]:
        to_permute_relation_names = config['clean_kg']["permute_kg_params"]
        if len(to_permute_relation_names) > 1:
            logging.info(f'Making permutations for relations {", ".join([rel for rel in to_permute_relation_names])}...')
        for rel in to_permute_relation_names:
            logging.info(f'Making permutations for relation {rel} with id {kg.rel2ix[rel]}.')
            kg = my_data_redundancy.permute_tails(kg, kg.rel2ix[rel])

    if config['clean_kg']['make_directed']:
        undirected_relations_names = config['clean_kg']['make_directed_params']
        relation_names = ", ".join([rel for rel in undirected_relations_names])
        logging.info(f'Adding reverse triplets for relations {relation_names}...')
        kg, undirected_relations_list = my_data_redundancy.add_inverse_relations(kg, [kg.rel2ix[key] for key in undirected_relations_names])
            
        if config['clean_kg']['check_DL1']:
            logging.info(f'Adding created reverses {[rel for rel in undirected_relations_names]} to the list of known redundant relations.')
            duplicated_relations_list.extend(undirected_relations_list)

    if config['clean_kg']['cold_start_split']:
        if config['clean_kg']['cold_start_split'] == "head":
            logging.info(f"Cold start split with relation {config['clean_kg']['cold_start_split_param']}.")
            kg_train, kg_val, kg_test = cold_start_by_head_frequency_bin(kg, config['clean_kg']['cold_start_split_param'])
        elif config['clean_kg']['cold_start_split'] == "tail":
            logging.info(f"Cold start split with relation {config['clean_kg']['cold_start_split_param']}.")
            kg_train, kg_val, kg_test = cold_start_by_tail_frequency_bin(kg, config['clean_kg']['cold_start_split_param'])
    else:
        logging.info("Splitting the dataset into train, validation and test sets...")
        kg_train, kg_val, kg_test = kg.split_kg(validation=True)

    kg_train_ok, _ = verify_entity_coverage(kg_train, kg)
    if not kg_train_ok:
        logging.info("Entity coverage verification failed...")
    else:
        logging.info("Entity coverage verified successfully.")

    if config['clean_kg']['clean_train_set']:
        logging.info("Cleaning the train set to avoid data leakage...")
        logging.info("Step 1: with respect to validation set.")
        kg_train = my_data_redundancy.clean_datasets(kg_train, kg_val, known_reverses=duplicated_relations_list)
        logging.info("Step 2: with respect to test set.")
        kg_train = my_data_redundancy.clean_datasets(kg_train, kg_test, known_reverses=duplicated_relations_list)
        if cartesian_rels:
            kg_train, kg_test = my_data_redundancy.clean_cartesians(kg_train, kg_test, cartesian_rels)


    kg_train_ok, _ = verify_entity_coverage(kg_train, kg)
    if not kg_train_ok:
        logging.info("Entity coverage verification failed...")
    else:
        logging.info("Entity coverage verified successfully.")

    new_train, new_val, new_test = my_data_redundancy.ensure_entity_coverage(kg_train, kg_val, kg_test)

    kg_train_ok, missing_entities = verify_entity_coverage(new_train, kg)
    if not kg_train_ok:
        logging.info(f"Entity coverage verification failed. {len(missing_entities)} entities are missing.")
        logging.info(f"Missing entities: {missing_entities}")
        raise ValueError('One or more entities are not covered in the training set after ensuring entity coverage...')
    else:
        logging.info("Entity coverage verified successfully.")

    if config.get("clean_kg", {}).get("compute_proportions", True):
        logging.info("Computing triplet proportions...")
        logging.info(my_data_redundancy.compute_triplet_proportions(kg_train, kg_test, kg_val))

    return new_train, new_val, new_test

def entity_stats(kg, target_rel: str, split_name: str, position: str = "head", n: int = 5):
    """
    Displays statistics for entities (head or tail) involved in a given target relation.
    
    Args:
    - kg: a KnowledgeGraph object with .head_idx, .tail_idx, .relations, .rel2ix
    - target_rel: name (str) of the target relation (e.g., "indication")
    - split_name: name (str) of the split for display (e.g., "Train")
    - position: "head" or "tail" to specify which part of the triplet to analyze
    """
    rel_id = kg.rel2ix[target_rel]
    is_target = kg.relations == rel_id
    
    if position == "head":
        entities = kg.head_idx[is_target].tolist()
        label = "head"
    elif position == "tail":
        entities = kg.tail_idx[is_target].tolist()
        label = "tail"
    else:
        raise ValueError("Position must be either 'head' or 'tail'.")

    counts = Counter(entities)

    logging.info(f"=== {split_name} ===")
    logging.info(f"Total number of target triplets : {len(entities)}")
    logging.info(f"Number of unique {label}s       : {len(counts)}")
    logging.info(f"Top {n} most frequent {label}s:")
    for i, (entity_id, freq) in enumerate(counts.most_common(n)):
        logging.info(f"  {i+1}. {label.capitalize()} ID {entity_id} : {freq} times")

def cold_start_by_head_frequency_bin(kg, target_rel, val_ratio=0.1, test_ratio=0.1, bins=None):
    """
    Split a Knowledge Graph for cold start learning based on the frequency of heads in a target relation.

    The function ensures:
    - Train contains all non-target triples and some target triples.
    - Validation and test sets only contain triples from the target relation.
    - Heads in val/test are exclusive (cold start) with respect to the target relation.
    - Distribution of heads in val/test is stratified by frequency bin.
    - All entities appear at least once in the train split.
    - No leakage occurs through "only-target" entities (i.e., entities that only appear
      in the target relation): such entities are forced into train and their triples
      cannot appear in val/test.

    Args:
        kg: KnowledgeGraph object with attributes `head_idx`, `tail_idx`, `relations`, etc.
        target_rel (str): Name of the target relation.
        val_ratio (float): Ratio of unique heads (per bin) to assign to the validation set.
        test_ratio (float): Ratio of unique heads (per bin) to assign to the test set.
        bins (dict, optional): Dictionary defining frequency bins.

    Returns:
        train_kg, val_kg, test_kg: Splits of the original KnowledgeGraph.
    """

    # Default bins if not provided
    if bins is None:
        bins = {
            0: (1, 1),
            1: (2, 2),
            2: (3, 5),
            3: (6, 10),
            4: (11, 50),
            5: (51, 100),
            6: (101, 129),
            7: (130, float('inf'))
        }

    # 1. Mask for target relation
    target_rel_ix = kg.rel2ix[target_rel]
    is_target = (kg.relations == target_rel_ix)

    # 2. Indices of target triples
    target_indices = is_target.nonzero(as_tuple=True)[0]

    # 3. Count head frequencies in the target relation
    target_heads = kg.head_idx[target_indices]
    unique_heads, counts = torch.unique(target_heads, return_counts=True)

    # 4. Assign each head to its frequency bin
    head_to_bin = {}
    for head, count in zip(unique_heads.tolist(), counts.tolist()):
        for b, (low, high) in bins.items():
            if low <= count <= high:
                head_to_bin[head] = b
                break

    # 5. Map heads to their target triple indices
    head_to_indices = defaultdict(list)
    for idx in target_indices.tolist():
        head = kg.head_idx[idx].item()
        head_to_indices[head].append(idx)

    # 6. Group heads by bins
    bin_to_heads = defaultdict(list)
    for head, b in head_to_bin.items():
        bin_to_heads[b].append(head)

    # 7. Prepare split masks
    mask_val = torch.zeros_like(is_target)
    mask_test = torch.zeros_like(is_target)

    # 8. Add all non-target triples to train
    mask_train = ~is_target.clone()

    # === 9. Identify "only-target" entities (heads AND tails) ===
    # An entity is "only-target" if it appears exclusively in the target relation.
    # Such entities MUST be kept in train (otherwise they disappear from the KG),
    # so we must:
    #   - exclude only-target heads from the val/test splitting pool
    #   - prevent any triple involving an only-target tail from being assigned to val/test

    unique_target_heads = torch.unique(kg.head_idx[target_indices])
    unique_target_tails = torch.unique(kg.tail_idx[target_indices])

    only_target_heads = set()
    for entity in unique_target_heads.tolist():
        entity_mask = (kg.head_idx == entity) | (kg.tail_idx == entity)
        if ((entity_mask & ~is_target).sum() == 0):
            only_target_heads.add(entity)

    only_target_tails = set()
    for entity in unique_target_tails.tolist():
        entity_mask = (kg.head_idx == entity) | (kg.tail_idx == entity)
        if ((entity_mask & ~is_target).sum() == 0):
            only_target_tails.add(entity)

    logging.info(f"Only-target heads (excluded from val/test pool): {len(only_target_heads)}")
    logging.info(f"Only-target tails (their triples forced to train): {len(only_target_tails)}")

    # Force all triples involving only-target entities to train
    forced_to_train = torch.zeros_like(is_target)
    if only_target_heads:
        head_mask = torch.zeros_like(is_target)
        for h in only_target_heads:
            head_mask |= (kg.head_idx == h)
        forced_to_train |= (head_mask & is_target)
    if only_target_tails:
        tail_mask = torch.zeros_like(is_target)
        for t in only_target_tails:
            tail_mask |= (kg.tail_idx == t)
        forced_to_train |= (tail_mask & is_target)

    mask_train |= forced_to_train

    # 10. Allocate val/test sets per bin (excluding only-target heads from pool,
    # and skipping triples whose tail is only-target)
    for b, heads in bin_to_heads.items():
        # Exclude only-target heads
        heads = [h for h in heads if h not in only_target_heads]
        random.shuffle(heads)
        n_total = len(heads)
        n_val = int(val_ratio * n_total)
        n_test = int(test_ratio * n_total)

        if n_val == 0 and n_total > 0:
            n_val = 1
        if n_test == 0 and n_total - n_val > 0:
            n_test = 1

        val_heads = heads[:n_val]
        test_heads = heads[n_val:n_val + n_test]

        # For each selected head, assign its triples to val/test ONLY if the tail
        # is not only-target (otherwise we'd lose the tail from the KG)
        for h in val_heads:
            for idx in head_to_indices[h]:
                if kg.tail_idx[idx].item() not in only_target_tails:
                    mask_val[idx] = True
        for h in test_heads:
            for idx in head_to_indices[h]:
                if kg.tail_idx[idx].item() not in only_target_tails:
                    mask_test[idx] = True

    # 11. Remaining target triples (those not assigned to val/test) go to train
    mask_train |= (is_target & ~(mask_val | mask_test))

    # === 11b. Sanity checks: enforce mutual exclusion of splits ===
    assert (mask_train & mask_val).sum() == 0, "Train/val overlap detected"
    assert (mask_train & mask_test).sum() == 0, "Train/test overlap detected"
    assert (mask_val & mask_test).sum() == 0, "Val/test overlap detected"

    # 12. Create final KG splits
    train_kg = my_knowledge_graph.KnowledgeGraph(
        kg={'heads': kg.head_idx[mask_train],
            'tails': kg.tail_idx[mask_train],
            'relations': kg.relations[mask_train]},
        ent2ix=kg.ent2ix, rel2ix=kg.rel2ix,
        dict_of_heads=kg.dict_of_heads,
        dict_of_tails=kg.dict_of_tails,
        dict_of_rels=kg.dict_of_rels)

    val_kg = my_knowledge_graph.KnowledgeGraph(
        kg={'heads': kg.head_idx[mask_val],
            'tails': kg.tail_idx[mask_val],
            'relations': kg.relations[mask_val]},
        ent2ix=kg.ent2ix, rel2ix=kg.rel2ix,
        dict_of_heads=kg.dict_of_heads,
        dict_of_tails=kg.dict_of_tails,
        dict_of_rels=kg.dict_of_rels)

    test_kg = my_knowledge_graph.KnowledgeGraph(
        kg={'heads': kg.head_idx[mask_test],
            'tails': kg.tail_idx[mask_test],
            'relations': kg.relations[mask_test]},
        ent2ix=kg.ent2ix, rel2ix=kg.rel2ix,
        dict_of_heads=kg.dict_of_heads,
        dict_of_tails=kg.dict_of_tails,
        dict_of_rels=kg.dict_of_rels)

    # 13. Logging
    logging.info("=== Head-based Cold Start Split Summary ===")
    for split_name, mask in zip(['Train', 'Validation', 'Test'], [mask_train, mask_val, mask_test]):
        idxs = mask.nonzero(as_tuple=True)[0]
        split_target_idxs = idxs[(kg.relations[idxs] == target_rel_ix)]
        split_heads = torch.unique(kg.head_idx[split_target_idxs])
        logging.info(f"{split_name} : {len(split_target_idxs)} target triplets, {len(split_heads)} unique heads")

    entity_stats(train_kg, target_rel=target_rel, split_name="Train", position="head")
    entity_stats(val_kg, target_rel=target_rel, split_name="Validation", position="head")
    entity_stats(test_kg, target_rel=target_rel, split_name="Test", position="head")

    entity_stats(train_kg, target_rel=target_rel, split_name="Train", position="tail")
    entity_stats(val_kg, target_rel=target_rel, split_name="Validation", position="tail")
    entity_stats(test_kg, target_rel=target_rel, split_name="Test", position="tail")

    return train_kg, val_kg, test_kg

def cold_start_by_tail_frequency_bin(kg, target_rel, val_ratio=0.1, test_ratio=0.1, bins=None):
    """
    Split a Knowledge Graph for cold start learning based on the frequency of *tails* (diseases)
    in a target relation (e.g. drug -indication-> disease).

    Same structure and logic as cold_start_by_head_frequency_bin, but flipped to cold start on tails.

    Now also handles "only-target" entities to prevent leakage.

    Returns:
        train_kg, val_kg, test_kg
    """
    if bins is None:
        bins = {
            0: (1, 1),
            1: (2, 3),
            2: (4, 7),
            3: (8, 15),
            4: (16, 30),
            5: (31, 60),
            6: (61, float('inf'))
        }

    target_rel_ix = kg.rel2ix[target_rel]
    is_target = (kg.relations == target_rel_ix)
    target_indices = is_target.nonzero(as_tuple=True)[0]

    # Count tail frequencies in target relation
    target_tails = kg.tail_idx[target_indices]
    unique_tails, counts = torch.unique(target_tails, return_counts=True)

    # Assign tails to bins
    tail_to_bin = {}
    for tail, count in zip(unique_tails.tolist(), counts.tolist()):
        for b, (low, high) in bins.items():
            if low <= count <= high:
                tail_to_bin[tail] = b
                break

    # Map tails to their target triple indices
    tail_to_indices = defaultdict(list)
    for idx in target_indices.tolist():
        tail = kg.tail_idx[idx].item()
        tail_to_indices[tail].append(idx)

    # Group tails by bins
    bin_to_tails = defaultdict(list)
    for tail, b in tail_to_bin.items():
        bin_to_tails[b].append(tail)

    # Prepare split masks
    mask_val = torch.zeros_like(is_target)
    mask_test = torch.zeros_like(is_target)
    mask_train = ~is_target.clone()

    # === Identify "only-target" entities ===
    unique_target_heads = torch.unique(kg.head_idx[target_indices])
    unique_target_tails = torch.unique(kg.tail_idx[target_indices])

    only_target_heads = set()
    for entity in unique_target_heads.tolist():
        entity_mask = (kg.head_idx == entity) | (kg.tail_idx == entity)
        if ((entity_mask & ~is_target).sum() == 0):
            only_target_heads.add(entity)

    only_target_tails = set()
    for entity in unique_target_tails.tolist():
        entity_mask = (kg.head_idx == entity) | (kg.tail_idx == entity)
        if ((entity_mask & ~is_target).sum() == 0):
            only_target_tails.add(entity)

    logging.info(f"Only-target tails (excluded from val/test pool): {len(only_target_tails)}")
    logging.info(f"Only-target heads (their triples forced to train): {len(only_target_heads)}")

    # Force triples involving only-target entities to train
    forced_to_train = torch.zeros_like(is_target)
    if only_target_heads:
        head_mask = torch.zeros_like(is_target)
        for h in only_target_heads:
            head_mask |= (kg.head_idx == h)
        forced_to_train |= (head_mask & is_target)
    if only_target_tails:
        tail_mask = torch.zeros_like(is_target)
        for t in only_target_tails:
            tail_mask |= (kg.tail_idx == t)
        forced_to_train |= (tail_mask & is_target)

    mask_train |= forced_to_train

    # Allocate val/test sets per bin (excluding only-target tails from pool,
    # and skipping triples whose head is only-target)
    for b, tails in bin_to_tails.items():
        tails = [t for t in tails if t not in only_target_tails]
        random.shuffle(tails)
        n_total = len(tails)
        n_val = int(val_ratio * n_total)
        n_test = int(test_ratio * n_total)

        if n_val == 0 and n_total > 0:
            n_val = 1
        if n_test == 0 and n_total - n_val > 0:
            n_test = 1

        val_tails = tails[:n_val]
        test_tails = tails[n_val:n_val + n_test]

        for t in val_tails:
            for idx in tail_to_indices[t]:
                if kg.head_idx[idx].item() not in only_target_heads:
                    mask_val[idx] = True
        for t in test_tails:
            for idx in tail_to_indices[t]:
                if kg.head_idx[idx].item() not in only_target_heads:
                    mask_test[idx] = True

    # Remaining target triples go to train
    mask_train |= (is_target & ~(mask_val | mask_test))

    # Sanity checks
    assert (mask_train & mask_val).sum() == 0, "Train/val overlap detected"
    assert (mask_train & mask_test).sum() == 0, "Train/test overlap detected"
    assert (mask_val & mask_test).sum() == 0, "Val/test overlap detected"

    # Create final KG splits
    train_kg = my_knowledge_graph.KnowledgeGraph(
        kg={'heads': kg.head_idx[mask_train],
            'tails': kg.tail_idx[mask_train],
            'relations': kg.relations[mask_train]},
        ent2ix=kg.ent2ix, rel2ix=kg.rel2ix,
        dict_of_heads=kg.dict_of_heads,
        dict_of_tails=kg.dict_of_tails,
        dict_of_rels=kg.dict_of_rels)

    val_kg = my_knowledge_graph.KnowledgeGraph(
        kg={'heads': kg.head_idx[mask_val],
            'tails': kg.tail_idx[mask_val],
            'relations': kg.relations[mask_val]},
        ent2ix=kg.ent2ix, rel2ix=kg.rel2ix,
        dict_of_heads=kg.dict_of_heads,
        dict_of_tails=kg.dict_of_tails,
        dict_of_rels=kg.dict_of_rels)

    test_kg = my_knowledge_graph.KnowledgeGraph(
        kg={'heads': kg.head_idx[mask_test],
            'tails': kg.tail_idx[mask_test],
            'relations': kg.relations[mask_test]},
        ent2ix=kg.ent2ix, rel2ix=kg.rel2ix,
        dict_of_heads=kg.dict_of_heads,
        dict_of_tails=kg.dict_of_tails,
        dict_of_rels=kg.dict_of_rels)

    # Logging
    logging.info("=== Tail-based Cold Start Split Summary ===")
    for split_name, mask in zip(['Train', 'Validation', 'Test'], [mask_train, mask_val, mask_test]):
        idxs = mask.nonzero(as_tuple=True)[0]
        split_target_idxs = idxs[(kg.relations[idxs] == target_rel_ix)]
        split_tails = torch.unique(kg.tail_idx[split_target_idxs])
        logging.info(f"{split_name} : {len(split_target_idxs)} target triplets, {len(split_tails)} unique tails")

    entity_stats(train_kg, target_rel=target_rel, split_name="Train", position="tail")
    entity_stats(val_kg, target_rel=target_rel, split_name="Validation", position="tail")
    entity_stats(test_kg, target_rel=target_rel, split_name="Test", position="tail")

    entity_stats(train_kg, target_rel=target_rel, split_name="Train", position="head")
    entity_stats(val_kg, target_rel=target_rel, split_name="Validation", position="head")
    entity_stats(test_kg, target_rel=target_rel, split_name="Test", position="head")

    return train_kg, val_kg, test_kg

def cold_start_split(kg, target_rel, val_ratio=0.1, test_ratio=0.1):
    # Create a mask for the target relation
    is_target = kg.relations == kg.rel2ix[target_rel]
    
    # Identify triplets related to the target relation
    target_indices = is_target.nonzero(as_tuple=True)[0]
    
    # Define empty masks for train, val, test sets
    mask_val = torch.zeros_like(is_target)
    mask_test = torch.zeros_like(is_target)
    mask_train_target = torch.zeros_like(is_target) # this one is for target triplets to put in train
    
    # Identify entities that appear only in target relation (these ones should go to train set)
    unique_target_heads = torch.unique(kg.head_idx[target_indices])
    unique_target_tails = torch.unique(kg.tail_idx[target_indices])
    unique_target_entities = torch.unique(torch.cat([unique_target_heads, unique_target_tails]))
    
    only_target_entities = []
    for entity in unique_target_entities:
        entity_in_triplet = (kg.head_idx == entity) | (kg.tail_idx == entity)
        if (entity_in_triplet & ~is_target).sum() == 0:  # entity only seen in target rel
            only_target_entities.append(entity.item())
            mask_train_target = mask_train_target | entity_in_triplet
    
    # Every triple that is not a target triple, or has an entity only seen in target rel, goes to train set
    mask_train = ~is_target | mask_train_target

    # If the triple was sent to train because of the tail, we need to add tro the train set every target triplets with the same head
    train_target_heads = torch.unique(kg.head_idx[mask_train_target & is_target])
    for head in train_target_heads:
        same_head_target = (kg.head_idx == head) & is_target
        mask_train_target = mask_train_target | same_head_target

    # We update mask_train to include those that are not target OR marked as train_target
    mask_train = ~is_target | mask_train_target

    
    # Group triples by head for the target relation
    remaining_target_mask = is_target & ~mask_train_target
    remaining_target_indices = remaining_target_mask.nonzero(as_tuple=True)[0]
    
    head_to_indices = defaultdict(list)
    for idx in remaining_target_indices.tolist():
        head = kg.head_idx[idx].item()
        head_to_indices[head].append(idx)
    
    # Split greedy based on the number of triplets per head
    remaining_heads = list(head_to_indices.keys())
    random.shuffle(remaining_heads)
    
    total_triplets = sum(len(v) for v in head_to_indices.values())
    target_val_count = int(val_ratio * total_triplets)
    target_test_count = int(test_ratio * total_triplets)
    
    val_heads = []
    test_heads = []
    val_total = 0
    test_total = 0
    
    for head in remaining_heads:
        triplets = head_to_indices[head]
        n = len(triplets)
        
        if val_total + n <= target_val_count:
            val_heads.append(head)
            val_total += n
        elif test_total + n <= target_test_count:
            test_heads.append(head)
            test_total += n
    
    # Update masks for val and test
    for h in val_heads:
        mask_val[head_to_indices[h]] = True
    
    for h in test_heads:
        mask_test[head_to_indices[h]] = True
    
    # Final update for the train mask (in case there are remaining target triplets)
    mask_train = mask_train | (~mask_val & ~mask_test & is_target)
    
    train_kg = my_knowledge_graph.KnowledgeGraph(
        kg={'heads': kg.head_idx[mask_train],
        'tails': kg.tail_idx[mask_train],
        'relations': kg.relations[mask_train]},
        ent2ix=kg.ent2ix, rel2ix=kg.rel2ix,
        dict_of_heads=kg.dict_of_heads,
        dict_of_tails=kg.dict_of_tails,
        dict_of_rels=kg.dict_of_rels)
    
    val_kg = my_knowledge_graph.KnowledgeGraph(
            kg={'heads': kg.head_idx[mask_val],
            'tails': kg.tail_idx[mask_val],
            'relations': kg.relations[mask_val]},
            ent2ix=kg.ent2ix, rel2ix=kg.rel2ix,
            dict_of_heads=kg.dict_of_heads,
            dict_of_tails=kg.dict_of_tails,
            dict_of_rels=kg.dict_of_rels)
    
    test_kg = my_knowledge_graph.KnowledgeGraph(
            kg={'heads': kg.head_idx[mask_test],
            'tails': kg.tail_idx[mask_test],
            'relations': kg.relations[mask_test]},
            ent2ix=kg.ent2ix, rel2ix=kg.rel2ix,
            dict_of_heads=kg.dict_of_heads,
            dict_of_tails=kg.dict_of_tails,
            dict_of_rels=kg.dict_of_rels)
    
    return train_kg, val_kg, test_kg

