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
from collections import defaultdict

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

def save_knowledge_graph(config, kg_train, kg_val, kg_test):
    """Save the knowledge graph to files."""
    pickle_filename = os.path.join(config['common']['out'], 'kg.pkl')
    logging.info(f"Saving results to {pickle_filename}...")
    with open(pickle_filename, 'wb') as file:
        pickle.dump(kg_train, file)
        pickle.dump(kg_val, file)
        pickle.dump(kg_test, file)

def prepare_knowledge_graph(config):
    """Prepare and clean the knowledge graph."""
    # Load knowledge graph
    input_file = config["common"]['input_csv']
    kg_df = pd.read_csv(input_file, sep="\t")[["my_x_id", "my_y_id", "relation"]]
    kg_df = kg_df.rename(columns={'my_x_id': 'from', 'my_y_id': 'to', 'relation': 'rel'})

    if config.get("clean_kg", {}).get("smaller_kg", False):
        logging.info(f"Keeping only relations {config['clean_kg']['keep_relations']}")
        kg_df = kg_df[kg_df['rel'].isin(config["clean_kg"]['keep_relations'])]

    kg = my_knowledge_graph.KnowledgeGraph(df=kg_df)

    # Clean and process knowledge graph
    kg_train, kg_val, kg_test = clean_knowledge_graph(kg, config)

    # Save results
    save_knowledge_graph(config, kg_train, kg_val, kg_test)

    return kg_train, kg_val, kg_test

def load_knowledge_graph(config):
    """Load the knowledge graph from pickle files."""
    pickle_filename = config["common"]['input_pkl']
    logging.info(f'Will not run the preparation step. Using KG stored in: {pickle_filename}')
    with open(pickle_filename, 'rb') as file:
        kg_train = pickle.load(file)
        kg_val = pickle.load(file)
        kg_test = pickle.load(file)
    return kg_train, kg_val, kg_test

def clean_knowledge_graph(kg, config):
    """Clean and prepare the knowledge graph according to the configuration."""

    set_random_seeds(config["common"]["seed"])

    id_to_rel_name = {v: k for k, v in kg.rel2ix.items()}

    if config["clean_kg"]['remove_duplicates_triplets']:
        logging.info("Removing duplicated triplets...")
        kg = my_data_redundancy.remove_duplicates_triplets(kg)

    duplicated_relations_list = []

    if config['clean_kg']['check_DL1']:
        logging.info("Checking for semantically redundant and Cartesian product relations...")
        theta1 = config['clean_kg']['check_DL1_params']['theta1']
        theta2 = config['clean_kg']['check_DL1_params']['theta2']

        duplicates_relations, rev_duplicates_relations = my_data_redundancy.duplicates(kg, theta1=theta1, theta2=theta2)
        if duplicates_relations:
            logging.info(f'Adding {len(duplicates_relations)} near-duplicate relations ({[id_to_rel_name[rel] for rel in duplicates_relations]}) to the list of known redundant relations.')
            duplicated_relations_list.extend(duplicates_relations)
        if rev_duplicates_relations:
            logging.info(f'Adding {len(rev_duplicates_relations)} near-reverse-duplicate relations ({[id_to_rel_name[rel] for rel in rev_duplicates_relations]}) to the list of known redundant relations.')
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

    if config['clean_kg']['zero_shot_split']:
        logging.info(f"Zero shot split with relation {config['clean_kg']['zero_shot_split_param']}.")
        kg_train, kg_val, kg_test = zero_shot_by_head_frequency_bin(kg, config['clean_kg']['zero_shot_split_param'])
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


def zero_shot_by_head_frequency_bin(kg, target_rel, val_ratio=0.1, test_ratio=0.1):
    # 1. Mask for target relation
    target_rel_ix = kg.rel2ix[target_rel]
    is_target = (kg.relations == target_rel_ix)
    
    # 2. Indices of target triples
    target_indices = is_target.nonzero(as_tuple=True)[0]
    
    # 3. Counting head frequency in target relation
    target_heads = kg.head_idx[target_indices]
    unique_heads, counts = torch.unique(target_heads, return_counts=True)
    
    # 4. Bins definition
    bins = {
        0: (1, 1),
        1: (2, 2),
        2: (3, 5),
        3: (6, 10),
        4: (11, 50),
        5: (51, 100),
        6: (101, float('inf'))
    }
    
    # 5. Associate each head to its corresponding bin
    head_to_bin = {}
    for head, count in zip(unique_heads.tolist(), counts.tolist()):
        for b, (low, high) in bins.items():
            if low <= count <= high:
                head_to_bin[head] = b
                break
    
    # 6. Find target triple index for each head
    head_to_indices = defaultdict(list)
    for idx in target_indices.tolist():
        head = kg.head_idx[idx].item()
        head_to_indices[head].append(idx)
    
    # 7. Fill each bin
    bin_to_heads = defaultdict(list)
    for head, b in head_to_bin.items():
        bin_to_heads[b].append(head)
    
    # 8. Masks
    mask_val = torch.zeros_like(is_target)
    mask_test = torch.zeros_like(is_target)
    mask_train_target = torch.zeros_like(is_target)
    
    # 9. Non-target triples goes in train
    mask_train = ~is_target.clone()
    
    # 10. Entities appearing only with target relation goes to train
    unique_target_entities = torch.unique(torch.cat([kg.head_idx[target_indices], kg.tail_idx[target_indices]]))
    only_target_entities = []
    for entity in unique_target_entities.tolist():
        entity_mask = (kg.head_idx == entity) | (kg.tail_idx == entity)
        if ((entity_mask & ~is_target).sum() == 0):
            only_target_entities.append(entity)
            mask_train_target |= entity_mask
    
    mask_train |= mask_train_target
    
    # 11. Allocate val/test sets proportionally by number of heads per bin 
    for b, heads in bin_to_heads.items():
        random.shuffle(heads)

        n_total_heads = len(heads)
        n_val_heads = int(val_ratio * n_total_heads)
        n_test_heads = int(test_ratio * n_total_heads)

        # Ensure at least one head goes to val if possible
        if n_val_heads == 0 and n_total_heads > 0:
            n_val_heads = 1
        # Ensure at least one head goes to test if possible and heads remain after val
        if n_test_heads == 0 and n_total_heads - n_val_heads > 0:
            n_test_heads = 1

        val_heads = heads[:n_val_heads]
        test_heads = heads[n_val_heads:n_val_heads + n_test_heads]

        for h in val_heads:
            mask_val[head_to_indices[h]] = True
        for h in test_heads:
            mask_test[head_to_indices[h]] = True

    # 12. Final train update : remaining triples goes in train
    mask_train |= (is_target & ~(mask_val | mask_test))
    
    # 13. Train, validation and test splits
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
    
    # 14. Logging 
    print("=== Split summary ===")
    for split_name, mask in zip(['Train', 'Validation', 'Test'], [mask_train, mask_val, mask_test]):
        idxs = mask.nonzero(as_tuple=True)[0]
        split_target_idxs = idxs[(kg.relations[idxs] == target_rel_ix)]
        split_heads = torch.unique(kg.head_idx[split_target_idxs])
        print(f"{split_name} : {len(split_target_idxs)} target triplets, {len(split_heads)} unique heads")
    
    return train_kg, val_kg, test_kg


def zero_shot_split(kg, target_rel, val_ratio=0.1, test_ratio=0.1):
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