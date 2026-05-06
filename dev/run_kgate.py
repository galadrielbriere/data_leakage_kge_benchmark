#!/usr/bin/env python3
"""
KGATE training and evaluation script.
Equivalent of the TorchKGE train_model pipeline, with made_directed_relations
and target_relations evaluation by frequency threshold.
"""

import argparse
import logging
import torch
import yaml
import os
import sys
import tomllib
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to TOML configuration file.')
    return parser.parse_args()


def run_evaluation(architect, made_directed_relations, target_relations, thresholds, output_dir):
    """
    Run full evaluation on test set, equivalent to TorchKGE pipeline.
    Saves results to evaluation_metrics.yaml.
    """
    architect.load_best_model()
    architect.evaluator = architect.initialize_evaluator()
    architect.eval()

    results = {}

    with torch.no_grad():

        # 1. Global + target_edges + remaining via architect.test()
        logging.info("Running architect.test()...")
        test_results = architect.test()
        results['Global_MRR'] = test_results.get('Global_metrics', 0)

        # 2. made_directed_relations
        if made_directed_relations:
            logging.info(f"Evaluating made_directed_relations: {made_directed_relations}")
            _, _, ind_mrr_directed, group_mrr_directed = architect.calculate_metrics_for_edges(
                architect.kg_test, made_directed_relations)
            results['made_directed_relations'] = {
                'Global_MRR': group_mrr_directed,
                'Individual_MRRs': ind_mrr_directed
            }

        # 3. target_relations
        if target_relations:
            logging.info(f"Evaluating target_relations: {target_relations}")
            _, _, ind_mrr_target, group_mrr_target = architect.calculate_metrics_for_edges(
                architect.kg_test, target_relations)
            results['target_relations'] = {
                'Global_MRR': group_mrr_target,
                'Individual_MRRs': ind_mrr_target
            }

        # 4. remaining_relations
        all_edges = set(architect.kg_test.edge_to_index.keys())
        remaining = list(all_edges - set(made_directed_relations) - set(target_relations))
        logging.info(f"Evaluating {len(remaining)} remaining relations...")
        _, _, ind_mrr_remaining, group_mrr_remaining = architect.calculate_metrics_for_edges(
            architect.kg_test, remaining)
        results['remaining_relations'] = {
            'Global_MRR': group_mrr_remaining,
            'Individual_MRRs': ind_mrr_remaining
        }

        # 5. target_relations by frequency threshold
        for rel in target_relations:
            for threshold in thresholds:
                logging.info(f"Evaluating {rel} with threshold={threshold}...")
                freq_idx, infreq_idx = architect.categorize_test_nodes(rel, threshold)
                freq_mrr, infreq_mrr = architect.calculate_metrics_for_categories(
                    freq_idx, infreq_idx)

                key = f"target_relations_by_frequency_{threshold}"
                results.setdefault(key, {})
                results[key][rel] = {
                    'Frequent_MRR': freq_mrr,
                    'Infrequent_MRR': infreq_mrr,
                    'Threshold': threshold
                }
                logging.info(f"  Frequent MRR: {freq_mrr}, Infrequent MRR: {infreq_mrr}")

    mrr_file = os.path.join(output_dir, 'evaluation_metrics.yaml')
    with open(mrr_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    logging.info(f"Evaluation results saved to {mrr_file}")

    return results


def run_inference(architect, orpha_path, output_dir):
    """
    Run evaluation on Orphanet inference set.
    Ground truth = only Orphanet triplets, consistent with TorchKGE pipeline.
    """
    import pandas as pd
    from kgate.knowledgegraph import KnowledgeGraph as KGATEKnowledgeGraph
    from kgate.evaluators import LinkPredictionEvaluator

    logging.info(f"Loading inference KG from {orpha_path}...")
    orpha_df = pd.read_csv(orpha_path, sep='\t').rename(
        columns={'from': 'head', 'to': 'tail', 'rel': 'edge'}
    )

    orpha_kg = KGATEKnowledgeGraph(
        dataframe=orpha_df,
        node_to_index=architect.kg_train.node_to_index,
        edge_to_index=architect.kg_train.edge_to_index
    )
    orpha_kg.removed_triplets = torch.zeros((4, 0), dtype=torch.long)

    # Ground truth = uniquement les triplets Orphanet, cohérent avec TorchKGE
    orpha_evaluator = LinkPredictionEvaluator(
        full_graphindices=orpha_kg.graphindices,
        embedding_dimensions=architect.node_embedding_dimensions
    )

    architect.eval()
    with torch.no_grad():
        head_preds, tail_preds = orpha_evaluator.evaluate(
            batch_size=architect.evaluation_batch_size,
            encoder=architect.encoder,
            decoder=architect.decoder,
            knowledge_graph=orpha_kg,
            node_embeddings=architect.node_embeddings,
            edge_embeddings=architect.edge_embeddings
        )

    inference_mrr   = (head_preds.mrr[1] + tail_preds.mrr[1]) / 2
    inference_hit10 = (head_preds.hit_at_k(10)[1] + tail_preds.hit_at_k(10)[1]) / 2

    logging.info(f"Inference MRR: {inference_mrr}")
    logging.info(f"Inference Hit@10: {inference_hit10}")

    results = {
        'Inference MRR': inference_mrr,
        'Inference hit@10': inference_hit10
    }

    inference_file = os.path.join(output_dir, 'inference_metrics.yaml')
    with open(inference_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    logging.info(f"Inference results saved to {inference_file}")

    return results


def main():
    args = parse_args()

    sys.path.append(str(Path(__file__).parent))
    from kgate import Architect

    with open(args.config, 'rb') as f:
        config = tomllib.load(f)

    made_directed_relations = config.get('evaluation', {}).get('made_directed_edges', [])
    target_relations        = config.get('evaluation', {}).get('target_edges', [])
    thresholds              = config.get('evaluation', {}).get('thresholds', [])
    output_dir              = config.get('output_directory', '.')
    orpha_path              = config.get('evaluation', {}).get('inference_kg', '')
    kg_dir                  = config.get('kg_pkl', '')

    # Chargement des KGs KGATE depuis les .pt
    logging.info(f"Chargement des KGs depuis {kg_dir}...")
    kg_train = torch.load(os.path.join(kg_dir, 'kg_train.pt'), weights_only=False)
    kg_val   = torch.load(os.path.join(kg_dir, 'kg_val.pt'),   weights_only=False)
    kg_test  = torch.load(os.path.join(kg_dir, 'kg_test.pt'),  weights_only=False)
    logging.info(f"KGs chargés : train={kg_train.triplet_count}, val={kg_val.triplet_count}, test={kg_test.triplet_count}")

    # Initialisation de l'Architect avec les KGs directement
    logging.info(f"Initializing Architect from {args.config}...")
    architect = Architect(
        config_path=args.config,
        kg=(kg_train, kg_val, kg_test)
    )

    # Training
    logging.info("Starting training...")
    architect.train_model()

    # Evaluation
    logging.info("Starting evaluation on test set...")
    run_evaluation(architect, made_directed_relations, target_relations, thresholds, output_dir)

    # Inference
    if orpha_path:
        logging.info("Starting inference evaluation...")
        run_inference(architect, orpha_path, output_dir)

if __name__ == '__main__':
    main()