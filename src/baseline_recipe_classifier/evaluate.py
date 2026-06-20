#!/usr/bin/env python3
"""
Evaluation script for recipe classifier.
Runs ablation study comparing baseline vs. context-enhanced predictions,
and prompt_v1 vs. prompt_v2 comparison with per-category accuracy.
"""

import sys
import os
import pandas as pd
from pathlib import Path
from baseline_recipe_classifier.__main__ import (
    load_config,
    load_recipe_dataset,
    predict_category
)


def evaluate_classifier(test_dishes, config, dataset=None, use_context=True):
    """
    Evaluate classifier on test dishes.

    Args:
        test_dishes: List of (dish_name, ground_truth_category) tuples
        config: Configuration dictionary
        dataset: Optional recipe dataset
        use_context: Whether to use context

    Returns:
        tuple: (accuracy, correct_count, total_count, predictions)
    """
    predictions = []
    correct = 0
    total = len(test_dishes)

    for dish_name, ground_truth in test_dishes:
        predicted = predict_category(
            dish_name,
            config,
            dataset=dataset,
            use_context=use_context
        )

        is_correct = predicted == ground_truth
        if is_correct:
            correct += 1

        predictions.append({
            'dish': dish_name,
            'ground_truth': ground_truth,
            'predicted': predicted,
            'correct': is_correct
        })

    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, correct, total, predictions


def print_results(mode, accuracy, correct, total, predictions, verbose=False):
    """Print evaluation results."""
    print(f"\n{'='*60}")
    print(f"{mode.upper()} MODE")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")

    if verbose:
        print(f"\nDetailed Results:")
        for pred in predictions:
            status = "OK" if pred['correct'] else "XX"
            print(f"  {status} {pred['dish'][:40]:40} | GT: {pred['ground_truth']} | Pred: {pred['predicted']}")


def run_prompt_comparison(config, labeled_csv_path, eval_csv_path, dataset):
    """
    Compare prompt_v1 (zero-shot) vs prompt_v2 (few-shot + wait_time field).

    Builds a balanced 15-pair eval set: A/B/C from prompt_eval.csv plus
    3 D and 3 E examples sampled from the labeled dataset.
    Prints per-category accuracy, confusion matrix, and a Week 5 decision.
    """
    ALL_CATS = ['A', 'B', 'C', 'D', 'E']

    # Load A/B/C golden pairs from prompt_eval.csv
    eval_df = pd.read_csv(eval_csv_path)
    # Sample 3 per present category to keep the set balanced
    abc_rows = []
    for cat in ['A', 'B', 'C']:
        sub = eval_df[eval_df['label'] == cat]
        abc_rows.append(sub.sample(n=min(3, len(sub)), random_state=42))
    abc_df = pd.concat(abc_rows, ignore_index=True)[['recipe', 'label']].copy()
    abc_df['wait_mins'] = 0

    # Sample 3 D and 3 E from the labeled dataset
    labeled = pd.read_csv(labeled_csv_path)
    de_rows = []
    for cat in ['D', 'E']:
        sub = labeled[labeled['Category Code'] == cat].dropna(subset=['Name', 'wait_mins'])
        sampled = sub.sample(n=min(3, len(sub)), random_state=42)
        for _, row in sampled.iterrows():
            de_rows.append({
                'recipe': row['Name'],
                'label': cat,
                'wait_mins': int(row['wait_mins'])
            })
    de_df = pd.DataFrame(de_rows)

    full_eval = pd.concat([abc_df, de_df], ignore_index=True)
    total = len(full_eval)
    print(f"\nEval set: {total} golden pairs")
    print("Distribution: " + ", ".join(
        f"{cat}={len(full_eval[full_eval['label']==cat])}" for cat in ALL_CATS
    ))

    results = {'v1': [], 'v2': []}

    for version in ['v1', 'v2']:
        print(f"\n[Running prompt_{version}] {total} recipes...")
        for _, row in full_eval.iterrows():
            pred = predict_category(
                row['recipe'],
                config,
                dataset=dataset,
                use_context=True,
                prompt_version=version,
                wait_time=int(row['wait_mins'])
            )
            results[version].append({
                'recipe': row['recipe'],
                'true': row['label'],
                'pred': pred,
                'correct': pred == row['label']
            })

    # --- Print comparison table ---
    print(f"\n{'='*65}")
    print("PROMPT v1 vs v2 — PER-CATEGORY ACCURACY")
    print(f"{'='*65}")
    header = f"{'Category':<12}" + "  ".join(f"{'v1':>6}  {'v2':>6}" for _ in ['']).replace("  ", "")
    print(f"{'Category':<12}  {'v1 acc':>8}  {'v2 acc':>8}  {'v1 n/N':>8}  {'v2 n/N':>8}")
    print("-" * 65)

    de_v1_correct = de_v1_total = de_v2_correct = de_v2_total = 0

    for cat in ALL_CATS:
        v1_rows = [r for r in results['v1'] if r['true'] == cat]
        v2_rows = [r for r in results['v2'] if r['true'] == cat]
        if not v1_rows:
            continue
        v1_c = sum(1 for r in v1_rows if r['correct'])
        v2_c = sum(1 for r in v2_rows if r['correct'])
        n = len(v1_rows)
        v1_acc = v1_c / n * 100
        v2_acc = v2_c / n * 100
        marker = " ← threshold" if cat in ('D', 'E') else ""
        print(f"  {cat:<10}  {v1_acc:>6.0f}%  {v2_acc:>6.0f}%  {v1_c:>3}/{n}      {v2_c:>3}/{n}{marker}")
        if cat in ('D', 'E'):
            de_v1_correct += v1_c; de_v1_total += n
            de_v2_correct += v2_c; de_v2_total += n

    v1_overall = sum(1 for r in results['v1'] if r['correct']) / total * 100
    v2_overall = sum(1 for r in results['v2'] if r['correct']) / total * 100
    print("-" * 65)
    print(f"  {'Overall':<10}  {v1_overall:>6.0f}%  {v2_overall:>6.0f}%")

    # D/E combined accuracy
    de_v1_acc = de_v1_correct / de_v1_total * 100 if de_v1_total else 0
    de_v2_acc = de_v2_correct / de_v2_total * 100 if de_v2_total else 0
    print(f"\n  D+E combined:  v1={de_v1_acc:.0f}%  v2={de_v2_acc:.0f}%  (threshold: 70%)")

    # Confusion matrix
    print(f"\n{'='*65}")
    print("CONFUSION MATRIX — prompt_v2")
    print(f"{'':>12}" + "".join(f"  pred_{c}" for c in ALL_CATS))
    for true_cat in ALL_CATS:
        row_str = f"  true_{true_cat:<6}"
        for pred_cat in ALL_CATS:
            count = sum(
                1 for r in results['v2']
                if r['true'] == true_cat and r['pred'] == pred_cat
            )
            row_str += f"  {count:>6}"
        print(row_str)

    # Decision
    print(f"\n{'='*65}")
    if de_v2_acc >= 70:
        print("RECOMMENDATION: Prompt v2 meets the 70% D/E threshold.")
        print("Prompt engineering is sufficient — Week 5 adapters are optional.")
    else:
        print("RECOMMENDATION: D/E accuracy below 70% threshold after prompt refinement.")
        print("Proceed to Week 5 LoRA adapters using this eval set as training signal.")
    print(f"{'='*65}\n")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate recipe classifier with ablation study"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to labeled dataset CSV"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of test samples (default: 10)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed predictions"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run prompt_v1 vs prompt_v2 comparison on balanced A-E eval set"
    )

    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine dataset path
    if args.dataset:
        dataset_path = args.dataset
    elif 'dataset' in config and 'path' in config['dataset']:
        package_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(os.path.dirname(package_dir))
        dataset_path = os.path.join(project_root, config['dataset']['path'])
    else:
        print("Error: No dataset path provided", file=sys.stderr)
        sys.exit(1)

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_recipe_dataset(dataset_path)

    if dataset is None:
        print("Error: Failed to load dataset", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(dataset)} recipes")

    # Check for ground truth labels
    if 'Category Code' not in dataset.columns:
        print("Error: Dataset missing 'Category Code' column", file=sys.stderr)
        sys.exit(1)

    # --compare: prompt_v1 vs prompt_v2 on balanced A-E eval set
    if args.compare:
        package_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(os.path.dirname(package_dir))
        eval_csv = os.path.join(project_root, 'data', 'prompt_eval.csv')
        labeled_csv = dataset_path
        if not os.path.exists(eval_csv):
            print(f"Error: eval set not found at {eval_csv}", file=sys.stderr)
            sys.exit(1)
        run_prompt_comparison(config, labeled_csv, eval_csv, dataset)
        return

    # Sample test dishes (filter out any with missing labels)
    labeled_dataset = dataset[dataset['Category Code'].notna()].copy()
    print(f"Found {len(labeled_dataset)} labeled recipes")

    if len(labeled_dataset) < args.num_samples:
        print(f"Warning: Only {len(labeled_dataset)} labeled recipes available", file=sys.stderr)
        num_samples = len(labeled_dataset)
    else:
        num_samples = args.num_samples

    # Random sample
    test_sample = labeled_dataset.sample(n=num_samples, random_state=args.seed)

    # Create test dishes list
    test_dishes = [
        (dish_name, row['Category Code'])
        for dish_name, row in test_sample.iterrows()
    ]

    print(f"\nRunning ablation study with {num_samples} test samples...")
    print(f"Using model: {config['model']['name']}")

    # Run baseline evaluation (no context)
    print("\n[1/2] Running baseline (no context)...")
    baseline_accuracy, baseline_correct, baseline_total, baseline_predictions = evaluate_classifier(
        test_dishes,
        config,
        dataset=None,
        use_context=False
    )

    # Run context-enhanced evaluation
    print("\n[2/2] Running context-enhanced...")
    context_accuracy, context_correct, context_total, context_predictions = evaluate_classifier(
        test_dishes,
        config,
        dataset=dataset,
        use_context=True
    )

    # Print results
    print_results("Baseline", baseline_accuracy, baseline_correct, baseline_total, baseline_predictions, args.verbose)
    print_results("Context-Enhanced", context_accuracy, context_correct, context_total, context_predictions, args.verbose)

    # Print comparison
    improvement = context_accuracy - baseline_accuracy
    print(f"\n{'='*60}")
    print(f"COMPARISON")
    print(f"{'='*60}")
    print(f"Baseline:         {baseline_accuracy:.1f}%")
    print(f"Context-Enhanced: {context_accuracy:.1f}%")
    print(f"Improvement:      {improvement:+.1f} percentage points")

    if improvement > 0:
        print(f"\n✓ Context improved accuracy!")
    elif improvement < 0:
        print(f"\n✗ Context decreased accuracy")
    else:
        print(f"\n− No change in accuracy")

    print(f"\nTest samples: {num_samples}")
    print(f"Random seed: {args.seed}")


if __name__ == '__main__':
    main()
