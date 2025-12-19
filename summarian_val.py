"""
Validator Evaluation Script - Fixed Version

Trains and evaluates the validator in one run to avoid pickle issues.

Usage:
    python evaluate_validator.py --train_data train.json --test_data test.json --pos_data pos.csv

Author: Pipeline Builder
Date: 2025
"""

import json
import argparse
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import random


class ValidatorEvaluator:
    """Evaluate validator performance on masked language modeling task."""
    
    def __init__(self, validator, test_data):
        """
        Initialize evaluator.
        
        Args:
            validator: Trained SumerianValidator object
            test_data: List of tokenized test examples
        """
        self.validator = validator
        self.test_data = test_data
        
        # Statistics
        self.stats = {
            'total_masks': 0,
            'correct_in_filtered': 0,
            'correct_rank_1': 0,
            'correct_rank_3': 0,
            'correct_rank_5': 0,
            'correct_rank_10': 0,
            'avg_candidates_before': 0,
            'avg_candidates_after': 0,
            'correct_filtered_by_subword': 0,
            'correct_filtered_by_word': 0,
            'correct_filtered_by_pos': 0
        }
    
    def create_candidate_pool(self, masked_token: str, all_tokens: set, pool_size: int = 100) -> List[str]:
        """Create a candidate pool simulating BERT predictions."""
        candidates = [masked_token]
        other_tokens = list(all_tokens - {masked_token})
        
        if masked_token.startswith('##'):
            continuation_tokens = [t for t in other_tokens if t.startswith('##')]
            random.shuffle(continuation_tokens)
            candidates.extend(continuation_tokens[:pool_size - 1])
            
            if len(candidates) < pool_size:
                starter_tokens = [t for t in other_tokens if not t.startswith('##')]
                random.shuffle(starter_tokens)
                candidates.extend(starter_tokens[:pool_size - len(candidates)])
        else:
            starter_tokens = [t for t in other_tokens if not t.startswith('##')]
            random.shuffle(starter_tokens)
            candidates.extend(starter_tokens[:pool_size - 1])
            
            if len(candidates) < pool_size:
                continuation_tokens = [t for t in other_tokens if t.startswith('##')]
                random.shuffle(continuation_tokens)
                candidates.extend(continuation_tokens[:pool_size - len(candidates)])
        
        random.shuffle(candidates)
        return candidates[:pool_size]
    
    def evaluate_single_mask(
        self,
        tokens: List[str],
        mask_index: int,
        candidate_pool: List[str]
    ) -> Dict:
        """Evaluate validator on a single masked position."""
        correct_token = tokens[mask_index]
        context_before = tokens[:mask_index]
        context_after = tokens[mask_index + 1:]
        
        scored_candidates = self.validator.validate_candidates(
            candidates=candidate_pool,
            context_before=context_before,
            context_after=context_after
        )
        
        filtered_tokens = [s.candidate for s in scored_candidates]
        
        result = {
            'correct_token': correct_token,
            'candidates_before': len(candidate_pool),
            'candidates_after': len(filtered_tokens),
            'correct_in_filtered': correct_token in filtered_tokens,
            'correct_rank': filtered_tokens.index(correct_token) + 1 if correct_token in filtered_tokens else -1,
            'reduction_ratio': 1 - (len(filtered_tokens) / len(candidate_pool)) if len(candidate_pool) > 0 else 0,
            'scored_candidates': scored_candidates
        }
        
        if correct_token not in filtered_tokens:
            test_validation = self.validator.validate_candidates(
                candidates=[correct_token],
                context_before=context_before,
                context_after=context_after
            )
            
            if not test_validation:
                result['filtered_at_level'] = 'subword'
            elif 'subword' not in test_validation[0].passed_levels:
                result['filtered_at_level'] = 'subword'
            elif 'word' not in test_validation[0].passed_levels:
                result['filtered_at_level'] = 'word'
            elif 'pos' not in test_validation[0].passed_levels:
                result['filtered_at_level'] = 'pos'
            else:
                result['filtered_at_level'] = 'unknown'
        else:
            result['filtered_at_level'] = None
        
        return result
    
    def evaluate_all(self, num_masks_per_example: int = 3, candidate_pool_size: int = 100):
        """Evaluate validator on all test data."""
        print(f"\nEvaluating validator...")
        print(f"  Masks per example: {num_masks_per_example}")
        print(f"  Candidate pool size: {candidate_pool_size}")
        print("=" * 70)
        
        all_tokens = set()
        for example in self.test_data:
            all_tokens.update(example['tokens'])
        
        print(f"Vocabulary size: {len(all_tokens)} tokens")
        
        results = []
        
        for example_idx, example in enumerate(self.test_data):
            tokens = example['tokens']
            
            if len(tokens) < 3:
                continue
            
            maskable_positions = list(range(1, len(tokens) - 1))
            if len(maskable_positions) < num_masks_per_example:
                mask_positions = maskable_positions
            else:
                mask_positions = random.sample(maskable_positions, num_masks_per_example)
            
            for mask_idx in mask_positions:
                masked_token = tokens[mask_idx]
                candidates = self.create_candidate_pool(masked_token, all_tokens, candidate_pool_size)
                
                result = self.evaluate_single_mask(
                    tokens=tokens,
                    mask_index=mask_idx,
                    candidate_pool=candidates
                )
                
                results.append(result)
                
                self.stats['total_masks'] += 1
                if result['correct_in_filtered']:
                    self.stats['correct_in_filtered'] += 1
                    
                    rank = result['correct_rank']
                    if rank == 1:
                        self.stats['correct_rank_1'] += 1
                    if rank <= 3:
                        self.stats['correct_rank_3'] += 1
                    if rank <= 5:
                        self.stats['correct_rank_5'] += 1
                    if rank <= 10:
                        self.stats['correct_rank_10'] += 1
                else:
                    if result['filtered_at_level'] == 'subword':
                        self.stats['correct_filtered_by_subword'] += 1
                    elif result['filtered_at_level'] == 'word':
                        self.stats['correct_filtered_by_word'] += 1
                    elif result['filtered_at_level'] == 'pos':
                        self.stats['correct_filtered_by_pos'] += 1
                
                self.stats['avg_candidates_before'] += result['candidates_before']
                self.stats['avg_candidates_after'] += result['candidates_after']
            
            if (example_idx + 1) % 10 == 0:
                print(f"Processed {example_idx + 1}/{len(self.test_data)} examples...")
        
        if self.stats['total_masks'] > 0:
            self.stats['avg_candidates_before'] /= self.stats['total_masks']
            self.stats['avg_candidates_after'] /= self.stats['total_masks']
        
        return results
    
    def print_report(self):
        """Print evaluation report."""
        print("\n" + "=" * 70)
        print("VALIDATOR EVALUATION REPORT")
        print("=" * 70)
        
        total = self.stats['total_masks']
        
        if total == 0:
            print("No masks evaluated!")
            return
        
        print(f"\nTotal masked positions evaluated: {total}")
        
        print("\n--- ACCURACY (Recall) ---")
        print(f"Correct token in filtered set: {self.stats['correct_in_filtered']}/{total} "
              f"({self.stats['correct_in_filtered']/total*100:.1f}%)")
        
        if self.stats['correct_in_filtered'] > 0:
            print(f"\nRanking (among filtered candidates):")
            print(f"  Rank 1:  {self.stats['correct_rank_1']}/{total} "
                  f"({self.stats['correct_rank_1']/total*100:.1f}%)")
            print(f"  Top-3:   {self.stats['correct_rank_3']}/{total} "
                  f"({self.stats['correct_rank_3']/total*100:.1f}%)")
            print(f"  Top-5:   {self.stats['correct_rank_5']}/{total} "
                  f"({self.stats['correct_rank_5']/total*100:.1f}%)")
            print(f"  Top-10:  {self.stats['correct_rank_10']}/{total} "
                  f"({self.stats['correct_rank_10']/total*100:.1f}%)")
        
        print("\n--- COVERAGE (Filtering) ---")
        print(f"Average candidates before: {self.stats['avg_candidates_before']:.1f}")
        print(f"Average candidates after:  {self.stats['avg_candidates_after']:.1f}")
        reduction = (1 - self.stats['avg_candidates_after'] / self.stats['avg_candidates_before']) * 100
        print(f"Reduction: {reduction:.1f}%")
        
        print("\n--- ERROR ANALYSIS ---")
        filtered_out = total - self.stats['correct_in_filtered']
        if filtered_out > 0:
            print(f"Correct token filtered out: {filtered_out}/{total} ({filtered_out/total*100:.1f}%)")
            print(f"  Filtered at subword level: {self.stats['correct_filtered_by_subword']} "
                  f"({self.stats['correct_filtered_by_subword']/filtered_out*100:.1f}% of errors)")
            print(f"  Filtered at word level:    {self.stats['correct_filtered_by_word']} "
                  f"({self.stats['correct_filtered_by_word']/filtered_out*100:.1f}% of errors)")
            print(f"  Filtered at POS level:     {self.stats['correct_filtered_by_pos']} "
                  f"({self.stats['correct_filtered_by_pos']/filtered_out*100:.1f}% of errors)")
        
        print("\n--- SUMMARY ---")
        print(f"✓ Recall (correct in filtered): {self.stats['correct_in_filtered']/total*100:.1f}%")
        print(f"✓ Reduction (candidates filtered): {reduction:.1f}%")
        print(f"✓ Top-10 Accuracy: {self.stats['correct_rank_10']/total*100:.1f}%")
        
        print("\n" + "=" * 70)
    
    def analyze_errors(self, results: List[Dict], top_n: int = 10):
        """Analyze common error patterns."""
        print("\n--- ERROR PATTERN ANALYSIS ---")
        
        errors = [r for r in results if not r['correct_in_filtered']]
        
        if not errors:
            print("No errors to analyze!")
            return
        
        print(f"\nTotal errors: {len(errors)} / {len(results)} ({len(errors)/len(results)*100:.1f}%)")
        
        continuation_errors = sum(1 for e in errors if e['correct_token'].startswith('##'))
        starter_errors = len(errors) - continuation_errors
        
        print(f"\nError by token type:")
        print(f"  Continuation tokens (##): {continuation_errors} ({continuation_errors/len(errors)*100:.1f}%)")
        print(f"  Starter tokens:           {starter_errors} ({starter_errors/len(errors)*100:.1f}%)")
        
        filtered_tokens = Counter([e['correct_token'] for e in errors])
        print(f"\nMost commonly filtered tokens:")
        for token, count in filtered_tokens.most_common(top_n):
            print(f"  '{token}': {count} times")


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate validator')
    parser.add_argument('--train_data', required=True, help='Path to training data (.json)')
    parser.add_argument('--test_data', required=True, help='Path to test data (.json)')
    parser.add_argument('--pos_data', default=None, help='Path to POS data (.csv)')
    parser.add_argument('--num_masks', type=int, default=3, help='Number of masks per example')
    parser.add_argument('--pool_size', type=int, default=100, help='Candidate pool size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_validator', default=None, help='Path to save trained validator')
    parser.add_argument('--min_frequency', type=int, default=2, help='Min frequency for patterns')
    parser.add_argument('--confidence_threshold', type=float, default=0.01, help='Min confidence threshold')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Import from your renamed file
    from summarian_val_trainer import SumerianValidatorTrainer, SumerianValidator
    
    # Train validator
    print("=" * 70)
    print("TRAINING VALIDATOR")
    print("=" * 70)
    trainer = SumerianValidatorTrainer(
        min_frequency=args.min_frequency,
        confidence_threshold=args.confidence_threshold
    )
    trainer.train(args.train_data, args.pos_data)
    
    # Get rules object directly (no pickle!)
    rules = trainer.get_rules()
    validator = SumerianValidator(rules)
    
    # Optionally save
    if args.save_validator:
        trainer.save_rules(args.save_validator)
        print(f"\nSaved validator to {args.save_validator}")
    
    # Load test data
    print(f"\nLoading test data from {args.test_data}...")
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test examples")
    
    # Create evaluator
    evaluator = ValidatorEvaluator(validator, test_data)
    
    # Run evaluation
    results = evaluator.evaluate_all(
        num_masks_per_example=args.num_masks,
        candidate_pool_size=args.pool_size
    )
    
    # Print report
    evaluator.print_report()
    evaluator.analyze_errors(results)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()