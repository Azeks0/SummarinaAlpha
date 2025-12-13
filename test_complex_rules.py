#!/usr/bin/env python3
"""
Comprehensive Context-Aware Rule Testing

Tests rules at multiple levels:
1. Level 1: Subword morpheme rules (suffix, prefix, infix)
2. Level 2: Word formation validation
3. Level 3: POS pattern rules
4. Combined: Full cascading validation
5. High-confidence only: Rules with ≥95% accuracy

Measures coverage and accuracy at each level.
"""

import pandas as pd
import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter


class ComprehensiveRuleTester:
    """Test context-aware rules at all levels."""
    
    def __init__(self, csv_path, context_rules_path, pos_rules_path):
        """
        Args:
            csv_path: Training data
            context_rules_path: Context-aware morpheme rules
            pos_rules_path: Multi-level POS rules
        """
        print(f"Loading data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Group tablets
        self.tablets = {}
        for tablet_id, group in self.df.groupby('tablet_id'):
            self.tablets[tablet_id] = group.sort_values('position').reset_index(drop=True)
        
        print(f"Loaded {len(self.tablets)} tablets")
        
        # Load context-aware rules
        print(f"\nLoading context-aware rules from {context_rules_path}...")
        with open(context_rules_path) as f:
            self.context_rules = json.load(f)
        
        print(f"  Suffix rules: {len(self.context_rules.get('suffix_rules', {}))}")
        print(f"  Prefix rules: {len(self.context_rules.get('prefix_rules', {}))}")
        print(f"  Infix rules: {len(self.context_rules.get('infix_rules', {}))}")
        
        # Load POS pattern rules
        print(f"\nLoading POS rules from {pos_rules_path}...")
        with open(pos_rules_path) as f:
            self.pos_rules = json.load(f)
        
        self.bigram_patterns = self._parse_patterns(
            self.pos_rules['pos_pattern_rules']['bigrams']
        )
        self.trigram_patterns = self._parse_patterns(
            self.pos_rules['pos_pattern_rules']['trigrams']
        )
        self.fourgram_patterns = self._parse_patterns(
            self.pos_rules['pos_pattern_rules']['fourgrams']
        )
        
        print(f"  Bigrams: {len(self.bigram_patterns)}")
        print(f"  Trigrams: {len(self.trigram_patterns)}")
        print(f"  4-grams: {len(self.fourgram_patterns)}")
        print()
    
    def _parse_patterns(self, patterns_dict):
        """Parse POS pattern strings."""
        parsed = {}
        for key, value in patterns_dict.items():
            context = tuple(key.split('___'))
            parsed[context] = value
        return parsed
    
    def run_comprehensive_tests(self, num_tablets=50):
        """Run all tests."""
        print("="*80)
        print("COMPREHENSIVE CONTEXT-AWARE RULE TESTING")
        print("="*80)
        print()
        
        # Sample tablets
        test_tablet_ids = random.sample(
            list(self.tablets.keys()),
            min(num_tablets, len(self.tablets))
        )
        
        # Run tests
        print("🔹 TEST 1: SUBWORD MORPHEME RULES")
        print("="*80)
        subword_results = self.test_subword_rules(test_tablet_ids)
        print()
        
        print("🔹 TEST 2: WORD FORMATION VALIDATION")
        print("="*80)
        word_results = self.test_word_formation(test_tablet_ids)
        print()
        
        print("🔹 TEST 3: POS PATTERN RULES")
        print("="*80)
        pos_results = self.test_pos_patterns(test_tablet_ids)
        print()
        
        print("🔹 TEST 4: CASCADING VALIDATION (ALL LEVELS)")
        print("="*80)
        cascade_results = self.test_cascading(test_tablet_ids)
        print()
        
        print("🔹 TEST 5: HIGH-CONFIDENCE RULES ONLY (≥95%)")
        print("="*80)
        high_conf_results = self.test_high_confidence_only(test_tablet_ids)
        print()
        
        # Summary
        self.print_overall_summary(subword_results, word_results, pos_results, 
                                   cascade_results, high_conf_results)
        
        return {
            'subword': subword_results,
            'word_formation': word_results,
            'pos_patterns': pos_results,
            'cascading': cascade_results,
            'high_confidence': high_conf_results
        }
    
    def test_subword_rules(self, tablet_ids):
        """Test Level 1: Context-aware morpheme rules."""
        print("Testing context-aware morpheme selection...")
        
        results = {
            'suffix': {'tested': 0, 'has_rule': 0, 'correct': 0, 'candidates': []},
            'prefix': {'tested': 0, 'has_rule': 0, 'correct': 0, 'candidates': []},
            'infix': {'tested': 0, 'has_rule': 0, 'correct': 0, 'candidates': []}
        }
        
        for tablet_id in tablet_ids:
            tablet = self.tablets[tablet_id]
            
            for idx, row in tablet.iterrows():
                word = str(row['form'])
                
                if '-' not in word:
                    continue
                
                parts = word.split('-')
                
                # Test suffix
                if len(parts) >= 2:
                    results['suffix']['tested'] += 1
                    stem = parts[0]
                    true_suffix = parts[-1]
                    
                    candidates = self._get_context_suffix_candidates(stem, parts[:-1])
                    
                    if candidates:
                        results['suffix']['has_rule'] += 1
                        results['suffix']['candidates'].append(len(candidates))
                        
                        if true_suffix in candidates:
                            results['suffix']['correct'] += 1
                
                # Test prefix
                if len(parts) >= 2:
                    results['prefix']['tested'] += 1
                    following = parts[1]
                    true_prefix = parts[0]
                    
                    candidates = self._get_context_prefix_candidates(following)
                    
                    if candidates:
                        results['prefix']['has_rule'] += 1
                        results['prefix']['candidates'].append(len(candidates))
                        
                        if true_prefix in candidates:
                            results['prefix']['correct'] += 1
                
                # Test infix
                if len(parts) >= 3:
                    results['infix']['tested'] += 1
                    prefix = parts[0]
                    suffix = parts[2]
                    true_infix = parts[1]
                    
                    candidates = self._get_context_infix_candidates(prefix, suffix)
                    
                    if candidates:
                        results['infix']['has_rule'] += 1
                        results['infix']['candidates'].append(len(candidates))
                        
                        if true_infix in candidates:
                            results['infix']['correct'] += 1
        
        # Print results
        self._print_level_results("SUBWORD MORPHEME RULES", results)
        
        return results
    
    def test_word_formation(self, tablet_ids):
        """Test Level 2: Word formation validation."""
        print("Testing word formation validation...")
        
        results = {
            'total': 0,
            'compound': 0,
            'valid': 0
        }
        
        for tablet_id in tablet_ids:
            tablet = self.tablets[tablet_id]
            
            for idx, row in tablet.iterrows():
                word = str(row['form'])
                results['total'] += 1
                
                if '-' not in word:
                    continue
                
                results['compound'] += 1
                parts = word.split('-')
                
                # Check if valid according to bigrams
                is_valid = True
                for i in range(len(parts) - 1):
                    bigram_key = f"{parts[i]}___{parts[i+1]}"
                    if bigram_key not in self.context_rules.get('morpheme_bigrams', {}):
                        is_valid = False
                        break
                
                if is_valid:
                    results['valid'] += 1
        
        # Print results
        print(f"\nWord Formation Validation:")
        print(f"  Total words: {results['total']}")
        print(f"  Compound words: {results['compound']}")
        print(f"  Valid formations: {results['valid']}/{results['compound']} ({results['valid']/results['compound']*100:.1f}%)")
        print()
        
        return results
    
    def test_pos_patterns(self, tablet_ids):
        """Test Level 3: POS pattern rules."""
        print("Testing POS pattern rules...")
        
        results = {
            'bigram': {'tested': 0, 'has_rule': 0, 'correct': 0},
            'trigram': {'tested': 0, 'has_rule': 0, 'correct': 0},
            'fourgram': {'tested': 0, 'has_rule': 0, 'correct': 0}
        }
        
        for tablet_id in tablet_ids:
            tablet = self.tablets[tablet_id]
            pos_sequence = [str(row['pos']) for _, row in tablet.iterrows()]
            words = [str(row['form']) for _, row in tablet.iterrows()]
            
            # Mask random positions
            maskable = list(range(1, len(tablet) - 1))
            if not maskable:
                continue
            
            num_to_mask = max(1, int(len(maskable) * 0.25))
            masked_positions = random.sample(maskable, num_to_mask)
            
            for pos in masked_positions:
                correct_word = words[pos]
                
                # Bigram
                if pos >= 1 and pos < len(pos_sequence) - 1:
                    results['bigram']['tested'] += 1
                    pattern = (pos_sequence[pos-1], pos_sequence[pos+1])
                    
                    if pattern in self.bigram_patterns:
                        results['bigram']['has_rule'] += 1
                        match = self.bigram_patterns[pattern]
                        
                        if match['word'] == correct_word:
                            results['bigram']['correct'] += 1
                
                # Trigram
                if pos >= 2 and pos < len(pos_sequence) - 1:
                    results['trigram']['tested'] += 1
                    pattern = (pos_sequence[pos-2], pos_sequence[pos-1], pos_sequence[pos+1])
                    
                    if pattern in self.trigram_patterns:
                        results['trigram']['has_rule'] += 1
                        match = self.trigram_patterns[pattern]
                        
                        if match['word'] == correct_word:
                            results['trigram']['correct'] += 1
                
                # 4-gram
                if pos >= 2 and pos < len(pos_sequence) - 2:
                    results['fourgram']['tested'] += 1
                    pattern = (pos_sequence[pos-2], pos_sequence[pos-1], 
                              pos_sequence[pos+1], pos_sequence[pos+2])
                    
                    if pattern in self.fourgram_patterns:
                        results['fourgram']['has_rule'] += 1
                        match = self.fourgram_patterns[pattern]
                        
                        if match['word'] == correct_word:
                            results['fourgram']['correct'] += 1
        
        # Print results
        print(f"\nPOS Pattern Rules:")
        for pattern_type in ['bigram', 'trigram', 'fourgram']:
            r = results[pattern_type]
            coverage = r['has_rule'] / r['tested'] * 100 if r['tested'] > 0 else 0
            accuracy = r['correct'] / r['has_rule'] * 100 if r['has_rule'] > 0 else 0
            
            print(f"\n  {pattern_type.upper()}:")
            print(f"    Coverage: {r['has_rule']}/{r['tested']} ({coverage:.1f}%)")
            print(f"    Accuracy: {r['correct']}/{r['has_rule']} ({accuracy:.1f}%)")
        print()
        
        return results
    
    def test_cascading(self, tablet_ids):
        """Test Level 4: Full cascading validation."""
        print("Testing cascading validation (all levels combined)...")
        
        results = {
            'total_positions': 0,
            'level1_constrained': 0,
            'level2_constrained': 0,
            'level3_constrained': 0,
            'final_correct': 0,
            'avg_reduction': []
        }
        
        for tablet_id in tablet_ids:
            tablet = self.tablets[tablet_id]
            pos_sequence = [str(row['pos']) for _, row in tablet.iterrows()]
            words = [str(row['form']) for _, row in tablet.iterrows()]
            
            for idx, row in tablet.iterrows():
                if idx == 0 or idx == len(tablet) - 1:
                    continue
                
                word = str(row['form'])
                if '-' not in word:
                    continue
                
                results['total_positions'] += 1
                parts = word.split('-')
                
                # Test suffix position
                if len(parts) >= 2:
                    stem = parts[0]
                    true_suffix = parts[-1]
                    
                    # Level 1: Subword
                    candidates_l1 = self._get_context_suffix_candidates(stem, parts[:-1])
                    initial_count = 2137
                    
                    if candidates_l1:
                        results['level1_constrained'] += 1
                        
                        # Level 2: Word formation
                        valid_words = []
                        for suffix in candidates_l1:
                            test_word = '-'.join(parts[:-1] + [suffix])
                            if self._is_valid_word_formation(test_word):
                                valid_words.append(suffix)
                        
                        if valid_words:
                            results['level2_constrained'] += 1
                            
                            # Level 3: POS patterns
                            if idx >= 1 and idx < len(pos_sequence) - 1:
                                pattern = (pos_sequence[idx-1], pos_sequence[idx+1])
                                
                                if pattern in self.bigram_patterns:
                                    results['level3_constrained'] += 1
                                    
                                    predicted = self.bigram_patterns[pattern]['word']
                                    if predicted == word:
                                        results['final_correct'] += 1
                            
                            # Track reduction
                            reduction = (1 - len(valid_words) / initial_count) * 100
                            results['avg_reduction'].append(reduction)
        
        # Print results
        print(f"\nCascading Validation (All Levels):")
        print(f"  Total positions tested: {results['total_positions']}")
        print(f"  Level 1 (Subword) constrained: {results['level1_constrained']} ({results['level1_constrained']/results['total_positions']*100:.1f}%)")
        print(f"  Level 2 (Word form) constrained: {results['level2_constrained']} ({results['level2_constrained']/results['total_positions']*100:.1f}%)")
        print(f"  Level 3 (POS) constrained: {results['level3_constrained']} ({results['level3_constrained']/results['total_positions']*100:.1f}%)")
        print(f"  Final accuracy: {results['final_correct']}/{results['level3_constrained']} ({results['final_correct']/results['level3_constrained']*100:.1f}%)" if results['level3_constrained'] > 0 else "  Final accuracy: N/A")
        
        if results['avg_reduction']:
            print(f"  Average candidate reduction: {np.mean(results['avg_reduction']):.1f}%")
        print()
        
        return results
    
    def test_high_confidence_only(self, tablet_ids, min_confidence=0.95):
        """Test Level 5: Only rules with ≥95% confidence."""
        print(f"Testing high-confidence rules only (≥{min_confidence*100:.0f}%)...")
        
        results = {
            'suffix_high_conf': {'tested': 0, 'has_rule': 0, 'correct': 0},
            'prefix_high_conf': {'tested': 0, 'has_rule': 0, 'correct': 0},
            'pos_high_conf': {'tested': 0, 'has_rule': 0, 'correct': 0}
        }
        
        for tablet_id in tablet_ids:
            tablet = self.tablets[tablet_id]
            pos_sequence = [str(row['pos']) for _, row in tablet.iterrows()]
            words = [str(row['form']) for _, row in tablet.iterrows()]
            
            for idx, row in tablet.iterrows():
                word = str(row['form'])
                
                if '-' not in word:
                    continue
                
                parts = word.split('-')
                
                # High-confidence suffix rules
                if len(parts) >= 2:
                    results['suffix_high_conf']['tested'] += 1
                    stem = parts[0]
                    true_suffix = parts[-1]
                    
                    candidates = self._get_high_confidence_suffixes(stem, parts[:-1], min_confidence)
                    
                    if candidates:
                        results['suffix_high_conf']['has_rule'] += 1
                        
                        if true_suffix in candidates:
                            results['suffix_high_conf']['correct'] += 1
                
                # High-confidence POS patterns
                if idx >= 1 and idx < len(pos_sequence) - 1:
                    results['pos_high_conf']['tested'] += 1
                    
                    # Try 4-gram first (most confident)
                    if idx >= 2 and idx < len(pos_sequence) - 2:
                        pattern = (pos_sequence[idx-2], pos_sequence[idx-1],
                                  pos_sequence[idx+1], pos_sequence[idx+2])
                        
                        if pattern in self.fourgram_patterns:
                            match = self.fourgram_patterns[pattern]
                            if match['accuracy'] >= min_confidence:
                                results['pos_high_conf']['has_rule'] += 1
                                
                                if match['word'] == word:
                                    results['pos_high_conf']['correct'] += 1
        
        # Print results
        print(f"\nHigh-Confidence Rules (≥{min_confidence*100:.0f}%):")
        for rule_type in ['suffix_high_conf', 'pos_high_conf']:
            r = results[rule_type]
            coverage = r['has_rule'] / r['tested'] * 100 if r['tested'] > 0 else 0
            accuracy = r['correct'] / r['has_rule'] * 100 if r['has_rule'] > 0 else 0
            
            print(f"\n  {rule_type.replace('_', ' ').upper()}:")
            print(f"    Coverage: {r['has_rule']}/{r['tested']} ({coverage:.1f}%)")
            print(f"    Accuracy: {r['correct']}/{r['has_rule']} ({accuracy:.1f}%)")
        print()
        
        return results
    
    def _get_context_suffix_candidates(self, stem, preceding_parts):
        """Get suffix candidates using context."""
        candidates = set()
        
        # Check after stem pattern
        pattern_key = f"after_stem_{stem}"
        if pattern_key in self.context_rules.get('suffix_rules', {}):
            for suffix, info in self.context_rules['suffix_rules'][pattern_key].items():
                candidates.add(suffix)
        
        return candidates
    
    def _get_context_prefix_candidates(self, following_morpheme):
        """Get prefix candidates using context."""
        candidates = set()
        
        # Check before morpheme pattern
        pattern_key = f"before_{following_morpheme}"
        if pattern_key in self.context_rules.get('prefix_rules', {}):
            for prefix, info in self.context_rules['prefix_rules'][pattern_key].items():
                candidates.add(prefix)
        
        return candidates
    
    def _get_context_infix_candidates(self, prefix, suffix):
        """Get infix candidates using context."""
        candidates = set()
        
        # Check between morphemes pattern
        pattern_key = f"between_{prefix}_and_{suffix}"
        if pattern_key in self.context_rules.get('infix_rules', {}):
            for infix, info in self.context_rules['infix_rules'][pattern_key].items():
                candidates.add(infix)
        
        return candidates
    
    def _get_high_confidence_suffixes(self, stem, preceding_parts, min_conf):
        """Get only high-confidence suffix candidates."""
        candidates = set()
        
        pattern_key = f"after_stem_{stem}"
        if pattern_key in self.context_rules.get('suffix_rules', {}):
            for suffix, info in self.context_rules['suffix_rules'][pattern_key].items():
                accuracy = info.get('accuracy', info.get('frequency', 0))
                if accuracy >= min_conf:
                    candidates.add(suffix)
        
        return candidates
    
    def _is_valid_word_formation(self, word):
        """Check if word formation is valid."""
        if '-' not in word:
            return True
        
        parts = word.split('-')
        for i in range(len(parts) - 1):
            bigram_key = f"{parts[i]}___{parts[i+1]}"
            if bigram_key not in self.context_rules.get('morpheme_bigrams', {}):
                return False
        
        return True
    
    def _print_level_results(self, level_name, results):
        """Print results for a testing level."""
        print(f"\n{level_name}:")
        
        for rule_type in ['suffix', 'prefix', 'infix']:
            if rule_type not in results:
                continue
            
            r = results[rule_type]
            tested = r['tested']
            has_rule = r['has_rule']
            correct = r['correct']
            
            coverage = has_rule / tested * 100 if tested > 0 else 0
            accuracy = correct / has_rule * 100 if has_rule > 0 else 0
            
            avg_candidates = np.mean(r['candidates']) if r['candidates'] else 0
            
            print(f"\n  {rule_type.upper()} Rules:")
            print(f"    Positions tested: {tested}")
            print(f"    Has rule: {has_rule} ({coverage:.1f}% coverage)")
            print(f"    Correct: {correct}/{has_rule} ({accuracy:.1f}% accuracy)")
            if avg_candidates > 0:
                reduction = (1 - avg_candidates / 2137) * 100
                print(f"    Avg candidates: {avg_candidates:.1f} ({reduction:.1f}% reduction from 2137)")
        print()
    
    def print_overall_summary(self, subword, word, pos, cascade, high_conf):
        """Print overall testing summary."""
        print("\n" + "="*80)
        print("OVERALL TESTING SUMMARY")
        print("="*80)
        print()
        
        print("📊 COVERAGE BY LEVEL:")
        print("-"*80)
        
        # Subword
        suffix_cov = subword['suffix']['has_rule'] / subword['suffix']['tested'] * 100 if subword['suffix']['tested'] > 0 else 0
        prefix_cov = subword['prefix']['has_rule'] / subword['prefix']['tested'] * 100 if subword['prefix']['tested'] > 0 else 0
        
        print(f"Level 1 (Subword):")
        print(f"  Suffix: {suffix_cov:.1f}% coverage")
        print(f"  Prefix: {prefix_cov:.1f}% coverage")
        
        # Word
        word_valid = word['valid'] / word['compound'] * 100 if word['compound'] > 0 else 0
        print(f"\nLevel 2 (Word Formation):")
        print(f"  Valid: {word_valid:.1f}% of compound words")
        
        # POS
        bigram_cov = pos['bigram']['has_rule'] / pos['bigram']['tested'] * 100 if pos['bigram']['tested'] > 0 else 0
        trigram_cov = pos['trigram']['has_rule'] / pos['trigram']['tested'] * 100 if pos['trigram']['tested'] > 0 else 0
        
        print(f"\nLevel 3 (POS Patterns):")
        print(f"  Bigrams: {bigram_cov:.1f}% coverage")
        print(f"  Trigrams: {trigram_cov:.1f}% coverage")
        
        print("\n" + "="*80)
        print("🎯 ACCURACY BY LEVEL:")
        print("-"*80)
        
        # Accuracies
        suffix_acc = subword['suffix']['correct'] / subword['suffix']['has_rule'] * 100 if subword['suffix']['has_rule'] > 0 else 0
        bigram_acc = pos['bigram']['correct'] / pos['bigram']['has_rule'] * 100 if pos['bigram']['has_rule'] > 0 else 0
        
        print(f"Level 1 (Subword): {suffix_acc:.1f}%")
        print(f"Level 3 (POS): {bigram_acc:.1f}%")
        
        # High confidence
        hc_suffix_cov = high_conf['suffix_high_conf']['has_rule'] / high_conf['suffix_high_conf']['tested'] * 100 if high_conf['suffix_high_conf']['tested'] > 0 else 0
        hc_suffix_acc = high_conf['suffix_high_conf']['correct'] / high_conf['suffix_high_conf']['has_rule'] * 100 if high_conf['suffix_high_conf']['has_rule'] > 0 else 0
        
        print(f"\nHigh-Confidence Rules (≥95%):")
        print(f"  Coverage: {hc_suffix_cov:.1f}%")
        print(f"  Accuracy: {hc_suffix_acc:.1f}%")
        
        print()


def main():
    """Run comprehensive testing."""
    
    csv_path = None
    for path in ["summerian_data/train_tablets.csv", "train_tablets.csv"]:
        if Path(path).exists():
            csv_path = path
            break
    
    if not csv_path:
        print("❌ CSV file not found!")
        return
    
    # Test different rule thresholds
    for threshold in [90, 85]:
        context_rules = f"context_aware_rules_{threshold}.json"
        pos_rules = f"multi_level_rules_{threshold}.json"
        
        if not Path(context_rules).exists():
            print(f"⚠️  {context_rules} not found, skipping...")
            continue
        
        if not Path(pos_rules).exists():
            print(f"⚠️  {pos_rules} not found, skipping...")
            continue
        
        print("\n" + "="*80)
        print(f"TESTING {threshold}% THRESHOLD RULES")
        print("="*80)
        print()
        
        tester = ComprehensiveRuleTester(csv_path, context_rules, pos_rules)
        results = tester.run_comprehensive_tests(num_tablets=50)
        
        print()


if __name__ == "__main__":
    main()