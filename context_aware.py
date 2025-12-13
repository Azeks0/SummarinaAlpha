#!/usr/bin/env python3
"""
Context-Aware Morphological Rule Extractor

Extracts rules that use CONTEXT to narrow morpheme candidates:
- Suffix after stem type and preceding morphemes
- Prefix before following morphemes and after preceding word
- Infix between specific morpheme pairs

Goal: Reduce 400 candidates → 3-10 candidates using context!
"""

import pandas as pd
import json
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path


class ContextAwareMorphemeExtractor:
    """Extract context-aware morpheme selection rules."""
    
    def __init__(self, csv_path, min_accuracy=0.85, min_occurrences=5):
        """
        Args:
            csv_path: Path to training data
            min_accuracy: Minimum confidence threshold
            min_occurrences: Minimum pattern frequency
        """
        self.min_accuracy = min_accuracy
        self.min_occurrences = min_occurrences
        
        print(f"Loading data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Group tablets
        self.tablets = {}
        for tablet_id, group in self.df.groupby('tablet_id'):
            self.tablets[tablet_id] = group.sort_values('position').reset_index(drop=True)
        
        print(f"Loaded {len(self.tablets)} tablets")
        print(f"Extraction criteria:")
        print(f"  - Minimum accuracy: {min_accuracy*100:.0f}%")
        print(f"  - Minimum occurrences: {min_occurrences}")
        print()
        
        self.rules = {
            'suffix_rules': {},
            'prefix_rules': {},
            'infix_rules': {},
            'morpheme_bigrams': {}
        }
    
    def extract_all_rules(self):
        """Extract context-aware rules at all levels."""
        print("="*80)
        print("CONTEXT-AWARE MORPHEME RULE EXTRACTION")
        print("="*80)
        print()
        
        print("🔹 SUFFIX RULES (context-aware)")
        print("-"*80)
        self.extract_context_aware_suffix_rules()
        print()
        
        print("🔹 PREFIX RULES (context-aware)")
        print("-"*80)
        self.extract_context_aware_prefix_rules()
        print()
        
        print("🔹 INFIX RULES (context-aware)")
        print("-"*80)
        self.extract_context_aware_infix_rules()
        print()
        
        print("🔹 MORPHEME BIGRAMS")
        print("-"*80)
        self.extract_morpheme_bigrams()
        print()
        
        self.print_summary()
        
        return self.rules
    
    def extract_context_aware_suffix_rules(self):
        """
        Extract suffix rules based on stem and preceding morphemes.
        
        Pattern: "stem-morph1-morph2-[SUFFIX]"
        Context: What suffix follows this stem+morpheme sequence?
        """
        print("Extracting suffix patterns...")
        
        # Pattern 1: After stem
        suffix_after_stem = defaultdict(Counter)
        
        # Pattern 2: After stem+morpheme
        suffix_after_sequence = defaultdict(Counter)
        
        # Pattern 3: By word POS type
        suffix_by_pos = defaultdict(Counter)
        
        for tablet_id, tablet in self.tablets.items():
            for idx, row in tablet.iterrows():
                word = str(row['form'])
                pos = str(row['pos'])
                
                if '-' not in word:
                    continue
                
                parts = word.split('-')
                if len(parts) < 2:
                    continue
                
                stem = parts[0]
                suffix = parts[-1]
                
                # Pattern 1: suffix after stem
                suffix_after_stem[stem][suffix] += 1
                
                # Pattern 2: suffix after stem+morpheme sequence
                if len(parts) >= 3:
                    preceding = tuple(parts[:-1])
                    suffix_after_sequence[preceding][suffix] += 1
                
                # Pattern 3: suffix by POS type
                # Extract base POS (N, V, PN, etc.)
                base_pos = pos.split('.')[0] if '.' in pos else pos
                suffix_by_pos[base_pos][suffix] += 1
        
        # Filter to high-confidence patterns
        filtered_suffix_rules = {}
        
        # After stem - return TOP candidates (not just dominant one)
        for stem, suffix_counts in suffix_after_stem.items():
            total = sum(suffix_counts.values())
            if total < self.min_occurrences:
                continue
            
            # Keep all suffixes that appear frequently enough
            # AND are reasonably common (≥5% of total)
            candidates = {}
            for suffix, count in suffix_counts.items():
                frequency = count / total
                
                # Keep if: (1) appears often enough, AND (2) reasonably common
                if count >= 3 and frequency >= 0.05:  # At least 5% frequency
                    candidates[suffix] = {
                        'count': count,
                        'total': total,
                        'frequency': frequency
                    }
            
            # Only create rule if we have candidates
            if candidates:
                filtered_suffix_rules[f"after_stem_{stem}"] = candidates
        
        # After sequence
        for sequence, suffix_counts in suffix_after_sequence.items():
            total = sum(suffix_counts.values())
            if total < self.min_occurrences:
                continue
            
            high_conf_suffixes = {}
            for suffix, count in suffix_counts.items():
                accuracy = count / total
                if accuracy >= self.min_accuracy:
                    high_conf_suffixes[suffix] = {
                        'count': count,
                        'total': total,
                        'accuracy': accuracy
                    }
            
            if high_conf_suffixes:
                seq_key = f"after_sequence_{'_'.join(sequence)}"
                filtered_suffix_rules[seq_key] = high_conf_suffixes
        
        # By POS type
        for pos_type, suffix_counts in suffix_by_pos.items():
            total = sum(suffix_counts.values())
            if total < self.min_occurrences * 3:  # Higher threshold for POS
                continue
            
            # Get top suffixes (don't require single suffix to dominate)
            top_suffixes = {}
            for suffix, count in suffix_counts.most_common(10):
                frequency = count / total
                if frequency >= 0.10:  # At least 10% frequency
                    top_suffixes[suffix] = {
                        'count': count,
                        'total': total,
                        'frequency': frequency
                    }
            
            if top_suffixes:
                filtered_suffix_rules[f"pos_type_{pos_type}"] = top_suffixes
        
        self.rules['suffix_rules'] = filtered_suffix_rules
        
        print(f"Found {len(filtered_suffix_rules)} context-aware suffix patterns")
        
        # Show examples
        if filtered_suffix_rules:
            print("\nExample patterns:")
            for pattern_key, suffixes in list(filtered_suffix_rules.items())[:5]:
                top_suffix = max(suffixes.items(), key=lambda x: x[1].get('accuracy', x[1].get('frequency', 0)))
                suffix, info = top_suffix
                acc = info.get('accuracy', info.get('frequency', 0))
                print(f"  {pattern_key} → '{suffix}' ({acc*100:.1f}%, {len(suffixes)} candidates)")
    
    def extract_context_aware_prefix_rules(self):
        """
        Extract prefix rules based on following morphemes and preceding word.
        
        Pattern 1: "[PREFIX]-morph1-morph2"
        Pattern 2: "prev_word [PREFIX]-..."
        Pattern 3: "word_ending [PREFIX]-..."
        """
        print("Extracting prefix patterns...")
        
        # Pattern 1: Before morpheme
        prefix_before_morpheme = defaultdict(Counter)
        
        # Pattern 2: Before sequence
        prefix_before_sequence = defaultdict(Counter)
        
        # Pattern 3: After word (inter-word context)
        prefix_after_word = defaultdict(Counter)
        
        # Pattern 4: After word ending
        prefix_after_ending = defaultdict(Counter)
        
        for tablet_id, tablet in self.tablets.items():
            words = [str(row['form']) for _, row in tablet.iterrows()]
            
            for word_idx, word in enumerate(words):
                if '-' not in word:
                    continue
                
                parts = word.split('-')
                if len(parts) < 2:
                    continue
                
                prefix = parts[0]
                
                # Pattern 1: prefix before first morpheme after it
                if len(parts) >= 2:
                    following = parts[1]
                    prefix_before_morpheme[following][prefix] += 1
                
                # Pattern 2: prefix before morpheme sequence
                if len(parts) >= 3:
                    following_seq = tuple(parts[1:])
                    prefix_before_sequence[following_seq][prefix] += 1
                
                # Pattern 3: prefix after previous word
                if word_idx > 0:
                    prev_word = words[word_idx - 1]
                    prefix_after_word[prev_word][prefix] += 1
                
                # Pattern 4: prefix after word ending pattern
                if word_idx > 0:
                    prev_word = words[word_idx - 1]
                    if '-' in prev_word:
                        ending = prev_word.split('-')[-1]
                        prefix_after_ending[f"ends_{ending}"][prefix] += 1
        
        # Filter patterns
        filtered_prefix_rules = {}
        
        # Before morpheme - return TOP candidates
        for morpheme, prefix_counts in prefix_before_morpheme.items():
            total = sum(prefix_counts.values())
            if total < self.min_occurrences:
                continue
            
            candidates = {}
            for prefix, count in prefix_counts.items():
                frequency = count / total
                
                # Keep if reasonably common (≥5% frequency)
                if count >= 3 and frequency >= 0.05:
                    candidates[prefix] = {
                        'count': count,
                        'total': total,
                        'frequency': frequency
                    }
            
            if candidates:
                filtered_prefix_rules[f"before_{morpheme}"] = candidates
        
        # After word ending
        for ending_pattern, prefix_counts in prefix_after_ending.items():
            total = sum(prefix_counts.values())
            if total < self.min_occurrences:
                continue
            
            # Get top prefixes (allow multiple)
            top_prefixes = {}
            for prefix, count in prefix_counts.most_common(10):
                frequency = count / total
                if frequency >= 0.15:  # At least 15%
                    top_prefixes[prefix] = {
                        'count': count,
                        'total': total,
                        'frequency': frequency
                    }
            
            if top_prefixes:
                filtered_prefix_rules[ending_pattern] = top_prefixes
        
        self.rules['prefix_rules'] = filtered_prefix_rules
        
        print(f"Found {len(filtered_prefix_rules)} context-aware prefix patterns")
        
        if filtered_prefix_rules:
            print("\nExample patterns:")
            for pattern_key, prefixes in list(filtered_prefix_rules.items())[:5]:
                top_prefix = max(prefixes.items(), key=lambda x: x[1].get('accuracy', x[1].get('frequency', 0)))
                prefix, info = top_prefix
                acc = info.get('accuracy', info.get('frequency', 0))
                print(f"  {pattern_key} → '{prefix}' ({acc*100:.1f}%, {len(prefixes)} candidates)")
    
    def extract_context_aware_infix_rules(self):
        """
        Extract infix rules based on surrounding morphemes.
        
        Pattern: "prefix-[INFIX]-suffix"
        """
        print("Extracting infix patterns...")
        
        # Infix between specific morphemes
        infix_between = defaultdict(Counter)
        
        for tablet_id, tablet in self.tablets.items():
            for idx, row in tablet.iterrows():
                word = str(row['form'])
                
                if '-' not in word:
                    continue
                
                parts = word.split('-')
                if len(parts) < 3:
                    continue
                
                # For each infix position
                for i in range(1, len(parts) - 1):
                    prefix = parts[i-1]
                    infix = parts[i]
                    suffix = parts[i+1]
                    
                    context = (prefix, suffix)
                    infix_between[context][infix] += 1
        
        # Filter patterns
        filtered_infix_rules = {}
        
        for context, infix_counts in infix_between.items():
            total = sum(infix_counts.values())
            if total < self.min_occurrences:
                continue
            
            candidates = {}
            for infix, count in infix_counts.items():
                frequency = count / total
                
                # Keep if reasonably common (≥5% frequency)
                if count >= 3 and frequency >= 0.05:
                    candidates[infix] = {
                        'count': count,
                        'total': total,
                        'frequency': frequency
                    }
            
            if candidates:
                context_key = f"between_{context[0]}_and_{context[1]}"
                filtered_infix_rules[context_key] = candidates
        
        self.rules['infix_rules'] = filtered_infix_rules
        
        print(f"Found {len(filtered_infix_rules)} context-aware infix patterns")
        
        if filtered_infix_rules:
            print("\nExample patterns:")
            for pattern_key, infixes in list(filtered_infix_rules.items())[:5]:
                top_infix = max(infixes.items(), key=lambda x: x[1]['frequency'])
                infix, info = top_infix
                print(f"  {pattern_key} → '{infix}' ({info['frequency']*100:.1f}%, {len(infixes)} candidates)")
    
    def extract_morpheme_bigrams(self):
        """Extract valid morpheme bigrams for word formation validation."""
        print("Extracting morpheme bigram patterns...")
        
        bigrams = defaultdict(Counter)
        
        for tablet_id, tablet in self.tablets.items():
            for idx, row in tablet.iterrows():
                word = str(row['form'])
                
                if '-' not in word:
                    continue
                
                parts = word.split('-')
                for i in range(len(parts) - 1):
                    bigram = (parts[i], parts[i+1])
                    bigrams[bigram]['count'] += 1
        
        # Filter
        filtered = {}
        for bigram, stats in bigrams.items():
            if stats['count'] >= self.min_occurrences:
                filtered[f"{bigram[0]}___{bigram[1]}"] = {
                    'count': stats['count']
                }
        
        self.rules['morpheme_bigrams'] = filtered
        
        print(f"Found {len(filtered)} valid morpheme bigrams")
    
    def print_summary(self):
        """Print extraction summary."""
        print("\n" + "="*80)
        print("CONTEXT-AWARE RULE SUMMARY")
        print("="*80)
        print()
        
        print(f"✅ Suffix rules (context-aware): {len(self.rules['suffix_rules'])}")
        print(f"✅ Prefix rules (context-aware): {len(self.rules['prefix_rules'])}")
        print(f"✅ Infix rules (context-aware): {len(self.rules['infix_rules'])}")
        print(f"✅ Morpheme bigrams: {len(self.rules['morpheme_bigrams'])}")
        print()
        
        total = sum(len(r) for r in self.rules.values())
        print(f"📊 TOTAL CONTEXT-AWARE RULES: {total}")
        print()
        
        print("💡 KEY DIFFERENCE FROM BEFORE:")
        print("   OLD: Return ALL 400 prefixes (useless)")
        print("   NEW: Return 2-10 prefixes based on context (useful!)")
        print()
    
    def save_rules(self, output_path='context_aware_rules.json'):
        """Save rules to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.rules, f, indent=2)
        
        print(f"✅ Rules saved to: {output_path}")


def main():
    """Extract context-aware rules."""
    
    csv_path = None
    for path in ["summerian_data/train_tablets.csv", "train_tablets.csv"]:
        if Path(path).exists():
            csv_path = path
            break
    
    if not csv_path:
        print("❌ CSV file not found!")
        return
    
    # Extract at different thresholds
    for threshold in [0.90, 0.85, 0.80]:
        print("\n" + "="*80)
        print(f"EXTRACTING RULES AT {threshold*100:.0f}% ACCURACY THRESHOLD")
        print("="*80)
        print()
        
        extractor = ContextAwareMorphemeExtractor(
            csv_path,
            min_accuracy=threshold,
            min_occurrences=5
        )
        
        rules = extractor.extract_all_rules()
        
        output_name = f"context_aware_rules_{int(threshold*100)}.json"
        extractor.save_rules(output_name)
        
        print()


if __name__ == "__main__":
    main()