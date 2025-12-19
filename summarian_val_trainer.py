"""
Sumerian Text Reconstruction Validator - Standalone Version

Complete implementation in a single file to avoid pickle import issues.
Train the validator and use it in the same script.

Author: Pipeline Builder
Date: 2025
"""

import json
import csv
import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path


@dataclass
class ValidationRule:
    """Represents a learned validation rule at any level."""
    rule_type: str  # 'subword', 'word', or 'pos'
    pattern: Tuple  # The pattern that this rule captures
    frequency: int  # How often this pattern appears
    confidence: float  # Confidence score (frequency / total)
    support_count: int  # Number of examples supporting this rule
    
    def __repr__(self):
        return f"Rule({self.rule_type}, {self.pattern}, freq={self.frequency}, conf={self.confidence:.3f})"


@dataclass
class ValidatorRules:
    """Container for all learned rules at all levels."""
    # Subword level rules
    subword_bigrams: Dict[Tuple[str, str], ValidationRule] = field(default_factory=dict)
    subword_trigrams: Dict[Tuple[str, str, str], ValidationRule] = field(default_factory=dict)
    token_starts: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    token_ends: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    continuation_tokens: Set[str] = field(default_factory=set)
    starter_tokens: Set[str] = field(default_factory=set)
    
    # Word level rules
    word_patterns: Dict[Tuple[str, ...], ValidationRule] = field(default_factory=dict)
    word_lengths: Dict[int, List[str]] = field(default_factory=lambda: defaultdict(list))
    word_boundaries: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    
    # POS level rules
    pos_bigrams: Dict[Tuple[str, str], ValidationRule] = field(default_factory=dict)
    pos_trigrams: Dict[Tuple[str, str, str], ValidationRule] = field(default_factory=dict)
    word_pos_map: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    
    # Statistics
    total_subword_sequences: int = 0
    total_word_sequences: int = 0
    total_pos_sequences: int = 0


@dataclass
class CandidateScore:
    """Score container for a candidate token/word."""
    candidate: str
    subword_score: float = 0.0
    word_score: float = 0.0
    pos_score: float = 0.0
    total_score: float = 0.0
    passed_levels: List[str] = None
    
    def __post_init__(self):
        if self.passed_levels is None:
            self.passed_levels = []
    
    def __repr__(self):
        return f"Candidate('{self.candidate}', total={self.total_score:.3f}, levels={self.passed_levels})"


class SumerianValidatorTrainer:
    """Trains a multi-level validator for Sumerian text reconstruction."""
    
    def __init__(self, min_frequency: int = 2, confidence_threshold: float = 0.01):
        self.min_frequency = min_frequency
        self.confidence_threshold = confidence_threshold
        self.rules = ValidatorRules()
        
    def train(self, tokenized_data_path: str, pos_data_path: Optional[str] = None):
        """Train the validator on tokenized data and optional POS data."""
        print("Loading tokenized data...")
        with open(tokenized_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} tokenized examples")
        
        print("\n=== Phase 1: Learning Subword Patterns ===")
        self._learn_subword_patterns(data)
        
        print("\n=== Phase 2: Learning Word Patterns ===")
        self._learn_word_patterns(data)
        
        if pos_data_path:
            print("\n=== Phase 3: Learning POS Patterns ===")
            self._learn_pos_patterns(data, pos_data_path)
        
        print("\n=== Training Complete ===")
        self._print_statistics()
        
    def _learn_subword_patterns(self, data: List[Dict]):
        """Learn patterns at the subword (token) level."""
        bigram_counts = Counter()
        trigram_counts = Counter()
        token_transitions = defaultdict(Counter)
        
        for example in data:
            tokens = example['tokens']
            
            for token in tokens:
                if token.startswith('##'):
                    self.rules.continuation_tokens.add(token)
                else:
                    self.rules.starter_tokens.add(token)
            
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i+1])
                bigram_counts[bigram] += 1
                token_transitions[tokens[i]][tokens[i+1]] += 1
                
                if i < len(tokens) - 2:
                    trigram = (tokens[i], tokens[i+1], tokens[i+2])
                    trigram_counts[trigram] += 1
        
        total_bigrams = sum(bigram_counts.values())
        for bigram, count in bigram_counts.items():
            if count >= self.min_frequency:
                confidence = count / total_bigrams
                if confidence >= self.confidence_threshold:
                    self.rules.subword_bigrams[bigram] = ValidationRule(
                        rule_type='subword', pattern=bigram, frequency=count,
                        confidence=confidence, support_count=count
                    )
        
        total_trigrams = sum(trigram_counts.values())
        for trigram, count in trigram_counts.items():
            if count >= self.min_frequency:
                confidence = count / total_trigrams
                if confidence >= self.confidence_threshold:
                    self.rules.subword_trigrams[trigram] = ValidationRule(
                        rule_type='subword', pattern=trigram, frequency=count,
                        confidence=confidence, support_count=count
                    )
        
        for token, next_tokens in token_transitions.items():
            self.rules.token_starts[token] = set(next_tokens.keys())
        
        for token, next_tokens in token_transitions.items():
            for next_token in next_tokens.keys():
                self.rules.token_ends[next_token].add(token)
        
        self.rules.total_subword_sequences = total_bigrams
        
        print(f"  Learned {len(self.rules.subword_bigrams)} bigram rules")
        print(f"  Learned {len(self.rules.subword_trigrams)} trigram rules")
        print(f"  Found {len(self.rules.continuation_tokens)} continuation tokens (##)")
        print(f"  Found {len(self.rules.starter_tokens)} starter tokens")
    
    def _learn_word_patterns(self, data: List[Dict]):
        """Learn patterns at the word level."""
        word_token_sequences = defaultdict(Counter)
        word_boundary_pairs = Counter()
        all_word_forms = set()
        
        for example in data:
            tokens = example['tokens']
            word_ids = example['word_ids']
            text = example['text']
            
            words = text.split()
            all_word_forms.update(words)
            
            word_tokens = defaultdict(list)
            for token, word_id in zip(tokens, word_ids):
                word_tokens[word_id].append(token)
            
            for word_id, word_token_list in word_tokens.items():
                if word_id < len(words):
                    word_text = words[word_id]
                    token_tuple = tuple(word_token_list)
                    word_token_sequences[word_text][token_tuple] += 1
                    
                    word_length = len(word_token_list)
                    if word_text not in self.rules.word_lengths[word_length]:
                        self.rules.word_lengths[word_length].append(word_text)
            
            sorted_word_ids = sorted(word_tokens.keys())
            for i in range(len(sorted_word_ids) - 1):
                current_word_id = sorted_word_ids[i]
                next_word_id = sorted_word_ids[i + 1]
                
                current_tokens = word_tokens[current_word_id]
                next_tokens = word_tokens[next_word_id]
                
                if current_tokens and next_tokens:
                    boundary = (current_tokens[-1], next_tokens[0])
                    word_boundary_pairs[boundary] += 1
        
        for word_text, token_seqs in word_token_sequences.items():
            for token_seq, count in token_seqs.items():
                if count >= self.min_frequency:
                    total = sum(token_seqs.values())
                    confidence = count / total
                    if confidence >= self.confidence_threshold:
                        self.rules.word_patterns[token_seq] = ValidationRule(
                            rule_type='word', pattern=token_seq, frequency=count,
                            confidence=confidence, support_count=count
                        )
        
        self.rules.word_boundaries = dict(word_boundary_pairs)
        self.rules.total_word_sequences = sum(word_boundary_pairs.values())
        
        print(f"  Learned {len(self.rules.word_patterns)} word token patterns")
        print(f"  Learned {len(self.rules.word_boundaries)} word boundary patterns")
        print(f"  Found {len(all_word_forms)} unique word forms")
    
    def _learn_pos_patterns(self, data: List[Dict], pos_data_path: str):
        """Learn patterns at the POS level."""
        pos_sequences = self._load_pos_data(pos_data_path)
        
        pos_bigram_counts = Counter()
        pos_trigram_counts = Counter()
        
        for example in data:
            tablet_id = example['tablet_id']
            row_index = example.get('row_index', 0)
            text = example['text']
            words = text.split()
            
            pos_key = f"{tablet_id}_{row_index}"
            if pos_key in pos_sequences:
                pos_tags = pos_sequences[pos_key]
                
                for word, pos in zip(words, pos_tags):
                    self.rules.word_pos_map[word][pos] += 1
                
                for i in range(len(pos_tags) - 1):
                    bigram = (pos_tags[i], pos_tags[i+1])
                    pos_bigram_counts[bigram] += 1
                    
                    if i < len(pos_tags) - 2:
                        trigram = (pos_tags[i], pos_tags[i+1], pos_tags[i+2])
                        pos_trigram_counts[trigram] += 1
        
        total_pos_bigrams = sum(pos_bigram_counts.values())
        for bigram, count in pos_bigram_counts.items():
            if count >= self.min_frequency:
                confidence = count / total_pos_bigrams
                if confidence >= self.confidence_threshold:
                    self.rules.pos_bigrams[bigram] = ValidationRule(
                        rule_type='pos', pattern=bigram, frequency=count,
                        confidence=confidence, support_count=count
                    )
        
        total_pos_trigrams = sum(pos_trigram_counts.values())
        for trigram, count in pos_trigram_counts.items():
            if count >= self.min_frequency:
                confidence = count / total_pos_trigrams
                if confidence >= self.confidence_threshold:
                    self.rules.pos_trigrams[trigram] = ValidationRule(
                        rule_type='pos', pattern=trigram, frequency=count,
                        confidence=confidence, support_count=count
                    )
        
        self.rules.total_pos_sequences = total_pos_bigrams
        
        print(f"  Learned {len(self.rules.pos_bigrams)} POS bigram rules")
        print(f"  Learned {len(self.rules.pos_trigrams)} POS trigram rules")
        print(f"  Mapped {len(self.rules.word_pos_map)} words to POS tags")
    
    def _load_pos_data(self, pos_data_path: str) -> Dict[str, List[str]]:
        """Load POS data from CSV file(s)."""
        pos_by_line = defaultdict(list)
        path = Path(pos_data_path)
        
        if path.is_dir():
            csv_files = list(path.glob("*.csv"))
        else:
            csv_files = [path]
        
        for csv_file in csv_files:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tablet_id = row.get('tablet_id', '')
                    line_id = row.get('line_id', '')
                    pos = row.get('pos', '')
                    
                    if tablet_id and line_id and pos:
                        parts = line_id.split('.')
                        if len(parts) >= 2:
                            base_line_id = '.'.join(parts[:2])
                        else:
                            base_line_id = line_id
                        
                        key = (tablet_id, base_line_id)
                        pos_by_line[key].append(pos)
        
        pos_sequences = {}
        tablet_lines = defaultdict(list)
        for (tablet_id, base_line_id), pos_tags in pos_by_line.items():
            tablet_lines[tablet_id].append((base_line_id, pos_tags))
        
        for tablet_id, lines in tablet_lines.items():
            sorted_lines = sorted(lines, key=lambda x: x[0])
            for row_index, (base_line_id, pos_tags) in enumerate(sorted_lines):
                key = f"{tablet_id}_{row_index}"
                pos_sequences[key] = pos_tags
        
        print(f"  Loaded {len(pos_sequences)} POS sequences from {len(csv_files)} file(s)")
        return pos_sequences
    
    def _print_statistics(self):
        """Print training statistics."""
        print("\n=== Validator Statistics ===")
        print(f"Subword Level:")
        print(f"  - Bigram rules: {len(self.rules.subword_bigrams)}")
        print(f"  - Trigram rules: {len(self.rules.subword_trigrams)}")
        print(f"  - Continuation tokens: {len(self.rules.continuation_tokens)}")
        print(f"  - Starter tokens: {len(self.rules.starter_tokens)}")
        print(f"\nWord Level:")
        print(f"  - Word patterns: {len(self.rules.word_patterns)}")
        print(f"  - Word boundaries: {len(self.rules.word_boundaries)}")
        print(f"\nPOS Level:")
        print(f"  - POS bigrams: {len(self.rules.pos_bigrams)}")
        print(f"  - POS trigrams: {len(self.rules.pos_trigrams)}")
        print(f"  - Word-POS mappings: {len(self.rules.word_pos_map)}")
    
    def save_rules(self, output_path: str):
        """Save learned rules to disk."""
        with open(output_path, 'wb') as f:
            pickle.dump(self.rules, f)
        print(f"\nRules saved to {output_path}")
    
    def get_rules(self):
        """Get the trained rules object."""
        return self.rules


class SumerianValidator:
    """Multi-level validator for filtering MLM candidates."""
    
    def __init__(self, rules_or_path):
        """
        Initialize validator with learned rules.
        
        Args:
            rules_or_path: Either a ValidatorRules object or path to pickled rules
        """
        if isinstance(rules_or_path, str):
            with open(rules_or_path, 'rb') as f:
                self.rules = pickle.load(f)
        else:
            self.rules = rules_or_path
        
        print(f"Validator loaded with:")
        print(f"  - {len(self.rules.subword_bigrams)} subword bigrams")
        print(f"  - {len(self.rules.word_patterns)} word patterns")
        print(f"  - {len(self.rules.pos_bigrams)} POS bigrams")
    
    def validate_candidates(
        self,
        candidates: List[str],
        context_before: List[str],
        context_after: List[str],
        word_context_before: Optional[List[str]] = None,
        word_context_after: Optional[List[str]] = None,
        pos_context_before: Optional[List[str]] = None,
        pos_context_after: Optional[List[str]] = None,
        weights: Dict[str, float] = None
    ) -> List[CandidateScore]:
        """Validate and score candidates at all three levels."""
        if weights is None:
            weights = {'subword': 1.0, 'word': 1.0, 'pos': 1.0}
        
        scored_candidates = []
        
        for candidate in candidates:
            score = CandidateScore(candidate=candidate)
            
            # Level 1: Subword validation
            subword_valid, subword_score = self._validate_subword_level(
                candidate, context_before, context_after
            )
            
            if not subword_valid:
                continue
            
            score.subword_score = subword_score
            score.passed_levels.append('subword')
            
            # Level 2: Word validation
            if word_context_before is not None or word_context_after is not None:
                word_valid, word_score = self._validate_word_level(
                    candidate, context_before, context_after,
                    word_context_before, word_context_after
                )
                
                if not word_valid:
                    continue
                
                score.word_score = word_score
                score.passed_levels.append('word')
            
            # Level 3: POS validation
            if pos_context_before is not None or pos_context_after is not None:
                pos_valid, pos_score = self._validate_pos_level(
                    candidate, pos_context_before, pos_context_after
                )
                
                if not pos_valid:
                    continue
                
                score.pos_score = pos_score
                score.passed_levels.append('pos')
            
            # Calculate total score
            score.total_score = (
                weights['subword'] * score.subword_score +
                weights['word'] * score.word_score +
                weights['pos'] * score.pos_score
            ) / sum(weights.values())
            
            scored_candidates.append(score)
        
        scored_candidates.sort(key=lambda x: x.total_score, reverse=True)
        return scored_candidates
    
    def _validate_subword_level(
        self,
        candidate: str,
        context_before: List[str],
        context_after: List[str]
    ) -> Tuple[bool, float]:
        """Validate candidate at subword level - can this token appear at this position?"""
        # Check if token is known
        if candidate.startswith('##'):
            if candidate not in self.rules.continuation_tokens:
                return False, 0.0
        else:
            if candidate not in self.rules.starter_tokens:
                return False, 0.0
        
        scores = []
        valid_context = False
        
        # Check with previous token
        if context_before:
            prev_token = context_before[-1]
            bigram = (prev_token, candidate)
            
            if bigram in self.rules.subword_bigrams:
                valid_context = True
                scores.append(self.rules.subword_bigrams[bigram].confidence)
            elif candidate in self.rules.token_starts.get(prev_token, set()):
                valid_context = True
                scores.append(0.1)
            else:
                return False, 0.0
        
        # Check with next token
        if context_after:
            next_token = context_after[0]
            bigram = (candidate, next_token)
            
            if bigram in self.rules.subword_bigrams:
                valid_context = True
                scores.append(self.rules.subword_bigrams[bigram].confidence)
            elif next_token in self.rules.token_starts.get(candidate, set()):
                valid_context = True
                scores.append(0.1)
            else:
                return False, 0.0
        
        # Check trigram
        if len(context_before) >= 1 and len(context_after) >= 1:
            trigram = (context_before[-1], candidate, context_after[0])
            if trigram in self.rules.subword_trigrams:
                valid_context = True
                scores.append(self.rules.subword_trigrams[trigram].confidence * 2.0)
        
        if not context_before and not context_after:
            return True, 0.5
        
        if not valid_context:
            return False, 0.0
        
        if scores:
            avg_score = np.mean(scores)
            return True, avg_score
        else:
            return True, 0.3
    
    def _validate_word_level(
        self,
        candidate: str,
        token_context_before: List[str],
        token_context_after: List[str],
        word_context_before: Optional[List[str]],
        word_context_after: Optional[List[str]]
    ) -> Tuple[bool, float]:
        """Validate candidate at word level - does this form a known word?"""
        scores = []
        
        is_word_start = not candidate.startswith('##')
        word_tokens = [candidate]
        
        # Reconstruct word
        if token_context_before:
            for i in range(len(token_context_before) - 1, -1, -1):
                token = token_context_before[i]
                if token.startswith('##'):
                    word_tokens.insert(0, token)
                else:
                    if is_word_start and i == len(token_context_before) - 1:
                        break
                    else:
                        word_tokens.insert(0, token)
                    break
        
        if token_context_after:
            for token in token_context_after:
                if token.startswith('##'):
                    word_tokens.append(token)
                else:
                    break
        
        word_tuple = tuple(word_tokens)
        
        if word_tuple in self.rules.word_patterns:
            rule = self.rules.word_patterns[word_tuple]
            scores.append(rule.confidence * 2.0)
        else:
            pattern_matches = 0
            for pattern, rule in self.rules.word_patterns.items():
                if candidate in pattern:
                    pattern_matches += 1
                    scores.append(rule.confidence * 0.3)
            
            if pattern_matches == 0:
                scores.append(0.2)
        
        # Check boundaries
        if token_context_before:
            prev_token = token_context_before[-1]
            boundary = (prev_token, candidate)
            if boundary in self.rules.word_boundaries:
                boundary_freq = self.rules.word_boundaries[boundary]
                boundary_score = min(boundary_freq / 50, 1.0)
                scores.append(boundary_score)
            elif not prev_token.startswith('##') and not candidate.startswith('##'):
                scores.append(0.5)
        
        if token_context_after:
            next_token = token_context_after[0]
            boundary = (candidate, next_token)
            if boundary in self.rules.word_boundaries:
                boundary_freq = self.rules.word_boundaries[boundary]
                boundary_score = min(boundary_freq / 50, 1.0)
                scores.append(boundary_score)
            elif not candidate.startswith('##') and not next_token.startswith('##'):
                scores.append(0.5)
        
        word_length = len(word_tokens)
        if word_length in self.rules.word_lengths:
            scores.append(0.7)
        
        if not scores:
            return True, 0.4
        
        avg_score = np.mean(scores)
        is_valid = avg_score > 0.2
        
        return is_valid, avg_score
    
    def _validate_pos_level(
        self,
        candidate: str,
        pos_context_before: Optional[List[str]],
        pos_context_after: Optional[List[str]]
    ) -> Tuple[bool, float]:
        """Validate candidate at POS level - does the word's POS fit the sequence?"""
        if not pos_context_before and not pos_context_after:
            return True, 0.5
        
        scores = []
        valid_pos_found = False
        
        # Get all known POS tags
        all_pos_tags = set()
        for bigram in self.rules.pos_bigrams.keys():
            all_pos_tags.add(bigram[0])
            all_pos_tags.add(bigram[1])
        
        if not all_pos_tags:
            return True, 0.5
        
        # Check each possible POS
        for possible_pos in all_pos_tags:
            pos_valid = True
            pos_scores = []
            
            if pos_context_before:
                prev_pos = pos_context_before[-1]
                bigram = (prev_pos, possible_pos)
                
                if bigram in self.rules.pos_bigrams:
                    pos_scores.append(self.rules.pos_bigrams[bigram].confidence)
                else:
                    pos_valid = False
            
            if pos_context_after and pos_valid:
                next_pos = pos_context_after[0]
                bigram = (possible_pos, next_pos)
                
                if bigram in self.rules.pos_bigrams:
                    pos_scores.append(self.rules.pos_bigrams[bigram].confidence)
                else:
                    pos_valid = False
            
            if len(pos_context_before) >= 1 and len(pos_context_after) >= 1 and pos_valid:
                trigram = (pos_context_before[-1], possible_pos, pos_context_after[0])
                if trigram in self.rules.pos_trigrams:
                    pos_scores.append(self.rules.pos_trigrams[trigram].confidence * 2.0)
            
            if pos_valid and pos_scores:
                valid_pos_found = True
                scores.append(np.mean(pos_scores))
        
        if not valid_pos_found:
            return False, 0.0
        
        if not scores:
            return True, 0.5
        
        max_score = max(scores)
        return True, max_score
    
    def filter_top_k(
        self,
        candidates: List[str],
        context_before: List[str],
        context_after: List[str],
        k: int = 10,
        **kwargs
    ) -> List[str]:
        """Get top-k filtered candidates."""
        scored = self.validate_candidates(
            candidates, context_before, context_after, **kwargs
        )
        return [score.candidate for score in scored[:k]]
    
    def get_validation_report(
        self,
        candidates: List[str],
        context_before: List[str],
        context_after: List[str],
        **kwargs
    ) -> Dict:
        """Get detailed validation report."""
        scored = self.validate_candidates(
            candidates, context_before, context_after, **kwargs
        )
        
        report = {
            'total_candidates': len(candidates),
            'passed_subword': sum(1 for s in scored if 'subword' in s.passed_levels),
            'passed_word': sum(1 for s in scored if 'word' in s.passed_levels),
            'passed_pos': sum(1 for s in scored if 'pos' in s.passed_levels),
            'fully_validated': sum(1 for s in scored if len(s.passed_levels) == 3),
            'top_10_scores': [
                {'candidate': s.candidate, 'score': s.total_score, 'levels': s.passed_levels}
                for s in scored[:10]
            ]
        }
        
        return report


if __name__ == "__main__":
    # This is just a library file - import and use the classes
    pass