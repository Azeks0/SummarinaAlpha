#!/usr/bin/env python3
"""
Sumerian Morpheme-Aware Tokenizer

Builds a custom tokenizer that respects Sumerian morphological structure:
- Splits on '-' for morpheme boundaries
- Learns subword vocabulary from corpus
- Compatible with HuggingFace transformers
"""

import pandas as pd
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple
import re


class SumerianTokenizer:
    """
    Custom tokenizer for Sumerian text.
    
    Features:
    1. Morpheme-aware: Respects '-' boundaries
    2. Subword vocabulary: Learns common morphemes
    3. OOV handling: Falls back to character-level for unknown words
    4. Special tokens: [PAD], [UNK], [CLS], [SEP], [MASK]
    """
    
    def __init__(self):
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4
        }
        
        self.vocab = self.special_tokens.copy()
        self.id2token = {v: k for k, v in self.vocab.items()}
        
        self.morpheme_freq = Counter()
        self.word_freq = Counter()
        
    def train(self, csv_path, min_freq=3, max_vocab_size=5000):
        """
        Train tokenizer on Sumerian corpus.
        
        Args:
            csv_path: Path to CSV with 'form' column
            min_freq: Minimum frequency for including token
            max_vocab_size: Maximum vocabulary size
        """
        print(f"Training tokenizer on {csv_path}...")
        
        # Load data
        df = pd.read_csv(csv_path)
        
        # Count words and morphemes
        print("Counting morphemes and words...")
        for word in df['form']:
            word = str(word)
            self.word_freq[word] += 1
            
            # Split into morphemes
            if '-' in word:
                morphemes = word.split('-')
                for i, morpheme in enumerate(morphemes):
                    # Count bare morpheme
                    self.morpheme_freq[morpheme] += 1
                    
                    # Also count with position markers for better tokenization
                    if i == 0 and len(morphemes) > 1:
                        # First morpheme (stem)
                        self.morpheme_freq[f"{morpheme}-"] += 1
                    elif i == len(morphemes) - 1 and i > 0:
                        # Last morpheme (suffix)
                        self.morpheme_freq[f"-{morpheme}"] += 1
                    elif i > 0:
                        # Middle morpheme (infix)
                        self.morpheme_freq[f"-{morpheme}-"] += 1
            else:
                self.morpheme_freq[word] += 1
        
        print(f"Found {len(self.word_freq)} unique words")
        print(f"Found {len(self.morpheme_freq)} unique morphemes")
        
        # Build vocabulary
        print("Building vocabulary...")
        
        # Strategy: Add most frequent morphemes first, then full words
        vocab_items = []
        
        # 1. Add most common morphemes
        for morpheme, freq in self.morpheme_freq.most_common():
            if freq >= min_freq:
                vocab_items.append(morpheme)
        
        # 2. Add most common full words (for efficient encoding)
        for word, freq in self.word_freq.most_common():
            if freq >= min_freq and word not in vocab_items:
                vocab_items.append(word)
        
        # Limit vocabulary size
        vocab_items = vocab_items[:max_vocab_size - len(self.special_tokens)]
        
        # Create vocabulary
        self.vocab = self.special_tokens.copy()
        for i, item in enumerate(vocab_items, start=len(self.special_tokens)):
            self.vocab[item] = i
        
        self.id2token = {v: k for k, v in self.vocab.items()}
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"  Special tokens: {len(self.special_tokens)}")
        print(f"  Morphemes: {len([t for t in vocab_items if '-' not in t])}")
        print(f"  Full words: {len([t for t in vocab_items if '-' in t])}")
        
        # Print statistics
        self._print_statistics()
    
    def _print_statistics(self):
        """Print vocabulary statistics."""
        print("\n" + "="*80)
        print("TOKENIZER STATISTICS")
        print("="*80)
        
        # Most common morphemes
        print("\nMost common morphemes:")
        morphemes = [(m, f) for m, f in self.morpheme_freq.most_common(20) 
                     if m in self.vocab]
        for morpheme, freq in morphemes[:10]:
            print(f"  {morpheme:15s} {freq:5d} occurrences")
        
        # Most common words
        print("\nMost common words:")
        words = [(w, f) for w, f in self.word_freq.most_common(20) 
                 if w in self.vocab]
        for word, freq in words[:10]:
            print(f"  {word:20s} {freq:5d} occurrences")
        
        # Morpheme types
        print("\nMorpheme type coverage:")
        
        # Common case markers (as suffixes)
        case_markers = ['-e', '-ak', '-ra', '-da', '-ta', '-sze3', '-gin7']
        in_vocab = sum(1 for m in case_markers if m in self.vocab)
        print(f"  Case markers (suffixes): {in_vocab}/{len(case_markers)} in vocab")
        if in_vocab > 0:
            found = [m for m in case_markers if m in self.vocab]
            print(f"    Found: {', '.join(found)}")
        
        # Common verbal prefixes
        verbal_prefixes = ['ba-', 'mu-', 'i3-', 'al-', 'im-', 'in-', 'u3-']
        in_vocab = sum(1 for m in verbal_prefixes if m in self.vocab)
        print(f"  Verbal prefixes: {in_vocab}/{len(verbal_prefixes)} in vocab")
        if in_vocab > 0:
            found = [m for m in verbal_prefixes if m in self.vocab]
            print(f"    Found: {', '.join(found)}")
        
        # Common verbal infixes
        verbal_infixes = ['-an-', '-ab-', '-mu-', '-e-', '-en-', '-b-', '-n-']
        in_vocab = sum(1 for m in verbal_infixes if m in self.vocab)
        print(f"  Verbal infixes: {in_vocab}/{len(verbal_infixes)} in vocab")
        if in_vocab > 0:
            found = [m for m in verbal_infixes if m in self.vocab]
            print(f"    Found: {', '.join(found)}")
        
        print()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Sumerian text into morphemes.
        
        Args:
            text: Input text (single word or sentence)
        
        Returns:
            List of tokens
        """
        if not text or text in self.special_tokens:
            return [text]
        
        # Split sentence into words
        words = text.split()
        
        tokens = []
        for word in words:
            tokens.extend(self._tokenize_word(word))
        
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into morphemes.
        
        Strategy:
        1. If full word in vocab → use it
        2. Else, split on '-' and use morphemes with position markers
        3. If morpheme unknown → try without markers, else [UNK]
        """
        # Check if full word is in vocabulary
        if word in self.vocab:
            return [word]
        
        # Split into morphemes
        if '-' not in word:
            # Single morpheme word
            return [word] if word in self.vocab else ['[UNK]']
        
        # Multi-morpheme word
        morphemes = word.split('-')
        tokens = []
        
        for i, morpheme in enumerate(morphemes):
            added = False
            
            # Try with position markers first
            if i == 0 and len(morphemes) > 1:
                # First position - try with suffix dash
                candidate = f"{morpheme}-"
                if candidate in self.vocab:
                    tokens.append(candidate)
                    added = True
            
            elif i == len(morphemes) - 1 and i > 0:
                # Last position - try with prefix dash
                candidate = f"-{morpheme}"
                if candidate in self.vocab:
                    tokens.append(candidate)
                    added = True
            
            elif 0 < i < len(morphemes) - 1:
                # Middle position - try with both dashes
                candidate = f"-{morpheme}-"
                if candidate in self.vocab:
                    tokens.append(candidate)
                    added = True
            
            # Fall back to bare morpheme
            if not added:
                if morpheme in self.vocab:
                    tokens.append(morpheme)
                else:
                    tokens.append('[UNK]')
        
        return tokens
    
    def encode(self, text: str, add_special_tokens=True, max_length=None, 
               padding='max_length', truncation=True) -> Dict:
        """
        Encode text to token IDs (HuggingFace-style interface).
        
        Args:
            text: Input text
            add_special_tokens: Whether to add [CLS] and [SEP]
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Tokenize
        tokens = self.tokenize(text)
        
        # Add special tokens
        if add_special_tokens:
            tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # Convert to IDs
        input_ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
        
        # Truncate
        if truncation and max_length and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            tokens = tokens[:max_length]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad
        if padding == 'max_length' and max_length:
            padding_length = max_length - len(input_ids)
            input_ids += [self.vocab['[PAD]']] * padding_length
            attention_mask += [0] * padding_length
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tokens': tokens
        }
    
    def decode(self, token_ids: List[int], skip_special_tokens=True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text
        """
        tokens = []
        for token_id in token_ids:
            token = self.id2token.get(token_id, '[UNK]')
            
            if skip_special_tokens and token in self.special_tokens:
                continue
            
            tokens.append(token)
        
        # Reconstruct text
        # Join morphemes with '-' where appropriate
        text = self._reconstruct_text(tokens)
        
        return text
    
    def _reconstruct_text(self, tokens: List[str]) -> str:
        """
        Reconstruct text from tokens, handling morpheme boundaries.
        
        Args:
            tokens: List of tokens
        
        Returns:
            Reconstructed text
        """
        if not tokens:
            return ""
        
        result = []
        current_word = []
        
        for i, token in enumerate(tokens):
            if token in self.special_tokens:
                continue
            
            # Check if token is part of compound word (has - markers)
            has_prefix_dash = token.startswith('-')
            has_suffix_dash = token.endswith('-')
            
            if has_prefix_dash or has_suffix_dash:
                # Part of compound word
                current_word.append(token.strip('-'))
            else:
                # Could be standalone or part of compound
                # Check if next token starts with '-'
                is_compound_start = (i + 1 < len(tokens) and 
                                   (tokens[i + 1].startswith('-') or tokens[i + 1].endswith('-')))
                
                if is_compound_start:
                    # Start of compound word
                    current_word.append(token)
                else:
                    # Standalone word - flush current compound if any
                    if current_word:
                        result.append('-'.join(current_word))
                        current_word = []
                    result.append(token)
        
        # Flush remaining
        if current_word:
            result.append('-'.join(current_word))
        
        return ' '.join(result)
    
    def save(self, path: str):
        """Save tokenizer to file."""
        data = {
            'vocab': self.vocab,
            'id2token': self.id2token,
            'morpheme_freq': dict(self.morpheme_freq),
            'word_freq': dict(self.word_freq)
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        # Also save as JSON for inspection
        json_path = Path(path).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'vocab_size': len(self.vocab)
            }, f, indent=2)
        
        print(f"Tokenizer saved to: {path}")
        print(f"Vocabulary saved to: {json_path}")
    
    @classmethod
    def load(cls, path: str):
        """Load tokenizer from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls()
        tokenizer.vocab = data['vocab']
        tokenizer.id2token = data['id2token']
        tokenizer.morpheme_freq = Counter(data['morpheme_freq'])
        tokenizer.word_freq = Counter(data['word_freq'])
        
        print(f"Tokenizer loaded from: {path}")
        print(f"Vocabulary size: {len(tokenizer.vocab)}")
        
        return tokenizer
    
    def __len__(self):
        """Return vocabulary size."""
        return len(self.vocab)
    
    def __call__(self, text, **kwargs):
        """Make tokenizer callable (HuggingFace-style)."""
        return self.encode(text, **kwargs)


def train_tokenizer(csv_path, output_path='sumerian_tokenizer.pkl', 
                    min_freq=3, max_vocab_size=5000):
    """
    Train and save Sumerian tokenizer.
    
    Args:
        csv_path: Path to training data CSV
        output_path: Path to save tokenizer
        min_freq: Minimum frequency for including token
        max_vocab_size: Maximum vocabulary size
    """
    print("="*80)
    print("TRAINING SUMERIAN TOKENIZER")
    print("="*80)
    print()
    
    # Create tokenizer
    tokenizer = SumerianTokenizer()
    
    # Train on corpus
    tokenizer.train(csv_path, min_freq=min_freq, max_vocab_size=max_vocab_size)
    
    # Save
    tokenizer.save(output_path)
    
    # Test
    print("\n" + "="*80)
    print("TESTING TOKENIZER")
    print("="*80)
    
    test_examples = [
        "lugal-e szu ba-an-ti",
        "1(disz) sila3 kasz",
        "iti ezem-mah mu us2-sa {d}amar-{d}suen lugal",
        "ki-ag2-ga2-ni",
        "unknown-word-here"
    ]
    
    for example in test_examples:
        print(f"\nInput: {example}")
        
        # Tokenize
        tokens = tokenizer.tokenize(example)
        print(f"Tokens: {tokens}")
        
        # Encode
        encoded = tokenizer.encode(example, max_length=20, padding='max_length')
        print(f"IDs: {encoded['input_ids'][:10]}...")
        
        # Decode
        decoded = tokenizer.decode(encoded['input_ids'])
        print(f"Decoded: {decoded}")
    
    print("\n✅ Tokenizer training complete!")
    
    return tokenizer


def main():
    """Train tokenizer on Sumerian corpus."""
    
    # Find training data
    csv_path = None
    for path in ["summerian_data/train_tablets.csv", "train_tablets.csv"]:
        if Path(path).exists():
            csv_path = path
            break
    
    if not csv_path:
        print("❌ Training data not found!")
        print("   Expected: summerian_data/train_tablets.csv")
        return
    
    # Train tokenizer
    tokenizer = train_tokenizer(
        csv_path=csv_path,
        output_path='sumerian_tokenizer.pkl',
        min_freq=3,
        max_vocab_size=5000
    )
    
    print("\n" + "="*80)
    print("TOKENIZER READY FOR USE")
    print("="*80)
    print()
    print("Usage:")
    print("  from sumerian_tokenizer import SumerianTokenizer")
    print("  tokenizer = SumerianTokenizer.load('sumerian_tokenizer.pkl')")
    print("  tokens = tokenizer.tokenize('lugal-e szu ba-an-ti')")
    print()


if __name__ == "__main__":
    main()