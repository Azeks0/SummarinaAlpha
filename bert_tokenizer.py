#!/usr/bin/env python3
"""
Tokenize Sumerian Tablets with mBERT

Processes all tablets and stores tokenized results for validator to use.
"""

import json
import pandas as pd
from transformers import BertTokenizerFast  # Changed from BertTokenizer
from pathlib import Path
from tqdm import tqdm


def load_sumerian_data(csv_path='summerian_data.csv'):
    """Load Sumerian tablets from CSV."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"  Found {len(df)} tablets")
    print(f"  Columns: {df.columns.tolist()}")
    
    return df


def tokenize_text_detailed(text, tokenizer):
    """
    Tokenize text and return detailed information.
    
    Returns:
        {
            'text': original text,
            'tokens': mBERT tokens (with ##),
            'token_ids': integer IDs,
            'word_ids': which original word each token belongs to,
            'char_spans': character positions for each token
        }
    """
    # Tokenize with detailed info
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False  # Don't add [CLS], [SEP]
    )
    
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    token_ids = encoding['input_ids']
    offsets = encoding['offset_mapping']
    
    # Map each token back to original word
    words = text.split()
    word_boundaries = []
    pos = 0
    for word in words:
        start = text.find(word, pos)
        end = start + len(word)
        word_boundaries.append((start, end))
        pos = end
    
    word_ids = []
    for token_start, token_end in offsets:
        # Find which word this token belongs to
        word_idx = None
        for i, (word_start, word_end) in enumerate(word_boundaries):
            if token_start >= word_start and token_end <= word_end:
                word_idx = i
                break
        word_ids.append(word_idx)
    
    return {
        'text': text,
        'tokens': tokens,
        'token_ids': token_ids,
        'word_ids': word_ids,
        'char_spans': offsets,
        'num_tokens': len(tokens)
    }


def tokenize_all_tablets(df, tokenizer):
    """
    Tokenize all tablets in dataset.
    
    Returns list of tokenized tablets with metadata.
    """
    print("\nTokenizing all tablets with mBERT...")
    
    tokenized_tablets = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
        # Get text (adjust column name based on your CSV)
        text_column = 'Original Text' if 'Original Text' in df.columns else 'text'
        text = str(row[text_column])
        
        # Skip empty
        if pd.isna(text) or text.strip() == '' or text == 'nan':
            continue
        
        # Tokenize
        tokenized = tokenize_text_detailed(text, tokenizer)
        
        # Add metadata
        tokenized['tablet_id'] = row.get('ID Code', f'tablet_{idx}')
        tokenized['row_index'] = idx
        
        # Add original morpheme segmentation if available
        if 'morphemes' in row:
            tokenized['original_morphemes'] = row['morphemes']
        
        tokenized_tablets.append(tokenized)
    
    print(f"  ✓ Tokenized {len(tokenized_tablets)} tablets")
    
    return tokenized_tablets


def analyze_tokenization(tokenized_tablets):
    """Analyze tokenization statistics."""
    print("\n" + "="*80)
    print("TOKENIZATION STATISTICS")
    print("="*80)
    print()
    
    total_tokens = sum(t['num_tokens'] for t in tokenized_tablets)
    avg_tokens = total_tokens / len(tokenized_tablets)
    
    print(f"Total tablets: {len(tokenized_tablets)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per tablet: {avg_tokens:.1f}")
    print()
    
    # Token distribution
    token_lengths = [t['num_tokens'] for t in tokenized_tablets]
    print(f"Min tokens: {min(token_lengths)}")
    print(f"Max tokens: {max(token_lengths)}")
    print(f"Median tokens: {sorted(token_lengths)[len(token_lengths)//2]}")
    print()
    
    # Show examples
    print("="*80)
    print("TOKENIZATION EXAMPLES")
    print("="*80)
    print()
    
    for i in range(min(5, len(tokenized_tablets))):
        tablet = tokenized_tablets[i]
        print(f"Tablet {i+1}: {tablet['tablet_id']}")
        print(f"  Original: {tablet['text']}")
        print(f"  Tokens ({tablet['num_tokens']}): {' '.join(tablet['tokens'])}")
        print()
    
    # Count unique tokens
    all_tokens = []
    for tablet in tokenized_tablets:
        all_tokens.extend(tablet['tokens'])
    
    unique_tokens = set(all_tokens)
    print(f"Unique tokens in corpus: {len(unique_tokens)}")
    
    # Most common tokens
    from collections import Counter
    token_counts = Counter(all_tokens)
    print()
    print("Most common tokens:")
    for token, count in token_counts.most_common(20):
        print(f"  {token:15s} {count:>6d}")
    print()


def create_token_vocabulary(tokenized_tablets):
    """Create vocabulary mapping for validator."""
    print("Creating token vocabulary...")
    
    # Collect all unique tokens
    all_tokens = set()
    for tablet in tokenized_tablets:
        all_tokens.update(tablet['tokens'])
    
    # Create mappings
    token_to_id = {token: i for i, token in enumerate(sorted(all_tokens))}
    id_to_token = {i: token for token, i in token_to_id.items()}
    
    vocab = {
        'tokens': sorted(all_tokens),
        'token_to_id': token_to_id,
        'id_to_token': id_to_token,
        'vocab_size': len(all_tokens)
    }
    
    print(f"  Vocabulary size: {len(all_tokens)}")
    
    return vocab


def save_tokenized_data(tokenized_tablets, vocab, output_dir='tokenized_data'):
    """Save tokenized data to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nSaving tokenized data to {output_dir}/...")
    
    # Save all tablets
    tablets_file = output_path / 'tokenized_tablets.json'
    with open(tablets_file, 'w', encoding='utf-8') as f:
        json.dump(tokenized_tablets, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved {tablets_file}")
    
    # Save vocabulary
    vocab_file = output_path / 'token_vocabulary.json'
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved {vocab_file}")
    
    # Save summary statistics
    stats = {
        'num_tablets': len(tokenized_tablets),
        'total_tokens': sum(t['num_tokens'] for t in tokenized_tablets),
        'vocab_size': vocab['vocab_size'],
        'avg_tokens_per_tablet': sum(t['num_tokens'] for t in tokenized_tablets) / len(tokenized_tablets)
    }
    
    stats_file = output_path / 'tokenization_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Saved {stats_file}")
    
    # Create splits (train/val/test)
    print("\nCreating train/val/test splits...")
    
    num_tablets = len(tokenized_tablets)
    train_size = int(0.9 * num_tablets)
    val_size = int(0.05 * num_tablets)
    
    train_tablets = tokenized_tablets[:train_size]
    val_tablets = tokenized_tablets[train_size:train_size+val_size]
    test_tablets = tokenized_tablets[train_size+val_size:]
    
    splits = {
        'train': train_tablets,
        'val': val_tablets,
        'test': test_tablets
    }
    
    for split_name, split_data in splits.items():
        split_file = output_path / f'tokenized_{split_name}.json'
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Saved {split_name}: {len(split_data)} tablets → {split_file}")
    
    print()
    print("="*80)
    print("TOKENIZATION COMPLETE!")
    print("="*80)
    print()
    print(f"Output directory: {output_dir}/")
    print()
    print("Files created:")
    print(f"  • tokenized_tablets.json    - All tokenized tablets")
    print(f"  • token_vocabulary.json     - Token vocabulary for validator")
    print(f"  • tokenization_stats.json   - Summary statistics")
    print(f"  • tokenized_train.json      - Training set ({len(train_tablets)} tablets)")
    print(f"  • tokenized_val.json        - Validation set ({len(val_tablets)} tablets)")
    print(f"  • tokenized_test.json       - Test set ({len(test_tablets)} tablets)")
    print()
    print("Next steps:")
    print("  1. Adapt validator to use these mBERT tokens")
    print("  2. Generate candidates using validator rules on tokens")
    print("  3. Fine-tune mBERT on this tokenized data")
    print()


def main():
    """Main tokenization pipeline."""
    
    print("="*80)
    print("SUMERIAN TOKENIZATION WITH mBERT")
    print("="*80)
    print()
    
    # Load mBERT tokenizer (FAST version for offset mapping)
    print("Loading mBERT tokenizer (Fast)...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
    print(f"  ✓ Loaded (vocab size: {tokenizer.vocab_size:,})")
    print()
    
    # Load data
    df = load_sumerian_data('summerian_data.csv')
    print()
    
    # Tokenize all tablets
    tokenized_tablets = tokenize_all_tablets(df, tokenizer)
    
    # Analyze
    analyze_tokenization(tokenized_tablets)
    
    # Create vocabulary
    vocab = create_token_vocabulary(tokenized_tablets)
    
    # Save
    save_tokenized_data(tokenized_tablets, vocab)
    
    print("="*80)
    print("SUCCESS!")
    print("="*80)
    print()
    print("Your data is now tokenized with mBERT.")
    print("The validator can now work with these consistent tokens.")
    print()


if __name__ == "__main__":
    main()