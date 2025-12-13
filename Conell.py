"""
Simple CoNLL to CSV Parser

Takes a CoNLL file and creates a CSV with:
tablet_id, line_id, form, segm, pos, features

Usage:
    python parse_conll_to_csv.py input.conll output.csv
"""

import csv
import sys
import re


def parse_conll_to_csv(input_file, output_file):
    """Parse CoNLL file and write to CSV"""
    
    results = []
    current_tablet = None
    
    print(f"Reading: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # New tablet
            if line.startswith('#new_text='):
                current_tablet = line.split('=')[1].strip()
                print(f"  Found tablet: {current_tablet}")
                continue
            
            # Skip header lines
            if line.startswith('#'):
                continue
            
            # Parse token line
            parts = line.split()
            
            if len(parts) < 4:
                continue  # Skip malformed lines
            
            line_id = parts[0]
            form = parts[1]
            segm = parts[2]
            pos = parts[3]
            
            # Extract features from PoS tag
            features = extract_features(pos, segm)
            
            results.append({
                'tablet_id': current_tablet,
                'line_id': line_id,
                'form': form,
                'segm': segm,
                'pos': pos,
                'features': features
            })
    
    print(f"\nParsed {len(results)} tokens from {len(set(r['tablet_id'] for r in results))} tablets")
    
    # Write to CSV
    print(f"Writing to: {output_file}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'tablet_id', 'line_id', 'form', 'segm', 'pos', 'features'
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ Done! Created {output_file}")


def extract_features(pos, segm):
    """Extract morphological features from PoS tag"""
    features = []
    
    # Parse PoS tag (e.g., "N.GEN.ABS" -> genitive, absolutive)
    if '.' in pos:
        parts = pos.split('.')
        
        case_markers = {
            'ABS': 'absolutive',
            'ERG': 'ergative', 
            'GEN': 'genitive',
            'DAT': 'dative',
            'LOC': 'locative',
            'ABL': 'ablative',
            'TERM': 'terminative',
            'COM': 'comitative',
            'L1': 'locative-1',
            'L3': 'locative-3'
        }
        
        for part in parts:
            if part in case_markers:
                features.append(case_markers[part])
            elif part == 'PN':
                features.append('proper_noun')
            elif part == 'DN':
                features.append('divine_name')
            elif part == 'SN':
                features.append('settlement_name')
            elif part == 'MN':
                features.append('month_name')
            elif part == 'RN':
                features.append('royal_name')
            elif 'SG' in part or 'PL' in part:
                features.append(part.lower())
    
    # Check for zero marker in segmentation
    if '-ø' in segm:
        features.append('zero_marker')
    
    return '; '.join(features) if features else ''


def main():
    if len(sys.argv) != 3:
        print("Usage: python parse_conll_to_csv.py input.conll output.csv")
        print("\nExample:")
        print("  python parse_conll_to_csv.py inscriptions-3.conll output.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        parse_conll_to_csv(input_file, output_file)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()