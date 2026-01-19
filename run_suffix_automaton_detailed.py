#!/usr/bin/env python3
"""
Run Suffix Automaton Analysis with Validation Enabled - Detailed Results
"""

from analyze_midi import MIDIAnalyzer
import json

def main():
    print('🔬 Running Suffix Automaton Analysis with Validation Enabled')
    print('='*60)

    # Initialize analyzer
    analyzer = MIDIAnalyzer()

    # Define MIDI files to analyze
    files = ['never.mid', 'queen.mid', 'toto-africa.mid']

    results = []

    for file in files:
        print(f'\n📁 Analyzing {file}...')
        file_results = analyzer.analyze_file(file, algorithm_index=4, validation_enabled=True)

        if file_results:
            results.append(file_results)
            print(f'  ✅ Completed analysis for {file}')

    print('\n📊 DETAILED RESULTS SUMMARY:')
    print('-'*60)

    for result in results:
        print(f'\nFile: {result["file"]}')
        algo_result = result['algorithms'][0]
        print(f'  Algorithm: {algo_result["name"]}')
        print(f'  Description: {algo_result["description"]}')
        print(f'  Coverage: {algo_result["coverage"]*100:.1f}%')
        print(f'  Processing time: {algo_result["processing_time"]:.3f}s')
        print(f'  Total notes: {algo_result["total_notes"]}')
        print(f'  Occupied notes: {algo_result["occupied_notes"]}')

        validation = algo_result.get('validation', {})
        print(f'\n  🔍 VALIDATION DETAILS:')
        print(f'    Validation enabled: {validation.get("validation_enabled", False)}')
        print(f'    Validated matches: {validation.get("validated_matches", 0)}')
        print(f'    Rejected matches: {validation.get("rejected_matches", 0)}')
        print(f'    Average similarity: {validation.get("average_similarity", 0):.3f}')
        print(f'    Similarity threshold: {algo_result.get("similarity_threshold", "N/A")}')

        similarity_dist = validation.get("similarity_distribution", [])
        if similarity_dist:
            print(f'    Similarity distribution (first 10): {similarity_dist[:10]}')
            print(f'    Similarity range: {min(similarity_dist):.3f} - {max(similarity_dist):.3f}')
        else:
            print(f'    Similarity distribution: []')

        # Show detailed match information
        details = algo_result.get('details', {})
        motif_matches = details.get('motif_matches', [])
        print(f'\n  📋 MATCH DETAILS:')
        print(f'    Total motif matches: {len(motif_matches)}')
        if motif_matches:
            print(f'    Sample match: {motif_matches[0] if len(motif_matches) > 0 else "None"}')

    print('\n🎯 Suffix Automaton Analysis with Validation - COMPLETED')

if __name__ == "__main__":
    main()
