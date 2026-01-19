#!/usr/bin/env python3
"""
Run Suffix Automaton Analysis with Validation Enabled
"""

from analyze_midi import MIDIAnalyzer

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

    print('\n📊 RESULTS SUMMARY:')
    print('-'*60)

    for result in results:
        print(f'\nFile: {result["file"]}')
        algo_result = result['algorithms'][0]
        print(f'  Algorithm: {algo_result["name"]}')
        print(f'  Coverage: {algo_result["coverage"]*100:.1f}%')
        print(f'  Processing time: {algo_result["processing_time"]:.3f}s')

        validation = algo_result.get('validation', {})
        if validation.get('validation_enabled', False):
            print(f'  Validation enabled: YES')
            print(f'  Validated matches: {validation.get("validated_matches", 0)}')
            print(f'  Rejected matches: {validation.get("rejected_matches", 0)}')
            print(f'  Average similarity: {validation.get("average_similarity", 0):.3f}')
            print(f'  Similarity threshold: {algo_result.get("similarity_threshold", "N/A")}')
            print(f'  Similarity distribution: {validation.get("similarity_distribution", [])[:5]}...' if validation.get("similarity_distribution") else '  Similarity distribution: []')

    print('\n🎯 Suffix Automaton Analysis with Validation - COMPLETED')

if __name__ == "__main__":
    main()
