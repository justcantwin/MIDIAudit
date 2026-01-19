#!/usr/bin/env python3
"""
Auto-tuning script for MIDI pattern recognition algorithms

This script uses the AutoTuningFramework to systematically test and optimize
algorithm parameters on real MIDI files for maximum coverage and similarity.
"""

import os
import sys
import time
import json
from typing import Dict, List
from analyze_midi import MIDIAnalyzer, UltraLongSectionMatcher, AutoTuningFramework

def main():
    print("🎼 MIDI Pattern Recognition Auto-Tuning")
    print("=" * 50)

    # Initialize frameworks
    analyzer = MIDIAnalyzer()
    tuner = AutoTuningFramework()

    # Find MIDI files in current directory
    midi_files = []
    for file in os.listdir('.'):
        if file.lower().endswith(('.mid', '.midi')):
            midi_files.append(file)

    if not midi_files:
        print("❌ No MIDI files found in current directory!")
        print("Please place some MIDI files (.mid or .midi) in the current directory.")
        return

    print(f"📁 Found {len(midi_files)} MIDI files:")
    for f in midi_files:
        tuner.add_test_file(f)
        print(f"  - {f}")

    # Define parameter ranges for UltraLongSectionMatcher
    param_ranges = {
        'min_section_bars': [4, 8, 12, 16],
        'max_section_bars': [32, 48, 64, 96],
        'similarity_threshold': [0.85, 0.90, 0.95, 0.98]
    }

    print("\n🎯 Starting parameter optimization...")
    print("Testing UltraLongSectionMatcher with different parameter combinations")

    # Run optimization
    start_time = time.time()
    optimal_params = tuner.optimize_parameters(
        algorithm_class=UltraLongSectionMatcher,
        param_ranges=param_ranges,
        target_metric='coverage'  # Optimize for maximum coverage
    )
    optimization_time = time.time() - start_time

    if optimal_params:
        print("\n✅ Optimization complete!")
        print(f"Optimization took {optimization_time:.2f} seconds")
        print("📊 Optimal parameters:")
        for param, value in optimal_params.items():
            print(f"  {param}: {value}")

        # Test optimal parameters on all files
        print("\n🔍 Testing optimal parameters on all files...")
        test_results = []

        for midi_file in midi_files:
            print(f"Testing {midi_file}...")

            # Create algorithm with optimal parameters
            algo = UltraLongSectionMatcher(**optimal_params)

            # Run analysis
            with open(midi_file, 'rb') as f:
                midi_data = f.read()

            # Create fresh auditor for each test
            from midi_auditor import MIDIAuditor
            import io
            auditor = MIDIAuditor(io.BytesIO(midi_data), verbose=False)
            result = algo.analyze(auditor)

            test_results.append({
                'file': midi_file,
                'coverage': result['coverage'],
                'large_matches': len(result.get('large_matches', [])),
                'avg_match_length': result.get('avg_match_length', 0),
                'processing_time': result['processing_time']
            })

        # Generate comprehensive report
        generate_tuning_report(optimal_params, test_results, optimization_time)

    else:
        print("❌ Optimization failed - no valid results found")

def generate_tuning_report(optimal_params: Dict, test_results: List[Dict], optimization_time: float):
    """Generate a comprehensive tuning report"""

    report = []
    report.append("🎼 MIDI Pattern Recognition Auto-Tuning Report")
    report.append("=" * 55)
    report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Optimization summary
    report.append("🎯 OPTIMIZATION SUMMARY")
    report.append("-" * 30)
    report.append(f"Optimization time: {optimization_time:.2f} seconds")
    report.append(f"Target metric: Coverage maximization")
    report.append("")

    # Optimal parameters
    report.append("📊 OPTIMAL PARAMETERS")
    report.append("-" * 30)
    for param, value in optimal_params.items():
        report.append(f"{param}: {value}")
    report.append("")

    # Test results
    report.append("📋 TEST RESULTS")
    report.append("-" * 30)

    total_coverage = 0
    total_matches = 0
    total_time = 0

    for result in test_results:
        report.append(f"File: {result['file']}")
        report.append(f"  Coverage: {result['coverage']*100:.1f}%")
        report.append(f"  Large matches: {result['large_matches']}")
        report.append(f"  Avg match length: {result['avg_match_length']:.1f} bars")
        report.append(f"  Processing time: {result['processing_time']:.3f}s")
        report.append("")

        total_coverage += result['coverage']
        total_matches += result['large_matches']
        total_time += result['processing_time']

    # Summary statistics
    avg_coverage = total_coverage / len(test_results)
    avg_matches = total_matches / len(test_results)
    avg_time = total_time / len(test_results)

    report.append("📈 SUMMARY STATISTICS")
    report.append("-" * 30)
    report.append(f"Files tested: {len(test_results)}")
    report.append(f"Average coverage: {avg_coverage*100:.1f}%")
    report.append(f"Average matches per file: {avg_matches:.1f}")
    report.append(f"Average processing time: {avg_time:.3f}s")
    report.append("")

    # Recommendations
    report.append("💡 RECOMMENDATIONS")
    report.append("-" * 30)

    if avg_coverage >= 0.8:
        report.append("✅ Excellent coverage achieved! Algorithm is well-tuned.")
    elif avg_coverage >= 0.6:
        report.append("⚠️ Good coverage, but room for improvement.")
        report.append("   Consider lowering similarity_threshold or adjusting min_section_bars.")
    else:
        report.append("❌ Coverage needs improvement.")
        report.append("   Try reducing similarity_threshold or min_section_bars.")

    if avg_matches < 2:
        report.append("⚠️ Low number of matches detected.")
        report.append("   Consider reducing min_section_bars to find more sections.")
    elif avg_matches > 20:
        report.append("⚠️ High number of matches detected.")
        report.append("   Consider increasing min_section_bars to focus on longer sections.")

    # Save report
    report_filename = f"tuning_report_{int(time.time())}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"\n📄 Report saved to: {report_filename}")

    # Also save as JSON for programmatic access
    json_filename = f"tuning_results_{int(time.time())}.json"
    json_data = {
        'optimal_params': optimal_params,
        'test_results': test_results,
        'summary': {
            'optimization_time': optimization_time,
            'average_coverage': avg_coverage,
            'average_matches': avg_matches,
            'average_time': avg_time
        },
        'timestamp': time.time()
    }

    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"📄 JSON results saved to: {json_filename}")

    # Print summary to console
    print("\n🎯 TUNING COMPLETE")
    print(f"Optimal coverage: {avg_coverage*100:.1f}%")
    print(f"Average matches: {avg_matches:.1f}")
    print(f"Parameters: {optimal_params}")

if __name__ == "__main__":
    main()
