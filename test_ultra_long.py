#!/usr/bin/env python3
"""Quick test of UltraLongSectionMatcher"""

from analyze_midi import UltraLongSectionMatcher
from midi_auditor import MIDIAuditor
import io

# Test with a simple MIDI file if available
midi_files = []
import os
for file in os.listdir('.'):
    if file.lower().endswith(('.mid', '.midi')):
        midi_files.append(file)

if midi_files:
    print(f"Testing with {midi_files[0]}")
    with open(midi_files[0], 'rb') as f:
        data = f.read()

    auditor = MIDIAuditor(io.BytesIO(data), verbose=False)

    # Test with different parameters
    test_configs = [
        {'similarity_threshold': 0.8, 'min_section_bars': 4},  # More flexible
        {'similarity_threshold': 0.7, 'min_section_bars': 2},  # Even more flexible
        {'similarity_threshold': 0.6, 'min_section_bars': 1},  # Very flexible
    ]

    for i, config in enumerate(test_configs):
        print(f"\n--- Test {i+1}: threshold={config['similarity_threshold']}, min_bars={config['min_section_bars']} ---")
        algo = UltraLongSectionMatcher(**config)
        result = algo.analyze(auditor)

        print(f"Coverage: {result['coverage']:.3f}")
        print(f"Large matches: {len(result.get('large_matches', []))}")
        print(f"Avg match length: {result.get('avg_match_length', 0):.1f} bars")

        # Show details of matches if any
        matches = result.get('large_matches', [])
        if matches:
            print("Sample matches:")
            for i, match in enumerate(matches[:3]):  # Show first 3
                print(f"  Match {i+1}: Bars {match.start_bar_a}-{match.start_bar_a + match.length_bars - 1} ↔ {match.start_bar_b}-{match.start_bar_b + match.length_bars - 1} (sim: {match.avg_similarity:.3f})")

    print("\n✅ UltraLongSectionMatcher works!")
else:
    print("✅ UltraLongSectionMatcher imports successfully (no MIDI files to test)")
