README.md: MIDI Auditor Implementation Plan

This document specifies the technical fixes and feature enhancements required for the MIDI Auditor. The goal is to transform the tool from a basic similarity finder into a precision DAW-integration utility for professional MIDI editing.
📋 Context & Requirements

    Use Case: Single-instrument MIDI files (one instrument processed at a time).

    Workflow: Identify repeated sections → evaluate performance quality → copy/paste best takes in DAW.

    Requirement: Sections must be musically interchangeable (fuzzy matching).

    Constraint: Support files with multiple MIDI channels (e.g., guitar layers, piano hands).

🛠 Phase 1: Critical Logic Fixes (High Priority)
BUG-1: Velocity Decoupling

Location: midi_auditor.py | _build_bar_features() Problem: Velocity is currently baked into features. Chorus (Forte) vs. Chorus (Piano) are not detected as matches. Fix:
Python

# Remove velocity component to allow matching across different dynamics
weight = math.sqrt(note["duration"]) 

BUG-2: Time Signature Awareness

Location: midi_auditor.py | _find_large_scale_repeats() Problem: Hardcoded 4/4 logic causes misalignment in Waltz (3/4) or Progressive (6/8) tracks. Fix:
Python

bar_ticks = self.ticks_per_bar  # Use dynamic value from MIDI header

BUG-3: Time Signature Map

Location: midi_auditor.py | _compute_average_ticks_per_bar() Problem: Averages time signatures (e.g., 4/4 and 3/4 becomes 3.5/4), breaking all bar boundaries. Fix: Implement a tick_to_bar mapping function.
Python

# Block: Time Map Logic
def tick_to_bar(self, tick: int) -> int:
    if not self.time_signatures:
        return tick // (self.ticks_per_beat * 4)
    
    current_bar = 0
    for i, (ts_tick, ts) in enumerate(self.time_signatures):
        next_tick = self.time_signatures[i+1][0] if i+1 < len(self.time_signatures) else float('inf')
        
        if tick < next_tick:
            bars_into_section = (tick - ts_tick) // ts.ticks_per_bar
            return current_bar + bars_into_section
        
        bars_in_section = (next_tick - ts_tick) // ts.ticks_per_bar
        current_bar += bars_in_section
    return current_bar

🔍 Phase 2: Interchangeability Validation
MISSING-1: Swappability Verification

Ensure matches aren't just similar, but technically swappable in a DAW.
Python

# Block: Validation
def _validate_interchangeability(self, lm: LargeMatch) -> bool:
    notes_a = self.notes_in_bar_range(lm.start_bar_a, lm.length_bars)
    notes_b = self.notes_in_bar_range(lm.start_bar_b, lm.length_bars)
    
    # 1. Channel Match: Piano cannot be swapped for Strings
    if set(n["channel"] for n in notes_a) != set(n["channel"] for n in notes_b):
        return False
        
    # 2. Pitch/Rhythm Similarity: Use SequenceMatcher for fuzzy verification
    from difflib import SequenceMatcher
    p_ratio = SequenceMatcher(None, [n["pitch"] for n in notes_a], [n["pitch"] for n in notes_b]).ratio()
    
    return p_ratio >= 0.85 # Threshold for interchangeability

MISSING-2: Boundary Alignment

Refine bar-level matches to precise note-level boundaries to handle pickup notes or trailing tails.

    Logic: Find the Longest Common Subsequence (LCS) of pitches.

    Action: Trim LargeMatch start/end ticks to the boundaries of that LCS.

📤 Phase 3: DAW Export & UI
EXPORT-1: Precision Markers

Problem: Markers currently use bar approximations. Fix: Export MIDI markers using exact min(tick) and max(tick) of the validated notes within a match.
FEATURE-1: Performance Quality Scoring

Analyze segments to recommend which take to "Keep."

    Timing Tightness: Variance from the 16th note grid.

    Velocity Consistency: Standard deviation of velocities.

    Note Density: Richness of the arrangement.

Python

# Block: Scoring
def _score_performance_quality(self, notes: List[Dict]) -> float:
    sixteenth = self.ticks_per_beat // 4
    timing_err = np.mean([abs(n["tick"] % sixteenth) for n in notes])
    timing_score = max(0.0, 1.0 - (timing_err / (sixteenth / 2)))
    
    return (0.4 * timing_score) + (0.6 * other_metrics)

⚙️ Recommended Configuration
Parameter	Value	Purpose
large_similarity	0.90	High threshold since velocity is ignored
min_large_bars	2	Catch short motifs
pitch_tolerance	0.85	Allow minor variations/ornaments
rhythm_bins	16	Resolution up to 16th notes
🚀 Execution Order

    Core Math: Fix Velocity (BUG-1) and Bar Ticks (BUG-2).

    Validation: Implement _validate_interchangeability to filter results.

    Alignment: Implement _align_segment_boundaries for sample-accurate DAW placement.

    UX: Add Performance Quality Scoring and Marker Export.

MIDI Auditor: Automated Test Suite Specification

This document defines the quantitative verification layer for the MIDI Auditor. It provides a blueprint for programmatically generating test data and measuring the performance of the auditor's logic gates.
🏗️ Test Data Generation

The following generator creates synthetic MIDI files with embedded "Ground Truth" to ensure fixes for BUG-1 through BUG-3 are verified objectively.
Python

# test_data_generator.py
import mido
import io
from mido import MidiFile, MidiTrack, Message, MetaMessage

class MIDITestGenerator:
    """Creates ground-truth MIDI files for unit testing auditor logic."""
    
    @staticmethod
    def create_velocity_test(ticks_per_beat=480):
        """
        Scenario: Identical melody at velocity 40 and velocity 110.
        Fix Target: BUG-1 (Velocity Independence).
        """
        mid = MidiFile(ticks_per_beat=ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)
        
        # Pattern: C-E-G-C (4 bars)
        def add_pattern(vel, start_tick):
            for i, pitch in enumerate([60, 64, 67, 72]):
                tick = start_tick + (i * ticks_per_beat)
                track.append(Message('note_on', note=pitch, velocity=vel, time=ticks_per_beat if i > 0 else start_tick))
                track.append(Message('note_off', note=pitch, velocity=0, time=ticks_per_beat // 2))
        
        add_pattern(40, 0)             # Section A
        add_pattern(110, 480 * 16)      # Section B (after 4 bar gap)
        
        return mid

📈 Quantitative Metrics

To evaluate the "Agentic IDE" performance, use these metrics to tune the auditor's fuzzy matching thresholds.
1. Match Detection Accuracy (F1-Score)

    Precision: Percentage of detected matches that are actually interchangeable.

    Recall: Percentage of total ground-truth repetitions discovered.

    Goal: >0.95 for quantized MIDI; >0.80 for human performances.

2. Alignment Drift (Ticks)

    Formula: Δ=∣TickFound​−TickActual​∣

    Goal: Δ<(TicksPerBeat/16) (Must be within 1/16th note for DAW usage).

3. Quality Ranking Correlation

    Test: Rank 3 versions of the same phrase (Quantized, 20ms Jitter, 50ms Jitter).

    Goal: Auditor must rank Quantized > 20ms > 50ms consistently.

🧪 Integration Test Suite
Execution Instructions for Agent

The Agent should execute the following test suite to validate the implementation of Phases 1-4.
Python

# test_suite.py
import unittest
from midi_auditor import MIDIAuditor

class MIDIAuditorTestSuite(unittest.TestCase):
    
    def test_velocity_normalization(self):
        """Verify BUG-1 fix: Matching is dynamics-agnostic."""
        # ... logic to run auditor on create_velocity_test()
        # assert len(matches) == 1
        
    def test_time_sig_alignment(self):
        """Verify BUG-2 fix: 3/4 waltz does not drift."""
        # ... generate 3/4 MIDI
        # assert match.length_bars == expected_bars
        
    def test_interchangeability_validation(self):
        """Verify MISSING-1: Rejects matches with different MIDI channels."""
        # ... generate MIDI with identical notes on Channel 1 vs Channel 2
        # assert len(validated_matches) == 0

    def test_boundary_alignment(self):
        """Verify MISSING-2: Aligns to note onsets, not just bar lines."""
        # ... generate pattern with 1/8th note pickup
        # assert match.start_tick % 480 != 0 

🛠️ Optimization Loop (For Agent Tuning)

If the test suite fails, the Agent should iterate on the following logic gates:

    If Precision is low: Increase pitch_tolerance or rhythm_tolerance.

    If Recall is low: Decrease large_similarity threshold (allow fuzzier matches).

    If Alignment is off: Check the tick_to_bar mapping logic for off-by-one errors in delta-time accumulation.