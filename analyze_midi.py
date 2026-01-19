#!/usr/bin/env python3
"""
Comprehensive MIDI Pattern Analysis and Algorithm Optimization Framework

This enhanced analyze_midi.py provides a systematic approach to testing and optimizing
different pattern recognition algorithms to achieve 90%+ coverage on real MIDI files.

Key Features:
- Multiple pattern recognition algorithms with different approaches
- Transposition-invariant and rhythmically flexible matching
- Comprehensive performance metrics and automated optimization
- Detailed reporting and visualization
- Systematic testing framework for algorithm comparison
- Similarity verification and validation framework
- Progress tracking and status indicators
- Performance optimization for large MIDI files
"""

import io
import os
import time
import numpy as np
import hashlib
import math
from typing import List, Dict, Tuple, Callable, Any
from collections import defaultdict
from difflib import SequenceMatcher
import copy
import sys
from functools import lru_cache

from midi_auditor import MIDIAuditor
from models import LargeMatch
import mido

# Import additional libraries for advanced analysis
try:
    import music21
    from music21 import converter, stream, note, chord, interval
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("Warning: music21 not available. Some advanced features will be disabled.")

class PerformanceOptimizer:
    """Performance optimization utilities"""

    @staticmethod
    def time_function(func):
        """Decorator to time function execution"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"    ⏱️  {func.__name__} completed in {end_time - start_time:.3f}s")
            return result
        return wrapper

    @staticmethod
    def optimize_numpy_operations():
        """Optimize numpy operations for better performance"""
        # Use faster numpy operations where possible
        np.seterr(divide='ignore', invalid='ignore')

    @staticmethod
    def limit_memory_usage():
        """Limit memory usage for large operations"""
        # Set reasonable limits for large operations
        MAX_PATTERN_LENGTH = 32
        MAX_SIMILARITY_CALCULATIONS = 1000
        return MAX_PATTERN_LENGTH, MAX_SIMILARITY_CALCULATIONS

class ProgressTracker:
    """Progress tracking and status reporting"""

    @staticmethod
    def show_progress(current: int, total: int, message: str = ""):
        """Show progress bar and status"""
        progress = current / total
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        status = f"\r[{bar}] {int(progress * 100)}% {message}"
        sys.stdout.write(status)
        sys.stdout.flush()

        if progress >= 1.0:
            print()  # New line when complete

    @staticmethod
    def print_status(message: str, level: str = "info"):
        """Print formatted status message"""
        symbols = {
            'info': 'ℹ️ ',
            'success': '✅ ',
            'warning': '⚠️ ',
            'error': '❌ ',
            'processing': '🔄 '
        }
        symbol = symbols.get(level, '  ')
        print(f"{symbol}{message}")

class SimilarityMetrics:
    """Comprehensive similarity metrics for pattern validation with performance optimizations"""

    @staticmethod
    @lru_cache(maxsize=1000)
    def pitch_sequence_similarity(seq1: tuple, seq2: tuple) -> float:
        """Calculate pitch sequence similarity (0-1) - optimized with caching"""
        if len(seq1) != len(seq2):
            return 0.0

        # Convert to numpy arrays for vectorized operations
        seq1_arr = np.array(seq1)
        seq2_arr = np.array(seq2)

        # Exact match score using vectorized comparison
        exact_matches = np.sum(seq1_arr == seq2_arr)
        exact_score = exact_matches / len(seq1_arr)

        # Interval similarity (transposition-aware)
        if len(seq1_arr) > 1:
            intervals1 = np.diff(seq1_arr)
            intervals2 = np.diff(seq2_arr)
            interval_matches = np.sum(intervals1 == intervals2)
            interval_score = interval_matches / len(intervals1)
        else:
            interval_score = exact_score

        # Combined score
        return 0.7 * exact_score + 0.3 * interval_score

    @staticmethod
    def rhythmic_similarity(rhythms1: List[Dict], rhythms2: List[Dict]) -> float:
        """Calculate rhythmic similarity (0-1) - optimized version"""
        if len(rhythms1) != len(rhythms2):
            return 0.0

        # Vectorized calculation
        durations1 = np.array([r['duration'] for r in rhythms1])
        durations2 = np.array([r['duration'] for r in rhythms2])
        positions1 = np.array([r['position'] for r in rhythms1])
        positions2 = np.array([r['position'] for r in rhythms2])

        # Vectorized similarity calculation
        duration_diffs = np.abs(durations1 - durations2)
        position_diffs = np.abs(positions1 - positions2)

        duration_sim = np.mean(np.maximum(0, 1 - duration_diffs / 4))
        position_sim = np.mean(np.maximum(0, 1 - position_diffs / 0.5))

        return 0.6 * duration_sim + 0.4 * position_sim

    @staticmethod
    def combined_musical_similarity(
        pitch_seq1: List[int], pitch_seq2: List[int],
        rhythms1: List[Dict], rhythms2: List[Dict],
        pitch_weight: float = 0.6, rhythm_weight: float = 0.4
    ) -> float:
        """Calculate comprehensive musical similarity - optimized"""
        # Use cached pitch similarity
        pitch_seq1_tuple = tuple(pitch_seq1)
        pitch_seq2_tuple = tuple(pitch_seq2)
        pitch_sim = SimilarityMetrics.pitch_sequence_similarity(pitch_seq1_tuple, pitch_seq2_tuple)
        rhythm_sim = SimilarityMetrics.rhythmic_similarity(rhythms1, rhythms2)

        return pitch_weight * pitch_sim + rhythm_weight * rhythm_sim

class PatternValidation:
    """Pattern validation and quality assessment with performance optimizations"""

    @staticmethod
    def validate_match_quality(
        auditor: MIDIAuditor,
        match: Dict,
        min_similarity: float = 0.7
    ) -> Dict:
        """Validate the quality of a detected match - optimized"""
        if match['type'] in ['transposition_invariant', 'multi_feature', 'suffix_automaton']:
            return PatternValidation._validate_repeating_pattern(auditor, match, min_similarity)
        elif match['type'] == 'rhythmic':
            return PatternValidation._validate_rhythmic_pattern(auditor, match, min_similarity)
        else:
            return {'valid': True, 'similarity': 1.0, 'confidence': 'high'}

    @staticmethod
    def _validate_repeating_pattern(
        auditor: MIDIAuditor,
        match: Dict,
        min_similarity: float = 0.7
    ) -> Dict:
        """Validate repeating patterns with similarity checking - optimized"""
        if len(match['occurrences']) < 2:
            return {'valid': False, 'similarity': 0.0, 'confidence': 'low', 'reason': 'insufficient_occurrences'}

        # Limit the number of similarity calculations for performance
        MAX_SIMILARITY_CALCULATIONS = 1000
        pitch_sequences = []
        rhythm_sequences = []

        for occ_start in match['occurrences']:
            end_idx = occ_start + match['length']
            if end_idx > len(auditor.notes):
                continue

            # Extract pitch sequence
            pitches = [auditor.notes[i]['pitch'] for i in range(occ_start, end_idx)]
            pitch_sequences.append(pitches)

            # Extract rhythmic information
            rhythms = []
            ticks_per_beat = auditor.ticks_per_beat
            for i in range(occ_start, end_idx):
                note_data = auditor.notes[i]
                rhythms.append({
                    'duration': note_data['duration'] / (ticks_per_beat / 4),
                    'position': (note_data['tick'] % ticks_per_beat) / ticks_per_beat
                })
            rhythm_sequences.append(rhythms)

            # Early exit if we have enough sequences
            if len(pitch_sequences) >= 10:  # Limit to 10 occurrences for performance
                break

        # Calculate pairwise similarities with sampling for large numbers
        similarities = []
        num_sequences = len(pitch_sequences)
        if num_sequences > 5:
            # For many sequences, sample a subset of pairs
            sample_size = min(MAX_SIMILARITY_CALCULATIONS, num_sequences * (num_sequences - 1) // 2)
            pairs_sampled = 0

            for i in range(num_sequences):
                for j in range(i + 1, num_sequences):
                    if i >= len(rhythm_sequences) or j >= len(rhythm_sequences):
                        continue

                    if pairs_sampled >= sample_size:
                        break

                    sim = SimilarityMetrics.combined_musical_similarity(
                        pitch_sequences[i], pitch_sequences[j],
                        rhythm_sequences[i], rhythm_sequences[j]
                    )
                    similarities.append(sim)
                    pairs_sampled += 1

                if pairs_sampled >= sample_size:
                    break
        else:
            # For small numbers, calculate all pairs
            for i in range(num_sequences):
                for j in range(i + 1, num_sequences):
                    if i >= len(rhythm_sequences) or j >= len(rhythm_sequences):
                        continue

                    sim = SimilarityMetrics.combined_musical_similarity(
                        pitch_sequences[i], pitch_sequences[j],
                        rhythm_sequences[i], rhythm_sequences[j]
                    )
                    similarities.append(sim)

        if not similarities:
            return {'valid': False, 'similarity': 0.0, 'confidence': 'low', 'reason': 'no_valid_pairs'}

        avg_similarity = np.mean(similarities)
        min_similarity_score = np.min(similarities)

        # Determine validation result
        if avg_similarity >= min_similarity:
            confidence = 'high' if avg_similarity >= 0.9 else ('medium' if avg_similarity >= 0.8 else 'low')
            return {
                'valid': True,
                'similarity': avg_similarity,
                'min_similarity': min_similarity_score,
                'confidence': confidence,
                'similarity_distribution': similarities
            }
        else:
            return {
                'valid': False,
                'similarity': avg_similarity,
                'min_similarity': min_similarity_score,
                'confidence': 'low',
                'reason': 'similarity_too_low'
            }

    @staticmethod
    def _validate_rhythmic_pattern(
        auditor: MIDIAuditor,
        match: Dict,
        min_similarity: float = 0.7
    ) -> Dict:
        """Validate rhythmic patterns - optimized"""
        if len(match['occurrences']) < 2:
            return {'valid': False, 'similarity': 0.0, 'confidence': 'low', 'reason': 'insufficient_occurrences'}

        # For rhythmic patterns, we focus on rhythmic similarity
        rhythmic_sequences = []

        for occ_start in match['occurrences']:
            end_idx = occ_start + match['length']
            if end_idx > len(auditor.notes):
                continue

            # Extract rhythmic information
            rhythms = []
            ticks_per_beat = auditor.ticks_per_beat
            for i in range(occ_start, end_idx):
                note_data = auditor.notes[i]
                rhythms.append({
                    'duration': note_data['duration'] / (ticks_per_beat / 4),
                    'position': (note_data['tick'] % ticks_per_beat) / ticks_per_beat
                })
            rhythmic_sequences.append(rhythms)

        # Calculate pairwise rhythmic similarities
        similarities = []
        for i in range(len(rhythmic_sequences)):
            for j in range(i + 1, len(rhythmic_sequences)):
                sim = SimilarityMetrics.rhythmic_similarity(rhythmic_sequences[i], rhythmic_sequences[j])
                similarities.append(sim)

        if not similarities:
            return {'valid': False, 'similarity': 0.0, 'confidence': 'low', 'reason': 'no_valid_pairs'}

        avg_similarity = np.mean(similarities)

        # Determine validation result
        if avg_similarity >= min_similarity:
            confidence = 'high' if avg_similarity >= 0.9 else ('medium' if avg_similarity >= 0.8 else 'low')
            return {
                'valid': True,
                'similarity': avg_similarity,
                'confidence': confidence,
                'similarity_distribution': similarities
            }
        else:
            return {
                'valid': False,
                'similarity': avg_similarity,
                'confidence': 'low',
                'reason': 'rhythmic_similarity_too_low'
            }

class PatternRecognitionAlgorithm:
    """Base class for pattern recognition algorithms"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.results = None
        self.similarity_threshold = 0.7
        self.validation_enabled = True
        self.performance_optimized = True

    def analyze(self, auditor: MIDIAuditor) -> Dict:
        """Analyze MIDI using this algorithm"""
        raise NotImplementedError

    def get_parameters(self) -> Dict:
        """Get current parameters"""
        return {
            'similarity_threshold': self.similarity_threshold,
            'validation_enabled': self.validation_enabled,
            'performance_optimized': self.performance_optimized
        }

    def set_parameters(self, params: Dict):
        """Set parameters"""
        if 'similarity_threshold' in params:
            self.similarity_threshold = params['similarity_threshold']
        if 'validation_enabled' in params:
            self.validation_enabled = params['validation_enabled']
        if 'performance_optimized' in params:
            self.performance_optimized = params['performance_optimized']

class ExactMatchingAlgorithm(PatternRecognitionAlgorithm):
    """Original exact matching algorithm (baseline)"""

    def __init__(self):
        super().__init__(
            "Exact Matching",
            "Original brute-force exact pattern matching"
        )

    def analyze(self, auditor: MIDIAuditor) -> Dict:
        start_time = time.time()
        ProgressTracker.print_status(f"  🔍 Analyzing with {self.name}...", "processing")

        # Use the original find_all_patterns method
        large_matches, motif_matches = auditor.find_all_patterns(
            min_motif_length=4,
            min_large_bars=2
        )

        processing_time = time.time() - start_time
        total_notes = len(auditor.notes)
        occupied_notes = len(auditor.occupied_indices)
        coverage = occupied_notes / total_notes if total_notes > 0 else 0

        ProgressTracker.print_status(f"  ✅ Completed {self.name} in {processing_time:.2f}s", "success")

        return {
            'large_matches': large_matches,
            'motif_matches': motif_matches,
            'coverage': coverage,
            'processing_time': processing_time,
            'total_notes': total_notes,
            'occupied_notes': occupied_notes
        }

class TranspositionInvariantAlgorithm(PatternRecognitionAlgorithm):
    """Transposition-invariant pattern matching with validation - optimized"""

    def __init__(self, similarity_threshold=0.7):
        super().__init__(
            "Transposition Invariant",
            "Detects patterns regardless of key/transposition with similarity validation"
        )
        self.similarity_threshold = similarity_threshold

    def analyze(self, auditor: MIDIAuditor) -> Dict:
        start_time = time.time()
        ProgressTracker.print_status(f"  🔍 Analyzing with {self.name}...", "processing")

        # Reset occupied indices for fair comparison
        auditor.occupied_indices = set()

        # Find transposition-invariant patterns with performance optimization
        matches = self._find_transposition_invariant_patterns_optimized(auditor)

        # Validate matches if enabled
        if self.validation_enabled:
            validated_matches = []
            similarity_scores = []
            rejected_count = 0

            for match in matches:
                validation_result = PatternValidation.validate_match_quality(
                    auditor, match, self.similarity_threshold
                )

                if validation_result['valid']:
                    validated_matches.append(match)
                    similarity_scores.append(validation_result['similarity'])
                    # Mark these notes as occupied only if validated
                    for occ_start in match['occurrences']:
                        for k in range(match['length']):
                            note_idx = occ_start + k
                            if note_idx < len(auditor.notes):
                                auditor.occupied_indices.add(note_idx)
                else:
                    rejected_count += 1
                    if rejected_count <= 5:  # Limit rejection messages for performance
                        print(f"    ❌ Rejected low-similarity match: {validation_result['similarity']:.3f}")
                    elif rejected_count == 6:
                        print(f"    ⚠️  Additional low-similarity matches rejected (suppressing further messages)")

            matches = validated_matches

            # Calculate validation statistics
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
            validation_stats = {
                'validated_matches': len(validated_matches),
                'rejected_matches': rejected_count,
                'average_similarity': avg_similarity,
                'similarity_distribution': similarity_scores
            }
        else:
            # Mark all matches as occupied if validation disabled
            for match in matches:
                for occ_start in match['occurrences']:
                    for k in range(match['length']):
                        note_idx = occ_start + k
                        if note_idx < len(auditor.notes):
                            auditor.occupied_indices.add(note_idx)
            validation_stats = {'validation_enabled': False}

        processing_time = time.time() - start_time
        total_notes = len(auditor.notes)
        occupied_notes = len(auditor.occupied_indices)
        coverage = occupied_notes / total_notes if total_notes > 0 else 0

        ProgressTracker.print_status(f"  ✅ Completed {self.name} in {processing_time:.2f}s", "success")

        return {
            'motif_matches': matches,
            'coverage': coverage,
            'processing_time': processing_time,
            'total_notes': total_notes,
            'occupied_notes': occupied_notes,
            'validation': validation_stats,
            'similarity_threshold': self.similarity_threshold
        }

    def _find_transposition_invariant_patterns_optimized(self, auditor: MIDIAuditor) -> List:
        """Find patterns that repeat in different keys - optimized version"""
        pitch_array = auditor.pitch_array
        n = len(pitch_array)
        matches = []
        match_id = 1

        # Limit pattern lengths for performance
        pattern_lengths = [8, 12, 16, 20, 24]  # Focus on musically relevant lengths

        for pattern_length in pattern_lengths:
            if pattern_length > n:
                continue

            pattern_dict = defaultdict(list)

            # Use step size for large arrays to reduce computation
            step_size = 1 if n < 5000 else 2
            for i in range(0, n - pattern_length + 1, step_size):
                if i in auditor.occupied_indices:
                    continue

                base_pattern = pitch_array[i:i + pattern_length]

                # Generate transposition-invariant signature
                if pattern_length > 1:
                    intervals = np.diff(base_pattern)
                    signature = tuple(intervals)
                else:
                    signature = tuple(base_pattern)

                pattern_dict[signature].append(i)

            # Find signatures that appear multiple times
            for signature, positions in pattern_dict.items():
                if len(positions) >= 2:
                    matches.append({
                        'id': match_id,
                        'length': pattern_length,
                        'occurrences': positions,
                        'signature': signature,
                        'type': 'transposition_invariant'
                    })
                    match_id += 1

        return matches

class RhythmicPatternAlgorithm(PatternRecognitionAlgorithm):
    """Rhythm-focused pattern detection with validation - optimized"""

    def __init__(self, rhythm_tolerance=0.2, similarity_threshold=0.75):
        super().__init__(
            "Rhythmic Patterns",
            "Focuses on rhythmic patterns and timing with similarity validation"
        )
        self.rhythm_tolerance = rhythm_tolerance
        self.similarity_threshold = similarity_threshold

    def analyze(self, auditor: MIDIAuditor) -> Dict:
        start_time = time.time()
        ProgressTracker.print_status(f"  🔍 Analyzing with {self.name}...", "processing")

        # Reset occupied indices
        auditor.occupied_indices = set()

        # Extract rhythmic information - optimized
        rhythms = self._extract_rhythmic_features_optimized(auditor)

        # Find rhythmic patterns - optimized
        matches = self._find_rhythmic_patterns_optimized(auditor, rhythms)

        # Validate matches if enabled
        if self.validation_enabled:
            validated_matches = []
            similarity_scores = []
            rejected_count = 0

            for match in matches:
                validation_result = PatternValidation.validate_match_quality(
                    auditor, match, self.similarity_threshold
                )

                if validation_result['valid']:
                    validated_matches.append(match)
                    similarity_scores.append(validation_result['similarity'])
                    # Mark these notes as occupied only if validated
                    for pos in match['occurrences']:
                        for k in range(match['length']):
                            note_idx = rhythms[pos + k]['index']
                            auditor.occupied_indices.add(note_idx)
                else:
                    rejected_count += 1
                    if rejected_count <= 5:  # Limit rejection messages
                        print(f"    ❌ Rejected low-similarity rhythmic match: {validation_result['similarity']:.3f}")
                    elif rejected_count == 6:
                        print(f"    ⚠️  Additional low-similarity matches rejected (suppressing further messages)")

            matches = validated_matches

            # Calculate validation statistics
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
            validation_stats = {
                'validated_matches': len(validated_matches),
                'rejected_matches': rejected_count,
                'average_similarity': avg_similarity,
                'similarity_distribution': similarity_scores
            }
        else:
            # Mark all matches as occupied if validation disabled
            for match in matches:
                for pos in match['occurrences']:
                    for k in range(match['length']):
                        note_idx = rhythms[pos + k]['index']
                        auditor.occupied_indices.add(note_idx)
            validation_stats = {'validation_enabled': False}

        processing_time = time.time() - start_time
        total_notes = len(auditor.notes)
        occupied_notes = len(auditor.occupied_indices)
        coverage = occupied_notes / total_notes if total_notes > 0 else 0

        ProgressTracker.print_status(f"  ✅ Completed {self.name} in {processing_time:.2f}s", "success")

        return {
            'motif_matches': matches,
            'coverage': coverage,
            'processing_time': processing_time,
            'total_notes': total_notes,
            'occupied_notes': occupied_notes,
            'validation': validation_stats,
            'similarity_threshold': self.similarity_threshold
        }

    def _extract_rhythmic_features_optimized(self, auditor: MIDIAuditor) -> List:
        """Extract rhythmic features - optimized version"""
        rhythms = []
        ticks_per_beat = auditor.ticks_per_beat

        # Use vectorized operations where possible
        for i, note_data in enumerate(auditor.notes):
            # Normalize duration to 16th notes
            duration_16th = note_data['duration'] / (ticks_per_beat / 4)

            # Position within beat (0-1)
            beat_pos = (note_data['tick'] % ticks_per_beat) / ticks_per_beat

            rhythms.append({
                'duration': duration_16th,
                'position': beat_pos,
                'index': i
            })

        return rhythms

    def _find_rhythmic_patterns_optimized(self, auditor: MIDIAuditor, rhythms: List) -> List:
        """Find repeating rhythmic patterns - optimized"""
        matches = []
        match_id = 1

        # Focus on most musically relevant pattern lengths
        pattern_lengths = [4, 6, 8, 10, 12]

        for pattern_length in pattern_lengths:
            if pattern_length > len(rhythms):
                continue

            pattern_dict = defaultdict(list)

            # Use step size for large rhythm arrays
            step_size = 1 if len(rhythms) < 2000 else 2
            for i in range(0, len(rhythms) - pattern_length + 1, step_size):
                if rhythms[i]['index'] in auditor.occupied_indices:
                    continue

                # Create rhythmic signature - use rounding for performance
                rhythmic_signature = tuple(
                    (round(rhythms[j]['duration'], 1), round(rhythms[j]['position'], 2))
                    for j in range(i, i + pattern_length)
                )

                pattern_dict[rhythmic_signature].append(rhythms[i]['index'])

            # Find repeating rhythmic patterns
            for signature, positions in pattern_dict.items():
                if len(positions) >= 2:
                    matches.append({
                        'id': match_id,
                        'length': pattern_length,
                        'occurrences': positions,
                        'signature': signature,
                        'type': 'rhythmic'
                    })
                    match_id += 1

        return matches

class MultiFeatureAlgorithm(PatternRecognitionAlgorithm):
    """Multi-feature pattern matching with comprehensive validation - optimized"""

    def __init__(self, pitch_weight=0.5, rhythm_weight=0.3, duration_weight=0.2, similarity_threshold=0.8):
        super().__init__(
            "Multi-Feature Matching",
            "Combines pitch, rhythm, and duration features with validation"
        )
        self.pitch_weight = pitch_weight
        self.rhythm_weight = rhythm_weight
        self.duration_weight = duration_weight
        self.similarity_threshold = similarity_threshold

    def analyze(self, auditor: MIDIAuditor) -> Dict:
        start_time = time.time()
        ProgressTracker.print_status(f"  🔍 Analyzing with {self.name}...", "processing")

        # Reset occupied indices
        auditor.occupied_indices = set()

        # Extract multi-dimensional features - optimized
        features = self._extract_multi_features_optimized(auditor)

        # Find multi-feature patterns - optimized
        matches = self._find_multi_feature_patterns_optimized(auditor, features)

        # Validate matches if enabled
        if self.validation_enabled:
            validated_matches = []
            similarity_scores = []
            rejected_count = 0

            for match in matches:
                validation_result = PatternValidation.validate_match_quality(
                    auditor, match, self.similarity_threshold
                )

                if validation_result['valid']:
                    validated_matches.append(match)
                    similarity_scores.append(validation_result['similarity'])
                    # Mark these notes as occupied only if validated
                    for occ_start in match['occurrences']:
                        for k in range(match['length']):
                            note_idx = occ_start + k
                            if note_idx < len(auditor.notes):
                                auditor.occupied_indices.add(note_idx)
                else:
                    rejected_count += 1
                    if rejected_count <= 3:  # Limit rejection messages
                        print(f"    ❌ Rejected low-similarity multi-feature match: {validation_result['similarity']:.3f}")
                    elif rejected_count == 4:
                        print(f"    ⚠️  Additional low-similarity matches rejected (suppressing further messages)")

            matches = validated_matches

            # Calculate validation statistics
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
            validation_stats = {
                'validated_matches': len(validated_matches),
                'rejected_matches': rejected_count,
                'average_similarity': avg_similarity,
                'similarity_distribution': similarity_scores
            }
        else:
            # Mark all matches as occupied if validation disabled
            for match in matches:
                for occ_start in match['occurrences']:
                    for k in range(match['length']):
                        note_idx = occ_start + k
                        if note_idx < len(auditor.notes):
                            auditor.occupied_indices.add(note_idx)
            validation_stats = {'validation_enabled': False}

        processing_time = time.time() - start_time
        total_notes = len(auditor.notes)
        occupied_notes = len(auditor.occupied_indices)
        coverage = occupied_notes / total_notes if total_notes > 0 else 0

        ProgressTracker.print_status(f"  ✅ Completed {self.name} in {processing_time:.2f}s", "success")

        return {
            'motif_matches': matches,
            'coverage': coverage,
            'processing_time': processing_time,
            'total_notes': total_notes,
            'occupied_notes': occupied_notes,
            'validation': validation_stats,
            'similarity_threshold': self.similarity_threshold
        }

    def _extract_multi_features_optimized(self, auditor: MIDIAuditor) -> List:
        """Extract combined pitch, rhythm, and duration features - optimized"""
        features = []
        ticks_per_beat = auditor.ticks_per_beat

        # Use vectorized operations where possible
        for i, note_data in enumerate(auditor.notes):
            # Pitch features
            pitch_class = note_data['pitch'] % 12
            pitch_octave = note_data['pitch'] // 12

            # Rhythm features
            duration_16th = note_data['duration'] / (ticks_per_beat / 4)
            beat_pos = (note_data['tick'] % ticks_per_beat) / ticks_per_beat

            # Combined feature vector
            feature_vector = np.array([
                pitch_class, pitch_octave,
                duration_16th, beat_pos
            ], dtype=np.float32)

            features.append({
                'vector': feature_vector,
                'index': i
            })

        return features

    def _find_multi_feature_patterns_optimized(self, auditor: MIDIAuditor, features: List) -> List:
        """Find patterns using multi-dimensional feature similarity - optimized"""
        matches = []
        match_id = 1

        # Focus on most effective pattern lengths
        pattern_lengths = [6, 8, 10, 12]

        # Limit the number of comparisons for large feature sets
        MAX_COMPARISONS = 5000
        comparisons_made = 0

        for pattern_length in pattern_lengths:
            if pattern_length > len(features):
                continue

            n = len(features)
            step_size = max(1, n // 1000)  # Adjust step size based on feature count

            for i in range(0, n - pattern_length + 1, step_size):
                if comparisons_made >= MAX_COMPARISONS:
                    break

                if features[i]['index'] in auditor.occupied_indices:
                    continue

                base_pattern = np.stack([features[j]['vector'] for j in range(i, i + pattern_length)])

                # Limit comparison range for performance
                comparison_range = min(n - pattern_length, i + 1000)
                for j in range(i + pattern_length, comparison_range, step_size):
                    if comparisons_made >= MAX_COMPARISONS:
                        break

                    if features[j]['index'] in auditor.occupied_indices:
                        continue

                    compare_pattern = np.stack([features[k]['vector'] for k in range(j, j + pattern_length)])

                    # Calculate weighted similarity
                    similarity = self._calculate_pattern_similarity(base_pattern, compare_pattern)
                    comparisons_made += 1

                    if similarity >= 0.8:  # Similarity threshold
                        matches.append({
                            'id': match_id,
                            'length': pattern_length,
                            'occurrences': [features[i]['index'], features[j]['index']],
                            'similarity': similarity,
                            'type': 'multi_feature'
                        })
                        match_id += 1
                        break  # Move to next base pattern

        return matches

    def _calculate_pattern_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate weighted similarity between two patterns"""
        # Separate feature components
        p1_pitch = pattern1[:, :2]
        p1_rhythm = pattern1[:, 2:]

        p2_pitch = pattern2[:, :2]
        p2_rhythm = pattern2[:, 2:]

        # Calculate component similarities using vectorized operations
        pitch_diffs = np.abs(p1_pitch - p2_pitch)
        rhythm_diffs = np.abs(p1_rhythm - p2_rhythm)

        pitch_sim = 1 - np.mean(pitch_diffs) / 12.0  # Normalize by pitch range
        rhythm_sim = 1 - np.mean(rhythm_diffs) / 4.0  # Normalize by rhythm range

        # Weighted combination
        return (
            self.pitch_weight * pitch_sim +
            self.rhythm_weight * rhythm_sim +
            self.duration_weight * rhythm_sim
        )

class SuffixAutomatonAlgorithm(PatternRecognitionAlgorithm):
    """Advanced suffix automaton-based pattern detection with validation - optimized"""

    def __init__(self, min_occurrences=2, min_length=4, similarity_threshold=0.7):
        super().__init__(
            "Suffix Automaton",
            "Leverages suffix automaton for efficient pattern detection with validation"
        )
        self.min_occurrences = min_occurrences
        self.min_length = min_length
        self.similarity_threshold = similarity_threshold

    def analyze(self, auditor: MIDIAuditor) -> Dict:
        start_time = time.time()
        ProgressTracker.print_status(f"  🔍 Analyzing with {self.name}...", "processing")

        # Reset occupied indices
        auditor.occupied_indices = set()

        # Use the suffix automaton to find patterns - optimized
        matches = self._find_sam_patterns_optimized(auditor)

        # Validate matches if enabled
        if self.validation_enabled:
            validated_matches = []
            similarity_scores = []
            rejected_count = 0

            for match in matches:
                validation_result = PatternValidation.validate_match_quality(
                    auditor, match, self.similarity_threshold
                )

                if validation_result['valid']:
                    validated_matches.append(match)
                    similarity_scores.append(validation_result['similarity'])
                    # Mark these notes as occupied only if validated
                    for occ_start in match['occurrences']:
                        for k in range(match['length']):
                            note_idx = occ_start + k
                            if note_idx < len(auditor.notes):
                                auditor.occupied_indices.add(note_idx)
                else:
                    rejected_count += 1
                    if rejected_count <= 5:  # Limit rejection messages
                        print(f"    ❌ Rejected low-similarity SAM match: {validation_result['similarity']:.3f}")
                    elif rejected_count == 6:
                        print(f"    ⚠️  Additional low-similarity matches rejected (suppressing further messages)")

            matches = validated_matches

            # Calculate validation statistics
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
            validation_stats = {
                'validated_matches': len(validated_matches),
                'rejected_matches': rejected_count,
                'average_similarity': avg_similarity,
                'similarity_distribution': similarity_scores
            }
        else:
            # Mark all matches as occupied if validation disabled
            for match in matches:
                for occ_start in match['occurrences']:
                    for k in range(match['length']):
                        note_idx = occ_start + k
                        if note_idx < len(auditor.notes):
                            auditor.occupied_indices.add(note_idx)
            validation_stats = {'validation_enabled': False}

        processing_time = time.time() - start_time
        total_notes = len(auditor.notes)
        occupied_notes = len(auditor.occupied_indices)
        coverage = occupied_notes / total_notes if total_notes > 0 else 0

        ProgressTracker.print_status(f"  ✅ Completed {self.name} in {processing_time:.2f}s", "success")

        return {
            'motif_matches': matches,
            'coverage': coverage,
            'processing_time': processing_time,
            'total_notes': total_notes,
            'occupied_notes': occupied_notes,
            'validation': validation_stats,
            'similarity_threshold': self.similarity_threshold
        }

    def _find_sam_patterns_optimized(self, auditor: MIDIAuditor) -> List:
        """Find patterns using suffix automaton - optimized"""
        matches = []
        match_id = 1

        # Focus on musically relevant pattern lengths
        pattern_lengths = [4, 6, 8, 10, 12, 16]

        for pattern_length in pattern_lengths:
            pattern_dict = defaultdict(list)

            # Use step size for large pitch arrays
            step_size = 1 if len(auditor.pitch_array) < 5000 else 2
            for i in range(0, len(auditor.pitch_array) - pattern_length + 1, step_size):
                if i in auditor.occupied_indices:
                    continue

                pattern = tuple(auditor.pitch_array[i:i + pattern_length])
                pattern_dict[pattern].append(i)

            # Find patterns that repeat
            for pattern, occurrences in pattern_dict.items():
                if len(occurrences) >= self.min_occurrences:
                    matches.append({
                        'id': match_id,
                        'length': pattern_length,
                        'occurrences': occurrences,
                        'pattern': pattern,
                        'type': 'suffix_automaton'
                    })
                    match_id += 1

        return matches

class CoverageOptimizedHybridAlgorithm(PatternRecognitionAlgorithm):
    """Coverage-optimized hybrid with lenient validation for maximum coverage"""

    def __init__(self, similarity_threshold=0.7):
        super().__init__(
            "Coverage-Optimized Hybrid",
            "Prioritizes coverage with lenient validation (Suffix Automaton 70% + Rhythmic Patterns 30%)"
        )
        # Initialize component algorithms with lenient validation
        self.suffix_automaton = SuffixAutomatonAlgorithm(min_occurrences=2, min_length=4, similarity_threshold=similarity_threshold)
        self.rhythmic_patterns = RhythmicPatternAlgorithm(rhythm_tolerance=0.3, similarity_threshold=similarity_threshold)

        # Weights: Suffix Automaton (70%), Rhythmic Patterns (30%)
        self.weights = {
            'suffix_automaton': 0.7,
            'rhythmic_patterns': 0.3
        }
        self.similarity_threshold = similarity_threshold

    def analyze(self, auditor: MIDIAuditor) -> Dict:
        start_time = time.time()
        ProgressTracker.print_status(f"  🔍 Analyzing with {self.name}...", "processing")

        # Reset occupied indices
        auditor.occupied_indices = set()

        all_matches = []
        validation_stats_list = []

        # Run Suffix Automaton (70% weight)
        temp_auditor = copy.copy(auditor)
        temp_auditor.occupied_indices = set()
        sa_result = self.suffix_automaton.analyze(temp_auditor)
        if 'motif_matches' in sa_result:
            all_matches.extend(sa_result['motif_matches'])
        validation_stats_list.append(('suffix_automaton', sa_result.get('validation', {'validation_enabled': False})))

        # Run Rhythmic Patterns (30% weight)
        temp_auditor = copy.copy(auditor)
        temp_auditor.occupied_indices = set()
        rp_result = self.rhythmic_patterns.analyze(temp_auditor)
        if 'motif_matches' in rp_result:
            all_matches.extend(rp_result['motif_matches'])
        validation_stats_list.append(('rhythmic_patterns', rp_result.get('validation', {'validation_enabled': False})))

        # Merge all matches (lenient approach - accept most matches)
        merged_matches = []
        occupied_indices = set()

        for match in all_matches:
            # Validate with lenient threshold
            validation_result = PatternValidation.validate_match_quality(
                auditor, match, max(0.6, self.similarity_threshold - 0.1)  # More lenient
            )

            if validation_result['valid']:
                # Check for minimal overlap
                match_indices = set()
                for occ_start in match['occurrences']:
                    for k in range(match['length']):
                        note_idx = occ_start + k
                        if note_idx < len(auditor.notes):
                            match_indices.add(note_idx)

                overlap = match_indices.intersection(occupied_indices)
                overlap_ratio = len(overlap) / len(match_indices) if match_indices else 0

                # Accept matches with up to 70% overlap (lenient)
                if overlap_ratio < 0.7:
                    merged_matches.append(match)
                    occupied_indices.update(match_indices)

        # Mark occupied indices from merged matches
        for match in merged_matches:
            for occ_start in match['occurrences']:
                for k in range(match['length']):
                    note_idx = occ_start + k
                    if note_idx < len(auditor.notes):
                        auditor.occupied_indices.add(note_idx)

        # Calculate validation statistics
        validation_stats = {
            'validated_matches': len(merged_matches),
            'rejected_matches': len(all_matches) - len(merged_matches),
            'average_similarity': 0.0,
            'similarity_distribution': [],
            'validation_enabled': True
        }

        # Calculate average similarity
        similarity_scores = []
        for match in merged_matches:
            validation_result = PatternValidation.validate_match_quality(
                auditor, match, self.similarity_threshold
            )
            if validation_result['valid']:
                similarity_scores.append(validation_result['similarity'])

        if similarity_scores:
            validation_stats['average_similarity'] = np.mean(similarity_scores)
            validation_stats['similarity_distribution'] = similarity_scores

        processing_time = time.time() - start_time
        total_notes = len(auditor.notes)
        occupied_notes = len(auditor.occupied_indices)
        coverage = occupied_notes / total_notes if total_notes > 0 else 0

        ProgressTracker.print_status(f"  ✅ Completed {self.name} in {processing_time:.2f}s", "success")
        ProgressTracker.print_status(f"    📈 Coverage: {coverage*100:.1f}% with {len(merged_matches)} matches", "success")

        return {
            'motif_matches': merged_matches,
            'coverage': coverage,
            'processing_time': processing_time,
            'total_notes': total_notes,
            'occupied_notes': occupied_notes,
            'validation': validation_stats,
            'similarity_threshold': self.similarity_threshold
        }

class BalancedHybridAlgorithm(PatternRecognitionAlgorithm):
    """Balanced hybrid with moderate validation for coverage-quality balance"""

    def __init__(self, similarity_threshold=0.75):
        super().__init__(
            "Balanced Hybrid",
            "Balances coverage and quality (Suffix Automaton 60% + Rhythmic Patterns 30% + Transposition Invariant 10%)"
        )
        # Initialize component algorithms with moderate validation
        self.suffix_automaton = SuffixAutomatonAlgorithm(min_occurrences=2, min_length=4, similarity_threshold=similarity_threshold)
        self.rhythmic_patterns = RhythmicPatternAlgorithm(rhythm_tolerance=0.2, similarity_threshold=similarity_threshold)
        self.transposition_invariant = TranspositionInvariantAlgorithm(similarity_threshold=similarity_threshold)

        # Weights: Suffix Automaton (60%), Rhythmic Patterns (30%), Transposition Invariant (10%)
        self.weights = {
            'suffix_automaton': 0.6,
            'rhythmic_patterns': 0.3,
            'transposition_invariant': 0.1
        }
        self.similarity_threshold = similarity_threshold

    def analyze(self, auditor: MIDIAuditor) -> Dict:
        start_time = time.time()
        ProgressTracker.print_status(f"  🔍 Analyzing with {self.name}...", "processing")

        # Reset occupied indices
        auditor.occupied_indices = set()

        all_matches = []
        validation_stats_list = []

        # Run all three algorithms
        for algo_name, algo in [('suffix_automaton', self.suffix_automaton),
                              ('rhythmic_patterns', self.rhythmic_patterns),
                              ('transposition_invariant', self.transposition_invariant)]:
            if self.weights[algo_name] > 0:
                temp_auditor = copy.copy(auditor)
                temp_auditor.occupied_indices = set()
                result = algo.analyze(temp_auditor)
                if 'motif_matches' in result:
                    all_matches.extend(result['motif_matches'])
                validation_stats_list.append((algo_name, result.get('validation', {'validation_enabled': False})))

        # Merge matches with balanced approach
        merged_matches = []
        occupied_indices = set()

        for match in all_matches:
            # Validate with standard threshold
            validation_result = PatternValidation.validate_match_quality(
                auditor, match, self.similarity_threshold
            )

            if validation_result['valid']:
                # Check for moderate overlap
                match_indices = set()
                for occ_start in match['occurrences']:
                    for k in range(match['length']):
                        note_idx = occ_start + k
                        if note_idx < len(auditor.notes):
                            match_indices.add(note_idx)

                overlap = match_indices.intersection(occupied_indices)
                overlap_ratio = len(overlap) / len(match_indices) if match_indices else 0

                # Accept matches with up to 50% overlap (balanced)
                if overlap_ratio < 0.5:
                    merged_matches.append(match)
                    occupied_indices.update(match_indices)
                else:
                    # For overlapping matches, keep the higher quality one
                    existing_validation = None
                    for existing_match in merged_matches:
                        existing_match_indices = set()
                        for occ_start in existing_match['occurrences']:
                            for k in range(existing_match['length']):
                                note_idx = occ_start + k
                                if note_idx < len(auditor.notes):
                                    existing_match_indices.add(note_idx)

                        if overlap.intersection(existing_match_indices):
                            existing_validation = PatternValidation.validate_match_quality(
                                auditor, existing_match, self.similarity_threshold
                            )
                            break

                    if existing_validation and validation_result['similarity'] > existing_validation.get('similarity', 0):
                        # Replace with higher quality match
                        merged_matches = [m for m in merged_matches if not overlap.intersection(set().union(*[set(range(occ_start, occ_start + m['length'])) for occ_start in m['occurrences']]))]
                        merged_matches.append(match)
                        occupied_indices.update(match_indices)

        # Mark occupied indices from merged matches
        for match in merged_matches:
            for occ_start in match['occurrences']:
                for k in range(match['length']):
                    note_idx = occ_start + k
                    if note_idx < len(auditor.notes):
                        auditor.occupied_indices.add(note_idx)

        # Calculate validation statistics
        validation_stats = {
            'validated_matches': len(merged_matches),
            'rejected_matches': len(all_matches) - len(merged_matches),
            'average_similarity': 0.0,
            'similarity_distribution': [],
            'validation_enabled': True
        }

        # Calculate average similarity
        similarity_scores = []
        for match in merged_matches:
            validation_result = PatternValidation.validate_match_quality(
                auditor, match, self.similarity_threshold
            )
            if validation_result['valid']:
                similarity_scores.append(validation_result['similarity'])

        if similarity_scores:
            validation_stats['average_similarity'] = np.mean(similarity_scores)
            validation_stats['similarity_distribution'] = similarity_scores

        processing_time = time.time() - start_time
        total_notes = len(auditor.notes)
        occupied_notes = len(auditor.occupied_indices)
        coverage = occupied_notes / total_notes if total_notes > 0 else 0

        ProgressTracker.print_status(f"  ✅ Completed {self.name} in {processing_time:.2f}s", "success")
        ProgressTracker.print_status(f"    📈 Coverage: {coverage*100:.1f}% with {len(merged_matches)} matches", "success")

        return {
            'motif_matches': merged_matches,
            'coverage': coverage,
            'processing_time': processing_time,
            'total_notes': total_notes,
            'occupied_notes': occupied_notes,
            'validation': validation_stats,
            'similarity_threshold': self.similarity_threshold
        }

class QualityOptimizedHybridAlgorithm(PatternRecognitionAlgorithm):
    """Quality-optimized hybrid with strict validation for high similarity"""

    def __init__(self, similarity_threshold=0.85):
        super().__init__(
            "Quality-Optimized Hybrid",
            "Prioritizes validation quality (Suffix Automaton 50% + Rhythmic Patterns 25% + Transposition Invariant 25%)"
        )
        # Initialize component algorithms with strict validation
        self.suffix_automaton = SuffixAutomatonAlgorithm(min_occurrences=2, min_length=4, similarity_threshold=similarity_threshold)
        self.rhythmic_patterns = RhythmicPatternAlgorithm(rhythm_tolerance=0.1, similarity_threshold=similarity_threshold)
        self.transposition_invariant = TranspositionInvariantAlgorithm(similarity_threshold=similarity_threshold)

        # Weights: Suffix Automaton (50%), Rhythmic Patterns (25%), Transposition Invariant (25%)
        self.weights = {
            'suffix_automaton': 0.5,
            'rhythmic_patterns': 0.25,
            'transposition_invariant': 0.25
        }
        self.similarity_threshold = similarity_threshold

    def analyze(self, auditor: MIDIAuditor) -> Dict:
        start_time = time.time()
        ProgressTracker.print_status(f"  🔍 Analyzing with {self.name}...", "processing")

        # Reset occupied indices
        auditor.occupied_indices = set()

        all_matches = []
        validation_stats_list = []

        # Run all three algorithms
        for algo_name, algo in [('suffix_automaton', self.suffix_automaton),
                              ('rhythmic_patterns', self.rhythmic_patterns),
                              ('transposition_invariant', self.transposition_invariant)]:
            if self.weights[algo_name] > 0:
                temp_auditor = copy.copy(auditor)
                temp_auditor.occupied_indices = set()
                result = algo.analyze(temp_auditor)
                if 'motif_matches' in result:
                    all_matches.extend(result['motif_matches'])
                validation_stats_list.append((algo_name, result.get('validation', {'validation_enabled': False})))

        # Merge matches with strict quality approach
        merged_matches = []
        occupied_indices = set()

        for match in all_matches:
            # Validate with strict threshold
            validation_result = PatternValidation.validate_match_quality(
                auditor, match, min(0.9, self.similarity_threshold + 0.05)  # More strict
            )

            if validation_result['valid'] and validation_result['similarity'] >= 0.8:
                # Check for minimal overlap
                match_indices = set()
                for occ_start in match['occurrences']:
                    for k in range(match['length']):
                        note_idx = occ_start + k
                        if note_idx < len(auditor.notes):
                            match_indices.add(note_idx)

                overlap = match_indices.intersection(occupied_indices)
                overlap_ratio = len(overlap) / len(match_indices) if match_indices else 0

                # Accept matches with up to 30% overlap (strict)
                if overlap_ratio < 0.3:
                    merged_matches.append(match)
                    occupied_indices.update(match_indices)
                else:
                    # For overlapping matches, only keep if significantly better quality
                    existing_validation = None
                    for existing_match in merged_matches:
                        existing_match_indices = set()
                        for occ_start in existing_match['occurrences']:
                            for k in range(existing_match['length']):
                                note_idx = occ_start + k
                                if note_idx < len(auditor.notes):
                                    existing_match_indices.add(note_idx)

                        if overlap.intersection(existing_match_indices):
                            existing_validation = PatternValidation.validate_match_quality(
                                auditor, existing_match, self.similarity_threshold
                            )
                            break

                    if existing_validation and validation_result['similarity'] > existing_validation.get('similarity', 0) + 0.05:
                        # Replace only if significantly better (0.05+ improvement)
                        merged_matches = [m for m in merged_matches if not overlap.intersection(set().union(*[set(range(occ_start, occ_start + m['length'])) for occ_start in m['occurrences']]))]
                        merged_matches.append(match)
                        occupied_indices.update(match_indices)

        # Mark occupied indices from merged matches
        for match in merged_matches:
            for occ_start in match['occurrences']:
                for k in range(match['length']):
                    note_idx = occ_start + k
                    if note_idx < len(auditor.notes):
                        auditor.occupied_indices.add(note_idx)

        # Calculate validation statistics
        validation_stats = {
            'validated_matches': len(merged_matches),
            'rejected_matches': len(all_matches) - len(merged_matches),
            'average_similarity': 0.0,
            'similarity_distribution': [],
            'validation_enabled': True
        }

        # Calculate average similarity
        similarity_scores = []
        for match in merged_matches:
            validation_result = PatternValidation.validate_match_quality(
                auditor, match, self.similarity_threshold
            )
            if validation_result['valid']:
                similarity_scores.append(validation_result['similarity'])

        if similarity_scores:
            validation_stats['average_similarity'] = np.mean(similarity_scores)
            validation_stats['similarity_distribution'] = similarity_scores

        processing_time = time.time() - start_time
        total_notes = len(auditor.notes)
        occupied_notes = len(auditor.occupied_indices)
        coverage = occupied_notes / total_notes if total_notes > 0 else 0

        ProgressTracker.print_status(f"  ✅ Completed {self.name} in {processing_time:.2f}s", "success")
        ProgressTracker.print_status(f"    📈 Coverage: {coverage*100:.1f}% with {len(merged_matches)} high-quality matches", "success")

        return {
            'motif_matches': merged_matches,
            'coverage': coverage,
            'processing_time': processing_time,
            'total_notes': total_notes,
            'occupied_notes': occupied_notes,
            'validation': validation_stats,
            'similarity_threshold': self.similarity_threshold
        }

class UltraLongSectionMatcher(PatternRecognitionAlgorithm):
    """Maximum-length multilayer section matching for DAW copy/paste workflow"""

    def __init__(self, min_section_bars=2, max_section_bars=32, similarity_threshold=0.75):
        super().__init__(
            "Ultra Long Section Matcher",
            "Finds maximum-length near-perfect multilayer sections for DAW replacement"
        )
        self.min_section_bars = min_section_bars
        self.max_section_bars = max_section_bars
        self.similarity_threshold = similarity_threshold
        self.max_comparisons = 50000  # Limit computational complexity

    def analyze(self, auditor: MIDIAuditor) -> Dict:
        start_time = time.time()
        ProgressTracker.print_status(f"  🔍 Analyzing with {self.name}...", "processing")

        # Reset occupied indices
        auditor.occupied_indices = set()

        # Find ultra-long section matches
        large_matches = self._find_ultra_long_sections(auditor)

        # Convert to LargeMatch objects and mark occupied indices
        for lm in large_matches:
            # Mark notes as occupied for coverage calculation
            for bar_offset in range(lm.length_bars):
                for bar_idx in [lm.start_bar_a + bar_offset, lm.start_bar_b + bar_offset]:
                    start_tick = bar_idx * auditor.ticks_per_bar
                    end_tick = (bar_idx + 1) * auditor.ticks_per_bar

                    for idx, note in enumerate(auditor.notes):
                        if start_tick <= note["tick"] < end_tick:
                            auditor.occupied_indices.add(idx)

        processing_time = time.time() - start_time
        total_notes = len(auditor.notes)
        occupied_notes = len(auditor.occupied_indices)
        coverage = occupied_notes / total_notes if total_notes > 0 else 0

        ProgressTracker.print_status(f"  ✅ Completed {self.name} in {processing_time:.2f}s", "success")
        ProgressTracker.print_status(f"    📈 Coverage: {coverage*100:.1f}% with {len(large_matches)} ultra-long sections", "success")

        return {
            'large_matches': large_matches,
            'motif_matches': [],  # No motifs - focus on sections only
            'coverage': coverage,
            'processing_time': processing_time,
            'total_notes': total_notes,
            'occupied_notes': occupied_notes,
            'validation': {'validation_enabled': False},  # We use strict similarity threshold
            'similarity_threshold': self.similarity_threshold,
            'avg_match_length': np.mean([lm.length_bars for lm in large_matches]) if large_matches else 0
        }

    def _find_ultra_long_sections(self, auditor: MIDIAuditor) -> List[LargeMatch]:
        """Find maximum-length near-perfect section matches"""
        matches = []

        # Convert parameters to note indices
        min_length_notes = self.min_section_bars * (len(auditor.notes) // max(1, auditor.num_bars))
        max_length_notes = min(self.max_section_bars * (len(auditor.notes) // max(1, auditor.num_bars)),
                              len(auditor.notes) // 2)

        # Start with longest possible sections and work down
        section_lengths = range(max_length_notes, max(min_length_notes - 1, 4), -4)  # Step by 4 notes

        comparisons_made = 0

        for length_notes in section_lengths:
            if comparisons_made >= self.max_comparisons:
                break

            # Find exact matches of this length
            pattern_dict = defaultdict(list)

            # Sample every Nth position to reduce comparisons
            step_size = max(1, length_notes // 8)  # Adaptive stepping

            for i in range(0, len(auditor.notes) - length_notes + 1, step_size):
                if i in auditor.occupied_indices:
                    continue

                # Create multilayer signature (pitch + rhythm + velocity + channel)
                signature = self._create_multilayer_signature(auditor.notes[i:i + length_notes])
                pattern_dict[signature].append(i)

            # Find patterns that appear multiple times
            for signature, positions in pattern_dict.items():
                if len(positions) >= 2 and comparisons_made < self.max_comparisons:
                    # Verify similarity between all pairs
                    valid_pairs = []
                    for i in range(len(positions)):
                        for j in range(i + 1, len(positions)):
                            if comparisons_made >= self.max_comparisons:
                                break

                            sim = self._calculate_multilayer_similarity(
                                auditor.notes[positions[i]:positions[i] + length_notes],
                                auditor.notes[positions[j]:positions[j] + length_notes]
                            )
                            comparisons_made += 1

                            if sim >= self.similarity_threshold:
                                valid_pairs.append((positions[i], positions[j]))

                    # Create LargeMatch objects for valid pairs
                    for pos_a, pos_b in valid_pairs:
                        # Convert note indices to bar positions
                        bar_a = auditor.notes[pos_a]["tick"] // auditor.ticks_per_bar
                        bar_b = auditor.notes[pos_b]["tick"] // auditor.ticks_per_bar

                        # Estimate section length in bars
                        section_ticks = auditor.notes[pos_a + length_notes - 1]["tick"] + \
                                      auditor.notes[pos_a + length_notes - 1]["duration"] - \
                                      auditor.notes[pos_a]["tick"]
                        length_bars = max(1, section_ticks // auditor.ticks_per_bar)

                        matches.append(LargeMatch(
                            id=len(matches) + 1,
                            start_bar_a=bar_a,
                            start_bar_b=bar_b,
                            length_bars=length_bars,
                            avg_similarity=self.similarity_threshold  # We already verified it's above threshold
                        ))

        return matches

    def _create_multilayer_signature(self, notes: List[Dict]) -> Tuple:
        """Create a multilayer signature for flexible matching"""
        signature_parts = []

        for note in notes:
            # More flexible: pitch class, velocity range, channel, duration range
            pitch_class = note['pitch'] % 12  # Pitch class instead of exact pitch
            velocity_range = note['velocity'] // 16  # Velocity in ranges of 16
            duration_range = min(7, note['duration'] // 32)  # Duration in ranges
            signature_parts.extend([
                pitch_class,  # 0-11 (pitch class)
                velocity_range,  # 0-7 (velocity ranges)
                note['channel'],  # Exact channel
                duration_range  # 0-7 (duration ranges)
            ])

        return tuple(signature_parts)

    def _calculate_multilayer_similarity(self, notes_a: List[Dict], notes_b: List[Dict]) -> float:
        """Calculate comprehensive multilayer similarity"""
        if len(notes_a) != len(notes_b):
            return 0.0

        total_features = 0
        matching_features = 0

        for na, nb in zip(notes_a, notes_b):
            # Pitch match (exact)
            if na['pitch'] == nb['pitch']:
                matching_features += 1
            total_features += 1

            # Velocity match (within tolerance)
            vel_diff = abs(na['velocity'] - nb['velocity'])
            if vel_diff <= 10:  # 10 velocity units tolerance
                matching_features += 1
            total_features += 1

            # Channel match (exact)
            if na['channel'] == nb['channel']:
                matching_features += 1
            total_features += 1

            # Duration match (within tolerance)
            duration_diff = abs(na['duration'] - nb['duration'])
            max_duration = max(na['duration'], nb['duration'])
            if max_duration > 0 and duration_diff / max_duration <= 0.1:  # 10% tolerance
                matching_features += 1
            total_features += 1

            # Timing alignment (relative within section)
            # This is handled by the signature matching

        return matching_features / total_features if total_features > 0 else 0.0


class AutoTuningFramework:
    """Framework for testing and auto-tuning algorithms on real MIDI files"""

    def __init__(self):
        self.test_files = []
        self.metrics = {
            'coverage': [],
            'similarity': [],
            'match_lengths': [],
            'processing_times': []
        }

    def add_test_file(self, filepath: str):
        """Add a MIDI file to the test suite"""
        if os.path.exists(filepath):
            self.test_files.append(filepath)

    def run_parameter_sweep(self, algorithm_class, param_ranges: Dict, midi_file: str):
        """Run parameter sweep on a single file"""
        results = []

        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_ranges)

        for params in param_combinations:
            # Create algorithm instance with parameters
            algo = algorithm_class(**params)

            # Run analysis
            try:
                with open(midi_file, 'rb') as f:
                    midi_data = f.read()

                auditor = MIDIAuditor(io.BytesIO(midi_data), verbose=False)

                start_time = time.time()
                result = algo.analyze(auditor)
                processing_time = time.time() - start_time

                results.append({
                    'params': params,
                    'coverage': result['coverage'],
                    'processing_time': processing_time,
                    'large_matches': len(result.get('large_matches', [])),
                    'avg_similarity': result.get('similarity_threshold', 0.95),
                    'avg_match_length': result.get('avg_match_length', 0)
                })

            except Exception as e:
                print(f"Error testing params {params}: {e}")
                continue

        return results

    def _generate_param_combinations(self, param_ranges: Dict) -> List[Dict]:
        """Generate all combinations of parameters"""
        if not param_ranges:
            return [{}]

        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]

        combinations = []
        for combo in np.ndindex(*[len(vals) for vals in param_values]):
            param_dict = {}
            for i, name in enumerate(param_names):
                param_dict[name] = param_values[i][combo[i]]
            combinations.append(param_dict)

        return combinations

    def optimize_parameters(self, algorithm_class, param_ranges: Dict, target_metric='coverage'):
        """Find optimal parameters across all test files"""
        all_results = []

        for midi_file in self.test_files:
            print(f"Optimizing on {midi_file}...")
            file_results = self.run_parameter_sweep(algorithm_class, param_ranges, midi_file)
            all_results.extend(file_results)

        if not all_results:
            return {}

        # Find best parameters by target metric
        if target_metric == 'coverage':
            best_result = max(all_results, key=lambda x: x['coverage'])
        elif target_metric == 'efficiency':
            best_result = max(all_results, key=lambda x: x['coverage'] / max(x['processing_time'], 0.1))
        elif target_metric == 'length':
            best_result = max(all_results, key=lambda x: x['avg_match_length'])
        else:
            best_result = max(all_results, key=lambda x: x['coverage'])

        print(f"Optimal parameters: {best_result['params']}")
        print(f"Coverage: {best_result['coverage']:.3f}")
        print(f"Processing time: {best_result['processing_time']:.3f}s")

        return best_result['params']


class DynamicUltraHybridAlgorithm(PatternRecognitionAlgorithm):
    """Dynamic ultra hybrid with adaptive tiered validation"""

    def __init__(self, similarity_threshold=0.75):
        super().__init__(
            "Dynamic Ultra Hybrid",
            "Adaptive hybrid with tiered validation and smart overlap resolution"
        )
        # Initialize component algorithms
        self.suffix_automaton = SuffixAutomatonAlgorithm(min_occurrences=2, min_length=4, similarity_threshold=similarity_threshold)
        self.rhythmic_patterns = RhythmicPatternAlgorithm(rhythm_tolerance=0.2, similarity_threshold=similarity_threshold)
        self.transposition_invariant = TranspositionInvariantAlgorithm(similarity_threshold=similarity_threshold)

        # Base weights that will be adjusted dynamically
        self.base_weights = {
            'suffix_automaton': 0.6,
            'rhythmic_patterns': 0.3,
            'transposition_invariant': 0.1
        }
        self.similarity_threshold = similarity_threshold

    def _analyze_file_characteristics(self, auditor: MIDIAuditor) -> Dict:
        """Analyze file characteristics for dynamic weighting"""
        # Calculate rhythmic complexity
        durations = [note['duration'] for note in auditor.notes]
        if len(durations) > 1:
            duration_std = np.std(durations)
            duration_mean = np.mean(durations)
            rhythmic_complexity = duration_std / duration_mean if duration_mean > 0 else 0
        else:
            rhythmic_complexity = 0

        # Calculate pitch variability
        pitches = [note['pitch'] for note in auditor.notes]
        if len(pitches) > 1:
            pitch_range = max(pitches) - min(pitches)
            pitch_variability = pitch_range / 12.0
        else:
            pitch_variability = 0

        return {
            'rhythmic_complexity': rhythmic_complexity,
            'pitch_variability': pitch_variability,
            'total_notes': len(auditor.notes)
        }

    def _calculate_dynamic_weights(self, characteristics: Dict) -> Dict:
        """Calculate dynamic weights based on file characteristics"""
        weights = self.base_weights.copy()

        # Adjust based on rhythmic complexity
        if characteristics['rhythmic_complexity'] > 0.5:
            weights['rhythmic_patterns'] = min(0.4, weights['rhythmic_patterns'] + 0.1)
            weights['suffix_automaton'] = max(0.5, weights['suffix_automaton'] - 0.1)
        elif characteristics['rhythmic_complexity'] < 0.3:
            weights['rhythmic_patterns'] = max(0.2, weights['rhythmic_patterns'] - 0.1)
            weights['suffix_automaton'] = min(0.7, weights['suffix_automaton'] + 0.1)

        # Adjust based on pitch variability
        if characteristics['pitch_variability'] > 2.0:
            weights['transposition_invariant'] = min(0.2, weights['transposition_invariant'] + 0.1)
            weights['suffix_automaton'] = max(0.5, weights['suffix_automaton'] - 0.05)

        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] = weights[key] / total_weight

        return weights

    def _tiered_validation(self, match: Dict, auditor: MIDIAuditor, match_source: str) -> Dict:
        """Tiered validation based on match source and characteristics"""
        # Different thresholds for different algorithm sources
        if match_source == 'suffix_automaton':
            # Suffix automaton gets standard validation
            return PatternValidation.validate_match_quality(auditor, match, self.similarity_threshold)
        elif match_source == 'rhythmic_patterns':
            # Rhythmic patterns get slightly more lenient validation
            return PatternValidation.validate_match_quality(auditor, match, max(0.65, self.similarity_threshold - 0.1))
        else:
            # Transposition invariant gets slightly stricter validation
            return PatternValidation.validate_match_quality(auditor, match, min(0.85, self.similarity_threshold + 0.1))

    def analyze(self, auditor: MIDIAuditor) -> Dict:
        start_time = time.time()
        ProgressTracker.print_status(f"  🔍 Analyzing with {self.name}...", "processing")

        # Reset occupied indices
        auditor.occupied_indices = set()

        # Analyze file characteristics
        characteristics = self._analyze_file_characteristics(auditor)
        dynamic_weights = self._calculate_dynamic_weights(characteristics)

        ProgressTracker.print_status(f"    📊 Characteristics: rhythmic={characteristics['rhythmic_complexity']:.3f}, " +
                                   f"pitch={characteristics['pitch_variability']:.3f}", "info")
        ProgressTracker.print_status(f"    🎚️  Weights: SA={dynamic_weights['suffix_automaton']:.1f}, " +
                                   f"RP={dynamic_weights['rhythmic_patterns']:.1f}, " +
                                   f"TI={dynamic_weights['transposition_invariant']:.1f}", "info")

        all_matches = []
        validation_stats_list = []

        # Run algorithms with dynamic weights
        for algo_name, algo in [('suffix_automaton', self.suffix_automaton),
                              ('rhythmic_patterns', self.rhythmic_patterns),
                              ('transposition_invariant', self.transposition_invariant)]:
            if dynamic_weights[algo_name] > 0:
                temp_auditor = copy.copy(auditor)
                temp_auditor.occupied_indices = set()
                result = algo.analyze(temp_auditor)
                if 'motif_matches' in result:
                    # Add source information to each match
                    for match in result['motif_matches']:
                        match['source'] = algo_name
                    all_matches.extend(result['motif_matches'])
                validation_stats_list.append((algo_name, result.get('validation', {'validation_enabled': False})))

        # Smart merging with tiered validation
        merged_matches = []
        occupied_indices = set()

        for match in all_matches:
            # Use tiered validation based on match source
            validation_result = self._tiered_validation(match, auditor, match['source'])

            if validation_result['valid']:
                # Smart overlap resolution
                match_indices = set()
                for occ_start in match['occurrences']:
                    for k in range(match['length']):
                        note_idx = occ_start + k
                        if note_idx < len(auditor.notes):
                            match_indices.add(note_idx)

                overlap = match_indices.intersection(occupied_indices)
                overlap_ratio = len(overlap) / len(match_indices) if match_indices else 0

                # Dynamic overlap threshold based on validation quality
                quality_based_threshold = 0.6 - (validation_result['similarity'] * 0.3)  # Better quality allows more overlap

                if overlap_ratio < quality_based_threshold:
                    merged_matches.append(match)
                    occupied_indices.update(match_indices)
                else:
                    # Quality-based replacement logic
                    existing_validation = None
                    for existing_match in merged_matches:
                        existing_match_indices = set()
                        for occ_start in existing_match['occurrences']:
                            for k in range(existing_match['length']):
                                note_idx = occ_start + k
                                if note_idx < len(auditor.notes):
                                    existing_match_indices.add(note_idx)

                        if overlap.intersection(existing_match_indices):
                            existing_validation = self._tiered_validation(existing_match, auditor, existing_match.get('source', 'unknown'))
                            break

                    if existing_validation:
                        # Replace if new match has better quality or similar quality but from higher-priority source
                        priority_order = ['suffix_automaton', 'rhythmic_patterns', 'transposition_invariant']
                        new_priority = priority_order.index(match['source']) if match['source'] in priority_order else 99
                        existing_priority = priority_order.index(existing_match.get('source', 'unknown')) if existing_match.get('source', 'unknown') in priority_order else 99

                        if (validation_result['similarity'] > existing_validation.get('similarity', 0) or
                            (abs(validation_result['similarity'] - existing_validation.get('similarity', 0)) < 0.05 and
                             new_priority < existing_priority)):
                            # Replace existing match
                            merged_matches = [m for m in merged_matches if not overlap.intersection(set().union(*[set(range(occ_start, occ_start + m['length'])) for occ_start in m['occurrences']]))]
                            merged_matches.append(match)
                            occupied_indices.update(match_indices)

        # Mark occupied indices
        for match in merged_matches:
            for occ_start in match['occurrences']:
                for k in range(match['length']):
                    note_idx = occ_start + k
                    if note_idx < len(auditor.notes):
                        auditor.occupied_indices.add(note_idx)

        # Calculate validation statistics
        validation_stats = {
            'validated_matches': len(merged_matches),
            'rejected_matches': len(all_matches) - len(merged_matches),
            'average_similarity': 0.0,
            'similarity_distribution': [],
            'validation_enabled': True
        }

        # Calculate average similarity
        similarity_scores = []
        for match in merged_matches:
            validation_result = self._tiered_validation(match, auditor, match.get('source', 'unknown'))
            if validation_result['valid']:
                similarity_scores.append(validation_result['similarity'])

        if similarity_scores:
            validation_stats['average_similarity'] = np.mean(similarity_scores)
            validation_stats['similarity_distribution'] = similarity_scores

        processing_time = time.time() - start_time
        total_notes = len(auditor.notes)
        occupied_notes = len(auditor.occupied_indices)
        coverage = occupied_notes / total_notes if total_notes > 0 else 0

        ProgressTracker.print_status(f"  ✅ Completed {self.name} in {processing_time:.2f}s", "success")
        ProgressTracker.print_status(f"    📈 Coverage: {coverage*100:.1f}% with {len(merged_matches)} matches (avg qual: {validation_stats['average_similarity']:.3f})", "success")

        return {
            'motif_matches': merged_matches,
            'coverage': coverage,
            'processing_time': processing_time,
            'total_notes': total_notes,
            'occupied_notes': occupied_notes,
            'validation': validation_stats,
            'similarity_threshold': self.similarity_threshold,
            'dynamic_weights': dynamic_weights,
            'file_characteristics': characteristics
        }

class IntelligentHybridAlgorithm(PatternRecognitionAlgorithm):
    """Intelligent hybrid approach with dynamic weighting and similarity-based merging"""

    def __init__(self, similarity_threshold=0.75):
        super().__init__(
            "Intelligent Hybrid",
            "Combines Suffix Automaton, Rhythmic Patterns, and Transposition Invariant with dynamic weighting and similarity-based merging"
        )
        # Initialize component algorithms with validation enabled
        self.suffix_automaton = SuffixAutomatonAlgorithm(min_occurrences=2, min_length=4, similarity_threshold=similarity_threshold)
        self.rhythmic_patterns = RhythmicPatternAlgorithm(rhythm_tolerance=0.2, similarity_threshold=similarity_threshold)
        self.transposition_invariant = TranspositionInvariantAlgorithm(similarity_threshold=similarity_threshold)

        # Default weights: Suffix Automaton (60%), Rhythmic Patterns (30%), Transposition Invariant (10%)
        self.base_weights = {
            'suffix_automaton': 0.6,
            'rhythmic_patterns': 0.3,
            'transposition_invariant': 0.1
        }
        self.similarity_threshold = similarity_threshold

    def _analyze_file_characteristics(self, auditor: MIDIAuditor) -> Dict:
        """Analyze file characteristics to determine dynamic weights"""
        # Calculate rhythmic complexity (variation in note durations)
        durations = [note['duration'] for note in auditor.notes]
        if len(durations) > 1:
            duration_std = np.std(durations)
            duration_mean = np.mean(durations)
            rhythmic_complexity = duration_std / duration_mean if duration_mean > 0 else 0
        else:
            rhythmic_complexity = 0

        # Calculate pitch variability
        pitches = [note['pitch'] for note in auditor.notes]
        if len(pitches) > 1:
            pitch_range = max(pitches) - min(pitches)
            pitch_variability = pitch_range / 12.0  # Normalize by octave size
        else:
            pitch_variability = 0

        # Calculate note density
        note_density = len(auditor.notes) / (auditor.num_bars if auditor.num_bars > 0 else 1)

        return {
            'rhythmic_complexity': rhythmic_complexity,
            'pitch_variability': pitch_variability,
            'note_density': note_density,
            'total_notes': len(auditor.notes)
        }

    def _calculate_dynamic_weights(self, characteristics: Dict) -> Dict:
        """Calculate dynamic weights based on file characteristics"""
        weights = self.base_weights.copy()

        # Adjust weights based on rhythmic complexity
        if characteristics['rhythmic_complexity'] > 0.5:  # High rhythmic complexity
            weights['rhythmic_patterns'] = min(0.4, weights['rhythmic_patterns'] + 0.1)
            weights['suffix_automaton'] = max(0.5, weights['suffix_automaton'] - 0.1)
        elif characteristics['rhythmic_complexity'] < 0.3:  # Low rhythmic complexity
            weights['rhythmic_patterns'] = max(0.2, weights['rhythmic_patterns'] - 0.1)
            weights['suffix_automaton'] = min(0.7, weights['suffix_automaton'] + 0.1)

        # Adjust weights based on pitch variability
        if characteristics['pitch_variability'] > 2.0:  # High pitch variability
            weights['transposition_invariant'] = min(0.2, weights['transposition_invariant'] + 0.1)
            weights['suffix_automaton'] = max(0.5, weights['suffix_automaton'] - 0.05)
        elif characteristics['pitch_variability'] < 1.0:  # Low pitch variability
            weights['transposition_invariant'] = max(0.05, weights['transposition_invariant'] - 0.05)
            weights['suffix_automaton'] = min(0.65, weights['suffix_automaton'] + 0.05)

        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] = weights[key] / total_weight

        return weights

    def _merge_matches_with_similarity_validation(self, all_matches: List[Dict], auditor: MIDIAuditor) -> List[Dict]:
        """Merge matches from different algorithms with similarity-based validation"""
        merged_matches = []
        occupied_indices = set()

        # Group matches by their coverage
        for match in all_matches:
            # Validate each match before merging
            validation_result = PatternValidation.validate_match_quality(
                auditor, match, self.similarity_threshold
            )

            if validation_result['valid']:
                # Check for overlap with existing matches
                match_indices = set()
                for occ_start in match['occurrences']:
                    for k in range(match['length']):
                        note_idx = occ_start + k
                        if note_idx < len(auditor.notes):
                            match_indices.add(note_idx)

                # Calculate overlap with existing occupied indices
                overlap = match_indices.intersection(occupied_indices)
                overlap_ratio = len(overlap) / len(match_indices) if match_indices else 0

                # Only add match if it doesn't overlap too much with existing matches
                if overlap_ratio < 0.5:  # Less than 50% overlap
                    merged_matches.append(match)
                    occupied_indices.update(match_indices)
                else:
                    # Check if this match has higher similarity than the overlapping one
                    existing_match_similarity = 0
                    for existing_match in merged_matches:
                        existing_match_indices = set()
                        for occ_start in existing_match['occurrences']:
                            for k in range(existing_match['length']):
                                note_idx = occ_start + k
                                if note_idx < len(auditor.notes):
                                    existing_match_indices.add(note_idx)

                        if overlap.intersection(existing_match_indices):
                            # This is the overlapping match
                            existing_validation = PatternValidation.validate_match_quality(
                                auditor, existing_match, self.similarity_threshold
                            )
                            existing_match_similarity = existing_validation.get('similarity', 0)
                            break

                    current_match_similarity = validation_result.get('similarity', 0)
                    if current_match_similarity > existing_match_similarity:
                        # Replace the existing match with this higher-quality one
                        merged_matches = [m for m in merged_matches if not overlap.intersection(set().union(*[set(range(occ_start, occ_start + m['length'])) for occ_start in m['occurrences']]))]
                        merged_matches.append(match)
                        occupied_indices.update(match_indices)

        return merged_matches

    def analyze(self, auditor: MIDIAuditor) -> Dict:
        start_time = time.time()
        ProgressTracker.print_status(f"  🔍 Analyzing with {self.name}...", "processing")

        # Reset occupied indices
        auditor.occupied_indices = set()

        # Analyze file characteristics for dynamic weighting
        characteristics = self._analyze_file_characteristics(auditor)
        dynamic_weights = self._calculate_dynamic_weights(characteristics)

        ProgressTracker.print_status(f"    📊 File characteristics: rhythmic_complexity={characteristics['rhythmic_complexity']:.3f}, " +
                                   f"pitch_variability={characteristics['pitch_variability']:.3f}, " +
                                   f"note_density={characteristics['note_density']:.1f}", "info")
        ProgressTracker.print_status(f"    🎚️  Dynamic weights: SA={dynamic_weights['suffix_automaton']:.1f}, " +
                                   f"RP={dynamic_weights['rhythmic_patterns']:.1f}, " +
                                   f"TI={dynamic_weights['transposition_invariant']:.1f}", "info")

        # Run each component algorithm with validation enabled
        component_results = []
        all_matches = []

        # Suffix Automaton
        if dynamic_weights['suffix_automaton'] > 0:
            temp_auditor = copy.copy(auditor)
            temp_auditor.occupied_indices = set()
            sa_result = self.suffix_automaton.analyze(temp_auditor)
            component_results.append(('suffix_automaton', sa_result))
            if 'motif_matches' in sa_result:
                all_matches.extend(sa_result['motif_matches'])

        # Rhythmic Patterns
        if dynamic_weights['rhythmic_patterns'] > 0:
            temp_auditor = copy.copy(auditor)
            temp_auditor.occupied_indices = set()
            rp_result = self.rhythmic_patterns.analyze(temp_auditor)
            component_results.append(('rhythmic_patterns', rp_result))
            if 'motif_matches' in rp_result:
                all_matches.extend(rp_result['motif_matches'])

        # Transposition Invariant
        if dynamic_weights['transposition_invariant'] > 0:
            temp_auditor = copy.copy(auditor)
            temp_auditor.occupied_indices = set()
            ti_result = self.transposition_invariant.analyze(temp_auditor)
            component_results.append(('transposition_invariant', ti_result))
            if 'motif_matches' in ti_result:
                all_matches.extend(ti_result['motif_matches'])

        # Merge matches with similarity-based validation
        merged_matches = self._merge_matches_with_similarity_validation(all_matches, auditor)

        # Mark occupied indices from merged matches
        for match in merged_matches:
            for occ_start in match['occurrences']:
                for k in range(match['length']):
                    note_idx = occ_start + k
                    if note_idx < len(auditor.notes):
                        auditor.occupied_indices.add(note_idx)

        # Calculate validation statistics
        validation_stats = {
            'validated_matches': len(merged_matches),
            'rejected_matches': len(all_matches) - len(merged_matches),
            'average_similarity': 0.0,
            'similarity_distribution': [],
            'validation_enabled': True
        }

        # Calculate average similarity for validated matches
        similarity_scores = []
        for match in merged_matches:
            validation_result = PatternValidation.validate_match_quality(
                auditor, match, self.similarity_threshold
            )
            if validation_result['valid']:
                similarity_scores.append(validation_result['similarity'])

        if similarity_scores:
            validation_stats['average_similarity'] = np.mean(similarity_scores)
            validation_stats['similarity_distribution'] = similarity_scores

        processing_time = time.time() - start_time
        total_notes = len(auditor.notes)
        occupied_notes = len(auditor.occupied_indices)
        coverage = occupied_notes / total_notes if total_notes > 0 else 0

        ProgressTracker.print_status(f"  ✅ Completed {self.name} in {processing_time:.2f}s", "success")
        ProgressTracker.print_status(f"    📈 Final coverage: {coverage*100:.1f}% with {len(merged_matches)} validated matches", "success")

        return {
            'motif_matches': merged_matches,
            'coverage': coverage,
            'processing_time': processing_time,
            'total_notes': total_notes,
            'occupied_notes': occupied_notes,
            'validation': validation_stats,
            'similarity_threshold': self.similarity_threshold,
            'dynamic_weights': dynamic_weights,
            'file_characteristics': characteristics
        }

class HybridAlgorithm(PatternRecognitionAlgorithm):
    """Hybrid approach combining multiple techniques with validation - optimized"""

    def __init__(self, similarity_threshold=0.75):
        super().__init__(
            "Hybrid Algorithm",
            "Combines exact matching, transposition-invariant, and rhythmic detection with validation"
        )
        self.algorithms = [
            ExactMatchingAlgorithm(),
            TranspositionInvariantAlgorithm(similarity_threshold),
            RhythmicPatternAlgorithm(similarity_threshold=similarity_threshold)
        ]
        self.similarity_threshold = similarity_threshold

    def analyze(self, auditor: MIDIAuditor) -> Dict:
        start_time = time.time()
        ProgressTracker.print_status(f"  🔍 Analyzing with {self.name}...", "processing")

        # Reset occupied indices
        auditor.occupied_indices = set()

        all_matches = []
        total_large = 0
        total_motif = 0
        validation_stats_list = []

        # Run each algorithm and combine results
        for algo in self.algorithms:
            # Create a copy of auditor for this algorithm - use shallow copy where possible
            temp_auditor = copy.copy(auditor)
            temp_auditor.occupied_indices = set(auditor.occupied_indices.copy())

            result = algo.analyze(temp_auditor)

            # Collect validation stats
            if 'validation' in result:
                validation_stats_list.append(result['validation'])

            # Merge occupied indices only for validated matches
            if self.validation_enabled and 'validation' in result:
                # Only merge if validation was performed and matches were validated
                for idx in temp_auditor.occupied_indices:
                    auditor.occupied_indices.add(idx)
            elif not self.validation_enabled:
                # Merge all if validation disabled
                for idx in temp_auditor.occupied_indices:
                    auditor.occupied_indices.add(idx)

            # Collect matches
            if 'large_matches' in result:
                all_matches.extend(result['large_matches'])
                total_large += len(result['large_matches'])

            if 'motif_matches' in result:
                all_matches.extend(result['motif_matches'])
                total_motif += len(result['motif_matches'])

        # Calculate overall validation statistics
        if validation_stats_list and self.validation_enabled:
            total_validated = sum(vs['validated_matches'] for vs in validation_stats_list)
            total_rejected = sum(vs['rejected_matches'] for vs in validation_stats_list)
            avg_similarities = [vs['average_similarity'] for vs in validation_stats_list if 'average_similarity' in vs]
            overall_avg_similarity = np.mean(avg_similarities) if avg_similarities else 0

            validation_stats = {
                'validated_matches': total_validated,
                'rejected_matches': total_rejected,
                'average_similarity': overall_avg_similarity,
                'validation_enabled': True
            }
        else:
            validation_stats = {'validation_enabled': False}

        processing_time = time.time() - start_time
        total_notes = len(auditor.notes)
        occupied_notes = len(auditor.occupied_indices)
        coverage = occupied_notes / total_notes if total_notes > 0 else 0

        ProgressTracker.print_status(f"  ✅ Completed {self.name} in {processing_time:.2f}s", "success")

        return {
            'large_matches': total_large,
            'motif_matches': total_motif,
            'all_matches': all_matches,
            'coverage': coverage,
            'processing_time': processing_time,
            'total_notes': total_notes,
            'occupied_notes': occupied_notes,
            'validation': validation_stats,
            'similarity_threshold': self.similarity_threshold
        }

class MIDIAnalyzer:
    """Comprehensive MIDI analysis framework with similarity validation and performance optimization"""

    def __init__(self):
        # Initialize with performance-optimized algorithms
        self.algorithms = [
            ExactMatchingAlgorithm(),
            TranspositionInvariantAlgorithm(similarity_threshold=0.8),
            RhythmicPatternAlgorithm(rhythm_tolerance=0.2, similarity_threshold=0.8),
            MultiFeatureAlgorithm(pitch_weight=0.5, rhythm_weight=0.3, duration_weight=0.2, similarity_threshold=0.85),
            SuffixAutomatonAlgorithm(min_occurrences=2, min_length=4, similarity_threshold=0.75),
            HybridAlgorithm(similarity_threshold=0.8),
            IntelligentHybridAlgorithm(similarity_threshold=0.8),
            CoverageOptimizedHybridAlgorithm(similarity_threshold=0.7),
            BalancedHybridAlgorithm(similarity_threshold=0.75),
            QualityOptimizedHybridAlgorithm(similarity_threshold=0.85),
            DynamicUltraHybridAlgorithm(similarity_threshold=0.75),
            UltraLongSectionMatcher(min_section_bars=2, max_section_bars=32, similarity_threshold=0.75)  # New algorithm
        ]
        self.results_cache = {}
        PerformanceOptimizer.optimize_numpy_operations()

    def analyze_file(self, filename: str, algorithm_index: int = None, validation_enabled: bool = True) -> Dict:
        """Analyze a single MIDI file with specified algorithm or all algorithms"""
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return None

        with open(filename, 'rb') as f:
            midi_data = f.read()

        # Create auditor instance
        auditor = MIDIAuditor(io.BytesIO(midi_data), verbose=False)

        results = {
            'file': filename,
            'total_notes': len(auditor.notes),
            'num_bars': auditor.num_bars,
            'algorithms': []
        }

        # Set validation mode for all algorithms
        for algo in self.algorithms:
            algo.validation_enabled = validation_enabled

        # Analyze with all algorithms or specific one
        algorithms_to_test = [self.algorithms[algorithm_index]] if algorithm_index is not None else self.algorithms

        for algo in algorithms_to_test:
            algo_result = algo.analyze(auditor)

            # Reset auditor for next algorithm - optimized reset
            auditor.occupied_indices = set()

            results['algorithms'].append({
                'name': algo.name,
                'description': algo.description,
                'coverage': algo_result['coverage'],
                'processing_time': algo_result['processing_time'],
                'total_notes': algo_result['total_notes'],
                'occupied_notes': algo_result['occupied_notes'],
                'large_matches': algo_result.get('large_matches', 0),
                'motif_matches': algo_result.get('motif_matches', 0),
                'validation': algo_result.get('validation', {'validation_enabled': False}),
                'similarity_threshold': algo_result.get('similarity_threshold', 'N/A'),
                'details': algo_result
            })

        return results

    def comprehensive_analysis(self, files: List[str], validation_enabled: bool = True) -> Dict:
        """Run comprehensive analysis on multiple files with similarity validation and performance optimization"""
        print("🔬 Running Comprehensive MIDI Pattern Analysis with Similarity Validation (Optimized)")
        print("=" * 80)

        overall_results = {
            'files': [],
            'summary': {},
            'best_performers': {},
            'validation_summary': {}
        }

        total_files = len(files)
        for file_idx, filename in enumerate(files, 1):
            print(f"\n📁 [{file_idx}/{total_files}] Analyzing {filename}...")
            ProgressTracker.show_progress(file_idx, total_files, f"Processing {filename}")

            file_results = self.analyze_file(filename, validation_enabled=validation_enabled)

            if file_results:
                overall_results['files'].append(file_results)

                # Print results for this file
                print(f"  Total notes: {file_results['total_notes']}")
                print(f"  Number of bars: {file_results['num_bars']}")
                print(f"  Algorithm performance:")

                for algo_result in file_results['algorithms']:
                    coverage_pct = algo_result['coverage'] * 100
                    validation_info = algo_result['validation']
                    if validation_info.get('validation_enabled', False):
                        val_info = f" (val: {validation_info.get('average_similarity', 0):.3f})"
                    else:
                        val_info = ""
                    print(f"    {algo_result['name']:25}: {coverage_pct:5.1f}%{val_info}")

        # Generate summary statistics
        self._generate_summary(overall_results)

        # Generate validation summary
        self._generate_validation_summary(overall_results)

        return overall_results

    def _generate_summary(self, results: Dict):
        """Generate summary statistics and identify best performers"""
        # Find best algorithm for each file
        best_performers = {}

        for file_result in results['files']:
            filename = file_result['file']
            best_algo = max(
                file_result['algorithms'],
                key=lambda x: x['coverage']
            )
            best_performers[filename] = best_algo

            # Add best performer info to file result
            file_result['best_algorithm'] = best_algo['name']
            file_result['best_coverage'] = best_algo['coverage']

        results['best_performers'] = best_performers

        # Calculate overall statistics
        total_files = len(results['files'])
        avg_coverage = np.mean([f['best_coverage'] for f in results['files']])
        max_coverage = max([f['best_coverage'] for f in results['files']])
        min_coverage = min([f['best_coverage'] for f in results['files']])

        results['summary'] = {
            'total_files': total_files,
            'average_coverage': avg_coverage,
            'max_coverage': max_coverage,
            'min_coverage': min_coverage,
            'target_achieved': avg_coverage >= 0.9
        }

    def _generate_validation_summary(self, results: Dict):
        """Generate validation statistics across all algorithms and files"""
        all_similarities = []
        validation_stats_by_algo = defaultdict(list)

        for file_result in results['files']:
            for algo_result in file_result['algorithms']:
                algo_name = algo_result['name']
                validation = algo_result['validation']

                if validation.get('validation_enabled', False) and 'average_similarity' in validation:
                    avg_sim = validation['average_similarity']
                    all_similarities.append(avg_sim)
                    validation_stats_by_algo[algo_name].append(avg_sim)

        # Calculate overall validation statistics
        if all_similarities:
            overall_avg_similarity = np.mean(all_similarities)
            overall_min_similarity = np.min(all_similarities)
            overall_max_similarity = np.max(all_similarities)
        else:
            overall_avg_similarity = 0.0
            overall_min_similarity = 0.0
            overall_max_similarity = 0.0

        # Calculate per-algorithm validation statistics
        algo_validation_stats = {}
        for algo_name, similarities in validation_stats_by_algo.items():
            algo_validation_stats[algo_name] = {
                'average_similarity': np.mean(similarities),
                'min_similarity': np.min(similarities),
                'max_similarity': np.max(similarities),
                'count': len(similarities)
            }

        results['validation_summary'] = {
            'overall_average_similarity': overall_avg_similarity,
            'overall_min_similarity': overall_min_similarity,
            'overall_max_similarity': overall_max_similarity,
            'total_validated_matches': len(all_similarities),
            'algorithms': algo_validation_stats
        }

    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive report of the analysis with similarity validation"""
        report_lines = []

        report_lines.append("🎼 COMPREHENSIVE MIDI PATTERN ANALYSIS REPORT WITH SIMILARITY VALIDATION (OPTIMIZED)")
        report_lines.append("=" * 85)
        report_lines.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Summary section
        report_lines.append("📊 SUMMARY")
        report_lines.append("-" * 45)
        summary = results.get('summary', {})
        report_lines.append(f"Files analyzed: {summary.get('total_files', 0)}")
        report_lines.append(f"Average coverage: {summary.get('average_coverage', 0) * 100:.1f}%")
        report_lines.append(f"Maximum coverage: {summary.get('max_coverage', 0) * 100:.1f}%")
        report_lines.append(f"Minimum coverage: {summary.get('min_coverage', 0) * 100:.1f}%")
        report_lines.append(f"Target achieved (90%+): {'YES' if summary.get('target_achieved', False) else 'NO'}")
        report_lines.append("")

        # Validation summary
        if 'validation_summary' in results:
            val_summary = results['validation_summary']
            report_lines.append("🔍 SIMILARITY VALIDATION SUMMARY")
            report_lines.append("-" * 45)
            report_lines.append(f"Overall average similarity: {val_summary.get('overall_average_similarity', 0):.3f}")
            report_lines.append(f"Similarity range: {val_summary.get('overall_min_similarity', 0):.3f} - {val_summary.get('overall_max_similarity', 0):.3f}")
            report_lines.append(f"Total validated matches: {val_summary.get('total_validated_matches', 0)}")
            report_lines.append("")

            report_lines.append("Algorithm Validation Performance:")
            for algo_name, algo_stats in val_summary.get('algorithms', {}).items():
                report_lines.append(f"  {algo_name:25}: avg={algo_stats['average_similarity']:.3f}, "
                                  f"min={algo_stats['min_similarity']:.3f}, "
                                  f"max={algo_stats['max_similarity']:.3f} ({algo_stats['count']} files)")
            report_lines.append("")

        # Best performers
        if 'best_performers' in results:
            report_lines.append("🏆 BEST PERFORMERS BY FILE")
            report_lines.append("-" * 45)
            for filename, algo_result in results['best_performers'].items():
                report_lines.append(f"{filename}:")
                report_lines.append(f"  Best algorithm: {algo_result['name']}")
                report_lines.append(f"  Coverage: {algo_result['coverage'] * 100:.1f}%")
                report_lines.append(f"  Processing time: {algo_result['processing_time']:.3f}s")
                if 'validation' in algo_result and algo_result['validation'].get('validation_enabled', False):
                    report_lines.append(f"  Average similarity: {algo_result['validation'].get('average_similarity', 0):.3f}")
                report_lines.append("")

        # Detailed results
        report_lines.append("📋 DETAILED RESULTS WITH VALIDATION")
        report_lines.append("-" * 45)

        for file_result in results.get('files', []):
            report_lines.append(f"File: {file_result['file']}")
            report_lines.append(f"  Total notes: {file_result['total_notes']}")
            report_lines.append(f"  Number of bars: {file_result['num_bars']}")
            report_lines.append(f"  Best coverage: {file_result['best_coverage'] * 100:.1f}%")
            report_lines.append("")

            report_lines.append("  Algorithm Performance:")
            for algo_result in file_result['algorithms']:
                coverage_pct = algo_result['coverage'] * 100
                validation_info = algo_result['validation']

                if validation_info.get('validation_enabled', False):
                    val_info = f" | val: {validation_info.get('average_similarity', 0):.3f} avg, " \
                             f"{validation_info.get('validated_matches', 0)} valid, " \
                             f"{validation_info.get('rejected_matches', 0)} rej"
                else:
                    val_info = " | validation: disabled"

                report_lines.append(f"    {algo_result['name']:25}: {coverage_pct:5.1f}%{val_info}")

            report_lines.append("")

        return "\n".join(report_lines)

def main():
    """Main analysis function with similarity validation and performance optimization"""
    print("🎼 COMPREHENSIVE MIDI PATTERN ANALYSIS WITH SIMILARITY VALIDATION (OPTIMIZED)")
    print("=" * 85)

    # Initialize analyzer with performance optimizations
    analyzer = MIDIAnalyzer()

    # Define MIDI files to analyze
    files = ['toto-africa.mid', 'never.mid', 'queen.mid']

    # Check which files exist
    existing_files = [f for f in files if os.path.exists(f)]
    if not existing_files:
        print("❌ No MIDI files found!")
        return

    print(f"Found {len(existing_files)} MIDI files to analyze:")
    for f in existing_files:
        print(f"  - {f}")

    # Run comprehensive analysis with validation enabled and performance optimization
    print("\n🔬 Running comprehensive analysis with similarity validation (optimized)...")
    results = analyzer.comprehensive_analysis(existing_files, validation_enabled=True)

    # Generate and display report
    report = analyzer.generate_report(results)
    print(report)

    # Check if target is achieved
    if results['summary']['target_achieved']:
        print("✅ SUCCESS: Target coverage of 90%+ achieved with similarity validation!")
    else:
        print("⚠️  Target not yet achieved. Running optimization with validation...")

        # Try to find optimal approach with validation
        optimal_results = analyzer.find_optimal_approach(existing_files, validation_enabled=True)
        optimal_report = analyzer.generate_report(optimal_results)
        print(optimal_report)

        if optimal_results['summary']['average_coverage'] >= 0.9:
            print("✅ SUCCESS: Optimal approach achieved target coverage with validation!")
        else:
            print("⚠️  Further optimization needed.")
            print(f"Current best: {optimal_results['summary']['max_coverage'] * 100:.1f}% coverage")
            print("Consider implementing additional algorithms or tuning parameters.")

if __name__ == "__main__":
    main()
