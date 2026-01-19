import mido
import io
from mido import MidiFile, MidiTrack, Message, MetaMessage
from typing import List, Dict, Tuple

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

    @staticmethod
    def create_time_signature_test(ticks_per_beat=480):
        """
        Scenario: Waltz pattern in 3/4 time signature.
        Fix Target: BUG-2 (Time Signature Awareness).
        """
        mid = MidiFile(ticks_per_beat=ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)

        # Set time signature to 3/4 (Waltz)
        track.append(MetaMessage('time_signature', numerator=3, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8))

        # Pattern: C-E-G repeated in 3/4 time
        pattern = [60, 64, 67]
        for bar in range(4):
            for i, pitch in enumerate(pattern):
                tick = bar * (ticks_per_beat * 3) + i * ticks_per_beat
                track.append(Message('note_on', note=pitch, velocity=80, time=ticks_per_beat if (bar > 0 or i > 0) else 0))
                track.append(Message('note_off', note=pitch, velocity=0, time=ticks_per_beat // 2))

        return mid

    @staticmethod
    def create_channel_mismatch_test(ticks_per_beat=480):
        """
        Scenario: Identical notes on different MIDI channels.
        Fix Target: MISSING-1 (Interchangeability Validation).
        """
        mid = MidiFile(ticks_per_beat=ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)

        # Pattern A: Channel 0 (Piano)
        for i, pitch in enumerate([60, 64, 67, 72]):
            tick = i * ticks_per_beat
            track.append(Message('note_on', note=pitch, velocity=80, time=ticks_per_beat if i > 0 else 0, channel=0))
            track.append(Message('note_off', note=pitch, velocity=0, time=ticks_per_beat // 2, channel=0))

        # Pattern B: Channel 1 (Strings) - should not match
        for i, pitch in enumerate([60, 64, 67, 72]):
            tick = (i + 4) * ticks_per_beat
            track.append(Message('note_on', note=pitch, velocity=80, time=ticks_per_beat if i > 0 else 0, channel=1))
            track.append(Message('note_off', note=pitch, velocity=0, time=ticks_per_beat // 2, channel=1))

        return mid

    @staticmethod
    def create_performance_quality_test(ticks_per_beat=480):
        """
        Scenario: Three versions of the same phrase with different timing quality.
        Fix Target: FEATURE-1 (Performance Quality Scoring).
        """
        mid = MidiFile(ticks_per_beat=ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)

        # Quantized version (perfect timing)
        for i, pitch in enumerate([60, 64, 67, 72]):
            tick = i * ticks_per_beat
            track.append(Message('note_on', note=pitch, velocity=80, time=ticks_per_beat if i > 0 else 0))
            track.append(Message('note_off', note=pitch, velocity=0, time=ticks_per_beat // 2))

        # 20ms jitter version
        for i, pitch in enumerate([60, 64, 67, 72]):
            tick = (i + 4) * ticks_per_beat + (i * 10)  # Add small jitter
            track.append(Message('note_on', note=pitch, velocity=80, time=ticks_per_beat if i > 0 else 0))
            track.append(Message('note_off', note=pitch, velocity=0, time=ticks_per_beat // 2))

        # 50ms jitter version (worse timing)
        for i, pitch in enumerate([60, 64, 67, 72]):
            tick = (i + 8) * ticks_per_beat + (i * 25)  # Add larger jitter
            track.append(Message('note_on', note=pitch, velocity=80, time=ticks_per_beat if i > 0 else 0))
            track.append(Message('note_off', note=pitch, velocity=0, time=ticks_per_beat // 2))

        return mid

    @staticmethod
    def create_boundary_alignment_test(ticks_per_beat=480):
        """
        Scenario: Pattern with pickup notes (1/8th note before bar).
        Fix Target: MISSING-2 (Boundary Alignment).
        """
        mid = MidiFile(ticks_per_beat=ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)

        # Pattern with pickup note (starts on 1/8th note before bar)
        pickup_note = 55  # A note before the main pattern
        pattern = [60, 64, 67, 72]

        # First occurrence
        track.append(Message('note_on', note=pickup_note, velocity=80, time=0))
        track.append(Message('note_off', note=pickup_note, velocity=0, time=ticks_per_beat // 2))

        for i, pitch in enumerate(pattern):
            tick = (i + 1) * ticks_per_beat
            track.append(Message('note_on', note=pitch, velocity=80, time=ticks_per_beat if i > 0 else 0))
            track.append(Message('note_off', note=pitch, velocity=0, time=ticks_per_beat // 2))

        # Second occurrence (repeated)
        track.append(Message('note_on', note=pickup_note, velocity=80, time=ticks_per_beat * 4))
        track.append(Message('note_off', note=pickup_note, velocity=0, time=ticks_per_beat // 2))

        for i, pitch in enumerate(pattern):
            tick = (i + 5) * ticks_per_beat
            track.append(Message('note_on', note=pitch, velocity=80, time=ticks_per_beat if i > 0 else 0))
            track.append(Message('note_off', note=pitch, velocity=0, time=ticks_per_beat // 2))

        return mid

    @staticmethod
    def create_full_song_test(ticks_per_beat=480):
        """
        Scenario: Complete song with verses, choruses, and repeating motifs.
        Tests large-scale pattern detection in realistic musical context.
        Song Structure: Intro - V1 - C1 - V2 - C2 - Bridge - C3 - Outro
        """
        mid = MidiFile(ticks_per_beat=ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)

        # Set tempo and time signature
        track.append(MetaMessage('set_tempo', tempo=500000))  # 120 BPM
        track.append(MetaMessage('time_signature', numerator=4, denominator=4))

        # Define musical elements
        verse_melody = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale
        chorus_melody = [64, 66, 68, 69, 71, 73, 74, 76]  # D major scale
        bridge_melody = [55, 57, 59, 60, 62, 64, 66, 67]  # A major scale

        harmony_progression = [
            [48, 52, 55],  # C major
            [50, 53, 57],  # D minor
            [52, 55, 59],  # E minor
            [48, 52, 55],  # C major
        ]

        current_tick = 0
        bar_duration = ticks_per_beat * 4  # 4 beats per bar

        def add_chord(bass_note, chord_notes, start_tick, duration, velocity=60):
            """Add chord with bass note"""
            # Bass note
            track.append(Message('note_on', note=bass_note, velocity=velocity, time=start_tick if start_tick > 0 else 0))
            track.append(Message('note_off', note=bass_note, velocity=0, time=duration))

            # Chord notes
            for i, note in enumerate(chord_notes):
                delta = ticks_per_beat // 8 if i > 0 else 0  # Slight arpeggio
                track.append(Message('note_on', note=note, velocity=velocity-10, time=delta))
                track.append(Message('note_off', note=note, velocity=0, time=duration - delta))

        def add_melody_line(melody, start_tick, bars=4, velocity=80):
            """Add melody line"""
            for bar in range(bars):
                for beat in range(4):  # 4 beats per bar
                    note_idx = (bar * 4 + beat) % len(melody)
                    tick = start_tick + bar * bar_duration + beat * ticks_per_beat
                    duration = ticks_per_beat // 2  # Eighth notes

                    track.append(Message('note_on', note=melody[note_idx], velocity=velocity,
                                       time=tick if tick > 0 else 0))
                    track.append(Message('note_off', note=melody[note_idx], velocity=0, time=duration))

        def add_section(melody, chords, start_tick, bars=4, variation=0):
            """Add complete section with melody and harmony"""
            # Add harmony
            for bar in range(bars):
                chord_idx = bar % len(chords)
                chord_start = start_tick + bar * bar_duration
                add_chord(chords[chord_idx][0], chords[chord_idx], chord_start, bar_duration)

            # Add melody with slight variation
            velocity = 75 + variation * 5  # Slight dynamics variation
            add_melody_line(melody, start_tick, bars, velocity)

        # Song structure
        sections = [
            ("Intro", bridge_melody, harmony_progression, 2, 0),
            ("Verse 1", verse_melody, harmony_progression, 4, 0),
            ("Chorus 1", chorus_melody, harmony_progression, 4, 1),
            ("Verse 2", verse_melody, harmony_progression, 4, 2),
            ("Chorus 2", chorus_melody, harmony_progression, 4, 3),
            ("Bridge", bridge_melody, harmony_progression, 3, 4),
            ("Chorus 3", chorus_melody, harmony_progression, 4, 5),
            ("Outro", verse_melody, harmony_progression, 2, 6),
        ]

        for section_name, melody, chords, bars, variation in sections:
            add_section(melody, chords, current_tick, bars, variation)
            current_tick += bars * bar_duration

            # Add small gap between sections
            current_tick += ticks_per_beat // 2

        # Add final end of track
        track.append(MetaMessage('end_of_track'))

        return mid

    @staticmethod
    def save_midi_file(mid: MidiFile, filename: str):
        """Save MIDI file to disk for manual inspection."""
        mid.save(filename)
        print(f"Saved test MIDI file: {filename}")

    @staticmethod
    def get_midi_bytes(mid: MidiFile) -> bytes:
        """Get MIDI file as bytes for direct testing."""
        buffer = io.BytesIO()
        mid.save(file=buffer)
        buffer.seek(0)
        return buffer.getvalue()
