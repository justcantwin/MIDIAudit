import io
import math
from typing import List, Dict, Tuple, Set, Optional
import os
import hashlib
import platform
from collections import defaultdict

import mido
import numpy as np
import pretty_midi
import soundfile as sf

from models import Match, LargeMatch, TimeSignature
from suffix_automaton import SuffixAutomaton


class MIDIAuditor:
    def __init__(
        self,
        file_stream: io.BytesIO,
        quantize_beats: float = 0.25,
        large_similarity: float = 0.90,
        motif_similarity: float = 0.70,
        verbose: bool = True,  # Optional logging flag
    ):
        self.verbose = verbose
        self.logs: List[str] = []

        self.mid = mido.MidiFile(file=file_stream)
        self.ticks_per_beat = self.mid.ticks_per_beat
        self.large_similarity = large_similarity
        self.motif_similarity = motif_similarity
        self.quantize_ticks = int(self.ticks_per_beat * quantize_beats)

        # Caching
        self._cache = {}

        # Tempo and time signatures
        self.tempos = self._extract_tempos()
        self.time_signatures = self._extract_time_signatures()
        self._tempo = self.tempos[0] if self.tempos else 500000

        self.notes = self._extract_notes()
        self.pitch_array = np.array([n["pitch"] for n in self.notes], dtype=np.int16)
        self.occupied_indices: Set[int] = set()

        self._build_suffix_automaton()
        self.ticks_per_bar = self._compute_average_ticks_per_bar()
        self.bar_features, self.num_bars = self._build_bar_features()

    def _log(self, msg: str):
        if self.verbose:
            self.logs.append(msg)

    def _extract_tempos(self) -> List[int]:
        tempos = [msg.tempo for track in self.mid.tracks for msg in track if msg.type == "set_tempo"]
        return tempos or [500000]

    def _extract_time_signatures(self) -> List[TimeSignature]:
        signatures = []
        for track in self.mid.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if msg.type == "time_signature":
                    ts = TimeSignature(
                        numerator=msg.numerator,
                        denominator=msg.denominator,
                        ticks_per_bar=int(self.ticks_per_beat * msg.numerator * 4 / msg.denominator)
                    )
                    signatures.append((abs_tick, ts))
        if not signatures:
            signatures = [(0, TimeSignature(4, 4, self.ticks_per_beat * 4))]
        return signatures

    def _compute_average_ticks_per_bar(self) -> int:
        total_ticks = sum(ts.ticks_per_bar for _, ts in self.time_signatures)
        return total_ticks // len(self.time_signatures) if self.time_signatures else self.ticks_per_beat * 4

    def _extract_notes(self) -> List[Dict]:
        all_notes = []
        for i, track in enumerate(self.mid.tracks):
            abs_tick = 0
            active_notes = {}
            count_note_on = 0
            count_note_off = 0
            for msg in track:
                abs_tick += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    note = {
                        "pitch": msg.note,
                        "tick": abs_tick,
                        "end_tick": abs_tick + self.ticks_per_beat,
                        "velocity": msg.velocity,
                        "duration": self.ticks_per_beat
                    }
                    all_notes.append(note)
                    active_notes[(msg.channel, msg.note)] = note
                    count_note_on += 1
                elif msg.type in ("note_off", "note_on") and msg.velocity == 0:
                    key = (msg.channel, msg.note)
                    if key in active_notes:
                        note = active_notes[key]
                        note["end_tick"] = abs_tick
                        note["duration"] = abs_tick - note["tick"]
                        del active_notes[key]
                        count_note_off += 1
            self._log(f"Track {i}: {len(track)} msgs, {count_note_on} note_ons, {count_note_off} matched note_offs, {len(active_notes)} unmatched active")

        all_notes.sort(key=lambda n: n["tick"])
        self._log(f"Extracted {len(all_notes)} notes from {len(self.mid.tracks)} tracks")
        return all_notes

    def _build_suffix_automaton(self):
        cache_key = hashlib.md5(self.pitch_array.tobytes()).hexdigest()
        if cache_key in self._cache:
            self.sam = self._cache[cache_key]
            self._log("Loaded suffix automaton from cache")
            return

        self.sam = SuffixAutomaton()
        for p in self.pitch_array:
            self.sam.extend(int(p))
        self.sam.finalize_occurrences()

        self._cache[cache_key] = self.sam
        self._log("Built suffix automaton")

        best_len, _ = self.sam.longest_repeated_substring(min_occ=2)
        if best_len > 0:
            self._log(f"Suffix automaton: longest repeat = {best_len} notes")

    def _build_bar_features(self) -> Tuple[np.ndarray, int]:
        if not self.notes:
            return np.zeros((0, 12), dtype=np.float32), 0

        bar_ticks = self.ticks_per_bar
        bar_indices = [n["tick"] // bar_ticks for n in self.notes]
        num_bars = int(max(bar_indices)) + 1 if bar_indices else 0

        chroma = np.zeros((num_bars, 12), dtype=np.float32)
        rhythm = np.zeros((num_bars, 4), dtype=np.float32)

        for note, bar in zip(self.notes, bar_indices):
            pitch_class = note["pitch"] % 12
            weight = note["velocity"] / 127.0 * math.sqrt(note["duration"])
            chroma[bar, pitch_class] += weight
            pos = (note["tick"] % bar_ticks) / bar_ticks
            quarter = int(pos * 4)
            if quarter < 4:
                rhythm[bar, quarter] += weight

        features = np.hstack([chroma, rhythm])
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0

        self._log(f"Built enhanced features for {num_bars} bars")
        return features / norms, num_bars

    def _find_exact_repeats_in_range(
        self,
        min_len: int,
        max_len: int,
        max_matches: int,
        start_id: int,
    ) -> List[Match]:
        """
        Optimized brute-force with early termination and chunking for large n.
        """
        n = len(self.pitch_array)
        results = []
        current_id = start_id

        # Chunk processing for large n
        chunk_size = 10000
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            sub_array = self.pitch_array[start:end]

            for length in range(max_len, min_len - 1, -1):
                if len(results) >= max_matches:
                    break

                pattern_dict = defaultdict(list)

                for i in range(len(sub_array) - length + 1):
                    if (start + i) in self.occupied_indices:
                        continue
                    pattern = tuple(sub_array[i:i + length])
                    pattern_dict[pattern].append(start + i)

                for pattern, indices in pattern_dict.items():
                    if len(indices) >= 2:
                        for idx in indices:
                            for k in range(length):
                                self.occupied_indices.add(idx + k)

                        bars = [
                            int(self.notes[idx]["tick"] / self.ticks_per_bar) + 1
                            for idx in indices
                        ]

                        results.append(Match(
                            id=current_id,
                            length=length,
                            occurrences=sorted(indices),
                            similarity=1.0,
                            bars=bars,
                        ))
                        current_id += 1

                        if len(results) >= max_matches:
                            return results

        return results

    def label_sections(self, large_matches: List[LargeMatch]):
        """
        Match-centric sections: for each LargeMatch, create two occurrences (A and B).
        Returns a list of dicts with:
            - match_id
            - occurrence ("A" or "B")
            - start_bar, end_bar
            - start_tick, end_tick
            - similarity
        """
        sections = []

        for i, lm in enumerate(large_matches, start=1):
            length = lm.length_bars

            # A occurrence
            start_a = lm.start_bar_a
            end_a = start_a + length - 1
            sections.append({
                "match_id": i,
                "occurrence": "A",
                "start_bar": start_a,
                "end_bar": end_a,
                "start_tick": start_a * self.ticks_per_bar,
                "end_tick": (end_a + 1) * self.ticks_per_bar,
                "similarity": lm.avg_similarity,
            })

            # B occurrence
            start_b = lm.start_bar_b
            end_b = start_b + length - 1
            sections.append({
                "match_id": i,
                "occurrence": "B",
                "start_bar": start_b,
                "end_bar": end_b,
                "start_tick": start_b * self.ticks_per_bar,
                "end_tick": (end_b + 1) * self.ticks_per_bar,
                "similarity": lm.avg_similarity,
            })

        sections.sort(key=lambda s: s["start_bar"])
        return sections

    def visualize_timeline(self, sections, total_bars):
        """
        Multi-row timeline showing overlapping large-scale repeats.
        Each match gets its own row; A/B occurrences are marked.
        """
        if not sections:
            return "No timeline available."

        match_ids = sorted(set(s["match_id"] for s in sections))
        rows = [[" "] * total_bars for _ in match_ids]

        for s in sections:
            row_idx = match_ids.index(s["match_id"])
            label = s["occurrence"]  # "A" or "B"
            for b in range(s["start_bar"], s["end_bar"] + 1):
                if 0 <= b < total_bars:
                    rows[row_idx][b] = label

        out = ["Timeline (overlapping repeats):"]
        for i, row in enumerate(rows, start=1):
            out.append(f"Match {i}: " + "".join(row))

        return "\n".join(out)

    def summarize_structure(self, sections):
        """
        Match-centric summary.
        Returns:
            - summary_text (multi-line string)
            - summary_lines (list of lines)
        """
        if not sections:
            return "No large-scale structure detected.", []

        from collections import defaultdict
        groups = defaultdict(list)
        for s in sections:
            groups[s["match_id"]].append(s)

        lines = []
        summary_list = []

        lines.append("Large-Scale Repeats\n")

        for match_id in sorted(groups.keys()):
            group = groups[match_id]
            # assume both occurrences have same length and similarity
            length = group[0]["end_bar"] - group[0]["start_bar"] + 1
            sim = group[0]["similarity"]

            lines.append(f"Match {match_id}: {length} bars (sim={sim:.2f})")

            for s in sorted(group, key=lambda x: x["occurrence"]):
                line = (
                    f"  {s['occurrence']}{match_id}: "
                    f"Bars {s['start_bar']}–{s['end_bar']} "
                    f"(Ticks {s['start_tick']}–{s['end_tick']})"
                )
                lines.append(line)
                summary_list.append(line)

            lines.append("")

        return "\n".join(lines), summary_list

    def plot_self_similarity(self):
        """
        Returns a Plotly figure showing the bar-level self-similarity matrix.
        Uses cosine similarity of bar_features.
        """

        if self.num_bars == 0:
            return go.Figure()

        S = self.bar_features @ self.bar_features.T

        fig = go.Figure(
            data=go.Heatmap(
                z=S,
                colorscale="Viridis",
                zmin=0,
                zmax=1,
                colorbar=dict(title="Similarity")
            )
        )

        fig.update_layout(
            title="Self-Similarity Matrix (Bars)",
            xaxis_title="Bar Index",
            yaxis_title="Bar Index",
            width=600,
            height=600
        )

        return fig

    def find_all_patterns(
        self,
        min_motif_length: int = 4,
        min_large_bars: int = 4,
        max_results: int = 100,
        allow_overlapping_repeats: bool = False,
    ) -> Tuple[List[LargeMatch], List[Match]]:
        """
        Complete hierarchical decomposition:
        1. Find large-scale bar repeats (sorted by quality)
        2. Find long motifs in gaps
        3. Find medium motifs in remaining gaps
        4. Find short motifs in final gaps
        """

        # Phase 1: Large-scale repeats
        large_matches = self._find_large_scale_repeats(
            min_bars=min_large_bars,
            max_matches=max_results // 2,
            allow_overlapping_repeats=allow_overlapping_repeats
        )

        # Phase 2: Hierarchical motif detection (skip if overlaps allowed)
        if allow_overlapping_repeats:
            motif_matches = []
            self.logs.append("Skipping motif detection (overlaps enabled)")
        else:
            motif_matches = self._find_hierarchical_motifs(
                min_length=min_motif_length,
                max_matches=max_results // 2
            )

        total_mapped = len(self.occupied_indices)
        n = len(self.pitch_array)
        coverage_pct = (total_mapped / n) * 100 if n > 0 else 0

        self.logs.append(
            f"FINAL: {len(large_matches)} large + {len(motif_matches)} motifs "
            f"= {coverage_pct:.1f}% coverage"
        )

        return large_matches, motif_matches

    def _find_large_scale_repeats(
        self,
        min_bars: int,
        max_matches: int,
        allow_overlapping_repeats: bool = False,
    ) -> List[LargeMatch]:
        """
        State-of-the-art large-scale repeat detection.
        Extracts ALL diagonals from the bar-level self-similarity matrix,
        scores them, prunes redundancy, and returns high-quality repeats.
        """

        features = self.bar_features
        B = self.num_bars

        if B < 2 * min_bars:
            self.logs.append(f"Skipping large-scale: only {B} bars")
            return []

        # Logging for overlap mode
        if allow_overlapping_repeats:
            self.logs.append("Large-scale: allowing overlapping repeats (legacy mode)")
        else:
            self.logs.append("Large-scale: filtering overlapping repeats (strict mode)")

        # ------------------------------------------------------------"
        # 1. Compute similarity matrix
        # ------------------------------------------------------------"
        S = features @ features.T

        # ------------------------------------------------------------"
        # 2. Extract ALL diagonals for offsets d = 1..B-1
        # ------------------------------------------------------------"
        candidates = []

        for d in range(1, B):  # offset j - i
            i = 0
            while i < B - d:
                # Find contiguous run where S[i+k, i+d+k] >= threshold
                if S[i, i + d] < self.large_similarity:
                    i += 1
                    continue

                start_i = i
                sims = []

                while i < B - d and S[i, i + d] >= self.large_similarity:
                    sims.append(S[i, i + d])
                    i += 1

                length = i - start_i
                if length >= min_bars:
                    avg_sim = float(np.mean(sims))
                    score = (length ** 1.5) * avg_sim

                    candidates.append({
                        "start_a": start_i,
                        "start_b": start_i + d,
                        "length": length,
                        "sim": avg_sim,
                        "score": score,
                    })

        if not candidates:
            self.logs.append("Large-scale: no diagonals found")
            return []

        # ------------------------------------------------------------"
        # 3. Sort by score (longer + more similar first)
        # ------------------------------------------------------------"
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # ------------------------------------------------------------"
        # 4. Greedy pruning of redundant overlaps
        # ------------------------------------------------------------"
        used = np.zeros(B, dtype=bool)
        results = []

        for cand in candidates:
            a0 = cand["start_a"]
            b0 = cand["start_b"]
            L = cand["length"]

            # Reject overlapping repeats if not allowed
            if not allow_overlapping_repeats and (a0 + L - 1 >= b0) and (b0 + L - 1 >= a0):
                continue

            bars_a = np.arange(a0, a0 + L)
            bars_b = np.arange(b0, b0 + L)

            # If both sides heavily overlap with already-selected repeats, skip
            if used[bars_a].sum() > L * 0.5 or used[bars_b].sum() > L * 0.5:
                continue

            # Accept
            results.append(LargeMatch(
                id=len(results) + 1,
                start_bar_a=a0,
                start_bar_b=b0,
                length_bars=L,
                avg_similarity=cand["sim"],
            ))

            # Mark bars as used
            used[bars_a] = True
            used[bars_b] = True

            if len(results) >= max_matches:
                break

        # ------------------------------------------------------------"
        # 5. Mark occupied note indices (same as before)
        # ------------------------------------------------------------"
        bar_ticks = self.ticks_per_beat * 4

        for lm in results:
            for bar_offset in range(lm.length_bars):
                for bar_idx in [lm.start_bar_a + bar_offset, lm.start_bar_b + bar_offset]:
                    start_tick = bar_idx * bar_ticks
                    end_tick = (bar_idx + 1) * bar_ticks

                    for idx, note in enumerate(self.notes):
                        if start_tick <= note["tick"] < end_tick:
                            self.occupied_indices.add(idx)

        # ------------------------------------------------------------"
        # 6. Logging
        # ------------------------------------------------------------"
        total_bars = sum(lm.length_bars * 2 for lm in results)
        coverage = (total_bars / B) * 100 if B > 0 else 0
        self.logs.append(
            f"Large-scale: {len(results)} matches (SOTA), {coverage:.1f}% bar coverage"
        )

        return results

    def _find_hierarchical_motifs(
        self,
        min_length: int,
        max_matches: int,
    ) -> List[Match]:
        """
        Find motifs in multiple passes from longest to shortest.
        Uses exact matching (brute force) for reliability.
        """

        n = len(self.pitch_array)
        unmapped = n - len(self.occupied_indices)

        if unmapped < min_length * 2:
            self.logs.append(f"Skipping motifs: only {unmapped} unmapped notes")
            return []

        max_len = min(unmapped // 2, 64)
        results = []
        match_id = 1

        # Tier 1: Long motifs (12-64 notes)
        tier1_min = max(min_length, 12)
        if tier1_min <= max_len:
            tier1 = self._find_exact_repeats_in_range(
                tier1_min, max_len, max_matches // 3, match_id
            )
            results.extend(tier1)
            match_id += len(tier1)
            self.logs.append(f"  Tier 1 (long): {len(tier1)} matches ({tier1_min}-{max_len} notes)")

        # Tier 2: Medium motifs (6-11 notes)
        tier2_min = max(min_length, 6)
        tier2_max = min(tier1_min - 1, max_len)
        if tier2_min <= tier2_max:
            tier2 = self._find_exact_repeats_in_range(
                tier2_min, tier2_max, max_matches // 3, match_id
            )
            results.extend(tier2)
            match_id += len(tier2)
            self.logs.append(f"  Tier 2 (med):  {len(tier2)} matches ({tier2_min}-{tier2_max} notes)")

        # Tier 3: Short motifs (min_length-5 notes)
        tier3_max = min(tier2_min - 1, max_len)
        if min_length <= tier3_max:
            tier3 = self._find_exact_repeats_in_range(
                min_length, tier3_max, max_matches // 3, match_id
            )
            results.extend(tier3)
            self.logs.append(f"  Tier 3 (short): {len(tier3)} matches ({min_length}-{tier3_max} notes)")

        return results

    def export_segment_as_midi(self, notes):
        """Export notes as MIDI data for browser playback"""
        from mido import MidiFile, MidiTrack, Message, MetaMessage

        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)

        # Add tempo
        track.append(MetaMessage('set_tempo', tempo=self._tempo))

        if not notes:
            track.append(MetaMessage('end_of_track'))
            buffer = io.BytesIO()
            mid.save(file=buffer)
            buffer.seek(0)
            return buffer.getvalue()

        # Sort notes by start tick
        notes_sorted = sorted(notes, key=lambda n: n["tick"])

        # Create events list: (tick_time, event_type, note_data)
        events = []

        for note in notes_sorted:
            # Note on event
            events.append((note["tick"], 'note_on', note))
            # Note off event
            events.append((note["tick"] + max(1, note["duration"]), 'note_off', note))

        # Sort all events by tick time
        events.sort(key=lambda x: x[0])

        # Remove duplicates at same tick (prefer note_off over note_on if conflict)
        filtered_events = []
        i = 0
        while i < len(events):
            current_tick = events[i][0]
            # Group events at same tick
            tick_events = []
            while i < len(events) and events[i][0] == current_tick:
                tick_events.append(events[i])
                i += 1

            # Prefer note_off over note_on for same note at same tick
            note_events = {}
            for tick, event_type, note in tick_events:
                key = (note["pitch"], event_type)
                if key not in note_events:
                    note_events[key] = (tick, event_type, note)

            # Add events, preferring note_off
            for (pitch, event_type), (tick, et, note) in note_events.items():
                filtered_events.append((tick, et, note))

        # Now create MIDI messages with proper delta times
        last_tick = 0
        for tick, event_type, note in filtered_events:
            delta_tick = max(0, tick - last_tick)  # Ensure non-negative

            if event_type == 'note_on':
                track.append(Message('note_on', note=note["pitch"], velocity=note["velocity"], time=delta_tick))
            elif event_type == 'note_off':
                track.append(Message('note_off', note=note["pitch"], velocity=0, time=delta_tick))

            last_tick = tick

        # End of track
        track.append(MetaMessage('end_of_track'))

        # Write to buffer
        buffer = io.BytesIO()
        mid.save(file=buffer)
        buffer.seek(0)
        return buffer.getvalue()

    def render_segment_as_wav(self, notes):
        """Render notes as WAV audio bytes using pretty_midi and soundfile"""
        if not notes:
            # Return empty WAV data
            return b''

        # Create pretty_midi object
        pm = pretty_midi.PrettyMIDI(resolution=self.ticks_per_beat)

        # Create instrument (default piano)
        instrument = pretty_midi.Instrument(program=0)

        # Convert notes to pretty_midi format
        for note in notes:
            # Convert tick time to seconds
            start_time = mido.tick2second(note["tick"], self.ticks_per_beat, self._tempo)
            end_time = mido.tick2second(note["tick"] + note["duration"], self.ticks_per_beat, self._tempo)

            # Create pretty_midi note
            pm_note = pretty_midi.Note(
                velocity=int(note["velocity"]),
                pitch=int(note["pitch"]),
                start=start_time,
                end=end_time
            )
            instrument.notes.append(pm_note)

        pm.instruments.append(instrument)

        # Synthesize audio
        audio_data = pm.synthesize(fs=44100)

        # Write to WAV buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, 44100, format='WAV')
        buffer.seek(0)
        return buffer.getvalue()

    def export_markers_as_midi(self, large_matches, motif_matches=None):
        from mido import MidiFile, MidiTrack, MetaMessage

        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)

        def sec_to_tick(sec):
            return int(mido.second2tick(sec, self.ticks_per_beat, self._tempo))

        for lm in large_matches:
            start_a_sec, end_a_sec = self.bar_range_to_seconds(lm.start_bar_a, lm.length_bars)
            start_b_sec, end_b_sec = self.bar_range_to_seconds(lm.start_bar_b, lm.length_bars)

            for sec, label in [
                (start_a_sec, f"L{lm.id}A Start"),
                (end_a_sec, f"L{lm.id}A End"),
                (start_b_sec, f"L{lm.id}B Start"),
                (end_b_sec, f"L{lm.id}B End"),
            ]:
                track.append(MetaMessage("marker", text=label, time=sec_to_tick(sec)))

        if motif_matches:
            for m in motif_matches:
                for idx, occ in enumerate(m.occurrences):
                    track.append(MetaMessage(
                        "marker",
                        text=f"Motif {m.id} Occ {idx+1}",
                        time=self.notes[occ]["tick"]
                    ))

        track.sort(key=lambda msg: msg.time)

        last = 0
        for msg in track:
            delta = msg.time - last
            last = msg.time
            msg.time = delta

        buffer = io.BytesIO()
        mid.save(file=buffer)
        buffer.seek(0)
        return buffer.getvalue()

    def build_timeline(self, window_seconds: float = 0.1) -> Tuple[List[float], List[int]]:
        if not self.notes:
            return [], []

        note_times = [
            mido.tick2second(n["tick"], self.ticks_per_beat, self._tempo)
            for n in self.notes
        ]

        duration = max(note_times) if note_times else 0.0
        if duration == 0:
            return [], []

        num_bins = max(1, int(math.ceil(duration / window_seconds)))
        densities = [0] * num_bins

        for t in note_times:
            densities[min(int(t // window_seconds), num_bins - 1)] += 1

        times = [(i + 0.5) * window_seconds for i in range(num_bins)]
        return times, densities

    def pattern_occurrences_in_seconds(self, match: Match) -> List[Tuple[float, float]]:
        result = []
        for start_idx in match.occurrences:
            start_tick = self.notes[start_idx]["tick"]
            end_tick = self.notes[start_idx + match.length - 1]["tick"]
            result.append((
                mido.tick2second(start_tick, self.ticks_per_beat, self._tempo),
                mido.tick2second(end_tick, self.ticks_per_beat, self._tempo)
            ))
        return result

    def bar_range_to_seconds(self, start_bar: int, length_bars: int) -> Tuple[float, float]:
        bar_ticks = self.ticks_per_bar
        start_tick = start_bar * bar_ticks
        end_tick = (start_bar + length_bars) * bar_ticks
        return (
            mido.tick2second(start_tick, self.ticks_per_beat, self._tempo),
            mido.tick2second(end_tick, self.ticks_per_beat, self._tempo)
        )

    def notes_in_bar_range(self, start_bar: int, length_bars: int):
        bar_ticks = self.ticks_per_bar
        start_tick = start_bar * bar_ticks
        end_tick = (start_bar + length_bars) * bar_ticks
        return [n for n in self.notes if start_tick <= n["tick"] < end_tick]

    def motif_signature(self, motif):
        """
        Transposition- and octave-invariant, ornament-tolerant signature.
        Uses only the *direction* of pitch movement: -1, 0, +1.
        """
        if not motif.occurrences:
            return ()

        start = motif.occurrences[0]
        end = start + motif.length

        pitches = [self.notes[i]["pitch"] for i in range(start, end)]
        if len(pitches) < 2:
            return ()

        dirs = []
        for a, b in zip(pitches, pitches[1:]):
            if b > a:
                dirs.append(1)
            elif b < a:
                dirs.append(-1)
            else:
                dirs.append(0)

        return tuple(dirs)

    def deduplicate_motifs(self, motifs):
        """Remove exact duplicates based on direction signature + length."""
        buckets = {}
        for m in motifs:
            sig = self.motif_signature(m)
            key = (sig, m.length)
            buckets.setdefault(key, []).append(m)

        deduped = []
        for key, group in buckets.items():
            best = max(group, key=lambda m: (m.length, m.similarity))
            deduped.append(best)

        return deduped

    def interval_distance(self, sig1, sig2):
        """Hamming distance on direction signatures."""
        if len(sig1) != len(sig2):
            return 999
        return sum(1 for a, b in zip(sig1, sig2) if a != b)

    def cluster_motifs(self, motifs, threshold=1):
        """
        Cluster motifs whose direction signatures differ by <= threshold.
        threshold=1 is a good starting point.
        """
        clusters = []
        used = set()

        sigs = [self.motif_signature(m) for m in motifs]

        for i, m in enumerate(motifs):
            if i in used:
                continue

            cluster = [m]
            used.add(i)

            for j in range(i + 1, len(motifs)):
                if j in used:
                    continue
                if self.interval_distance(sigs[i], sigs[j]) <= threshold:
                    cluster.append(motifs[j])
                    used.add(j)

            clusters.append(cluster)

        return clusters

    def pick_representatives(self, clusters):
        """Pick the best motif from each cluster."""
        reps = []
        for cluster in clusters:
            best = max(cluster, key=lambda m: (m.length, m.similarity))
            reps.append(best)
        return reps

    def postprocess_motifs(self, motifs):
        """Full pipeline: dedupe → cluster → pick representatives."""
        if not motifs:
            return motifs

        motifs = self.deduplicate_motifs(motifs)
        clusters = self.cluster_motifs(motifs, threshold=1)
        reps = self.pick_representatives(clusters)

        self.logs.append(
            f"Motif clustering: {len(motifs)} → {len(reps)} representatives across {len(clusters)} clusters"
        )

        return reps
