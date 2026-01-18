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
        verbose: bool = True,
        per_layer: bool = False,
    ):
        self.verbose = verbose
        self.logs: List[str] = []

        self.mid = mido.MidiFile(file=file_stream)
        self.ticks_per_beat = self.mid.ticks_per_beat
        self.large_similarity = large_similarity
        self.motif_similarity = motif_similarity
        self.quantize_ticks = int(self.ticks_per_beat * quantize_beats)
        self.per_layer = per_layer

        self._cache = {}

        self.tempos = self._extract_tempos()
        self.time_signatures = self._extract_time_signatures()
        self._tempo = self.tempos[0] if self.tempos else 500000

        self.notes = self._extract_notes()
        self.pitch_array = np.array([n["pitch"] for n in self.notes], dtype=np.int16)
        self.occupied_indices: Set[int] = set()

        self._build_suffix_automaton()
        self.ticks_per_bar = self._compute_average_ticks_per_bar()
        self.bar_features, self.num_bars = self._build_bar_features(self.per_layer)

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
                        "duration": self.ticks_per_beat,
                        "channel": msg.channel
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
            self._log(
                f"Track {i}: {len(track)} msgs, "
                f"{count_note_on} note_ons, "
                f"{count_note_off} matched note_offs, "
                f"{len(active_notes)} unmatched active"
            )

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

    def _build_bar_features(self, per_layer: bool = False) -> Tuple[np.ndarray, int]:
        if not self.notes:
            return np.zeros((0, 12), dtype=np.float32), 0

        bar_ticks = self.ticks_per_bar
        bar_indices = [n["tick"] // bar_ticks for n in self.notes]
        num_bars = int(max(bar_indices)) + 1 if bar_indices else 0

        if per_layer:
            num_layers = max(n["channel"] for n in self.notes) + 1
            chroma = np.zeros((num_bars, 12 * num_layers), dtype=np.float32)
        else:
            chroma = np.zeros((num_bars, 12), dtype=np.float32)

        rhythm = np.zeros((num_bars, 4), dtype=np.float32)

        for note, bar in zip(self.notes, bar_indices):
            pitch_class = note["pitch"] % 12
            weight = note["velocity"] / 127.0 * math.sqrt(note["duration"])
            if per_layer:
                idx = pitch_class + 12 * note["channel"]
            else:
                idx = pitch_class
            chroma[bar, idx] += weight

            pos = (note["tick"] % bar_ticks) / bar_ticks
            quarter = int(pos * 4)
            if quarter < 4:
                rhythm[bar, quarter] += weight

        features = np.hstack([chroma, rhythm])
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0

        self._log(
            f"Built {'per-layer' if per_layer else 'summed-layer'} features for {num_bars} bars"
        )
        return features / norms, num_bars
    def _find_exact_repeats_in_range(
        self,
        min_len: int,
        max_len: int,
        max_matches: int,
        start_id: int,
    ) -> List[Match]:
        n = len(self.pitch_array)
        results = []
        current_id = start_id

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
                            int(self.notes[idx]["tick"] / self.ticks_per_bar)
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

    def trim_notes_to_pattern(
        self,
        notes: List[Dict],
        start_tick: int,
        end_tick: int
    ) -> List[Dict]:
        """
        STRICT trim.
        No re-timing, no shifting.
        Guarantees no pre-roll contamination.
        """
        return [
            n for n in notes
            if start_tick <= n["tick"] < end_tick
        ]

    def find_all_patterns(
        self,
        min_motif_length: int = 4,
        min_large_bars: int = 4,
        max_results: int = 100,
        allow_overlapping_repeats: bool = False,
        strict_full_layers: bool = False,
    ) -> Tuple[List[LargeMatch], List[Match]]:

        large_matches = self._find_large_scale_repeats(
            min_bars=min_large_bars,
            max_matches=max_results // 2,
            allow_overlapping_repeats=allow_overlapping_repeats,
            strict_full_layers=strict_full_layers,
        )

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
        strict_full_layers: bool = False,
    ) -> List[LargeMatch]:

        features = self.bar_features
        B = self.num_bars

        if B < 2 * min_bars:
            self.logs.append(f"Skipping large-scale: only {B} bars")
            return []

        S = features @ features.T
        candidates = []

        for d in range(1, B):
            i = 0
            while i < B - d:
                if S[i, i + d] < self.large_similarity:
                    i += 1
                    continue

                start_i = i
                sims = []

                while i < B - d and S[i, i + d] >= self.large_similarity:
                    if strict_full_layers:
                        if not np.allclose(features[i], features[i + d]):
                            break
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

        candidates.sort(key=lambda x: x["score"], reverse=True)

        used = np.zeros(B, dtype=bool)
        results = []

        for cand in candidates:
            a0 = cand["start_a"]
            b0 = cand["start_b"]
            L = cand["length"]

            if not allow_overlapping_repeats:
                if (a0 < b0 + L) and (b0 < a0 + L):
                    continue

            bars_a = np.arange(a0, a0 + L)
            bars_b = np.arange(b0, b0 + L)

            if used[bars_a].sum() > L * 0.5 or used[bars_b].sum() > L * 0.5:
                continue

            results.append(LargeMatch(
                id=len(results) + 1,
                start_bar_a=a0,
                start_bar_b=b0,
                length_bars=L,
                avg_similarity=cand["sim"],
            ))

            used[bars_a] = True
            used[bars_b] = True

            if len(results) >= max_matches:
                break

        bar_ticks = self.ticks_per_bar
        for lm in results:
            for bar_offset in range(lm.length_bars):
                for bar_idx in (lm.start_bar_a + bar_offset,
                                lm.start_bar_b + bar_offset):
                    start_tick = bar_idx * bar_ticks
                    end_tick = start_tick + bar_ticks
                    for idx, note in enumerate(self.notes):
                        if start_tick <= note["tick"] < end_tick:
                            self.occupied_indices.add(idx)

        return results

    def label_sections(self, large_matches: List[LargeMatch]):
        sections = []

        for lm in large_matches:
            length = lm.length_bars
            bar_ticks = self.ticks_per_bar

            for occ, start_bar in (("A", lm.start_bar_a), ("B", lm.start_bar_b)):
                start_tick = start_bar * bar_ticks
                end_tick = start_tick + length * bar_ticks

                sections.append({
                    "match_id": lm.id,
                    "occurrence": occ,
                    "start_bar": start_bar,
                    "end_bar": start_bar + length - 1,
                    "start_tick": start_tick,
                    "end_tick": end_tick,
                    "similarity": lm.avg_similarity,
                })

        sections.sort(key=lambda s: s["start_tick"])
        return sections
        
    def notes_in_bar_range(self, start_bar: int, length_bars: int):
        """
        LEGACY helper (bar-aligned).
        Kept for visualization only.
        DO NOT use for audio or export.
        """
        bar_ticks = self.ticks_per_bar
        start_tick = start_bar * bar_ticks
        end_tick = start_tick + length_bars * bar_ticks
        return [
            n for n in self.notes
            if start_tick <= n["tick"] < end_tick
        ]

    # =========================================================
    # EXPORT — STRICTLY TRIMMED
    # =========================================================
    def export_segment_as_midi(
        self,
        notes: List[Dict],
        start_tick: int,
        end_tick: int,
    ) -> bytes:
        """
        MIDI export with:
        - strict trimming
        - time re-based to 0
        - no phantom silence
        """

        trimmed = self.trim_notes_to_pattern(notes, start_tick, end_tick)
        if not trimmed:
            return b""

        from mido import MidiFile, MidiTrack, Message, MetaMessage

        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)

        track.append(MetaMessage("set_tempo", tempo=self._tempo, time=0))

        events = []
        for n in trimmed:
            on = n["tick"] - start_tick
            off = on + max(1, n["duration"])
            events.append((on, "on", n))
            events.append((off, "off", n))

        events.sort(key=lambda e: (e[0], e[1] == "on"))

        last = 0
        for t, kind, n in events:
            delta = max(0, t - last)
            if kind == "on":
                track.append(
                    Message(
                        "note_on",
                        note=int(n["pitch"]),
                        velocity=int(n["velocity"]),
                        time=delta,
                    )
                )
            else:
                track.append(
                    Message(
                        "note_off",
                        note=int(n["pitch"]),
                        velocity=0,
                        time=delta,
                    )
                )
            last = t

        track.append(MetaMessage("end_of_track", time=0))

        buf = io.BytesIO()
        mid.save(file=buf)
        buf.seek(0)
        return buf.getvalue()

    def render_segment_as_wav(
        self,
        notes: List[Dict],
        start_tick: int,
        end_tick: int,
        sample_rate: int = 44100,
    ) -> bytes:
        """
        WAV render with:
        - strict trimming
        - tick re-base
        - sample-accurate alignment
        """

        trimmed = self.trim_notes_to_pattern(notes, start_tick, end_tick)
        if not trimmed:
            return b""

        pm = pretty_midi.PrettyMIDI(resolution=self.ticks_per_beat)
        inst = pretty_midi.Instrument(program=0)

        for n in trimmed:
            start = mido.tick2second(
                n["tick"] - start_tick,
                self.ticks_per_beat,
                self._tempo,
            )
            end = mido.tick2second(
                (n["tick"] - start_tick) + n["duration"],
                self.ticks_per_beat,
                self._tempo,
            )
            inst.notes.append(
                pretty_midi.Note(
                    velocity=int(n["velocity"]),
                    pitch=int(n["pitch"]),
                    start=start,
                    end=end,
                )
            )

        pm.instruments.append(inst)
        audio = pm.synthesize(fs=sample_rate)

        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        buf.seek(0)
        return buf.getvalue()

    # =========================================================
    # TRUE MIXED PLAYBACK (SIMULTANEOUS)
    # =========================================================
    def render_mixed_segment_as_wav(
        self,
        notes_a: List[Dict],
        notes_b: List[Dict],
        start_tick: int,
        end_tick: int,
        sample_rate: int = 44100,
    ) -> bytes:
        """
        Layered mix.
        NOT concatenation.
        """

        a = self.trim_notes_to_pattern(notes_a, start_tick, end_tick)
        b = self.trim_notes_to_pattern(notes_b, start_tick, end_tick)

        if not a and not b:
            return b""

        pm = pretty_midi.PrettyMIDI(resolution=self.ticks_per_beat)

        for notes in (a, b):
            inst = pretty_midi.Instrument(program=0)
            for n in notes:
                start = mido.tick2second(
                    n["tick"] - start_tick,
                    self.ticks_per_beat,
                    self._tempo,
                )
                end = mido.tick2second(
                    (n["tick"] - start_tick) + n["duration"],
                    self.ticks_per_beat,
                    self._tempo,
                )
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=int(n["velocity"]),
                        pitch=int(n["pitch"]),
                        start=start,
                        end=end,
                    )
                )
            pm.instruments.append(inst)

        audio = pm.synthesize(fs=sample_rate)
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        buf.seek(0)
        return buf.getvalue()
