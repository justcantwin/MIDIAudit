# ================================================================
# STREAMLIT MIDI STRUCTURAL AUDITOR (Full Version)
# ================================================================

import streamlit as st
import io
import os
import plotly.graph_objects as go
import mido
import hashlib

from midi_auditor import MIDIAuditor
from visualization import (
    plot_structure_waveform,
    plot_large_scale_repeats,
    plot_timeline_with_overlaps,
    plot_piano_roll
)

# Top-level Audix declaration
from streamlit_advanced_audio import audix
audix_player = audix  # single shared instance

# Import for error handling
try:
    from tornado.websocket import WebSocketClosedError
    from tornado.iostream import StreamClosedError
except ImportError:
    WebSocketClosedError = Exception
    StreamClosedError = Exception

# =========================
# SESSION STATE INIT
# =========================
if 'audio_cache' not in st.session_state:
    st.session_state.audio_cache = {}

def init_audio_cache():
    """Initialize audio cache in session state safely."""
    if 'audio_cache' not in st.session_state:
        st.session_state.audio_cache = {}

def calculate_audio_duration(audio_bytes):
    """Calculate duration from WAV audio bytes."""
    if not audio_bytes:
        return 0.1
    try:
        import soundfile as sf
        import io
        import numpy as np

        buffer = io.BytesIO(audio_bytes)
        data, samplerate = sf.read(buffer)

        # Remove leading silence
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        threshold = 0.01
        start_idx = 0
        for i in range(len(data)):
            if abs(data[i]) > threshold:
                start_idx = i
                break
        trimmed_data = data[start_idx:]
        return len(trimmed_data) / samplerate
    except Exception:
        return 0.1

def render_segment_audio(notes, ticks_per_beat, tempo):
    """Render notes as WAV audio bytes using pretty_midi and soundfile."""
    if not notes:
        return b''
    try:
        import pretty_midi
        import soundfile as sf
        import numpy as np

        pm = pretty_midi.PrettyMIDI(resolution=ticks_per_beat)
        instrument = pretty_midi.Instrument(program=0)

        for note in notes:
            start_time = mido.tick2second(note["tick"], ticks_per_beat, tempo)
            end_time = mido.tick2second(note["tick"] + note["duration"], ticks_per_beat, tempo)
            pm_note = pretty_midi.Note(
                velocity=int(note["velocity"]),
                pitch=int(note["pitch"]),
                start=start_time,
                end=end_time
            )
            instrument.notes.append(pm_note)

        pm.instruments.append(instrument)

        # Try fluidsynth first, fallback to synthesize
        try:
            sf2_path = "FluidR3_GM.sf2"
            audio_data = pm.fluidsynth(sf2_path) if os.path.exists(sf2_path) else pm.synthesize(fs=44100)
        except Exception:
            audio_data = pm.synthesize(fs=44100)

        # Trim leading silence
        threshold = 0.01
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        start_idx = 0
        for i in range(len(audio_data)):
            if abs(audio_data[i]) > threshold:
                start_idx = i
                break
        trimmed_audio = audio_data[start_idx:]

        buffer = io.BytesIO()
        sf.write(buffer, trimmed_audio, 44100, format='WAV')
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        st.error(f"Error rendering audio: {e}")
        return b''

def render_mixed_audio(notes_a, notes_b, ticks_per_beat, tempo):
    """Render mixed audio by overlaying segments A and B simultaneously."""
    if not notes_a and not notes_b:
        return b''

    try:
        import pretty_midi
        import soundfile as sf
        import numpy as np

        # Create pretty_midi object
        pm = pretty_midi.PrettyMIDI(resolution=ticks_per_beat)

        # Create instrument (default piano)
        instrument = pretty_midi.Instrument(program=0)

        # Find min tick for each segment to align to start at 0
        min_tick_a = min(note["tick"] for note in notes_a) if notes_a else 0
        min_tick_b = min(note["tick"] for note in notes_b) if notes_b else 0

        # Add notes from segment A, aligned to start at 0
        for note in notes_a:
            start_time = mido.tick2second(note["tick"] - min_tick_a, ticks_per_beat, tempo)
            end_time = mido.tick2second(note["tick"] - min_tick_a + note["duration"], ticks_per_beat, tempo)
            pm_note = pretty_midi.Note(
                velocity=int(note["velocity"]),
                pitch=int(note["pitch"]),
                start=start_time,
                end=end_time
            )
            instrument.notes.append(pm_note)

        # Add notes from segment B, aligned to start at 0 (overlay)
        for note in notes_b:
            start_time = mido.tick2second(note["tick"] - min_tick_b, ticks_per_beat, tempo)
            end_time = mido.tick2second(note["tick"] - min_tick_b + note["duration"], ticks_per_beat, tempo)
            pm_note = pretty_midi.Note(
                velocity=int(note["velocity"]),
                pitch=int(note["pitch"]),
                start=start_time,
                end=end_time
            )
            instrument.notes.append(pm_note)

        pm.instruments.append(instrument)

        # Try fluidsynth first, fallback to synthesize
        try:
            sf2_path = "FluidR3_GM.sf2"
            audio_data = pm.fluidsynth(sf2_path) if os.path.exists(sf2_path) else pm.synthesize(fs=44100)
        except Exception:
            audio_data = pm.synthesize(fs=44100)

        # Trim leading silence
        threshold = 0.01
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        start_idx = 0
        for i in range(len(audio_data)):
            if abs(audio_data[i]) > threshold:
                start_idx = i
                break
        trimmed_audio = audio_data[start_idx:]

        buffer = io.BytesIO()
        sf.write(buffer, trimmed_audio, 44100, format='WAV')
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        st.error(f"Error rendering mixed audio: {e}")
        return b''

def initialize_audio_cache_for_analysis(auditor, large_matches, motif_matches):
    """Pre-generate and cache all audio during initialization phase."""
    # Use safe session state access
    audio_cache = st.session_state.get('audio_cache', {})
    if 'audio_cache' not in st.session_state:
        st.session_state.audio_cache = audio_cache

    for lm in large_matches:
        # Segment A
        notes_a = auditor.notes_in_bar_range(lm.start_bar_a, lm.length_bars)
        key_a = f"a_{lm.id}_{hash(str(notes_a) + str(auditor.ticks_per_beat) + str(auditor._tempo))}"
        if key_a not in audio_cache:
            wav_bytes = render_segment_audio(notes_a, auditor.ticks_per_beat, auditor._tempo)
            audio_cache[key_a] = {
                'audio_bytes': wav_bytes,
                'duration': calculate_audio_duration(wav_bytes)
            }

        # Segment B
        notes_b = auditor.notes_in_bar_range(lm.start_bar_b, lm.length_bars)
        key_b = f"b_{lm.id}_{hash(str(notes_b) + str(auditor.ticks_per_beat) + str(auditor._tempo))}"
        if key_b not in audio_cache:
            wav_bytes = render_segment_audio(notes_b, auditor.ticks_per_beat, auditor._tempo)
            audio_cache[key_b] = {
                'audio_bytes': wav_bytes,
                'duration': calculate_audio_duration(wav_bytes)
            }

        # Mixed
        mixed_notes = (notes_a or []) + (notes_b or [])
        key_mixed = f"mixed_{lm.id}_{hash(str(mixed_notes) + str(auditor.ticks_per_beat) + str(auditor._tempo))}"
        if key_mixed not in audio_cache:
            wav_bytes = render_mixed_audio(notes_a, notes_b, auditor.ticks_per_beat, auditor._tempo)
            audio_cache[key_mixed] = {
                'audio_bytes': wav_bytes,
                'duration': calculate_audio_duration(wav_bytes)
            }

    # Update session state with the complete cache
    st.session_state.audio_cache = audio_cache

# =========================
# AUDIO PLAYER COMPONENT
# =========================
def audio_player_component(notes_a, notes_b=None, label="Segment A", ticks_per_beat=480, tempo=500000, match_id=None):
    """Read-only audio player displaying pre-cached audio with unique keys."""
    col1, col2 = st.columns(2) if notes_b else (st.container(), None)

    # Segment A
    with col1:
        st.markdown(f"**🎵 {label}**")
        if notes_a:
            key_a = f"a_{match_id}_{hash(str(notes_a) + str(ticks_per_beat) + str(tempo))}"
            cached = st.session_state.audio_cache.get(key_a, {})
            wav_bytes = cached.get("audio_bytes")
            duration = cached.get("duration", 0.1)
            if wav_bytes:
                try:
                    audix_player(wav_bytes, key=f"audio_player_a_{match_id}", sample_rate=44100)
                    st.caption(f"Duration: {duration:.1f}s")
                except (WebSocketClosedError, StreamClosedError) as e:
                    st.warning("WebSocket connection lost. Please refresh the page.")
                except Exception as e:
                    st.warning(f"Audio playback error: {str(e)}")
            else:
                st.info(f"Duration: {duration:.1f}s | No audio data generated.")
        else:
            st.info("No notes to play.")

    # Segment B
    if notes_b and col2:
        with col2:
            st.markdown("**🎵 Segment B**")
            key_b = f"b_{match_id}_{hash(str(notes_b) + str(ticks_per_beat) + str(tempo))}"
            cached = st.session_state.audio_cache.get(key_b, {})
            wav_bytes = cached.get("audio_bytes")
            duration = cached.get("duration", 0.1)
            if wav_bytes:
                try:
                    audix_player(wav_bytes, key=f"audio_player_b_{match_id}", sample_rate=44100)
                    st.caption(f"Duration: {duration:.1f}s")
                except (WebSocketClosedError, StreamClosedError) as e:
                    st.warning("WebSocket connection lost. Please refresh the page.")
                except Exception as e:
                    st.warning(f"Audio playback error: {str(e)}")
            else:
                st.info(f"Duration: {duration:.1f}s | No audio data generated.")

    # Mixed Playback
    if notes_b:
        st.markdown("**🎼 Mixed Playback**")
        mixed_notes = (notes_a or []) + (notes_b or [])
        key_mixed = f"mixed_{match_id}_{hash(str(mixed_notes) + str(ticks_per_beat) + str(tempo))}"
        cached = st.session_state.audio_cache.get(key_mixed, {})
        wav_bytes = cached.get("audio_bytes")
        duration = cached.get("duration", 0.1)
        if wav_bytes:
            try:
                audix_player(wav_bytes, key=f"audio_player_mixed_{match_id}", sample_rate=44100)
                st.caption(f"Duration: {duration:.1f}s")
            except (WebSocketClosedError, StreamClosedError) as e:
                st.warning("WebSocket connection lost. Please refresh the page.")
            except Exception as e:
                st.warning(f"Audio playback error: {str(e)}")
        else:
            st.info(f"Duration: {duration:.1f}s | No audio data generated.")

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="MIDI Structural Auditor", layout="wide", page_icon="🎼")

st.markdown("""
<style>
    .main-header {text-align: center; color: #2E86AB; font-size: 2.5em; margin-bottom: 1em;}
</style>
""", unsafe_allow_html=True)
st.markdown('<h1 class="main-header">🎼 MIDI Structural Auditor</h1>', unsafe_allow_html=True)
st.markdown("*Advanced musical structure analysis for MIDI files*")

col1, col2 = st.columns([1,2])
with col1:
    uploaded_file = st.file_uploader("Upload MIDI file", type=["mid","midi"])
with col2:
    if uploaded_file:
        st.success(f"📁 **{uploaded_file.name}** uploaded successfully!")
        st.info(f"File size: {len(uploaded_file.getvalue())/1024:.1f} KB")
    else:
        st.info("Upload a MIDI file to begin analysis")

if uploaded_file:
    midi_stream = io.BytesIO(uploaded_file.getvalue())

    # Sidebar
    st.sidebar.title("⚙️ Analysis Settings")
    with st.sidebar.expander("🔍 Detection Parameters", expanded=True):
        large_sim = st.slider("Large Section Similarity", 0.70,0.98,0.90,0.02)
        motif_sim = st.slider("Motif Similarity (unused)", 0.60,0.98,0.70,0.02)
        min_large_bars = st.slider("Min Section Length (bars)", 2,16,4)
        min_motif_notes = st.slider("Min Motif Length (notes)", 3,16,4)
    with st.sidebar.expander("🔧 Advanced Options"):
        allow_overlapping_repeats = st.checkbox("Allow overlapping repeats (legacy mode)", value=False)
        per_layer_features = st.checkbox("Per-Layer Features", value=False)

    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("🔄 Initializing analysis...")
    progress_bar.progress(10)

    auditor = MIDIAuditor(midi_stream, large_similarity=large_sim, motif_similarity=motif_sim, per_layer=per_layer_features)
    progress_bar.progress(30)
    status_text.text("📊 Building features...")
    progress_bar.progress(50)
    status_text.text("🔍 Finding patterns...")

    auditor.occupied_indices = set()
    large_matches, motif_matches = auditor.find_all_patterns(
        min_motif_length=min_motif_notes,
        min_large_bars=min_large_bars,
        max_results=100,
        allow_overlapping_repeats=allow_overlapping_repeats
    )
    motif_matches = auditor.postprocess_motifs(motif_matches)
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")

    # Audio cache
    init_audio_cache()
    initialize_audio_cache_for_analysis(auditor, large_matches, motif_matches)

    sections = auditor.label_sections(large_matches)
    summary_text, summary_lines = auditor.summarize_structure(sections)
    timeline = auditor.visualize_timeline(sections, auditor.num_bars)
    times, densities = auditor.build_timeline()

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview","🔁 Large-Scale Analysis","🧬 Motif Analysis","📋 Exports & Logs"])

    # -----------------------
    # TAB 1: Overview
    # -----------------------
    with tab1:
        st.header("📊 Analysis Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Notes", len(auditor.notes))
        with col2: st.metric("Total Bars", auditor.num_bars)
        with col3: st.metric("Large Sections", len(large_matches))
        coverage = len(auditor.occupied_indices)/len(auditor.notes)*100 if auditor.notes else 0
        with col4: st.metric("Coverage", f"{coverage:.1f}%")
        st.success(f"🎯 Found {len(large_matches)} large-scale sections and {len(motif_matches)} motifs")
        st.subheader("🎼 Structural Summary")
        st.code(summary_text, language="text")
        st.subheader("📈 Structure Timeline")
        if sections:
            st.plotly_chart(plot_timeline_with_overlaps(sections, auditor), use_container_width=True)
        else:
            st.info("No sections detected to display in timeline.")
        if times and densities:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=densities, mode="lines", line=dict(color="#2E86AB"), name="Note Density"))
            fig.update_layout(title="Overall Note Density Timeline", xaxis_title="Time (s)", yaxis_title="Notes per 0.1s", height=300)
            st.plotly_chart(fig, use_container_width=True)

    # -----------------------
    # TAB 2: Large-Scale
    # -----------------------
    with tab2:
        st.header("🔁 Large-Scale Repeats")
        if large_matches:
            match_options = {f"Match {lm.id}: {lm.length_bars} bars (Sim: {lm.avg_similarity:.2f})": lm.id for lm in large_matches}
            selected = st.selectbox("Focus on specific repeat:", ["Show all"] + list(match_options.keys()))
            large_matches_filtered = large_matches
            zoom_range = None
            if selected != "Show all":
                lm = next(m for m in large_matches if m.id==match_options[selected])
                start_a, end_a = auditor.bar_range_to_seconds(lm.start_bar_a, lm.length_bars)
                start_b, end_b = auditor.bar_range_to_seconds(lm.start_bar_b, lm.length_bars)
                zoom_range = (min(start_a,start_b)-0.5, max(end_a,end_b)+0.5)
                large_matches_filtered = [lm]
            if times and densities:
                st.plotly_chart(plot_large_scale_repeats(times,densities,large_matches_filtered,auditor,zoom_range), use_container_width=True)

            st.subheader("📋 Detailed Matches")
            for lm in sorted(large_matches_filtered, key=lambda x:x.length_bars, reverse=True):
                with st.expander(f"🔄 Match {lm.id}: {lm.length_bars} bars", expanded=(len(large_matches_filtered)==1)):
                    notes_a = auditor.notes_in_bar_range(lm.start_bar_a, lm.length_bars)
                    notes_b = auditor.notes_in_bar_range(lm.start_bar_b, lm.length_bars)

                    colA, colB = st.columns(2)
                    with colA: st.plotly_chart(plot_piano_roll(notes_a, auditor.ticks_per_beat, auditor._tempo, "Segment A"), use_container_width=True)
                    with colB: st.plotly_chart(plot_piano_roll(notes_b, auditor.ticks_per_beat, auditor._tempo, "Segment B"), use_container_width=True)

                    audio_player_component(notes_a, notes_b, "Segment A", auditor.ticks_per_beat, auditor._tempo, lm.id)

                    start_tick_a = lm.start_bar_a * auditor.ticks_per_bar
                    end_tick_a   = (lm.start_bar_a + lm.length_bars) * auditor.ticks_per_bar
                    trimmed_notes_a = auditor.trim_notes_to_pattern(notes_a, start_tick_a, end_tick_a)

                    start_tick_b = lm.start_bar_b * auditor.ticks_per_bar
                    end_tick_b   = (lm.start_bar_b + lm.length_bars) * auditor.ticks_per_bar
                    trimmed_notes_b = auditor.trim_notes_to_pattern(notes_b, start_tick_b, end_tick_b)

                    dl1, dl2 = st.columns(2)
                    with dl1:
                        st.download_button("Download Segment A (MIDI)", auditor.export_segment_as_midi(trimmed_notes_a), f"large_{lm.id}_A.mid", "audio/midi")
                    with dl2:
                        st.download_button("Download Segment B (MIDI)", auditor.export_segment_as_midi(trimmed_notes_b), f"large_{lm.id}_B.mid", "audio/midi")

        else:
            st.info("No large-scale repeats detected with current settings.")


    # -----------------------
    # TAB 3: Motif Analysis
    # -----------------------
    with tab3:
        st.header("🧬 Motif Analysis")
        if motif_matches:
            st.plotly_chart(plot_structure_waveform(times,densities,motif_matches,auditor), use_container_width=True)
            st.subheader("📋 Detected Motifs")
            for m in motif_matches:
                with st.expander(f"🎵 Motif {m.id}: {m.length} notes ({len(m.occurrences)} occurrences)"):
                    best_idx = m.occurrences[0]
                    segment = [auditor.notes[i] for i in range(best_idx,best_idx+m.length)]
                    times_motif = [mido.tick2second(n["tick"],auditor.ticks_per_beat,auditor._tempo) for n in segment]
                    pitches = [n["pitch"] for n in segment]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=times_motif, y=pitches, mode="lines+markers", line=dict(color="#FF6B6B", width=3), marker=dict(size=8, color="#FF6B6B")))
                    fig.update_layout(title=f"Motif {m.id} Pattern", xaxis_title="Time (s)", yaxis_title="Pitch", height=250)
                    st.plotly_chart(fig, use_container_width=True)
                    st.download_button("Download Motif (MIDI)", auditor.export_segment_as_midi(segment), f"motif_{m.id}.mid", "audio/midi")
        else:
            st.info("No motifs detected with current settings.")

    # -----------------------
    # TAB 4: Exports & Logs
    # -----------------------
    with tab4:
        st.header("📋 Exports & Technical Details")
        if large_matches:
            st.download_button("Download DAW Markers (Cubase/Logic)", auditor.export_markers_as_midi(large_matches, motif_matches), "markers.mid", "audio/midi")
        st.subheader("📊 Analysis Summary")
        st.json({
            "File": uploaded_file.name,
            "Total Notes": len(auditor.notes),
            "Total Bars": auditor.num_bars,
            "Large Sections Found": len(large_matches),
            "Motifs Found": len(motif_matches),
            "Coverage %": f"{coverage:.1f}",
            "Analysis Timestamp": "2026-01-18"
        })
        with st.expander("🔧 Debug Logs", expanded=False):
            if auditor.logs:
                st.code("\n".join(auditor.logs), language="text")
            else:
                st.write("No logs available")

    st.markdown("---")
    st.markdown("*Built with ❤️ for musical analysis*")

else:
    # Welcome
    st.markdown("""
    ## Welcome to MIDI Structural Auditor! 🎼

    This tool analyzes MIDI files to identify:
    - **Large-scale repeats** (verse-chorus, thematic sections)
    - **Motifs** (short musical patterns)
    - **Structural coverage** (how much of the piece is accounted for)

    ### How to use:
    1. Upload a MIDI file using the uploader above
    2. Adjust analysis parameters in the sidebar if needed
    3. Explore the results in the different tabs
    """)
