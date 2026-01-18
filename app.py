# Top-level Audix declaration
from streamlit_advanced_audio import audix

# Declare once
audix_player = audix

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



# Top-level session state initialization - MUST be at the very top before any components
if 'audio_cache' not in st.session_state:
    st.session_state.audio_cache = {}

# Safe session state initialization for audio cache
def init_audio_cache():
    """Initialize audio cache in session state safely."""
    if 'audio_cache' not in st.session_state:
        st.session_state.audio_cache = {}

def initialize_audio_cache_for_analysis(auditor, large_matches, motif_matches):
    """Pre-generate and cache all audio during initialization phase."""
    # Ensure session state exists
    if 'audio_cache' not in st.session_state:
        st.session_state.audio_cache = {}

    # Pre-cache all large match audio
    for lm in large_matches:
        # Segment A
        notes_a = auditor.notes_in_bar_range(lm.start_bar_a, lm.length_bars)
        cache_key_a = f"a_{lm.id}_{hash(str(notes_a) + str(auditor.ticks_per_beat) + str(auditor._tempo))}"
        if cache_key_a not in st.session_state.audio_cache:
            wav_bytes_a = render_segment_audio(notes_a, auditor.ticks_per_beat, auditor._tempo)
            # Calculate duration from audio bytes
            duration_a = calculate_audio_duration(wav_bytes_a)
            # Store both audio and duration in cache
            st.session_state.audio_cache[cache_key_a] = {
                'audio_bytes': wav_bytes_a,
                'duration': duration_a
            }

        # Segment B
        notes_b = auditor.notes_in_bar_range(lm.start_bar_b, lm.length_bars)
        cache_key_b = f"b_{lm.id}_{hash(str(notes_b) + str(auditor.ticks_per_beat) + str(auditor._tempo))}"
        if cache_key_b not in st.session_state.audio_cache:
            wav_bytes_b = render_segment_audio(notes_b, auditor.ticks_per_beat, auditor._tempo)
            # Calculate duration from audio bytes
            duration_b = calculate_audio_duration(wav_bytes_b)
            # Store both audio and duration in cache
            st.session_state.audio_cache[cache_key_b] = {
                'audio_bytes': wav_bytes_b,
                'duration': duration_b
            }

        # Mixed
        mixed_notes = (notes_a or []) + (notes_b or [])
        cache_key_mixed = f"mixed_{lm.id}_{hash(str(mixed_notes) + str(auditor.ticks_per_beat) + str(auditor._tempo))}"
        if cache_key_mixed not in st.session_state.audio_cache:
            wav_bytes_mixed = render_mixed_audio(notes_a, notes_b, auditor.ticks_per_beat, auditor._tempo)
            # Calculate duration from audio bytes
            duration_mixed = calculate_audio_duration(wav_bytes_mixed)
            # Store both audio and duration in cache
            st.session_state.audio_cache[cache_key_mixed] = {
                'audio_bytes': wav_bytes_mixed,
                'duration': duration_mixed
            }

def calculate_audio_duration(audio_bytes):
    """Calculate duration from WAV audio bytes."""
    if not audio_bytes:
        return 0.1

    try:
        import soundfile as sf
        import io

        # Read WAV data to get duration
        buffer = io.BytesIO(audio_bytes)
        data, samplerate = sf.read(buffer)
        return len(data) / samplerate
    except Exception as e:
        # Fallback to default duration if calculation fails
        return 0.1

def render_segment_audio(notes, ticks_per_beat, tempo):
    """Render notes as WAV audio bytes using pretty_midi and soundfile"""
    if not notes:
        return b''

    try:
        # Import pretty_midi and soundfile locally to avoid issues
        import pretty_midi
        import soundfile as sf

        # Create pretty_midi object
        pm = pretty_midi.PrettyMIDI(resolution=ticks_per_beat)

        # Create instrument (default piano)
        instrument = pretty_midi.Instrument(program=0)

        # Convert notes to pretty_midi format
        for note in notes:
            # Convert tick time to seconds
            start_time = mido.tick2second(note["tick"], ticks_per_beat, tempo)
            end_time = mido.tick2second(note["tick"] + note["duration"], ticks_per_beat, tempo)

            # Create pretty_midi note
            pm_note = pretty_midi.Note(
                velocity=int(note["velocity"]),
                pitch=int(note["pitch"]),
                start=start_time,
                end=end_time
            )
            instrument.notes.append(pm_note)

        pm.instruments.append(instrument)

        # Synthesize audio with sound font if available
        try:
            # Try to use the FluidR3_GM.sf2 sound font if it exists
            sf2_path = "FluidR3_GM.sf2"
            if os.path.exists(sf2_path):
                audio_data = pm.fluidsynth(sf2_path)
            else:
                audio_data = pm.synthesize(fs=44100)
        except Exception as e:
            # Fallback to default synthesis if fluidsynth fails
            audio_data = pm.synthesize(fs=44100)

        # Write to WAV buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, 44100, format='WAV')
        buffer.seek(0)
        return buffer.getvalue()

    except Exception as e:
        st.error(f"Error rendering audio: {e}")
        return b''

def render_mixed_audio(notes_a, notes_b, ticks_per_beat, tempo):
    """Render mixed audio by numerically mixing WAV bytes"""
    if not notes_a and not notes_b:
        return b''

    # Combine notes
    mixed_notes = notes_a + notes_b if notes_a and notes_b else (notes_a or notes_b or [])

    # For now, just render the mixed notes (proper numerical mixing would be more complex)
    return render_segment_audio(mixed_notes, ticks_per_beat, tempo)


# Audio Player Component
def audio_player_component(notes_a, notes_b=None, label="Segment A", ticks_per_beat=480, tempo=500000, match_id=None):
    """Read-only audio player displaying pre-cached audio with unique keys."""
    col1, col2 = st.columns(2) if notes_b else (st.container(), None)

    # Segment A
    with col1:
        st.markdown(f"**🎵 {label}**")
        if notes_a:
            cache_key_a = f"a_{match_id}_{hash(str(notes_a) + str(ticks_per_beat) + str(tempo))}"
            cached_data_a = st.session_state.audio_cache.get(cache_key_a, {})
            wav_bytes_a = cached_data_a.get("audio_bytes")
            duration_a = cached_data_a.get("duration", 0.1)

            if wav_bytes_a:
                with st.container():
                    audix_player(wav_bytes_a, key=f"audio_player_a_{match_id}", sample_rate=44100)
                    st.caption(f"Duration: {duration_a:.1f}s")
            else:
                st.info(f"Duration: {duration_a:.1f}s | No audio data generated.")
        else:
            st.info("No notes to play.")

    # Segment B
    if notes_b and col2:
        with col2:
            st.markdown("**🎵 Segment B**")
            if notes_b:
                cache_key_b = f"b_{match_id}_{hash(str(notes_b) + str(ticks_per_beat) + str(tempo))}"
                cached_data_b = st.session_state.audio_cache.get(cache_key_b, {})
                wav_bytes_b = cached_data_b.get("audio_bytes")
                duration_b = cached_data_b.get("duration", 0.1)

                if wav_bytes_b:
                    with st.container():
                        audix_player(wav_bytes_b, key=f"audio_player_b_{match_id}", sample_rate=44100)
                        st.caption(f"Duration: {duration_b:.1f}s")
                else:
                    st.info(f"Duration: {duration_b:.1f}s | No audio data generated.")
            else:
                st.info("No notes to play.")

    # Mixed Playback
    if notes_b:
        st.markdown("**🎼 Mixed Playback**")
        mixed_notes = (notes_a or []) + (notes_b or [])
        if mixed_notes:
            cache_key_mixed = f"mixed_{match_id}_{hash(str(mixed_notes) + str(ticks_per_beat) + str(tempo))}"
            cached_data_mixed = st.session_state.audio_cache.get(cache_key_mixed, {})
            wav_bytes_mixed = cached_data_mixed.get("audio_bytes")
            duration_mixed = cached_data_mixed.get("duration", 0.1)

            if wav_bytes_mixed:
                with st.container():
                    audix_player(wav_bytes_mixed, key=f"audio_player_mixed_{match_id}", sample_rate=44100)
                    st.caption(f"Duration: {duration_mixed:.1f}s")
            else:
                st.info(f"Duration: {duration_mixed:.1f}s | No audio data generated.")
        else:
            st.info("No notes to mix.")


# ================================================================
# STREAMLIT UI
# ================================================================

st.set_page_config(page_title="MIDI Structural Auditor", layout="wide", page_icon="🎼")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5em;
        margin-bottom: 1em;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1em;
        border-radius: 0.5em;
        margin: 0.5em 0;
    }
    .tab-content {
        padding: 1em 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎼 MIDI Structural Auditor</h1>', unsafe_allow_html=True)
st.markdown("*Advanced musical structure analysis for MIDI files*")

# File upload with better UX
col1, col2 = st.columns([1, 2])
with col1:
    uploaded_file = st.file_uploader("Upload MIDI file", type=["mid", "midi"])
with col2:
    if uploaded_file:
        st.success(f"📁 **{uploaded_file.name}** uploaded successfully!")
        st.info(f"File size: {len(uploaded_file.getvalue()) / 1024:.1f} KB")
    else:
        st.info("Upload a MIDI file to begin analysis")

if uploaded_file:
    midi_stream = io.BytesIO(uploaded_file.getvalue())

    # Sidebar configuration with better organization
    st.sidebar.title("⚙️ Analysis Settings")

    with st.sidebar.expander("🔍 Detection Parameters", expanded=True):
        st.markdown("**Similarity Thresholds**")
        large_sim = st.slider(
            "Large Section Similarity",
            0.70, 0.98, 0.90, 0.02,
            help="Minimum similarity for large-scale repeats (higher = stricter matching)"
        )
        motif_sim = st.slider(
            "Motif Similarity (unused)",
            0.60, 0.98, 0.70, 0.02,
            help="Currently using exact matching for motifs"
        )

        st.markdown("**Size Thresholds**")
        min_large_bars = st.slider(
            "Min Section Length (bars)",
            2, 16, 4,
            help="Minimum length for large-scale sections"
        )
        min_motif_notes = st.slider(
            "Min Motif Length (notes)",
            3, 16, 4,
            help="Minimum notes per motif pattern"
        )

    with st.sidebar.expander("🔧 Advanced Options"):
        allow_overlaps = st.checkbox(
            "Allow overlapping matches",
            value=False,
            help="Show all matches even if they overlap (may show redundant structure)"
        )

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("🔄 Initializing analysis...")
    progress_bar.progress(10)

    # Create auditor
    auditor = MIDIAuditor(midi_stream, large_similarity=large_sim, motif_similarity=motif_sim)
    progress_bar.progress(30)

    status_text.text("📊 Building features...")
    progress_bar.progress(50)

    status_text.text("🔍 Finding patterns...")
    # Run analysis
    auditor.occupied_indices = set()  # RESET
    large_matches, motif_matches = auditor.find_all_patterns(
        min_motif_length=min_motif_notes,
        min_large_bars=min_large_bars,
        max_results=100,
        allow_overlaps=allow_overlaps
    )
    motif_matches = auditor.postprocess_motifs(motif_matches)
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")

    # Initialize audio cache safely after analysis
    init_audio_cache()

    # Pre-generate all audio during initialization phase
    initialize_audio_cache_for_analysis(auditor, large_matches, motif_matches)

    # Prepare data
    sections = auditor.label_sections(large_matches)
    summary_text, summary_lines = auditor.summarize_structure(sections)
    timeline = auditor.visualize_timeline(sections, auditor.num_bars)
    times, densities = auditor.build_timeline()

    # Main content with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔁 Large-Scale Analysis", "🧬 Motif Analysis", "📋 Exports & Logs"])

    with tab1:  # Overview
        st.header("📊 Analysis Overview")

        # Key metrics in a nice layout
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Notes", len(auditor.notes))
        with col2:
            st.metric("Total Bars", auditor.num_bars)
        with col3:
            st.metric("Large Sections", len(large_matches))
        with col4:
            coverage = len(auditor.occupied_indices) / len(auditor.notes) * 100 if auditor.notes else 0
            st.metric("Coverage", ".1f")

        st.success(f"🎯 Found {len(large_matches)} large-scale sections and {len(motif_matches)} motifs")

        # Structural summary
        st.subheader("🎼 Structural Summary")
        st.code(summary_text, language="text")

        # Timeline visualization
        st.subheader("📈 Structure Timeline")
        if sections:
            st.plotly_chart(plot_timeline_with_overlaps(sections, auditor), use_container_width=True)
        else:
            st.info("No sections detected to display in timeline.")

        # Quick timeline plot if data available
        if times and densities:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=times, y=densities,
                mode="lines",
                line=dict(color="#2E86AB"),
                name="Note Density"
            ))
            fig.update_layout(
                title="Overall Note Density Timeline",
                xaxis_title="Time (s)",
                yaxis_title="Notes per 0.1s",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:  # Large-Scale Analysis
        st.header("🔁 Large-Scale Repeats")

        if large_matches:
            # Filter selector
            match_options = {
                f"Match {lm.id}: {lm.length_bars} bars (Similarity: {lm.avg_similarity:.2f})": lm.id
                for lm in large_matches
            }

            selected = st.selectbox(
                "Focus on specific repeat:",
                ["Show all"] + list(match_options.keys()),
                help="Zoom into a specific large-scale repeat"
            )

            large_matches_filtered = large_matches
            zoom_range = None

            if selected != "Show all":
                lm = next(m for m in large_matches if m.id == match_options[selected])
                start_a, end_a = auditor.bar_range_to_seconds(lm.start_bar_a, lm.length_bars)
                start_b, end_b = auditor.bar_range_to_seconds(lm.start_bar_b, lm.length_bars)
                zoom_range = (min(start_a, start_b) - 0.5, max(end_a, end_b) + 0.5)
                large_matches_filtered = [lm]

            # Visualization
            if times and densities:
                st.plotly_chart(
                    plot_large_scale_repeats(times, densities, large_matches_filtered, auditor, zoom_range),
                    use_container_width=True
                )

            # Detailed list
            st.subheader("📋 Detailed Matches")
            for lm in sorted(large_matches_filtered, key=lambda x: x.length_bars, reverse=True):
                with st.expander(f"🔄 Match {lm.id}: {lm.length_bars} bars", expanded=(len(large_matches_filtered) == 1)):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"**Segment A:** Bars {lm.start_bar_a+1}–{lm.start_bar_a+lm.length_bars}")
                        start_a, end_a = auditor.bar_range_to_seconds(lm.start_bar_a, lm.length_bars)
                        st.markdown(f"**Time:** {start_a:.1f}s - {end_a:.1f}s")

                        # Check for overlaps and offer the full overlapping version for copying
                        overlap_start = max(lm.start_bar_a, lm.start_bar_b)
                        overlap_end = min(lm.start_bar_a + lm.length_bars, lm.start_bar_b + lm.length_bars)
                        if overlap_start < overlap_end:
                            overlap_bars = overlap_end - overlap_start
                            st.markdown(f"🎼 *Musically overlaps with Segment B by {overlap_bars} bars (bars {overlap_start+1}-{overlap_end}) - creates layered texture*")
                            # Offer the full segment including overlap for authentic reproduction
                            st.markdown(f"💡 **Copy-ready version (with overlap):** Bars {lm.start_bar_a+1}-{lm.start_bar_a+lm.length_bars} ({lm.length_bars} bars, includes musical layering)")

                    with col2:
                        st.markdown(f"**Segment B:** Bars {lm.start_bar_b+1}–{lm.start_bar_b+lm.length_bars}")
                        start_b, end_b = auditor.bar_range_to_seconds(lm.start_bar_b, lm.length_bars)
                        st.markdown(f"**Time:** {start_b:.1f}s - {end_b:.1f}s")

                        # Check for overlaps and offer the full overlapping version for copying
                        overlap_start = max(lm.start_bar_a, lm.start_bar_b)
                        overlap_end = min(lm.start_bar_a + lm.length_bars, lm.start_bar_b + lm.length_bars)
                        if overlap_start < overlap_end:
                            overlap_bars = overlap_end - overlap_start
                            st.markdown(f"🎼 *Musically overlaps with Segment A by {overlap_bars} bars (bars {overlap_start+1}-{overlap_end}) - creates layered texture*")
                            # Offer the full segment including overlap for authentic reproduction
                            st.markdown(f"💡 **Copy-ready version (with overlap):** Bars {lm.start_bar_b+1}-{lm.start_bar_b+lm.length_bars} ({lm.length_bars} bars, includes musical layering)")

                    st.markdown(f"**Similarity:** {lm.avg_similarity:.3f}")

                    # Piano rolls
                    notes_a = auditor.notes_in_bar_range(lm.start_bar_a, lm.length_bars)
                    notes_b = auditor.notes_in_bar_range(lm.start_bar_b, lm.length_bars)

                    if notes_a and notes_b:
                        colA, colB = st.columns(2)
                        with colA:
                            st.plotly_chart(plot_piano_roll(notes_a, auditor.ticks_per_beat, auditor._tempo, "Segment A"), use_container_width=True)
                        with colB:
                            st.plotly_chart(plot_piano_roll(notes_b, auditor.ticks_per_beat, auditor._tempo, "Segment B"), use_container_width=True)

                        # Server-rendered audio preview
                        audio_player_component(notes_a, notes_b, "Segment A", auditor.ticks_per_beat, auditor._tempo, lm.id)

                        # Downloads
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.download_button(
                                "Download Segment A (MIDI)",
                                auditor.export_segment_as_midi(notes_a),
                                f"large_{lm.id}_A.mid",
                                "audio/midi"
                            )
                        with col_dl2:
                            st.download_button(
                                "Download Segment B (MIDI)",
                                auditor.export_segment_as_midi(notes_b),
                                f"large_{lm.id}_B.mid",
                                "audio/midi"
                            )
        else:
            st.info("No large-scale repeats detected with current settings.")

    with tab3:  # Motif Analysis
        st.header("🧬 Motif Analysis")

        if motif_matches:
            st.plotly_chart(plot_structure_waveform(times, densities, motif_matches, auditor), use_container_width=True)

            st.subheader("📋 Detected Motifs")
            for m in motif_matches:
                with st.expander(f"🎵 Motif {m.id}: {m.length} notes ({len(m.occurrences)} occurrences)"):
                    best_idx = m.occurrences[0]
                    best_bar = int(auditor.notes[best_idx]["tick"] / auditor.ticks_per_bar) + 1
                    st.markdown(f"**First occurrence:** Bar {best_bar}")

                    other_occurrences = [f"Bar {int(auditor.notes[idx]['tick'] / auditor.ticks_per_bar) + 1}"
                                       for idx in m.occurrences[1:]]
                    if other_occurrences:
                        st.markdown(f"**Other occurrences:** {', '.join(other_occurrences)}")

                    # Motif visualization
                    segment = [auditor.notes[i] for i in range(best_idx, best_idx + m.length)]
                    pitches = [n["pitch"] for n in segment]
                    times_motif = [mido.tick2second(n["tick"], auditor.ticks_per_beat, auditor._tempo) for n in segment]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=times_motif,
                        y=pitches,
                        mode="lines+markers",
                        line=dict(color="#FF6B6B", width=3),
                        marker=dict(size=8, color="#FF6B6B"),
                        name=f"Motif {m.id}"
                    ))
                    fig.update_layout(
                        title=f"Motif {m.id} Pattern",
                        xaxis_title="Time (s)",
                        yaxis_title="Pitch",
                        height=250
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Download
                    st.download_button(
                        "Download Motif (MIDI)",
                        auditor.export_segment_as_midi(segment),
                        f"motif_{m.id}.mid",
                        "audio/midi"
                    )
        else:
            st.info("No motifs detected with current settings.")

    with tab4:  # Exports & Logs
        st.header("📋 Exports & Technical Details")

        # Export section
        st.subheader("📤 Export Options")
        if large_matches:
            st.download_button(
                "Download DAW Markers (Cubase/Logic)",
                auditor.export_markers_as_midi(large_matches, motif_matches),
                "markers.mid",
                "audio/midi",
                help="Export markers for use in digital audio workstations"
            )

        # Analysis summary for export
        st.subheader("📊 Analysis Summary")
        summary_data = {
            "File": uploaded_file.name,
            "Total Notes": len(auditor.notes),
            "Total Bars": auditor.num_bars,
            "Large Sections Found": len(large_matches),
            "Motifs Found": len(motif_matches),
            "Coverage %": f"{coverage:.1f}",
            "Analysis Timestamp": "2026-01-18"
        }

        st.json(summary_data)

        # Debug logs
        with st.expander("🔧 Debug Logs", expanded=False):
            if auditor.logs:
                st.code("\n".join(auditor.logs), language="text")
            else:
                st.write("No logs available")

    # Footer
    st.markdown("---")
    st.markdown("*Built with ❤️ for musical analysis*")

else:
    # Welcome screen
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

    ### Tips:
    - Works best with multi-track MIDI files
    - Try different similarity thresholds for different levels of detail
    - Use the export features to integrate with DAW software

    *Ready to analyze your music! 🎵*
    """)
