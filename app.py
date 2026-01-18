import streamlit as st
import streamlit.components.v1 as components
import io
import plotly.graph_objects as go
import mido
import base64
import json

from midi_auditor import MIDIAuditor
from visualization import (
    plot_structure_waveform,
    plot_large_scale_repeats,
    plot_timeline_with_overlaps,
    plot_piano_roll
)

# Convert MIDI notes to JSON events for browser playback
def notes_to_json_events(notes, tempo=500000):
    """Convert MIDI notes to JSON format for Tone.js playback"""
    import mido

    events = []
    for note in notes:
        # Convert tick time to seconds
        time_seconds = mido.tick2second(note["tick"], 480, tempo)  # Assume 480 ticks per beat
        duration_seconds = mido.tick2second(note["duration"], 480, tempo)

        # Convert MIDI note number to note name (C4, D#4, etc.)
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (note["pitch"] // 12) - 1
        note_name = note_names[note["pitch"] % 12]
        full_note_name = f"{note_name}{octave}"

        events.append({
            "time": time_seconds,
            "duration": duration_seconds,
            "note": full_note_name,
            "velocity": note["velocity"] / 127.0  # Normalize to 0-1
        })

    return json.dumps(events)

# Custom component for browser-based MIDI synthesis using Tone.js
def midi_player_component(notes_a, notes_b=None, label="Play MIDI"):
    """Professional MIDI player using Tone.js and Streamlit components"""

    # Convert notes to JSON for browser
    events_a_json = notes_to_json_events(notes_a)

    if notes_b:
        events_b_json = notes_to_json_events(notes_b)
        # For mixed playback, combine both note sets
        all_notes = notes_a + notes_b
        # Adjust timing so they start at the same time
        events_mixed_json = notes_to_json_events(all_notes)
    else:
        events_b_json = "[]"
        events_mixed_json = events_a_json

    # HTML component using Tone.js
    html_code = f"""
    <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 8px; background: #f9f9f9;">
        <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 10px;">
            <button id="play-a-btn" style="
                background: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
            ">Play {label}</button>
            {"<button id='play-b-btn' style='background: #2196F3; border: none; color: white; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 14px;'>Play Segment B</button>" if notes_b else ""}
            {"<button id='play-mixed-btn' style='background: #FF9800; border: none; color: white; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 14px;'>Play Mixed</button>" if notes_b else ""}
            <button id="stop-btn" style="
                background: #f44336;
                border: none;
                color: white;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                display: none;
            ">Stop</button>
        </div>
        <div id="status" style="font-size: 14px; color: #666;"></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.8.49/Tone.js"></script>
    <script>
    (function() {{
        const playABtn = document.getElementById('play-a-btn');
        const playBBtn = document.getElementById('play-b-btn');
        const playMixedBtn = document.getElementById('play-mixed-btn');
        const stopBtn = document.getElementById('stop-btn');
        const statusDiv = document.getElementById('status');

        // Note events from Python
        const eventsA = {events_a_json};
        const eventsB = {events_b_json};
        const eventsMixed = {events_mixed_json};

        let synth = null;
        let partA = null;
        let partB = null;
        let partMixed = null;
        let isPlaying = false;

        function updateStatus(message, color = '#666') {{
            statusDiv.textContent = message;
            statusDiv.style.color = color;
            console.log('Audio Player:', message);
        }}

        // Initialize synth
        function initSynth() {{
            if (!synth) {{
                synth = new Tone.PolySynth(Tone.Synth, {{
                    oscillator: {{ type: 'sawtooth' }},
                    envelope: {{
                        attack: 0.01,
                        decay: 0.1,
                        sustain: 0.3,
                        release: 0.2
                    }}
                }}).toDestination();
                console.log('Synth initialized');
            }}
        }}

        // Play specific events
        async function playEvents(events, partRef, label) {{
            try {{
                updateStatus('Starting ' + label.toLowerCase() + '...');

                // Start audio context if needed
                if (Tone.context.state !== 'running') {{
                    await Tone.start();
                }}

                initSynth();

                // Dispose existing part
                if (partRef) {{
                    partRef.dispose();
                }}

                // Create new part
                partRef = new Tone.Part((time, note) => {{
                    synth.triggerAttackRelease(note.note, note.duration, time, note.velocity);
                }}, events).start(0);

                updateStatus('Playing ' + label.toLowerCase() + '...');

                // Start transport
                Tone.Transport.start();

                isPlaying = true;
                showStopButton();

                // Auto-stop after playback
                const duration = Math.max(...events.map(e => e.time + e.duration)) + 0.5;
                Tone.Transport.schedule(() => {{
                    stopPlayback();
                }}, duration);

            }} catch (error) {{
                console.error('Error playing ' + label.toLowerCase() + ':', error);
                updateStatus('Error: ' + error.message, 'red');
            }}
        }}

        // Button event listeners
        playABtn.addEventListener('click', () => playEvents(eventsA, partA, 'Segment A'));
        if (playBBtn) {{
            playBBtn.addEventListener('click', () => playEvents(eventsB, partB, 'Segment B'));
        }}
        if (playMixedBtn) {{
            playMixedBtn.addEventListener('click', () => playEvents(eventsMixed, partMixed, 'Mixed'));
        }}

        stopBtn.addEventListener('click', stopPlayback);

        function showStopButton() {{
            [playABtn, playBBtn, playMixedBtn].forEach(btn => {{
                if (btn) btn.style.display = 'none';
            }});
            stopBtn.style.display = 'inline-block';
        }}

        function showPlayButtons() {{
            [playABtn, playBBtn, playMixedBtn].forEach(btn => {{
                if (btn) btn.style.display = 'inline-block';
            }});
            stopBtn.style.display = 'none';
        }}

        function stopPlayback() {{
            Tone.Transport.stop();
            Tone.Transport.cancel();

            // Dispose parts
            [partA, partB, partMixed].forEach(part => {{
                if (part) {{
                    part.dispose();
                    part = null;
                }}
            }});

            isPlaying = false;
            showPlayButtons();
            updateStatus('Stopped');
        }}

        // Cleanup
        window.addEventListener('beforeunload', () => {{
            stopPlayback();
            if (synth) synth.dispose();
        }});

        console.log('Audio player initialized with', eventsA.length, 'A notes', eventsB.length, 'B notes');
    }})();
    </script>
    """

    components.html(html_code, height=120)


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

                        # Browser-based MIDI audio preview
                        col_audio1, col_audio2 = st.columns(2)

                        with col_audio1:
                            st.markdown("**🎵 Segment A Audio Preview**")
                            midi_player_component(notes_a, notes_b, "Segment A")

                        with col_audio2:
                            st.markdown("**🎵 Segment B Audio Preview**")
                            midi_player_component(notes_b, notes_a, "Segment B")

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
