# MIDI Structural Auditor

A Streamlit web application for advanced musical structure analysis of MIDI files. This tool identifies large-scale repeats (like verse-chorus sections) and motifs (short musical patterns) in MIDI compositions.

## Features

- **Large-scale repeat detection**: Finds thematic sections and structural repeats
- **Motif analysis**: Identifies recurring short musical patterns
- **Interactive visualizations**: Plotly-based charts showing structure and timeline
- **Audio synthesis**: Preview segments using FluidSynth
- **Export capabilities**: Download MIDI segments and DAW markers

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload a MIDI file
2. Adjust analysis parameters in the sidebar
3. Explore results in the tabs:
   - Overview: Summary and timeline
   - Large-Scale Analysis: Detailed repeat sections
   - Motif Analysis: Pattern details
   - Exports: Download options and logs

## Dependencies

- Python 3.8+
- mido: MIDI file handling
- numpy: Numerical computations
- plotly: Interactive visualizations
- streamlit: Web interface
- pretty_midi: MIDI synthesis
- pyfluidsynth: Audio rendering

## Algorithm

Uses a suffix automaton for efficient pattern matching combined with bar-level self-similarity matrices for large-scale structure detection.
