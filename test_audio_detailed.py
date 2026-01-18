import streamlit as st
import numpy as np
import io
import soundfile as sf
from streamlit_advanced_audio import audix

# Initialize session state for audio cache
if 'audio_cache' not in st.session_state:
    st.session_state.audio_cache = {}

# Create test audio data and cache it
sample_rate = 44100
duration = 2.0
t = np.linspace(0, duration, int(sample_rate * duration), False)
audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

# Convert to bytes
buffer = io.BytesIO()
sf.write(buffer, audio_data, sample_rate, format='WAV')
buffer.seek(0)
wav_bytes = buffer.getvalue()

# Cache the audio data
st.session_state.audio_cache['test_audio'] = {
    'audio_bytes': wav_bytes,
    'duration': duration
}

st.title("Detailed Audio Test")
st.write("Testing audio playback with session state caching")

# Test the audio_player_component pattern from MIDIAuditor
def test_audio_player_component(match_id=1):
    """Test function that mimics the MIDIAuditor audio_player_component"""
    st.subheader(f"Testing Match ID: {match_id}")

    # Segment A
    st.markdown("**🎵 Segment A**")
    cache_key_a = f"a_{match_id}_test"
    cached_data_a = st.session_state.audio_cache.get(cache_key_a, {})
    wav_bytes_a = cached_data_a.get('audio_bytes', wav_bytes)  # Fallback to test audio
    duration_a = cached_data_a.get('duration', duration)

    if wav_bytes_a:
        with st.container():
            try:
                audix(f"audio_player_a_{match_id}", wav_bytes_a, sample_rate=44100)
                st.caption(f"Duration: {duration_a:.1f}s")
            except Exception as e:
                st.error(f"Error with audio_player_a_{match_id}: {e}")

    # Segment B
    st.markdown("**🎵 Segment B**")
    cache_key_b = f"b_{match_id}_test"
    cached_data_b = st.session_state.audio_cache.get(cache_key_b, {})
    wav_bytes_b = cached_data_b.get('audio_bytes', wav_bytes)  # Fallback to test audio
    duration_b = cached_data_b.get('duration', duration)

    if wav_bytes_b:
        with st.container():
            try:
                audix(f"audio_player_b_{match_id}", wav_bytes_b, sample_rate=44100)
                st.caption(f"Duration: {duration_b:.1f}s")
            except Exception as e:
                st.error(f"Error with audio_player_b_{match_id}: {e}")

    # Mixed Playback
    st.markdown("**🎼 Mixed Playback**")
    cache_key_mixed = f"mixed_{match_id}_test"
    cached_data_mixed = st.session_state.audio_cache.get(cache_key_mixed, {})
    wav_bytes_mixed = cached_data_mixed.get('audio_bytes', wav_bytes)  # Fallback to test audio
    duration_mixed = cached_data_mixed.get('duration', duration)

    if wav_bytes_mixed:
        with st.container():
            try:
                audix(f"audio_player_mixed_{match_id}", wav_bytes_mixed, sample_rate=44100)
                st.caption(f"Duration: {duration_mixed:.1f}s")
            except Exception as e:
                st.error(f"Error with audio_player_mixed_{match_id}: {e}")

# Test with multiple match IDs
test_audio_player_component(1)
test_audio_player_component(2)
