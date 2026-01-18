import streamlit as st
import numpy as np
from streamlit_advanced_audio import audix

# Create simple test audio data
sample_rate = 44100
duration = 2.0
t = np.linspace(0, duration, int(sample_rate * duration), False)
audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

st.title("Simple Audio Test")
st.write("Testing basic audio playback")

# Test with a simple key
audix("test_audio", audio_data, sample_rate=sample_rate)

# Test with multiple keys to simulate the MIDIAuditor scenario
st.write("Testing multiple players")
audix("audio_player_a_1", audio_data, sample_rate=sample_rate)
audix("audio_player_b_1", audio_data, sample_rate=sample_rate)
audix("audio_player_mixed_1", audio_data, sample_rate=sample_rate)
