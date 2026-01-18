import io
import os
import soundfile as sf
import pretty_midi
import mido

def render_segment_audio(notes, ticks_per_beat, tempo):
    """Render notes as WAV audio bytes using pretty_midi and soundfile"""
    if not notes:
        return b''

    try:
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
        print(f"Error rendering audio: {e}")
        return b''

def render_mixed_audio(notes_a, notes_b, ticks_per_beat, tempo):
    """Render mixed audio by numerically mixing WAV bytes"""
    if not notes_a and not notes_b:
        return b''

    # Combine notes
    mixed_notes = notes_a + notes_b if notes_a and notes_b else (notes_a or notes_b or [])

    # For now, just render the mixed notes (proper numerical mixing would be more complex)
    return render_segment_audio(mixed_notes, ticks_per_beat, tempo)

# Sample notes for testing
sample_notes = [
    {"tick": 0, "duration": 480, "pitch": 60, "velocity": 100},
    {"tick": 480, "duration": 480, "pitch": 64, "velocity": 100},
    {"tick": 960, "duration": 480, "pitch": 67, "velocity": 100},
]

ticks_per_beat = 480
tempo = 500000

print("Testing render_segment_audio...")
wav_bytes = render_segment_audio(sample_notes, ticks_per_beat, tempo)
print(f"Returned type: {type(wav_bytes)}")
print(f"Length: {len(wav_bytes)} bytes")

if wav_bytes:
    # Try to read the WAV bytes
    buffer = io.BytesIO(wav_bytes)
    try:
        data, samplerate = sf.read(buffer)
        print(f"✓ Valid WAV: {len(data)} samples at {samplerate} Hz")
        print(f"Duration: {len(data)/samplerate:.2f}s")
    except Exception as e:
        print(f"✗ Invalid WAV: {e}")
else:
    print("✗ No WAV bytes returned")

print("\nTesting render_mixed_audio...")
wav_bytes_mixed = render_mixed_audio(sample_notes, sample_notes, ticks_per_beat, tempo)
print(f"Returned type: {type(wav_bytes_mixed)}")
print(f"Length: {len(wav_bytes_mixed)} bytes")

if wav_bytes_mixed:
    buffer = io.BytesIO(wav_bytes_mixed)
    try:
        data, samplerate = sf.read(buffer)
        print(f"✓ Valid WAV: {len(data)} samples at {samplerate} Hz")
        print(f"Duration: {len(data)/samplerate:.2f}s")
    except Exception as e:
        print(f"✗ Invalid WAV: {e}")
else:
    print("✗ No WAV bytes returned")
