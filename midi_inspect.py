import mido

mid = mido.MidiFile('never.mid')
print('Tracks:', len(mid.tracks))
for i, t in enumerate(mid.tracks):
    note_on_count = sum(1 for m in t if m.type == "note_on" and getattr(m, "velocity", 0) > 0)
    note_off_count = sum(1 for m in t if m.type == "note_off")
    note_on_zero = sum(1 for m in t if m.type == "note_on" and getattr(m, "velocity", 0) == 0)
    print(f'Track {i}: {len(t)} messages, {note_on_count} note_ons, {note_off_count} note_offs, {note_on_zero} note_on vel=0')
    # Find first note_on
    for j, msg in enumerate(t):
        if msg.type == "note_on" and msg.velocity > 0:
            print(f'First note_on at index {j}: {msg}')
            break
    print('First 20 msgs:', [str(msg) for msg in t[:20]])
