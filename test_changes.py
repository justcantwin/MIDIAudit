import io
from midi_auditor import MIDIAuditor

# Test with one of the sample files
with open('never.mid', 'rb') as f:
    data = f.read()

stream = io.BytesIO(data)
auditor = MIDIAuditor(stream)

print(f"Tempos: {auditor.tempos}")
print(f"Time Signatures: {[(tick, ts.numerator, ts.denominator) for tick, ts in auditor.time_signatures]}")
print(f"Notes extracted: {len(auditor.notes)}")
print(f"Bars: {auditor.num_bars}")
print(f"Features shape: {auditor.bar_features.shape}")
print("Logs:")
for log in auditor.logs:
    print(f"  {log}")

# Test analysis
auditor.occupied_indices = set()  # RESET
large_matches, motif_matches = auditor.find_all_patterns()

print(f"Large matches: {len(large_matches)}")
print(f"Motif matches: {len(motif_matches)}")

print("Additional Logs:")
for log in auditor.logs[-10:]:  # Last 10 logs
    print(f"  {log}")
