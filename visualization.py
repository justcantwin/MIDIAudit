import plotly.graph_objects as go
import mido
from typing import List, Tuple
from models import Match, LargeMatch
from midi_auditor import MIDIAuditor


def plot_structure_waveform(times, densities, matches, auditor):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=times, y=densities,
        mode="lines",
        line=dict(color="lightgray"),
        name="Note Density"
    ))

    colors = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#1A535C", "#FF9F1C", "#9B5DE5", "#00BBF9", "#F15BB5"]
    max_density = max(densities) if densities else 1

    for match in matches:
        occ_secs = auditor.pattern_occurrences_in_seconds(match)
        color = colors[(match.id - 1) % len(colors)]

        for idx, (start, end) in enumerate(occ_secs):
            fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.25, line_width=1, line_color=color, layer="below")
            fig.add_trace(go.Scatter(
                x=[(start + end) / 2],
                y=[max_density * 0.9],
                mode="markers+text",
                marker=dict(color=color, size=8),
                text=[f"M{match.id}-{idx+1}"],
                textposition="top center",
                showlegend=False,
                hovertemplate=f"Motif {match.id} Occurrence {idx+1}<br>{start:.3f}s - {end:.3f}s<br>{match.length} notes<extra></extra>"
            ))
            # Add vertical lines at start and end to clearly mark boundaries
            fig.add_vline(x=start, line_width=1, line_dash="dot", line_color=color, opacity=0.7)
            fig.add_vline(x=end, line_width=1, line_dash="dot", line_color=color, opacity=0.7)

    fig.update_layout(
        title="Motif-Level Timeline",
        xaxis_title="Time (s)",
        yaxis_title="Density",
        height=400
    )
    return fig


def plot_large_scale_repeats(times, densities, large_matches, auditor, zoom_range=None):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=times, y=densities,
        mode="lines",
        line=dict(color="lightgray"),
        name="Note Density"
    ))

    colors = ["#2EC4B6", "#E71D36", "#FF9F1C", "#011627", "#9B5DE5", "#00BBF9", "#F15BB5", "#FF6B6B"]
    max_density = max(densities) if densities else 1

    for lm in large_matches:
        color = colors[(lm.id - 1) % len(colors)]
        start_a, end_a = auditor.bar_range_to_seconds(lm.start_bar_a, lm.length_bars)
        start_b, end_b = auditor.bar_range_to_seconds(lm.start_bar_b, lm.length_bars)

        for (start, end, label, start_bar) in [(start_a, end_a, f"L{lm.id}A", lm.start_bar_a), (start_b, end_b, f"L{lm.id}B", lm.start_bar_b)]:
            end_bar = start_bar + lm.length_bars - 1
            fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.25, line_width=1, line_color=color, layer="below")
            fig.add_trace(go.Scatter(
                x=[(start + end) / 2],
                y=[max_density * 0.9],
                mode="markers+text",
                marker=dict(color=color, size=9),
                text=[label],
                textposition="top center",
                showlegend=False,
                hovertemplate=f"{label}: {start:.3f}s - {end:.3f}s<br>Bars {start_bar+1}-{end_bar+1}<br>Similarity: {lm.avg_similarity:.3f}<extra></extra>"
            ))
            # Add vertical lines at start and end to clearly mark boundaries
            fig.add_vline(x=start, line_width=1, line_dash="dot", line_color=color, opacity=0.7)
            fig.add_vline(x=end, line_width=1, line_dash="dot", line_color=color, opacity=0.7)

    if zoom_range:
        fig.update_xaxes(range=zoom_range)

    fig.update_layout(
        title="Large-Scale Repeats",
        xaxis_title="Time (s)",
        yaxis_title="Density",
        height=400
    )
    return fig


def plot_timeline_with_overlaps(sections, auditor):
    """
    Create a graphical timeline showing sections positioned by their actual start times,
    with overlaps stacked vertically for clarity.
    """
    if not sections:
        return go.Figure()

    fig = go.Figure()

    # Sort sections by start time
    sections_sorted = sorted(sections, key=lambda s: s["start_tick"])

    colors = ["#2EC4B6", "#E71D36", "#FF9F1C", "#011627", "#9B5DE5", "#00BBF9", "#F15BB5", "#FF6B6B"]

    # Track occupied vertical positions to stack overlapping sections
    occupied_ranges = []  # List of (start_time, end_time, y_level)

    for section in sections_sorted:
        start_sec = mido.tick2second(section["start_tick"], auditor.ticks_per_beat, auditor._tempo)
        end_sec = mido.tick2second(section["end_tick"], auditor.ticks_per_beat, auditor._tempo)

        # Find the lowest available Y level that doesn't conflict
        y_level = 0
        while True:
            conflict = False
            for occ_start, occ_end, occ_y in occupied_ranges:
                if occ_y == y_level and (start_sec < occ_end and end_sec > occ_start):
                    conflict = True
                    break
            if not conflict:
                break
            y_level += 1

        # Add this section to occupied ranges
        occupied_ranges.append((start_sec, end_sec, y_level))

        # Plot the section
        color = colors[section["match_id"] % len(colors)]
        label = f"{section['occurrence']}{section['match_id']}"

        fig.add_vrect(
            x0=start_sec, x1=end_sec,
            y0=y_level - 0.4, y1=y_level + 0.4,
            fillcolor=color, opacity=0.7,
            line_width=2, line_color=color,
            annotation_text=label,
            annotation_position="top",
            showlegend=False
        )

        # Add text label at the center
        fig.add_trace(go.Scatter(
            x=[(start_sec + end_sec) / 2],
            y=[y_level],
            mode="text",
            text=[label],
            textposition="middle center",
            showlegend=False,
            hovertemplate=f"{label}: {start_sec:.1f}s - {end_sec:.1f}s<br>Bars {section['start_bar']+1}-{section['end_bar']+1}<extra></extra>"
        ))

    # Set Y axis to show integer levels
    max_y = max((y for _, _, y in occupied_ranges), default=0)
    fig.update_yaxes(
        tickvals=list(range(max_y + 1)),
        ticktext=[f"Layer {i+1}" for i in range(max_y + 1)],
        range=[-0.5, max_y + 0.5]
    )

    fig.update_layout(
        title="Musical Structure Timeline (with overlaps)",
        xaxis_title="Time (s)",
        yaxis_title="Playback Layer",
        height=300,
        showlegend=False
    )

    return fig


def plot_piano_roll(notes, ticks_per_beat, tempo, title):
    if not notes:
        return go.Figure()

    xs = [mido.tick2second(n["tick"], ticks_per_beat, tempo) for n in notes]
    ys = [n["pitch"] for n in notes]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=6, color="cyan")))
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Pitch", height=250)
    return fig
