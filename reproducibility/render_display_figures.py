"""Render the article figures from the aggregate tables.

Presentation only: every plotted coordinate is read from the CSV files in tables/.
No timing value is recomputed. Figure 1 is a schematic drawn from the Methods text.

Usage (from the repository root):
    python reproducibility/render_display_figures.py --tables tables --out new_figures
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
POLICY = {'fifo': 'FIFO', 'legacy': 'Historical DATS', 'latest': 'Latest-pending',
          'drop_newest': 'Drop-newest', 'periodic': 'Periodic'}
COLOR = {'fifo': '#4D4D4D', 'legacy': '#A6761D', 'latest': '#0072B2', 'drop_newest': '#D55E00', 'periodic': '#009E73'}
MARK = {'fifo': 'D', 'legacy': 'v', 'latest': 'o', 'drop_newest': 's', 'periodic': '^'}


def save(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.png'), dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def box(ax, x, y, w, h, title, lines, fc='#F4F6F8', ec='#4D4D4D', tc='black', dy=0.105):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.012', fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h - 0.022, title, ha='center', va='top', fontsize=8.2, fontweight='bold', color=tc)
    ax.text(x + w / 2, y + h - dy, '\n'.join(lines), ha='center', va='top', fontsize=7.2, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, **kw):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', lw=1.0, color=kw.get('color', '#333333'), ls=kw.get('ls', '-'),
                                shrinkA=0, shrinkB=0))


def schematic(out):
    """Figure 1: benchmark pipeline (a) and the four admission policies (b)."""
    fig = plt.figure(figsize=(7.2, 5.0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(0.005, 0.975, 'a', fontsize=10, fontweight='bold', va='top')
    ax.text(0.005, 0.385, 'b', fontsize=10, fontweight='bold', va='top')
    # panel a: pipeline, left to right
    y, h = 0.70, 0.235
    xs = [0.03, 0.225, 0.42, 0.615, 0.81]; w = 0.16
    box(ax, xs[0], y, w, h, 'Prerecorded\nsources',
        ['Cows2021 clip', 'June and November', 'NVR recordings', 'frames decoded,', 'resized, held in RAM'])
    box(ax, xs[1], y, w, h, 'Scheduled\nreplay clock',
        ['artificial 30 or 80', 'arrivals/s', 'or native 15 or 30', 'arrivals/s per camera'])
    box(ax, xs[2], y, w, h, 'Frame\nadmission',
        ['one of four', 'policies (b)', 'discards are', 'logged with reason'], fc='#E8F1F8', ec='#0072B2')
    box(ax, xs[3], y, w, h, 'Single worker\non shared GPU',
        ['YOLOv8 detection', '+ centroid/boundary', 'proxy, one frame', 'at a time'])
    box(ax, xs[4], y, w, h, 'Timing\nrecords',
        ['arrival, service', 'start, completion', 'per delivered frame'])
    for i in range(4):
        arrow(ax, xs[i] + w + 0.008, y + h / 2, xs[i + 1] - 0.008, y + h / 2)
    # co-tenant below the worker
    box(ax, xs[3], 0.455, w, 0.15, 'Co-tenant process',
        ['matrix multiplication', 'or detector training'], fc='#FBEDE3', ec='#D55E00', dy=0.062)
    arrow(ax, xs[3] + w / 2, 0.612, xs[3] + w / 2, y - 0.008, color='#D55E00')
    ax.text(xs[3] + w / 2 + 0.012, 0.652, 'same GPU', fontsize=7, color='#D55E00', va='center')
    # discarded frames below admission
    arrow(ax, xs[2] + w / 2, y - 0.008, xs[2] + w / 2, 0.56, color='#0072B2', ls='--')
    ax.text(xs[2] + w / 2, 0.545, 'discarded frames', fontsize=7.2, color='#0072B2', ha='center', va='top')
    # endpoints below the records
    box(ax, xs[4], 0.455, w, 0.15, 'Endpoints',
        ['delivered latency,', 'information age,', 'frame loss'], fc='#EAF5EF', ec='#009E73', dy=0.058)
    arrow(ax, xs[4] + w / 2, y - 0.008, xs[4] + w / 2, 0.612, color='#009E73')
    # panel b: policies
    rows = [('fifo', 'FIFO', 'Unbounded queue. Every arriving frame is enqueued and served in arrival order.'),
            ('latest', 'Latest-pending', 'One pending slot. An arriving frame replaces the waiting frame; the frame in service finishes.'),
            ('drop_newest', 'Drop-newest', 'One pending slot. An arriving frame is rejected while a frame is already waiting.'),
            ('periodic', 'Periodic', 'A fixed fraction of arrivals, spread evenly by index, is admitted to a FIFO queue.')]
    y0, rh = 0.295, 0.072
    for k, (key, name, text) in enumerate(rows):
        yy = y0 - k * rh
        ax.add_patch(FancyBboxPatch((0.03, yy - 0.025), 0.94, 0.056, boxstyle='round,pad=0.004,rounding_size=0.01',
                                    fc='white', ec=COLOR[key], lw=0.9))
        ax.plot([0.06], [yy + 0.003], marker=MARK[key], color=COLOR[key], ms=6, mfc=COLOR[key])
        ax.text(0.08, yy + 0.003, name, fontsize=8.2, fontweight='bold', va='center', color=COLOR[key])
        ax.text(0.235, yy + 0.003, text, fontsize=7.4, va='center')
    ax.text(0.03, 0.345, 'Admission policies compared; the multi-camera runs keep one pending slot per camera.',
            fontsize=7.4, va='center', style='italic')
    save(fig, out / 'Figure1')


def repeated(tables, out):
    """Figure 2: all 72 repeated arms; shared logarithmic axis."""
    d = pd.read_csv(tables / 'repeated_all_arms.csv'); assert len(d) == 72
    names = {0: 'Cows2021', 1: 'June NVR', 2: 'November NVR'}; order = ['fifo', 'latest', 'drop_newest', 'periodic']
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharey=True)
    for r, fps in enumerate([80, 30]):
        for c, src in enumerate([0, 1, 2]):
            ax = axes[r, c]; g = d[(d.fps == fps) & (d.source == src)]
            for k, pol in enumerate(order):
                v = g[g.policy == pol].sort_values('replicate').p99_ms.values; assert len(v) == 3
                ax.scatter(k + np.array([-.16, 0, .16]), v, s=14, color=COLOR[pol], marker=MARK[pol], zorder=3)
                ax.hlines(v.mean(), k - .3, k + .3, color=COLOR[pol], lw=1.2)
            ax.axhline(33, ls='--', lw=.6, color='0.45'); ax.set_yscale('log'); ax.grid(axis='y', alpha=.2)
            ax.set_xticks(range(4), ['FIFO', 'Latest-\npending', 'Drop-\nnewest', 'Periodic'], fontsize=7.5)
            ax.set_title(f'{names[src]}, {fps} arrivals/s', fontsize=8.5, loc='center')
            ax.text(-.02, 1.04, 'abcdef'[r * 3 + c], transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='right')
        axes[r, 0].set_ylabel('Delivered latency p99 (ms)')
    fig.tight_layout(); save(fig, out / 'Figure2')


def grid(tables, out):
    """Figure 3: admission-fraction grid; final discard percentage on the horizontal axis."""
    d = pd.read_csv(tables / 'admission_grid_all_arms.csv'); assert len(d) == 36
    style = {'latest': dict(marker='o', mfc='none', ms=9), 'drop_newest': dict(marker='s', mfc='none', ms=7),
             'periodic': dict(marker='^', ms=4.5)}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, metric, lab, s in zip(axes, ['p99_ms', 'mean_aoi_ms'], ['Delivered latency p99 (ms)', 'Mean information age (ms)'], 'ab'):
        for pol in ['latest', 'drop_newest', 'periodic']:
            g = d[d.policy == pol]
            ax.scatter(g.drop_pct, g[metric], s=6, color=COLOR[pol], alpha=.35, zorder=2)
            m = g.groupby('admission_fraction')[['drop_pct', metric]].mean().sort_index()
            ax.plot(m.drop_pct, m[metric], color=COLOR[pol], lw=1.1, label=POLICY[pol], zorder=3, **style[pol])
        ax.set_yscale('log'); ax.set_xlabel('Final discarded arrivals (%)'); ax.set_ylabel(lab); ax.grid(alpha=.2)
        ax.text(-.14, 1.03, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); save(fig, out / 'Figure3')


def earlier(tables, out):
    """Figure 4: initial replication at 80 arrivals/s (a) and the ten-minute alternating-load run (b, c)."""
    d = pd.read_csv(tables / 'initial_replication_all_arms.csv'); d = d[d.fps == 80]
    b = pd.read_csv(tables / 'gpu1_ten_minute_one_second_bins.csv')
    t = pd.read_csv(tables / 'gpu1_extension_gpu_telemetry.csv'); t = t[t.label == 's1_80_latest_burst600']
    assert len(b) == 598 and len(t) > 100
    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1, 1], hspace=0.6)
    ax = fig.add_subplot(gs[0])
    for arm in ['fifo', 'legacy', 'latest']:
        g = d[d.arm == arm].sort_values('rep')
        assert len(g) == 6
        ax.plot(g.rep, g.p99_ms, marker=MARK[arm], color=COLOR[arm], lw=1.3, ms=5, label=POLICY[arm])
    ax.set_yscale('log'); ax.set_xlabel('Counterbalanced repetition'); ax.set_ylabel('Delivered latency p99 (ms)')
    ax.set_xticks(range(1, 7)); ax.grid(axis='y', alpha=.2); ax.legend(frameon=False, loc='center right')
    ax1 = fig.add_subplot(gs[1]); ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax1.plot(b.second, b.mean_aoi_ms, lw=.9, color='#0072B2', label='One-second mean age')
    ax1.plot(b.second, b.p99_ms, lw=.7, color='#E69F00', alpha=.85, label='One-second delivered p99')
    ax1.set_ylabel('Milliseconds'); ax1.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(.5, 1.22), fontsize=8)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.plot(t.elapsed_s, t.power_w, lw=.9, color='#D55E00')
    ax2.set_ylabel('GPU power (W)'); ax2.set_xlabel('Elapsed replay time (s)')
    for a, s in zip([ax, ax1, ax2], 'abc'):
        a.grid(alpha=.2); a.text(-.085, 1.03, s, transform=a.transAxes, fontsize=10, fontweight='bold', va='bottom')
    save(fig, out / 'Figure4')


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--tables', default='tables'); a.add_argument('--out', default='figures'); a = a.parse_args()
    schematic(Path(a.out)); repeated(Path(a.tables), Path(a.out)); grid(Path(a.tables), Path(a.out)); earlier(Path(a.tables), Path(a.out))
    print('rendered')
