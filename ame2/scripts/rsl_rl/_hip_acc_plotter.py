#!/usr/bin/env python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Live matplotlib plot of hip-link linear-acceleration norms.

Spawned as a subprocess by ``play.py``. Reads one JSON line per simulation step
from stdin:

    {"t": <float seconds>, "norms": [<float m/s²>, ...]}

Uses the matplotlib WebAgg backend (plot served over HTTP in a browser tab).
Close the figure window to stop; the parent then stops publishing.
"""

import argparse
import json
import math
import os
import sys
import threading
from collections import deque
from queue import Empty, Queue

# WebAgg needs no GUI toolkit (avoids Qt/xcb plugin errors inside Isaac Sim);
# the parent also sets MPLBACKEND=WebAgg, so do not clear it.
import matplotlib  # noqa: E402
matplotlib.use("WebAgg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body_names", required=True, help="Comma-separated body names (plot order).")
    parser.add_argument("--threshold", type=float, default=100.0, help="Reference horizontal line (m/s²).")
    parser.add_argument("--window", type=float, default=10.0, help="Rolling time window shown on the x-axis (s).")
    args = parser.parse_args()

    body_names = [n for n in args.body_names.split(",") if n]
    n = len(body_names)
    if n == 0:
        print("[hip_acc_plotter] no body names supplied — exiting", file=sys.stderr)
        return

    histsize = max(2000, int(args.window * 600) + 200)
    times: deque = deque(maxlen=histsize)
    series = [deque(maxlen=histsize) for _ in range(n)]
    q: Queue = Queue(maxsize=20000)

    # parent sends per-episode time; buffers are cleared on each episode reset
    state = {"last_local_t": None}

    def reader() -> None:
        """Read stdin line-by-line in a thread so matplotlib's GUI loop never blocks on I/O."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                q.put((float(msg["t"]), list(msg["norms"])))
            except Exception:
                continue
        q.put(None)  # EOF sentinel

    threading.Thread(target=reader, daemon=True).start()

    fig, ax = plt.subplots(figsize=(3, 2.5))
    lines = []
    for name in body_names:
        ln, = ax.plot([], [], label=name, linewidth=1.4)
        lines.append(ln)
    ax.axhline(args.threshold, color="red", linestyle="--", alpha=0.4,
               label=f"threshold {args.threshold:.0f} m/s²")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration [m/s²]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    print(f"[hip_acc_plotter] backend={plt.get_backend()}", file=sys.stderr)

    def animate(_frame):
        # drain everything currently available; EOF leaves the figure idle (user closes manually)
        while True:
            try:
                msg = q.get_nowait()
            except Empty:
                break
            if msg is None:
                continue  # parent closed; keep plot visible for inspection
            local_t, norms = msg

            # episode reset: time wraps backwards, so restart the plot from t=0
            if state["last_local_t"] is not None and local_t < state["last_local_t"]:
                times.clear()
                for s in series:
                    s.clear()

            times.append(local_t)
            for s, v in zip(series, norms):
                s.append(float(v))
            state["last_local_t"] = local_t

        if not times:
            return lines

        t_arr = list(times)
        for ln, s in zip(lines, series):
            ln.set_data(t_arr, list(s))

        # x-range: rolling window on per-episode time
        xmax = t_arr[-1]
        xmin = max(0.0, xmax - args.window)
        ax.set_xlim(xmin, xmax + 0.1)

        # y-autoscale over the visible window
        visible_max = 0.0
        tail_len = max(1, int(args.window * 600))
        for s in series:
            if not s:
                continue
            for v in list(s)[-tail_len:]:
                if not math.isnan(v) and v > visible_max:
                    visible_max = v
        ax.set_ylim(0.0, max(visible_max * 1.2, args.threshold * 1.2))
        return lines

    _anim = FuncAnimation(fig, animate, interval=50, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
