#!/usr/bin/env python3
"""Walk-forward pipeline monitor dashboard.

Usage:
    python3 scripts/monitor_wf.py [--pid PID] [--log /tmp/wf_run.log] [--refresh 5]

Reads the log file + process stats and renders a live Rich dashboard.
Press Ctrl-C to exit.
"""

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich import box

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FOLDS       = list(range(2013, 2022))          # 9 folds
MODELS      = ["ridge", "lasso", "logistic", "rf", "xgb", "lgbm", "ensemble"]
TRACKS      = ["track_b", "track_a"]
N_FOLDS     = len(FOLDS)
N_TRACKS    = len(TRACKS)
TOTAL_JOBS  = N_FOLDS * N_TRACKS               # 18 fold-track pairs

# Rough per-fold time in seconds (calibrated from observed ~5-6 min/fold)
SECS_PER_FOLD = 330

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_log(log_path: str) -> dict:
    """Extract structured data from the log file."""
    state = {
        "lines": [],
        "folds_done": [],          # list of fold_id strings completed
        "current_fold": None,
        "current_track": None,
        "ic_table": {},            # {(track, fold): {model: ic}}
        "track_b_done": False,
        "track_a_done": False,
        "finished": False,
        "saved_path": None,
        "errors": [],
    }
    try:
        text = Path(log_path).read_text(errors="replace")
    except FileNotFoundError:
        return state

    state["lines"] = text.splitlines()

    # Current track
    for line in state["lines"]:
        if "track_b" in line.lower() and "target" in line.lower():
            state["current_track"] = "track_b"
        if "track_a" in line.lower() and "target" in line.lower():
            state["current_track"] = "track_a"

    # Current / completed folds
    fold_pattern = re.compile(r"Fold\s+(\d{4})\s+train=")
    ic_pattern   = re.compile(r"IC\s+(\w+)\s*=\s*([+-]?\d+\.\d+)")
    saved_pattern = re.compile(r"Saved\s+→\s+(\S+ic_by_fold\S+)")
    track_pattern = re.compile(r"Track:\s+(TRACK_[AB]|track_[ab])", re.IGNORECASE)

    current_fold = None
    current_track = "track_b"

    for line in state["lines"]:
        t_m = track_pattern.search(line)
        if t_m:
            raw = t_m.group(1).lower()
            current_track = "track_b" if "b" in raw else "track_a"

        f_m = fold_pattern.search(line)
        if f_m:
            current_fold = f_m.group(1)
            state["current_fold"] = current_fold

        ic_m = ic_pattern.search(line)
        if ic_m and current_fold:
            model_name = ic_m.group(1)
            ic_val     = float(ic_m.group(2))
            key = (current_track, current_fold)
            state["ic_table"].setdefault(key, {})[model_name] = ic_val

        if saved_pattern.search(line):
            state["saved_path"] = saved_pattern.search(line).group(1)
            state["finished"] = True

        if "Total elapsed" in line:
            state["finished"] = True

        if "Track B done" in line or "track_b done" in line.lower():
            state["track_b_done"] = True
        if "Track A done" in line or "track_a done" in line.lower():
            state["track_a_done"] = True

        if "Traceback" in line or "Error" in line:
            state["errors"].append(line.strip())

    # Count completed folds
    completed = set()
    for (track, fold) in state["ic_table"]:
        if "ensemble" in state["ic_table"][(track, fold)]:
            completed.add((track, fold))
    state["folds_done"] = sorted(completed)

    return state


def get_proc_stats(pid: int) -> dict:
    """Get process CPU / memory via ps."""
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "pid=,etime=,pcpu=,pmem=,rss="],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        parts = out.split()
        if len(parts) < 5:
            return {}
        return {
            "pid":     parts[0],
            "elapsed": parts[1],
            "cpu":     float(parts[2]),
            "mem_pct": float(parts[3]),
            "rss_mb":  int(parts[4]) / 1024,
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def ic_color(val: float) -> str:
    if val >= 0.04:  return "bold green"
    if val >= 0.02:  return "green"
    if val >= 0.00:  return "yellow"
    return "red"


def build_ic_table(ic_table: dict, track: str) -> Table:
    t = Table(
        title=f"IC by Fold — {track.upper().replace('_', ' ')}",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold cyan",
        title_style="bold white",
        min_width=70,
    )
    t.add_column("Fold", style="bold white", width=6)
    for m in MODELS:
        t.add_column(m, justify="right", width=9)

    folds_with_data = sorted({f for (tr, f) in ic_table if tr == track})
    for fold in folds_with_data:
        row = [fold]
        data = ic_table.get((track, fold), {})
        for m in MODELS:
            v = data.get(m)
            if v is None:
                row.append(Text("—", style="dim"))
            else:
                row.append(Text(f"{v:+.4f}", style=ic_color(v)))
        t.add_row(*row)

    if not folds_with_data:
        t.add_row(*["—"] * (len(MODELS) + 1))

    return t


def build_summary_table(ic_table: dict) -> Table:
    t = Table(
        title="Mean IC across completed folds",
        box=box.SIMPLE_HEAVY,
        header_style="bold magenta",
        title_style="bold white",
        min_width=55,
    )
    t.add_column("Model", style="bold white", width=12)
    t.add_column("Track B", justify="right", width=10)
    t.add_column("Track A", justify="right", width=10)

    import numpy as np
    for m in MODELS:
        vals = {}
        for track in TRACKS:
            fold_ics = [
                ic_table[(tr, f)][m]
                for (tr, f) in ic_table
                if tr == track and m in ic_table[(tr, f)]
            ]
            vals[track] = np.mean(fold_ics) if fold_ics else None

        row = [m]
        for track in TRACKS:
            v = vals[track]
            row.append(
                Text(f"{v:+.4f}", style=ic_color(v))
                if v is not None else Text("—", style="dim")
            )
        t.add_row(*row)
    return t


def build_layout(state: dict, proc: dict, pid: int, log_path: str) -> Layout:
    import numpy as np

    # --- Header ---
    n_done = len(state["folds_done"])
    pct    = n_done / TOTAL_JOBS * 100
    bar_filled = int(pct / 5)
    bar    = "█" * bar_filled + "░" * (20 - bar_filled)

    elapsed_s = None
    eta_str   = "estimating…"
    if proc:
        # Parse elapsed time from ps output (format: [[dd-]hh:]mm:ss)
        e = proc["elapsed"]
        parts = re.split(r"[-:]", e)
        secs_list = [int(p) for p in parts]
        multipliers = [1, 60, 3600, 86400]
        elapsed_s = sum(s * m for s, m in zip(reversed(secs_list), multipliers))
        if n_done > 0:
            secs_per_job = elapsed_s / n_done
            remaining = (TOTAL_JOBS - n_done) * secs_per_job
            eta_dt = datetime.now() + timedelta(seconds=remaining)
            eta_str = eta_dt.strftime("%H:%M:%S")
        elif elapsed_s > 0:
            # No folds done yet — estimate based on typical per-fold time
            remaining = TOTAL_JOBS * SECS_PER_FOLD - elapsed_s
            if remaining > 0:
                eta_dt = datetime.now() + timedelta(seconds=remaining)
                eta_str = f"~{eta_dt.strftime('%H:%M:%S')}"

    status_color = "green" if not state["errors"] else "red"
    if state["finished"]:
        status_str = "[bold green]COMPLETE ✓[/bold green]"
    elif state["errors"]:
        status_str = "[bold red]ERROR[/bold red]"
    elif proc:
        status_str = f"[bold {status_color}]RUNNING[/bold {status_color}]"
    else:
        status_str = "[bold yellow]PROCESS ENDED[/bold yellow]"

    track_str = (state["current_track"] or "—").upper().replace("_", " ")
    fold_str  = state["current_fold"] or "loading…"

    header_text = (
        f"[bold white]Walk-Forward CV Monitor[/bold white]  │  "
        f"PID [cyan]{pid}[/cyan]  │  Status: {status_str}\n"
        f"[dim]Log:[/dim] [dim]{log_path}[/dim]\n\n"
        f"Progress  [{bar}]  {n_done}/{TOTAL_JOBS} jobs  ({pct:.0f}%)\n"
        f"Elapsed: [cyan]{proc.get('elapsed', '—')}[/cyan]  │  "
        f"ETA: [yellow]{eta_str}[/yellow]  │  "
        f"CPU: [green]{proc.get('cpu', 0):.0f}%[/green]  │  "
        f"RAM: [green]{proc.get('rss_mb', 0):.0f} MB[/green]\n"
        f"Current: Track [magenta]{track_str}[/magenta]  │  Fold [magenta]{fold_str}[/magenta]"
    )

    # --- Errors panel ---
    err_text = ""
    if state["errors"]:
        err_text = "\n".join(state["errors"][-3:])

    # --- Recent log lines ---
    clean_lines = [
        l for l in state["lines"]
        if not any(x in l for x in [
            "Warning", "RuntimeWarning", "LinAlgWarning",
            "site-packages", "return f(", "nanvar", "nanmean",
        ])
    ]
    recent = "\n".join(clean_lines[-12:]) if clean_lines else "[dim]waiting for output…[/dim]"

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=7),
        Layout(name="body"),
        Layout(name="log", size=16),
    )
    layout["body"].split_row(
        Layout(name="ic_b"),
        Layout(name="ic_a"),
        Layout(name="summary", ratio=1),
    )

    layout["header"].update(Panel(header_text, title="[bold cyan]ML Paper — Week 4[/bold cyan]", border_style="cyan"))
    layout["ic_b"].update(Panel(build_ic_table(state["ic_table"], "track_b"), border_style="blue"))
    layout["ic_a"].update(Panel(build_ic_table(state["ic_table"], "track_a"), border_style="green"))
    layout["summary"].update(Panel(build_summary_table(state["ic_table"]), border_style="magenta"))
    layout["log"].update(Panel(
        (f"[bold red]ERRORS:[/bold red]\n{err_text}\n\n" if err_text else "") + recent,
        title="[bold]Recent Log[/bold]",
        border_style="dim",
    ))

    return layout


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid",     type=int,  default=17010)
    parser.add_argument("--log",     type=str,  default="/tmp/wf_run.log")
    parser.add_argument("--refresh", type=int,  default=5, help="Refresh interval (s)")
    args = parser.parse_args()

    console = Console()

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            state = parse_log(args.log)
            proc  = get_proc_stats(args.pid)

            layout = build_layout(state, proc, args.pid, args.log)
            live.update(layout)

            if state["finished"] and not proc:
                # Final render then exit
                time.sleep(2)
                break

            time.sleep(args.refresh)

    # Print final summary outside of Live
    console.print()
    console.print("[bold green]Pipeline finished.[/bold green]")
    if state.get("saved_path"):
        console.print(f"Output: [cyan]{state['saved_path']}[/cyan]")


if __name__ == "__main__":
    main()
