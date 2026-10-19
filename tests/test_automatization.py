"""
run_afm_exploration.py
──────────────────────
Automatic AFM exploration script.

Edit data/config/automation_default.yaml (or your own copy) with the
parameters and pass the file as an argument:

    python run_afm_exploration.py                              # uses automation_default.yaml
    python run_afm_exploration.py my_experiment.yaml          # uses your copy

Press  S + Enter  at any time to stop the session.
Ctrl+C also stops it cleanly.
"""

import sys
import time
import signal
import threading
from pathlib import Path

# ── Add src/ to the path ───────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent   # tests/
_SRC  = _ROOT.parent / "src"              # project_root/src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ── Clean Ctrl+C ─────────────────────────────────────────────────────────────
def _sigint(sig, frame):
    print(f"\n\n\033[91m  Interrupted (Ctrl+C). Sending stop signal…\033[0m")
    try:
        from automation.automation import stop_exploration
        stop_exploration()
    except Exception:
        pass
    sys.exit(130)
signal.signal(signal.SIGINT, _sigint)

# ── Colors ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"; RED    = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; RESET  = "\033[0m";  BOLD   = "\033[1m"; DIM = "\033[2m"

PHASE_COLOR = {
    "idle"          : DIM,
    "connected"     : GREEN,
    "connecting"    : YELLOW,
    "planning"      : YELLOW,
    "scanning_large": CYAN,
    "detecting"     : YELLOW,
    "scanning_zoom" : GREEN,
    "finalising"    : YELLOW,
    "done"          : GREEN,
    "error"         : RED,
}

# Script lives in tests/ — walk up one level to reach the project root
_PROJECT_ROOT = _ROOT.parent
_DEFAULT_YAML = _PROJECT_ROOT / "data" / "config" / "automation_default.yaml"

# ── Load config from YAML ─────────────────────────────────────────────────────
def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _validate(cfg: dict) -> list:
    errors = []
    if not cfg.get("gallery_dir"):
        errors.append("gallery_dir is empty")
    zoom_on = cfg.get("zoom_enabled", True)
    yolo = cfg.get("yolo_model_path", "")
    if zoom_on and not yolo:
        errors.append("yolo_model_path is empty (required when zoom_enabled=true)")
    elif yolo and not Path(yolo).is_file():
        errors.append(f"yolo_model_path not found: {yolo}")
    if zoom_on and not cfg.get("yolo_classes"):
        errors.append("yolo_classes is empty (required when zoom_enabled=true)")
    strategy = cfg.get("strategy", "raster")
    if strategy not in ("raster", "spiral", "probabilistic"):
        errors.append(f"invalid strategy: '{strategy}'")
    overlap = cfg.get("overlap_fraction", 0.1)
    if overlap >= 1.0:
        errors.append(f"overlap_fraction must be < 1.0 (current: {overlap})")
    conf = cfg.get("yolo_conf_threshold", 0.7)
    if not (0.0 < conf <= 1.0):
        errors.append(f"yolo_conf_threshold must be between 0 and 1 (current: {conf})")
    factor = cfg.get("zoom_bbox_factor", 1.5)
    if factor <= 0:
        errors.append(f"zoom_bbox_factor must be > 0 (current: {factor})")
    return errors


# ── Monitor ───────────────────────────────────────────────────────────────────
_stop_monitor = threading.Event()


def _monitor_loop(get_status, poll_s: float = 3.0) -> None:
    last_msg = ""
    while not _stop_monitor.is_set():
        time.sleep(poll_s)
        result = get_status()
        if result["status"] != "ok":
            continue
        d      = result["data"]
        phase  = d.get("phase", "?")
        msg    = d.get("last_message", "")
        lf     = d.get("large_frames", 0)
        zf     = d.get("zoom_frames", 0)
        counts = d.get("counts_per_class", {})
        color  = PHASE_COLOR.get(phase, RESET)
        if msg != last_msg:
            counts_str = "  ".join(f"{c}:{n}" for c, n in counts.items()) if counts else "—"
            print(
                f"  {color}[{phase}]{RESET}  "
                f"{DIM}L:{lf} Z:{zf} | {counts_str}{RESET}  "
                f"{msg[:72]}",
                flush=True,
            )
            last_msg = msg
        if phase in ("done", "error"):
            _stop_monitor.set()
            break


def _keyboard_listener(stop_fn) -> None:
    while not _stop_monitor.is_set():
        try:
            line = input()
        except EOFError:
            break
        if line.strip().lower() in ("s", "stop", "parar", "abort"):
            print(f"\n  {YELLOW}Sending stop signal…{RESET}")
            r = stop_fn()
            print(f"  {r['message']}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Resolve YAML path from argument or default
    if len(sys.argv) > 1:
        yaml_path = Path(sys.argv[1])
        if not yaml_path.is_absolute():
            yaml_path = _ROOT / yaml_path
    else:
        yaml_path = _DEFAULT_YAML

    print(f"\n{BOLD}{CYAN}{'═'*62}")
    print("  SimuScan — Automatic AFM Exploration")
    print(f"{'═'*62}{RESET}\n")

    if not yaml_path.is_file():
        print(f"  {RED}Configuration file not found:{RESET}")
        print(f"  {yaml_path}\n")
        sys.exit(1)

    print(f"  {DIM}Config: {yaml_path}{RESET}\n")
    cfg = _load_yaml(yaml_path)

    # ── Validate ──────────────────────────────────────────────────────────────
    errors = _validate(cfg)
    if errors:
        print(f"{RED}{BOLD}  Configuration errors:{RESET}")
        for e in errors:
            print(f"  {RED}✗  {e}{RESET}")
        print(f"\n  Fix {yaml_path.name} and run again.\n")
        sys.exit(1)

    # ── Print summary ─────────────────────────────────────────────────────────
    ox = cfg.get("origin_x_um", 0.0)
    oy = cfg.get("origin_y_um", 0.0)
    max_d = cfg.get("max_distance_um", 0.0)

    print(f"{BOLD}  Configuration:{RESET}")
    print(f"  {'gallery_dir':<28} {cfg['gallery_dir']}")
    print(f"  {'gwy_base_name':<28} {cfg.get('gwy_base_name','scan')}")
    print(f"  {'yolo_model':<28} {Path(cfg['yolo_model_path']).name}")
    print(f"  {'yolo_classes':<28} {', '.join(cfg['yolo_classes'])}")
    print(f"  {'yolo_conf / max_iou':<28} {cfg.get('yolo_conf_threshold',0.7)} / {cfg.get('yolo_overlap_threshold',0.3)}")
    print(f"  {'large frame':<28} {cfg.get('large_scan_size_um',50.0)} µm  |  {cfg.get('large_scan_rate_hz',1.0)} Hz  |  {cfg.get('large_resolution_px',256)} px")
    print(f"  {'zoom':<28} ×{cfg.get('zoom_bbox_factor',1.5)} bbox  |  {cfg.get('zoom_scan_rate_hz',0.5)} Hz  |  {cfg.get('zoom_resolution_px',512)} px")
    print(f"  {'stop':<28} {cfg.get('stop_n_per_class',20)} zooms/class  |  max {cfg.get('max_large_frames',200)} large frames")
    print(f"  {'strategy':<28} {cfg.get('strategy','raster')}  |  area {cfg.get('area_to_cover_um',200.0)} µm  |  overlap {cfg.get('overlap_fraction',0.1)}")
    print(f"  {'origin (tip position)':<28} ({ox}, {oy}) µm")
    if max_d > 0:
        print(f"  {'max radius from origin':<28} {max_d} µm")

    # ── Import automation ─────────────────────────────────────────────────────
    try:
        from automation.automation import (
            connect_microscope, run_exploration,
            get_exploration_status, stop_exploration,
        )
    except ImportError as e:
        print(f"\n  {RED}Could not import automation.automation:\n  {e}{RESET}\n")
        sys.exit(1)

    # ── Connect ───────────────────────────────────────────────────────────────
    print(f"\n  {YELLOW}Conectando a Nanosurf Studio…{RESET}")
    conn = connect_microscope()
    if conn["status"] != "ok":
        print(f"\n  {RED}✗  Connection failed: {conn['message']}{RESET}")
        print(f"  Make sure Nanosurf Studio is open and logged in.\n")
        sys.exit(1)
    print(f"  {GREEN}✓  Connected.{RESET}")
    print(f"  {DIM}Origin: ({ox}, {oy}) µm{RESET}")

    # ── Confirm ───────────────────────────────────────────────────────────────
    print(f"\n  {DIM}Press  S + Enter  to stop at any time.{RESET}\n")
    input("  Press Enter to start the exploration… ")

    # ── Launch ────────────────────────────────────────────────────────────────
    max_d_m = max_d * 1e-6 if max_d > 0 else None

    result = run_exploration({
        "gallery_dir"           : cfg["gallery_dir"],
        "gwy_base_name"         : cfg.get("gwy_base_name", "scan"),
        "yolo_model_path"       : cfg["yolo_model_path"],
        "yolo_classes"          : cfg["yolo_classes"],
        "large_scan_size_m"     : cfg.get("large_scan_size_um",  50.0) * 1e-6,
        "large_scan_rate_hz"    : cfg.get("large_scan_rate_hz",   1.0),
        "large_resolution_px"   : cfg.get("large_resolution_px", 256),
        "zoom_enabled"          : cfg.get("zoom_enabled",         True),
        "zoom_bbox_factor"      : cfg.get("zoom_bbox_factor",     1.5),
        "zoom_scan_rate_hz"     : cfg.get("zoom_scan_rate_hz",    0.5),
        "zoom_resolution_px"    : cfg.get("zoom_resolution_px",  512),
        "yolo_conf_threshold"   : cfg.get("yolo_conf_threshold",  0.7),
        "yolo_overlap_threshold": cfg.get("yolo_overlap_threshold", 0.3),
        "stop_n_per_class"      : cfg.get("stop_n_per_class",     20),
        "max_large_frames"      : cfg.get("max_large_frames",    200),
        "strategy"              : cfg.get("strategy",        "raster"),
        "area_to_cover_m"       : cfg.get("area_to_cover_um", 200.0) * 1e-6,
        "overlap_fraction"      : cfg.get("overlap_fraction",    0.1),
        "max_distance_m"        : max_d_m,
        "origin_x_m"            : ox * 1e-6,
        "origin_y_m"            : oy * 1e-6,
        "scan_direction"        : cfg.get("scan_direction",  "forward"),
        "gwy_wait_timeout_s"    : cfg.get("gwy_wait_timeout_s",  300.0),
        "gwy_poll_interval_s"   : cfg.get("gwy_poll_interval_s",   2.0),
        "flatten_perc"          : cfg.get("flatten_perc",          40.0),
        "triplet_model_path"    : cfg.get("triplet_model_path",    ""),
        "config_yaml_path"      : str(yaml_path),
    })

    if result["status"] != "ok":
        print(f"\n  {RED}Error al arrancar: {result['message']}{RESET}\n")
        sys.exit(1)

    print(f"\n  {GREEN}Session started.{RESET}\n")

    # ── Monitor + keyboard ────────────────────────────────────────────────────
    threading.Thread(
        target=_monitor_loop,
        args=(get_exploration_status, cfg.get("monitor_poll_s", 3.0)),
        daemon=True,
    ).start()
    threading.Thread(
        target=_keyboard_listener,
        args=(stop_exploration,),
        daemon=True,
    ).start()
    _stop_monitor.wait()

    # ── Final summary ─────────────────────────────────────────────────────────
    final  = get_exploration_status()["data"]
    phase  = final.get("phase", "?")
    counts = final.get("counts_per_class", {})
    out    = final.get("output_dir", "—")
    color  = PHASE_COLOR.get(phase, RESET)
    stop_n = cfg.get("stop_n_per_class", 20)

    print(f"\n{BOLD}{'═'*62}")
    print(f"  FINISHED — {color}{phase.upper()}{RESET}{BOLD}")
    print(f"{'═'*62}{RESET}")
    print(f"  {CYAN}Origin:{RESET}                   ({ox}, {oy}) µm")
    print(f"  {CYAN}Large frames:{RESET}           {final.get('large_frames', 0)}")
    print(f"  {CYAN}Zooms captured:{RESET}           {final.get('zoom_frames', 0)}")
    if counts:
        print(f"  {CYAN}Objects per class:{RESET}")
        for cls, n in counts.items():
            bar = "█" * min(n, 40)
            print(f"    {cls:<22} {GREEN}{bar}{RESET} {n}/{stop_n}")
    print(f"  {CYAN}Output:{RESET}                   {out}")
    if final.get("error"):
        print(f"  {RED}Error:{RESET}                    {final['error']}")
    print(f"{BOLD}{'═'*62}{RESET}\n")


if __name__ == "__main__":
    main()