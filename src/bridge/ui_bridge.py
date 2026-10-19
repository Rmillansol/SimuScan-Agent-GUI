"""
src/bridge/ui_bridge.py
───────────────────────
Single communication object between the agent (background thread)
and the UI (main thread).

Usage in agent/visualizer code:
    from src.bridge.ui_bridge import ui

    ui.log("INFO", "Starting training...")
    ui.log("OK",   "Training complete.")
    ui.log("WARN", "Low GPU memory.")
    ui.log("ERR",  "File not found.")
    ui.log("TOOL", "[train_yolo] {epochs: 100}")

    ui.chat("Response text, **markdown** supported.")

    ui.plot(matplotlib_figure)        # pass a Figure object
    ui.plot_path("/path/to/file.png") # or a saved PNG path

All methods are thread-safe — they use Qt signals to cross
from the agent thread to the main UI thread.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional
from PyQt6.QtCore import QObject, pyqtSignal


# ── Internal signal carrier ───────────────────────────────────────────────────

class _Signals(QObject):
    log_line         = pyqtSignal(str, str)          # (level, text)
    chat_message     = pyqtSignal(str)               # markdown text
    plot_path        = pyqtSignal(str)               # path to PNG
    plot_live        = pyqtSignal(str, str, bool)    # (path, key, focus) — named updating slot (not recorded)
    plot_preview     = pyqtSignal(str)               # path to PNG — appended as a new preview image (capped, not recorded)
    crop_request     = pyqtSignal(str, str, str, str)  # (image_path, out_path, mask_path, label)
    explorer_request = pyqtSignal(str)               # library dir path
    pid_tuner_request = pyqtSignal()                 # open PID tuner window


# ── Public bridge object ──────────────────────────────────────────────────────

class UIBridge:
    """
    Singleton-style bridge. Instantiated once in main_gui.py,
    then imported everywhere via `from src.bridge.ui_bridge import ui`.
    """

    def __init__(self) -> None:
        self._signals    = _Signals()
        self._fig_dir    = self._make_fig_dir()
        self._fig_counter = 0
        self._preview_counter = 0
        self._session_logger = None  # set by agent at session start

        # Callbacks registered by the UI panels (set in main_window.py)
        self._log_cb:  Optional[Callable[[str, str], None]] = None
        self._chat_cb: Optional[Callable[[str], None]]      = None
        self._plot_cb: Optional[Callable[[str], None]]      = None

    @staticmethod
    def _make_fig_dir() -> "Path":
        import tempfile
        from pathlib import Path
        d = Path(tempfile.gettempdir()) / "simuscan_plots"
        d.mkdir(exist_ok=True)
        return d

    # ── Called by main_window to wire up the panels ───────────────────────────

    def connect(
        self,
        log_cb:       Callable[[str, str], None],
        chat_cb:      Callable[[str], None],
        plot_cb:      Callable[[str], None],
        crop_cb:      Optional[Callable[[str, str, str, str], None]] = None,
        explorer_cb:  Optional[Callable[[str], None]] = None,
        pid_tuner_cb: Optional[Callable[[], None]] = None,
        plot_live_cb: Optional[Callable[[str], None]] = None,
        preview_cb:   Optional[Callable[[str], None]] = None,
    ) -> None:
        """Wire the bridge to the UI panel callbacks."""
        self._signals.log_line.connect(log_cb)
        self._signals.chat_message.connect(chat_cb)
        self._signals.plot_path.connect(plot_cb)
        if plot_live_cb is not None:
            self._signals.plot_live.connect(plot_live_cb)
        if preview_cb is not None:
            self._signals.plot_preview.connect(preview_cb)
        if crop_cb is not None:
            self._signals.crop_request.connect(crop_cb)
        if explorer_cb is not None:
            self._signals.explorer_request.connect(explorer_cb)
        if pid_tuner_cb is not None:
            self._signals.pid_tuner_request.connect(pid_tuner_cb)

    # ── Session logger (set by agent at session start/end) ────────────────────

    def set_session_logger(self, logger) -> None:
        """Attach or detach the ChatLogger for the current session."""
        self._session_logger = logger

    # ── Agent-facing API ──────────────────────────────────────────────────────

    def log(self, level: str, text: str) -> None:
        """
        Send a line to the Log panel and persist it to session.log.
        level: INFO | OK | WARN | ERR | TOOL | SYS
        """
        lvl = level.upper()
        self._signals.log_line.emit(lvl, str(text))
        if self._session_logger is not None:
            try:
                self._session_logger.log_line(lvl, text)
            except Exception:
                pass

    def chat(self, text: str) -> None:
        """Send an agent reply (Markdown) to the Chat panel."""
        self._signals.chat_message.emit(str(text))

    def plot(self, fig) -> None:
        """
        Save a matplotlib Figure to a PNG, send it to the Plot panel,
        and copy it into the session plots/ folder.
        """
        try:
            import matplotlib.pyplot as plt
            from datetime import datetime
            self._fig_counter += 1
            ts   = datetime.now().strftime("%H%M%S_%f")
            path = str(self._fig_dir / f"fig_{ts}_{self._fig_counter}.png")
            fig.savefig(path, dpi=120, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            self._signals.plot_path.emit(path)
            if self._session_logger is not None:
                try:
                    self._session_logger.record_figure(path)
                except Exception:
                    pass
        except Exception as e:
            self.log("ERR", f"Could not save figure: {e}")

    def plot_live(self, fig, key: str = "training", focus: bool = False) -> None:
        """
        Update a NAMED 'live' plot slot in the Plot panel WITHOUT recording it to
        the session gallery. Each `key` has its own reusable slot and its own fixed
        PNG, so independent live streams (e.g. 'training' loss curves and 'preview'
        synthetic images) never overwrite each other, and none grows the gallery.

        focus=True forces the view to jump to this slot (use for user-requested
        content such as previews). focus=False follows the slot only until the user
        navigates away (use for background streams like the live training curve).
        """
        try:
            import matplotlib.pyplot as plt
            safe_key = "".join(c if (c.isalnum() or c in "-_") else "_"
                               for c in str(key)) or "live"
            path = str(self._fig_dir / f"live_{safe_key}.png")
            fig.savefig(path, dpi=110, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            self._signals.plot_live.emit(path, safe_key, bool(focus))
        except Exception as e:
            self.log("ERR", f"Could not update live plot: {e}")

    def plot_preview(self, fig) -> None:
        """
        Append a synthetic-preview image as a NEW slot in the Plot panel and jump
        to it, so successive previews can be compared with Prev/Next. The panel
        keeps only the most recent few previews (older ones are evicted). Uses a
        unique filename per preview (they coexist in the carousel) and does NOT
        record to the session gallery.
        """
        try:
            import matplotlib.pyplot as plt
            self._preview_counter += 1
            path = str(self._fig_dir / f"preview_{self._preview_counter}.png")
            fig.savefig(path, dpi=110, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            self._signals.plot_preview.emit(path)
        except Exception as e:
            self.log("ERR", f"Could not show preview: {e}")

    def open_crop_dialog(self, image_path: str, out_path: str, mask_path: str, label: str) -> None:
        """Ask the main thread to open the interactive crop dialog."""
        self._signals.crop_request.emit(image_path, out_path, mask_path, label)

    def open_pid_tuner(self) -> None:
        """Ask the main thread to open the PID tuner window."""
        self._signals.pid_tuner_request.emit()

    def open_explorer_dialog(self, lib_dir: str) -> None:
        """Ask the main thread to open the UMAP explorer (blocks until closed)."""
        self._explorer_done = threading.Event()
        self._signals.explorer_request.emit(lib_dir)
        self._explorer_done.wait()

    def _explorer_closed(self) -> None:
        """Called by the main thread after the explorer dialog closes."""
        if hasattr(self, "_explorer_done"):
            self._explorer_done.set()

    def plot_path(self, path: str) -> None:
        """Send an existing PNG path directly to the Plot panel and record it."""
        self._signals.plot_path.emit(str(path))
        if self._session_logger is not None:
            try:
                self._session_logger.record_image_path(path)
            except Exception:
                pass

    def set_agent_queue(self, queue) -> None:
            """Called by main_window when the agent starts."""
            self._agent_queue = queue

    def inject_to_agent(self, text: str) -> None:
        """Inject a message into the agent loop as if the user typed it."""
        if hasattr(self, '_agent_queue') and self._agent_queue is not None:
            self._agent_queue.put(text)

# ── Global singleton ──────────────────────────────────────────────────────────
# Import this everywhere:  from src.bridge.ui_bridge import ui

ui = UIBridge()
