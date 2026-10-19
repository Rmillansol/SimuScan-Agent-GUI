"""
src/ui/main_window.py
─────────────────────
Main window: layout, menu, agent thread lifecycle.
Wires the UIBridge singleton to the three panels.
"""

from __future__ import annotations
import re
from queue import Queue
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout, QStatusBar, QDialog,
)
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal, QTimer
from PyQt6.QtGui import QAction

from .chat_panel import ChatPanel
from .plot_panel  import PlotPanel
from .log_panel   import LogPanel
from .dialogs.settings_dialog import SettingsDialog, load_settings
from .dialogs.about_dialog    import AboutDialog
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Agent worker ──────────────────────────────────────────────────────────────

class AgentWorker(QObject):
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(self, api_key: str, model: str, input_queue: Queue) -> None:
        super().__init__()
        self.api_key      = api_key
        self.model        = model
        self.input_queue  = input_queue
        self._running     = True

    def stop(self) -> None:
        self._running = False
        self.input_queue.put("exit")

    def run(self) -> None:
        import asyncio
        from src.agent.agent import run_gui as agent_run_gui
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                agent_run_gui(
                    api_key     = self.api_key,
                    model       = self.model,
                    input_queue = self.input_queue,
                )
            )
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            try:
                loop.close()
            except Exception:
                pass
            self.finished.emit()


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SimuScan Agent")
        self.resize(1400, 860)
        self._apply_style()

        # Panels
        self.chat  = ChatPanel()
        self.plot  = PlotPanel()
        self.log   = LogPanel()

        # Layout
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.addWidget(self.chat)
        h_split.addWidget(self.plot)
        h_split.setSizes([560, 700])
        h_split.setHandleWidth(4)

        v_split = QSplitter(Qt.Orientation.Vertical)
        v_split.addWidget(h_split)
        v_split.addWidget(self.log)
        v_split.setSizes([630, 200])
        v_split.setHandleWidth(4)

        root.addWidget(v_split)

        # Status bar
        self._sb = QStatusBar()
        self.setStatusBar(self._sb)
        self._sb.showMessage("Ready — configure API key in ⚙ Settings")

        # Wire UIBridge → panels
        from src.bridge.ui_bridge import ui
        ui.connect(
            log_cb       = self.log.append_line,
            chat_cb      = self.chat.add_agent_message,
            plot_cb      = self.plot.show_image,
            crop_cb      = self._open_crop_dialog,
            explorer_cb  = self._open_explorer_dialog,
            pid_tuner_cb = self._open_pid_tuner_dialog,
            plot_live_cb = self.plot.update_live_plot,
            preview_cb   = self.plot.add_preview_image,
        )

        # Menu
        self._build_menu()

        # Agent state
        self._worker: AgentWorker | None = None
        self._thread: QThread      | None = None
        self._threads: list = []
        self._queue:  Queue = Queue()

        # Chat send → agent
        self.chat.user_message_sent.connect(self._on_user_message)

        # Dev translate panel (Spanish ↔ English) — enabled via env var
        self._dev_panel = None
        import os as _os
        if _os.environ.get("SIMUSCAN_DEV_TRANSLATE", "").lower() in ("1", "true", "yes", "on"):
            QTimer.singleShot(0, self._open_dev_translate_panel)

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # File
        fm = mb.addMenu("📁  File")
        a5 = QAction("Load Conversation…", self)
        a5.triggered.connect(self._load_conversation)
        fm.addAction(a5)
        fm.addSeparator()
        a6 = QAction("Exit", self)
        a6.triggered.connect(self.close)
        fm.addAction(a6)

        # Settings
        sm = mb.addMenu("⚙  Settings")
        a = QAction("API Key && Model…", self)
        a.triggered.connect(self._open_settings)
        sm.addAction(a)
        sm.addSeparator()
        a2 = QAction("Clear Log", self)
        a2.triggered.connect(self.log.clear)
        sm.addAction(a2)
        a3 = QAction("Clear Chat", self)
        a3.triggered.connect(self.chat.clear_history)
        sm.addAction(a3)
        sm.addSeparator()
        a_dev = QAction("🌐 Dev: Spanish ↔ English panel", self)
        a_dev.triggered.connect(self._open_dev_translate_panel)
        sm.addAction(a_dev)

        # About
        am = mb.addMenu("ℹ  About")
        a4 = QAction("About SimuScan…", self)
        a4.triggered.connect(lambda: AboutDialog(self).exec())
        am.addAction(a4)

    def _open_settings(self) -> None:
        if SettingsDialog(self).exec():
            self._sb.showMessage("Settings saved.")

    def _open_dev_translate_panel(self) -> None:
        from .dialogs.dev_translate_panel import DevTranslatePanel
        if self._dev_panel is None:
            self._dev_panel = DevTranslatePanel(self.chat, self)
        self._dev_panel.show()
        self._dev_panel.raise_()
        self._dev_panel.activateWindow()

    def _on_user_message(self, text: str) -> None:
        if self._worker is None:
            s = load_settings()
            key   = s.get("OPENAI_API_KEY", "")
            model = s.get("OPENAI_MODEL", "gpt-5.2")
            if not key:
                self._sb.showMessage("⚠  No API key — open ⚙ Settings first")
                self.chat.add_system_message(
                    "Please set your OpenAI API key in ⚙ Settings before chatting."
                )
                return
            self._start_agent(key, model)
        self._queue.put(text)

    def _start_agent(self, api_key: str, model: str) -> None:
        self._worker = AgentWorker(api_key, model, self._queue)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_agent_done)
        self._worker.error.connect(self._on_agent_error)
        self._threads.append(self._thread)
        self._thread.start()
        self._thread.start()
        from src.bridge.ui_bridge import ui
        ui.set_agent_queue(self._queue)   
        self._sb.showMessage(f"Agent running  [{model}]")

    def _on_agent_done(self) -> None:
        self._sb.showMessage("Agent session ended.")
        self._worker = None
        self._thread = None

    def _on_agent_error(self, msg: str) -> None:
        self._sb.showMessage(f"Agent error: {msg}")
        from src.bridge.ui_bridge import ui
        ui.log("ERR", f"Agent error: {msg}")

    def closeEvent(self, event) -> None:
        if self._worker:
            self._worker.stop()
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            if not self._thread.wait(5000):
                self._thread.terminate()
                self._thread.wait(2000)
        event.accept()

    def _load_conversation(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        import json

        # Let the user pick the session folder directly
        chats_dir = str(_ROOT / "data" / "chats")
        session_dir = QFileDialog.getExistingDirectory(
            self, "Select Session Folder", chats_dir,
        )
        if not session_dir:
            return

        session_path = Path(session_dir)
        conv_file    = session_path / "conversation.json"
        log_file     = session_path / "session.log"
        images_file  = session_path / "images.json"

        if not conv_file.exists():
            self.chat.add_system_message(
                f"No conversation.json found in {session_dir}"
            )
            return

        # ── 1. Chat messages ──────────────────────────────────────────────────
        try:
            with open(conv_file, encoding="utf-8") as f:
                entries = json.load(f)
            self.chat.clear_history()
            self.chat.add_system_message(
                f"Session loaded: **{session_path.name}**"
            )
            for entry in entries:
                role    = entry.get("role", "")
                content = entry.get("content", "")
                if not content:
                    continue
                if role == "user":
                    self.chat.add_user_message(content)
                elif role == "assistant":
                    self.chat.add_agent_message(content)
                elif role == "system" and not content.startswith("[GUI SESSION"):
                    self.chat.add_system_message(content)
        except Exception as e:
            self.chat.add_system_message(f"Could not load chat: {e}")
            return

        # ── 2. Log panel ──────────────────────────────────────────────────────
        if log_file.exists():
            try:
                self.log.clear()
                self.log.append_line("SYS", f"--- Session log: {session_path.name} ---")
                for raw_line in log_file.read_text(encoding="utf-8").splitlines():
                    # Format: [HH:MM:SS] LEVEL  text
                    # Re-parse level and text to get correct colouring
                    m = re.match(r"\[(\d{2}:\d{2}:\d{2})\]\s+(\w+)\s+(.*)", raw_line)
                    if m:
                        _ts, level, text = m.groups()
                        self.log.append_line(level, f"{_ts}  {text}")
                    else:
                        self.log.append_line("INFO", raw_line)
            except Exception as e:
                self.log.append_line("WARN", f"Could not load session.log: {e}")

        # ── 3. Images ─────────────────────────────────────────────────────────
        if images_file.exists():
            try:
                with open(images_file, encoding="utf-8") as f:
                    image_entries = json.load(f)

                loaded = 0
                missing = 0
                for entry in image_entries:
                    img_path = Path(entry.get("path", ""))
                    if img_path.exists():
                        self.plot.show_image(str(img_path))
                        loaded += 1
                    else:
                        missing += 1

                msg = f"Images: {loaded} loaded"
                if missing:
                    msg += f", {missing} not found (files may have moved)"
                self.chat.add_system_message(msg)
            except Exception as e:
                self.chat.add_system_message(f"Could not load images: {e}")

        self._sb.showMessage(f"Session loaded: {session_path.name}")

    def _open_crop_dialog(self, image_path: str, out_path: str, mask_path: str, label: str) -> None:
        from .dialogs.crop_dialog import CropDialog
        from src.bridge.ui_bridge import ui
        import json as _json

        # image_path field is re-used to carry JSON payload from the new tool signature
        try:
            data = _json.loads(image_path)
            raw_images  = data["raw_images"]
            start_index = data.get("start_index", 0)
            out_dir     = data["out_dir"]
            prefix      = data["prefix"]
        except Exception:
            # Legacy single-image fallback
            raw_images  = [image_path]
            start_index = 0
            out_dir     = str(Path(out_path).parent)
            prefix      = label

        ui.chat(
            f"**Crop tool open** — {len(raw_images)} image(s) available\n\n"
            "1. Draw a rectangle around the object of interest.\n"
            "2. Drag corners/edges to resize, drag inside to move.\n"
            "3. Press **Enter** or **Save Crop** to save that crop.\n"
            "4. Use **◀ ▶** (or the ← → keys) to switch image.\n"
            "5. Repeat steps 1-3 for more crops.\n"
            "6. Press **Esc** or **Done** when you are finished."
        )
        ui.log("INFO", f"Crop dialog open — {len(raw_images)} image(s)  (Enter=save, Esc=done)")

        def on_saved(crops: list) -> None:
            if not crops:
                ui.log("INFO", "No crops saved.")
                ui.inject_to_agent("[CROP DONE] crops: []")
                return
            for c in crops:
                ui.plot_path(c["crop_path"])
                ui.log("OK", f"Crop saved: {c['crop_file']}")
            image_folder = data.get("image_folder", prefix) if isinstance(data, dict) else prefix
            image_files  = [c["crop_file"] for c in crops]
            summary = {
                "crops":        crops,
                "image_folder": image_folder,
                "image_files":  image_files,
            }
            ui.inject_to_agent(
                f"[CROP DONE] {len(crops)} crop(s) saved to ref_objects/{image_folder}/. "
                f"image_folder={image_folder!r}  image_files={image_files}. "
                f"You can call add_image_object directly or tool_generate_ref_image "
                f"with reference_image=<crop_path> to generate more variants. "
                f"Full data: {_json.dumps(summary)}"
            )

        dlg = CropDialog(
            raw_images = raw_images,
            out_dir    = out_dir,
            prefix     = prefix,
            on_saved   = on_saved,
            parent     = self,
        )
        # Start on the requested image
        if start_index > 0:
            dlg._img_idx = start_index
            dlg._load_current()
        dlg.exec()

    def _open_explorer_dialog(self, lib_dir: str) -> None:
        from src.bridge.ui_bridge import ui
        from src.trainer.triplet_explorer import _run_explorer
        from pathlib import Path as _Path
        try:
            _run_explorer(_Path(lib_dir))
        except Exception as e:
            ui.log("ERR", f"Explorer error: {e}")
        finally:
            ui._explorer_closed()

    def _open_pid_tuner_dialog(self) -> None:
        from src.simuscan_tools.pid_tuner import open_pid_tuner_on_main_thread
        try:
            open_pid_tuner_on_main_thread()
        except Exception as e:
            from src.bridge.ui_bridge import ui
            ui.log("ERR", f"PID Tuner error: {e}")

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow { background:#0e1117; }
            QSplitter::handle { background:#1e2530; }
            QMenuBar { background:#0e1117; color:#c9d1d9;
                       font-family:'Consolas',monospace; font-size:13px;
                       padding:4px 0; border-bottom:1px solid #1e2530; }
            QMenuBar::item:selected { background:#1f6feb; color:#fff; border-radius:4px; }
            QMenu { background:#161b22; color:#c9d1d9; border:1px solid #30363d;
                    font-family:'Consolas',monospace; font-size:13px; }
            QMenu::item:selected { background:#1f6feb; color:#fff; }
            QStatusBar { background:#0e1117; color:#8b949e;
                         font-family:'Consolas',monospace; font-size:11px;
                         border-top:1px solid #1e2530; padding:2px 8px; }
        """)
