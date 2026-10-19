# SimuScan-Agent

**Language-model orchestration of synthetic-data-driven autonomous atomic force microscopy.**

SimuScan-Agent is an LLM-orchestrated framework that turns natural-language
objectives into adaptive atomic force microscopy (AFM) workflows. From a single
chat interface it:

1. **Generates annotated synthetic AFM datasets** without manual per-image
   labelling, combining multimodal morphological sources (parametric sections,
   reference images, STL meshes, user Python generators) with physics-informed
   simulation of the tip, substrate, PID feedback and scan artefacts.
2. **Trains and deploys models**: YOLO instance segmentation, UNet semantic
   segmentation and triplet networks for morphological representation learning.
3. **Guides acquisition on a real AFM** (Nanosurf), using online detections and
   class-specific sampling objectives (multiclass probabilistic active search)
   instead of exhaustive raster scanning.
4. **Analyses the results**: class-specific spatial reconstruction and a
   morphological library that ranks representative and atypical nanostructures.

In simulations, probabilistic active exploration reduced the number of
measurements and the sampled area by approximately **71 %** relative to raster
scanning. Experimentally, SimuScan-Agent explored a **~10 mm²** region by
sampling only **17.2 %** of the available positions. The LLM (GPT models
through the OpenAI API) acts as the orchestration layer; the tools are exposed
to it through the Model Context Protocol (MCP).

This repository contains the complete software package described in the
article: the graphical user interface, the orchestration framework and MCP
tools, the SimuScan synthesis engine, the machine-learning workflows,
morphological analysis, spatial reconstruction and adaptive AFM exploration.

**Author:** Rubén Millán-Solsona

### Documentation

| Document | What it covers |
|---|---|
| [SimuScan manual](docs/simuscan_manual.md) | Full user guide: workflow, configuration and tools |
| [Parameter reference](docs/parameter_reference.md) | Every synthesis parameter, with ranges and meaning |
| [AFM artefacts](docs/afm_artifacts.md) | Physics of the simulated tip, feedback, noise and scan artefacts |
| [Examples](docs/examples.md) | End-to-end examples of agent conversations and configs |
| [Generator examples](docs/generator_examples.md) | Writing your own Python object generators |
| [Domain / grain generator](docs/domains_generator.md) | Synthetic topographic grains or domains (`shape: domains`) |
| [User vocabulary](docs/user_vocabulary.md) | How everyday phrases map to parameters for the agent |

The same documents are indexed by the agent (RAG), so you can also ask it
directly in the chat.

### Related resources

- Synthetic dataset and SimuScan configuration used in the article: <https://zenodo.org/records/23482159>
- Trained machine-learning models: <https://zenodo.org/records/23481955>
- Supplementary videos of the workflow: <https://zenodo.org/records/23481112>
- KLayout script for the nanostructure mask layout: <https://github.com/Rmillansol/probabilistic-gds-mask>
- Generative-model workflow for AFM-like reference images: <https://github.com/Rmillansol/Generative-AFM-like_workflow>

---

## 1. Requirements

- **Windows 10/11** (the GUI uses PyQt6; needs a desktop session, not headless).
- **Python ≥ 3.11** (managed automatically by `uv`).
- **[uv](https://docs.astral.sh/uv/)** as the environment/package manager.
- **NVIDIA GPU** (optional but recommended for training). This setup targets
  **CUDA 12.4+** (tested on an RTX 2000 Ada, driver 12.5).
- **Nanosurf Studio** — only on the microscope machine, for exploration.
- **An OpenAI API key**.

---

## 2. Install uv

In PowerShell:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Close and reopen the terminal so `uv` is on the PATH.

---

## 3. About CUDA — read this before "reinstalling"

**You do NOT need to install the CUDA Toolkit.** The PyTorch wheels (cu124) ship
their own CUDA libraries. The only thing your system needs is the
**NVIDIA driver** (which already includes the CUDA runtime).

Check what you have:

```powershell
nvidia-smi
```

- If it shows the GPU and `CUDA Version: 12.x` (where x ≥ 4) → **you're ready, do not reinstall anything**.
  (That "CUDA Version" is the maximum your driver supports, not an installed toolkit.)
- If `nvidia-smi` is **not recognized** or the GPU is missing → the **driver** is
  missing/broken: download the latest driver for your GPU from
  [nvidia.com/Download](https://www.nvidia.com/Download/index.aspx) and install it.
  You do NOT need the SDK "CUDA Toolkit", just the driver (Studio or Game Ready).

> Bottom line: **driver yes, toolkit no.** An up-to-date driver supports any CUDA 12.x build.

---

## 4. Project installation

```powershell
cd <path-to-this-repository>

uv lock      # resolves dependencies (torch/torchvision from the cu124 index on Windows)
uv sync      # creates .venv and installs everything
```

`uv.lock` reproduces the full environment (torch, ultralytics, nanosurf,
chromadb, etc.). On Windows, torch/torchvision install the **CUDA (cu124)** build;
on Linux they resolve from PyPI with their own CUDA dependencies.

### Verify the GPU works

```powershell
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

It should print `CUDA: True` and your GPU name. If it prints `False`, see
[§7 Troubleshooting](#7-troubleshooting).

---

## 5. Configuration (`.env`)

Create a `.env` file at the project root. Copy the
template and fill in your values:

```env
OPENAI_API_KEY=sk-...your-key...
OPENAI_MODEL=gpt-5.2
OPENAI_IMAGE_MODEL=gpt-image-1
WORKING_DIR=C:\data\simuscan
```

- `OPENAI_MODEL`: agent model (use a full model, not nano/mini).
- `OPENAI_IMAGE_MODEL`: reference-image generation model.
- Both can also be changed later from ⚙ Settings in the GUI.

> ⚠️ **Never commit `.env` to git** and do not reuse a key that has been
> exposed — regenerate it at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
> Make sure `.env` is in `.gitignore` (it is). A `.env.example` template is provided.

---

## 6. Run

```powershell
uv run python main_gui.py
```

The first run may take a moment while it indexes the documentation (RAG).

### General workflow
1. Define reference objects (images / STL / Python generators).
2. Configure and generate the synthetic dataset (COCO → YOLO).
3. Train YOLO / UNet / Triplet.
4. Configure and launch microscope exploration.
5. Analyze results (PNG or GWY folders).

### Microscope (Nanosurf)
- Only on the machine connected to the AFM, with **Nanosurf Studio open**.
- Recommended bring-up: Mode 1 (scan only) → Mode 2 (+detection) →
  Mode 3 (+zoom); start with `strategy: raster` before `probabilistic`.
- Verify first: connection, coordinates (`origin_x_um`/`origin_y_um`),
  scan direction (`scan_direction`), and `gallery_dir`.

---

## 7. Troubleshooting

**`torch.cuda.is_available()` returns `False`:**
1. Run `nvidia-smi`. If it fails → install/update the NVIDIA driver (§3).
2. Confirm `uv sync` installed the CUDA build, not CPU:
   ```powershell
   uv run python -c "import torch; print(torch.__version__)"
   ```
   It should end in `+cu124` (e.g. `2.x.x+cu124`). If not, force the CUDA build:
   ```powershell
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
3. If your driver predates 12.4, update it (no hardware change needed).

**GUI won't open / Qt error:** make sure you run in a desktop session
(not over SSH/headless).

**`uv lock` fails to resolve torch:** check connectivity and that the index
`https://download.pytorch.org/whl/cu124` is reachable (corporate proxy/firewall).

**Agent uses the wrong model:** check `OPENAI_MODEL` in `.env` or ⚙ Settings
(avoid `nano`/`mini` tiers for multi-tool tasks).

---

## 8. Notes

- Reproducible environment via `uv.lock`; don't hand-edit versions unless needed.
- On Linux, `uv sync` resolves torch from PyPI with CUDA (without the cu124 index).
- `nanosurf` installs automatically; connecting requires Studio to be running.

---

## How to cite

If you use SimuScan-Agent, its synthetic-data generator or any part of this
code in your work, please cite the article:

> R. Millan-Solsona, M. Checa, Y. Liu, B. Srijanto, R. K. Vasudevan,
> S. Retterer and L. Collins. *SimuScan-Agent: language-model orchestration of
> synthetic-data-driven autonomous atomic force microscopy.* Manuscript under
> review (2026).

```bibtex
@unpublished{MillanSolsona2026SimuScanAgent,
  title  = {SimuScan-Agent: language-model orchestration of synthetic-data-driven autonomous atomic force microscopy},
  author = {Millan-Solsona, Ruben and Checa, Marti and Liu, Yongtao and Srijanto, Bernadeta and Vasudevan, Rama K. and Retterer, Scott and Collins, Liam},
  note   = {Manuscript under review},
  year   = {2026}
}
```

The reference will be updated with the journal and DOI once the article is
published.

---

## Author

Developed by **Rubén Millán-Solsona**, Center for Nanophase Materials Sciences,
Oak Ridge National Laboratory.
