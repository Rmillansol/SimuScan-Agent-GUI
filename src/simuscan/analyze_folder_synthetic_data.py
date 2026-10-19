import os
import glob
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime


# ------------------------------------------------------------
# 1. MATHEMATICAL FUNCTIONS
# ------------------------------------------------------------

def compute_HHCF(arr):
    """
    Height–Height Correlation Function: <[h(x+r)-h(x)]^2>.
    Computed on the central line of the image.
    """
    arr = arr.astype(np.float64)
    h = arr - np.mean(arr)
    line = h[h.shape[0] // 2, :]
    N = len(line)

    hhcf = np.zeros(N)
    for r in range(N):
        if r == 0:
            hhcf[0] = 0
        else:
            diff = line[:-r] - line[r:]
            hhcf[r] = np.mean(diff**2)

    return np.arange(N), hhcf


def compute_PSD(arr):
    """
    1D radial PSD derived from the averaged 2D PSD.
    """
    arr = arr.astype(np.float64)
    arr = arr - np.mean(arr)

    F = np.fft.fftshift(np.fft.fft2(arr))
    P = np.abs(F)**2

    ny, nx = arr.shape
    cy, cx = ny // 2, nx // 2

    y, x = np.indices((ny, nx))
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(np.int32)

    rmax = r.max()
    radial_psd = np.zeros(rmax + 1)
    counts = np.zeros(rmax + 1)

    for i in range(ny):
        for j in range(nx):
            radial_psd[r[i, j]] += P[i, j]
            counts[r[i, j]] += 1

    radial_psd = radial_psd / np.maximum(counts, 1)

    return np.arange(rmax + 1), radial_psd


# ------------------------------------------------------------
# 2. MAIN FUNCTION
# ------------------------------------------------------------

def analyze_folder_HHCF_PSD(folder):
    """
    Analyses a folder of PNGs:
      * HHCF (Height–Height Correlation Function)
      * PSD (Power Spectral Density)
      * Basic roughness

    Saves a folder inside `folder`:
        out_PSD_HHCF_YYYYMMDD_HHMMSS/

    Where it writes:
      - summary_psd_hhcf.csv
      - figures *_PSD.png and *_HHCF.png

    Returns the output folder.
    """
    folder = os.path.abspath(folder)

    # ---------- Create folder with timestamp ----------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(folder, f"out_PSD_HHCF_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    pngs = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not pngs:
        print(f"No PNG in {folder}")
        return out_dir

    # ---------- CSV ----------
    csv_path = os.path.join(out_dir, "summary_psd_hhcf.csv")
    fcsv = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(fcsv)
    writer.writerow([
        "filename",
        "mean",
        "std",
        "Ra",
        "Rq",
        "corr_length_px",
        "psd_slope"
    ])

    # ---------- Process images ----------
    for path in pngs:
        name = os.path.basename(path)

        img = Image.open(path).convert("L")
        arr = np.array(img).astype(np.float64)

        # Normalise intensities for relative analysis
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)

        h = arr - np.mean(arr)

        mean = np.mean(arr)
        std = np.std(arr)
        Ra = np.mean(np.abs(h))
        Rq = np.sqrt(np.mean(h**2))

        # HHCF
        r, hhcf = compute_HHCF(arr)
        max_hhcf = np.max(hhcf)
        halfsat = 0.5 * max_hhcf
        idx = np.where(hhcf >= halfsat)[0]
        Lcorr = idx[0] if len(idx) else np.nan

        # PSD
        k, psd = compute_PSD(arr)
        valid = (k > 5) & (k < 50)
        if np.sum(valid) > 5:
            slope, _ = np.polyfit(np.log(k[valid]), np.log(psd[valid]), 1)
        else:
            slope = np.nan

        # ---------- Save CSV ----------
        writer.writerow([name, mean, std, Ra, Rq, Lcorr, slope])

        # ---------- Plot PSD ----------
        plt.figure(figsize=(5, 4))
        plt.loglog(k[1:], psd[1:], lw=2)
        plt.xlabel("Spatial frequency (1/px)")
        plt.ylabel("PSD")
        plt.grid(True, ls="--", alpha=0.3)
        plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_PSD.png"), dpi=180)
        plt.close()

        # ---------- Plot HHCF ----------
        plt.figure(figsize=(5, 4))
        plt.plot(r, hhcf, lw=2)
        plt.xlabel("r (pixels)")
        plt.ylabel("HHCF")
        plt.grid(True, ls="--", alpha=0.3)
        plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_HHCF.png"), dpi=180)
        plt.close()

    fcsv.close()

    print("\n-------------------------------------")
    print(f" Analysis completed.")
    print(f" Results in: {out_dir}")
    print("-------------------------------------\n")

    return out_dir

# ============================================================
#          🔥 GLOBAL ANALYSIS + SAVING RESULTS 🔥
# ============================================================
def compute_global_psd_hhcf(folder):
    """
    Analyses all PNG images in a folder:
        - Computes PSD and HHCF for each image
        - Averages all curves (mean ± std)
        - Computes global Ra, Rq and Lc
        - Saves:
            * PSD_global.png
            * HHCF_global.png
            * global_metrics.csv
            * PSD_global_curve.csv
            * HHCF_global_curve.csv

    Automatically creates a folder:
        folder/out_PSD_HHCF_YYYYMMDD_HHMMSS/
    """

    folder = os.path.abspath(folder)

    # --- Create folder with timestamp ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(folder, f"out_PSD_HHCF_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # --- Find PNGs ---
    paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not paths:
        raise ValueError(f"No PNG files found in folder: {folder}")

    psd_list = []
    hhcf_list = []
    Ra_list = []
    Rq_list = []
    Lc_list = []
    freq_axis = None
    dist_axis = None

    # ============================================================
    #                          LOOP
    # ============================================================
    for p in paths:
        arr = np.array(Image.open(p).convert("L")).astype(float)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)
        h = arr - np.mean(arr)

        # Roughness
        Ra = np.mean(np.abs(h))
        Rq = np.sqrt(np.mean(h**2))

        # HHCF
        r, hhcf = compute_HHCF(arr)
        if dist_axis is None: dist_axis = r
        hhcf_list.append(hhcf)

        # correlation length
        half = np.max(hhcf) * 0.5
        idx = np.where(hhcf >= half)[0]
        Lcorr = idx[0] if len(idx) else np.nan

        # PSD
        k, psd = compute_PSD(arr)
        if freq_axis is None: freq_axis = k
        psd_list.append(psd)

        Ra_list.append(Ra)
        Rq_list.append(Rq)
        Lc_list.append(Lcorr)

    # Convert to arrays
    psd_arr = np.vstack(psd_list)
    hhcf_arr = np.vstack(hhcf_list)

    PSD_mean = psd_arr.mean(axis=0)
    PSD_std  = psd_arr.std(axis=0)

    HHCF_mean = hhcf_arr.mean(axis=0)
    HHCF_std  = hhcf_arr.std(axis=0)

    # ============================================================
    #                        SAVE CSVs
    # ============================================================

    # --- Global metrics ---
    with open(os.path.join(out_dir, "global_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std"])
        w.writerow(["Ra", np.mean(Ra_list), np.std(Ra_list)])
        w.writerow(["Rq", np.mean(Rq_list), np.std(Rq_list)])
        w.writerow(["Lc", np.nanmean(Lc_list), np.nanstd(Lc_list)])

    # --- PSD curve ---
    with open(os.path.join(out_dir, "PSD_global_curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "PSD_mean", "PSD_std"])
        for i in range(len(freq_axis)):
            w.writerow([freq_axis[i], PSD_mean[i], PSD_std[i]])

    # --- HHCF curve ---
    with open(os.path.join(out_dir, "HHCF_global_curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["r", "HHCF_mean", "HHCF_std"])
        for i in range(len(dist_axis)):
            w.writerow([dist_axis[i], HHCF_mean[i], HHCF_std[i]])


    # ============================================================
    #                        SAVE FIGURES
    # ============================================================

    # PSD
    plt.figure(figsize=(6,5))
    plt.loglog(freq_axis[1:], PSD_mean[1:], "b", lw=2, label="Mean PSD")
    plt.fill_between(freq_axis[1:], PSD_mean[1:]-PSD_std[1:], PSD_mean[1:]+PSD_std[1:], 
                     color="b", alpha=0.25)
    plt.xlabel("Spatial frequency (1/pixel)")
    plt.ylabel("PSD")
    plt.title("Global PSD (mean ± std)")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "PSD_global.png"), dpi=200)
    plt.close()

    # HHCF
    plt.figure(figsize=(6,5))
    plt.plot(dist_axis, HHCF_mean, "r", lw=2, label="HHCF mean")
    plt.fill_between(dist_axis, HHCF_mean-HHCF_std, HHCF_mean+HHCF_std,
                     color="r", alpha=0.25)
    plt.xlabel("r (pixels)")
    plt.ylabel("HHCF")
    plt.title("Global HHCF (mean ± std)")
    plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "HHCF_global.png"), dpi=200)
    plt.close()

    print(f"\n✔ Global analysis completed.")
    print(f"✔ Results saved in:\n   {out_dir}\n")

    return out_dir

if __name__ == '__main__':
    # out_exp = analyze_folder_HHCF_PSD(r"C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\DataTest\Bacillus_on_pillars_2")
    # out_syn = analyze_folder_HHCF_PSD(r"C:\Users\z78\Documents\projects\SimuScan\Datasets\20251126_094806_YOLO_SEG\images\train")

    #exp_data = compute_global_psd_hhcf(r"C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\DataTest\Bacillus_on_pillars_2")
    sim_data = compute_global_psd_hhcf(r"C:\Users\z78\Documents\projects\SimuScan\Datasets\20251126_094806_YOLO_SEG\images\train")