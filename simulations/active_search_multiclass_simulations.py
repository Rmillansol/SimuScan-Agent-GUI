#!/usr/bin/env python3
"""
active_search_multiclass_simulations.py
======================================

Simulation code for two-class probabilistic active search on a 20 x 20 grid.

Implements:
  1. Single-case comparison with figures:
     - ground truth class A and B
     - trajectories for baseline, unknown exploration, lower movement penalty, raster
     - accumulated measured maps
     - cumulative detections
  2. Monte Carlo comparison over N random spatial distributions:
     - no max_scans cap by default, so all methods run until targets are reached
     - summary tables and publication-style plots

Core active-search method:
  - first image at the grid center
  - then N random images near the center
  - then class-aware probabilistic active search
  - optional epsilon_unknown exploration of uncertain regions
  - max_step_distance constraint

Author: Rubén Millán-Solsona
"""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

@dataclass
class Params:
    grid_size: int = 20
    target_A: int = 95
    target_B: int = 24

    # Initial acquisition
    initial_random_near_center: int = 5
    initial_center_radius: float = 4.5

    # Movement and acquisition limits
    max_step_distance: float = 6.0
    max_scans: Optional[int] = None  # None = no cap, run until target or no valid cell

    # Bayesian map initialization
    alpha_init: float = 1.0
    beta_init: float = 1.0

    # Spatial update model
    sigma_A_model: float = 2.0
    sigma_B_model: float = 1.6
    negative_update_scale: float = 1.4

    # Utility function
    kappa_A: float = 0.55
    kappa_B: float = 0.70
    baseline_lambda_move: float = 0.018
    lower_lambda_move: float = 0.005
    unknown_epsilon: float = 0.10
    focus_balance_threshold: float = 0.18

    # Synthetic distributions
    background_A_min: float = 0.05
    background_A_max: float = 0.18
    background_B_min: float = 0.01
    background_B_max: float = 0.05
    n_centers_A_min: int = 2
    n_centers_A_max_exclusive: int = 5  # 2-4
    n_centers_B_min: int = 1
    n_centers_B_max_exclusive: int = 4  # 1-3

    # Monte Carlo
    n_repeats: int = 100
    seed: int = 20260619
    max_generation_attempts_per_replicate: int = 1000


# -----------------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------------

def beta_mean(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return alpha / (alpha + beta)


def beta_std(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.sqrt((alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1)))


def choose_focus_from_contrib(a: float, b: float, threshold: float) -> str:
    """Return 'both', 'A', or 'B' from class utility contributions."""
    if max(a, b) == 0:
        return "both"
    rel_diff = abs(a - b) / max(a, b)
    if rel_diff < threshold:
        return "both"
    return "A" if a > b else "B"


# -----------------------------------------------------------------------------
# Simulator
# -----------------------------------------------------------------------------

class ActiveSearchSimulator:
    def __init__(self, params: Params):
        self.p = params
        self.grid_size = params.grid_size
        self.yy, self.xx = np.mgrid[0:self.grid_size, 0:self.grid_size]
        self.center_x = self.grid_size // 2
        self.center_y = self.grid_size // 2

    # ----------------------------
    # Synthetic data generation
    # ----------------------------

    def generate_random_distribution(self, rng: np.random.Generator):
        """
        Generate a different spatial distribution for A and B.
        A is globally more abundant. B is rarer and has fewer/narrower clusters.
        """
        p = self.p
        rate_A = np.full(
            (self.grid_size, self.grid_size),
            rng.uniform(p.background_A_min, p.background_A_max),
            dtype=float,
        )
        rate_B = np.full(
            (self.grid_size, self.grid_size),
            rng.uniform(p.background_B_min, p.background_B_max),
            dtype=float,
        )

        n_centers_A = rng.integers(p.n_centers_A_min, p.n_centers_A_max_exclusive)
        n_centers_B = rng.integers(p.n_centers_B_min, p.n_centers_B_max_exclusive)

        centers_A, centers_B = [], []

        for _ in range(n_centers_A):
            cx = rng.uniform(2, self.grid_size - 3)
            cy = rng.uniform(2, self.grid_size - 3)
            amp = rng.uniform(3.5, 8.0)
            sigma = rng.uniform(1.7, 3.2)
            rate_A += amp * np.exp(-((self.xx - cx) ** 2 + (self.yy - cy) ** 2) / (2 * sigma ** 2))
            centers_A.append((float(cx), float(cy), float(amp), float(sigma)))

        for _ in range(n_centers_B):
            cx = rng.uniform(2, self.grid_size - 3)
            cy = rng.uniform(2, self.grid_size - 3)
            amp = rng.uniform(1.2, 3.5)
            sigma = rng.uniform(1.1, 2.4)
            rate_B += amp * np.exp(-((self.xx - cx) ** 2 + (self.yy - cy) ** 2) / (2 * sigma ** 2))
            centers_B.append((float(cx), float(cy), float(amp), float(sigma)))

        true_A = rng.poisson(rate_A)
        true_B = rng.poisson(rate_B)
        return true_A, true_B, rate_A, rate_B, centers_A, centers_B

    def generate_reachable_distribution(self, rng: np.random.Generator):
        """Generate a distribution with enough total objects to reach both targets."""
        for attempt in range(1, self.p.max_generation_attempts_per_replicate + 1):
            true_A, true_B, rate_A, rate_B, centers_A, centers_B = self.generate_random_distribution(rng)
            if true_A.sum() >= self.p.target_A and true_B.sum() >= self.p.target_B:
                return true_A, true_B, rate_A, rate_B, centers_A, centers_B, attempt
        raise RuntimeError("Could not generate a reachable distribution within the attempt limit.")

    # ----------------------------
    # Initial scans
    # ----------------------------

    def make_initial_cells(self, rng: np.random.Generator) -> List[Tuple[int, int]]:
        """
        First point is always the center. Then take random cells near center.
        Cells are returned as (y, x).
        """
        dist_from_center = np.sqrt((self.xx - self.center_x) ** 2 + (self.yy - self.center_y) ** 2)
        allowed = np.argwhere(dist_from_center <= self.p.initial_center_radius)
        allowed = [tuple(map(int, v)) for v in allowed if tuple(v) != (self.center_y, self.center_x)]
        rng.shuffle(allowed)
        cells = [(self.center_y, self.center_x)]
        cells.extend(allowed[: self.p.initial_random_near_center])
        return cells

    # ----------------------------
    # Bayesian map update
    # ----------------------------

    def kernel_around(self, x0: int, y0: int, sigma: float) -> np.ndarray:
        dist2 = (self.xx - x0) ** 2 + (self.yy - y0) ** 2
        return np.exp(-dist2 / (2 * sigma ** 2))

    def update_class(self, alpha: np.ndarray, beta: np.ndarray, x: int, y: int, observed_count: int, sigma: float):
        k = self.kernel_around(x, y, sigma)
        if observed_count > 0:
            alpha += observed_count * k
        else:
            beta += self.p.negative_update_scale * k

    # ----------------------------
    # Cell selection
    # ----------------------------

    def choose_best_cell(
        self,
        alpha_A: np.ndarray,
        beta_A: np.ndarray,
        alpha_B: np.ndarray,
        beta_B: np.ndarray,
        scanned: np.ndarray,
        current_x: int,
        current_y: int,
        found_A: int,
        found_B: int,
        lambda_move: float,
    ) -> Optional[Tuple[int, int, str]]:
        p = self.p
        mu_A = beta_mean(alpha_A, beta_A)
        sd_A = beta_std(alpha_A, beta_A)
        mu_B = beta_mean(alpha_B, beta_B)
        sd_B = beta_std(alpha_B, beta_B)

        deficit_A = max(0.0, p.target_A - found_A) / p.target_A
        deficit_B = max(0.0, p.target_B - found_B) / p.target_B

        dist = np.sqrt((self.xx - current_x) ** 2 + (self.yy - current_y) ** 2)
        valid = (~scanned) & (dist <= p.max_step_distance)
        if not np.any(valid):
            return None

        contrib_A = deficit_A * (mu_A + p.kappa_A * sd_A)
        contrib_B = deficit_B * (mu_B + p.kappa_B * sd_B)
        score = contrib_A + contrib_B - lambda_move * dist
        score = np.where(valid, score, -np.inf)

        y, x = np.unravel_index(np.argmax(score), score.shape)
        focus = choose_focus_from_contrib(contrib_A[y, x], contrib_B[y, x], p.focus_balance_threshold)
        return int(y), int(x), focus

    def choose_unknown_cell(
        self,
        alpha_A: np.ndarray,
        beta_A: np.ndarray,
        alpha_B: np.ndarray,
        beta_B: np.ndarray,
        scanned: np.ndarray,
        current_x: int,
        current_y: int,
        found_A: int,
        found_B: int,
    ) -> Optional[Tuple[int, int, str]]:
        p = self.p
        sd_A = beta_std(alpha_A, beta_A)
        sd_B = beta_std(alpha_B, beta_B)

        deficit_A = max(0.0, p.target_A - found_A) / p.target_A
        deficit_B = max(0.0, p.target_B - found_B) / p.target_B

        dist = np.sqrt((self.xx - current_x) ** 2 + (self.yy - current_y) ** 2)
        valid = (~scanned) & (dist <= p.max_step_distance)
        if not np.any(valid):
            return None

        unknown_score = deficit_A * sd_A + deficit_B * sd_B
        unknown_score = np.where(valid, unknown_score, -np.inf)

        y, x = np.unravel_index(np.argmax(unknown_score), unknown_score.shape)
        focus = choose_focus_from_contrib(deficit_A * sd_A[y, x], deficit_B * sd_B[y, x], p.focus_balance_threshold)
        return int(y), int(x), focus

    # ----------------------------
    # Strategy execution
    # ----------------------------

    def run_active_strategy(
        self,
        true_A: np.ndarray,
        true_B: np.ndarray,
        initial_cells: List[Tuple[int, int]],
        name: str,
        epsilon_unknown: float,
        lambda_move: float,
        rng: np.random.Generator,
        keep_maps: bool = True,
    ) -> Dict[str, Any]:
        p = self.p
        alpha_A = np.full((self.grid_size, self.grid_size), p.alpha_init, dtype=float)
        beta_A = np.full((self.grid_size, self.grid_size), p.beta_init, dtype=float)
        alpha_B = np.full((self.grid_size, self.grid_size), p.alpha_init, dtype=float)
        beta_B = np.full((self.grid_size, self.grid_size), p.beta_init, dtype=float)

        scanned = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        measured_A = np.zeros((self.grid_size, self.grid_size), dtype=float)
        measured_B = np.zeros((self.grid_size, self.grid_size), dtype=float)

        found_A = 0
        found_B = 0
        current_x, current_y = self.center_x, self.center_y
        scan_no = 0
        total_move = 0.0
        focus_counts = {"both": 0, "A": 0, "B": 0}
        choice_counts = {"initial": 0, "best_score": 0, "forced_unknown": 0}
        history = []

        # Initial scans: center + random near-center
        for iy, ix in initial_cells:
            step_move = float(np.sqrt((ix - current_x) ** 2 + (iy - current_y) ** 2))
            total_move += step_move
            scan_no += 1

            nA, nB = int(true_A[iy, ix]), int(true_B[iy, ix])
            found_A += nA
            found_B += nB
            scanned[iy, ix] = True
            measured_A[iy, ix] = nA
            measured_B[iy, ix] = nB

            self.update_class(alpha_A, beta_A, ix, iy, nA, p.sigma_A_model)
            self.update_class(alpha_B, beta_B, ix, iy, nB, p.sigma_B_model)

            current_x, current_y = ix, iy
            focus_counts["both"] += 1
            choice_counts["initial"] += 1
            history.append(self._history_row(name, scan_no, ix, iy, nA, nB, found_A, found_B, "both", "initial", step_move))

            if found_A >= p.target_A and found_B >= p.target_B:
                break

        # Active loop
        while not (found_A >= p.target_A and found_B >= p.target_B):
            if p.max_scans is not None and scan_no >= p.max_scans:
                break

            if rng.random() < epsilon_unknown:
                chosen = self.choose_unknown_cell(alpha_A, beta_A, alpha_B, beta_B, scanned, current_x, current_y, found_A, found_B)
                choice_type = "forced_unknown"
            else:
                chosen = self.choose_best_cell(alpha_A, beta_A, alpha_B, beta_B, scanned, current_x, current_y, found_A, found_B, lambda_move)
                choice_type = "best_score"

            if chosen is None:
                break

            iy, ix, focus = chosen
            step_move = float(np.sqrt((ix - current_x) ** 2 + (iy - current_y) ** 2))
            total_move += step_move
            scan_no += 1

            nA, nB = int(true_A[iy, ix]), int(true_B[iy, ix])
            found_A += nA
            found_B += nB
            scanned[iy, ix] = True
            measured_A[iy, ix] = nA
            measured_B[iy, ix] = nB

            self.update_class(alpha_A, beta_A, ix, iy, nA, p.sigma_A_model)
            self.update_class(alpha_B, beta_B, ix, iy, nB, p.sigma_B_model)

            current_x, current_y = ix, iy
            focus_counts[focus] += 1
            choice_counts[choice_type] += 1
            history.append(self._history_row(name, scan_no, ix, iy, nA, nB, found_A, found_B, focus, choice_type, step_move))

        success = (found_A >= p.target_A) and (found_B >= p.target_B)
        out = {
            "strategy": name,
            "history": pd.DataFrame(history),
            "found_A": found_A,
            "found_B": found_B,
            "success": success,
            "scans_used": scan_no,
            "total_stage_movement": total_move,
            "average_move_per_scan": total_move / max(scan_no, 1),
            "fraction_grid_scanned": scan_no / (self.grid_size * self.grid_size),
            "focus_both": focus_counts["both"],
            "focus_A": focus_counts["A"],
            "focus_B": focus_counts["B"],
            "forced_unknown_steps": choice_counts["forced_unknown"],
            "best_score_steps": choice_counts["best_score"],
            "initial_steps": choice_counts["initial"],
        }
        if keep_maps:
            out.update({
                "measured_A": measured_A,
                "measured_B": measured_B,
                "measured_total": measured_A + measured_B,
            })
        return out

    def run_raster(self, true_A: np.ndarray, true_B: np.ndarray, keep_maps: bool = True) -> Dict[str, Any]:
        p = self.p
        measured_A = np.zeros((self.grid_size, self.grid_size), dtype=float)
        measured_B = np.zeros((self.grid_size, self.grid_size), dtype=float)
        found_A = 0
        found_B = 0
        current_x, current_y = self.center_x, self.center_y
        scan_no = 0
        total_move = 0.0
        focus_counts = {"both": 0, "A": 0, "B": 0}
        history = []

        for y in range(self.grid_size):
            xs = list(range(self.grid_size)) if y % 2 == 0 else list(range(self.grid_size - 1, -1, -1))
            for x in xs:
                if p.max_scans is not None and scan_no >= p.max_scans:
                    break

                step_move = float(np.sqrt((x - current_x) ** 2 + (y - current_y) ** 2))
                total_move += step_move
                current_x, current_y = x, y
                scan_no += 1

                nA, nB = int(true_A[y, x]), int(true_B[y, x])
                found_A += nA
                found_B += nB
                measured_A[y, x] = nA
                measured_B[y, x] = nB

                deficit_A = max(0.0, p.target_A - found_A) / p.target_A
                deficit_B = max(0.0, p.target_B - found_B) / p.target_B
                focus = choose_focus_from_contrib(deficit_A, deficit_B, p.focus_balance_threshold)
                focus_counts[focus] += 1
                history.append(self._history_row("raster", scan_no, x, y, nA, nB, found_A, found_B, focus, "raster", step_move))

                if found_A >= p.target_A and found_B >= p.target_B:
                    break
            if found_A >= p.target_A and found_B >= p.target_B:
                break
            if p.max_scans is not None and scan_no >= p.max_scans:
                break

        success = (found_A >= p.target_A) and (found_B >= p.target_B)
        out = {
            "strategy": "raster",
            "history": pd.DataFrame(history),
            "found_A": found_A,
            "found_B": found_B,
            "success": success,
            "scans_used": scan_no,
            "total_stage_movement": total_move,
            "average_move_per_scan": total_move / max(scan_no, 1),
            "fraction_grid_scanned": scan_no / (self.grid_size * self.grid_size),
            "focus_both": focus_counts["both"],
            "focus_A": focus_counts["A"],
            "focus_B": focus_counts["B"],
            "forced_unknown_steps": 0,
            "best_score_steps": 0,
            "initial_steps": 0,
        }
        if keep_maps:
            out.update({
                "measured_A": measured_A,
                "measured_B": measured_B,
                "measured_total": measured_A + measured_B,
            })
        return out

    @staticmethod
    def _history_row(strategy, scan, x, y, nA, nB, cumA, cumB, focus, choice_type, step_move):
        return {
            "strategy": strategy,
            "scan": scan,
            "x": x,
            "y": y,
            "found_A_in_scan": nA,
            "found_B_in_scan": nB,
            "cum_A": cumA,
            "cum_B": cumB,
            "focus": focus,
            "choice_type": choice_type,
            "step_move": step_move,
        }


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def save_ground_truth(sim: ActiveSearchSimulator, true_A, true_B, outdir: Path):
    paths = []
    for arr, cls in [(true_A, "A"), (true_B, "B")]:
        plt.figure(figsize=(7, 6))
        plt.imshow(arr, origin="lower")
        plt.colorbar(label=f"Ground truth objects of class {cls}")
        plt.title(f"Ground truth distribution - class {cls}")
        plt.xlabel("Grid x")
        plt.ylabel("Grid y")
        p = outdir / f"singlecase_groundtruth_class_{cls}.png"
        plt.tight_layout()
        plt.savefig(p, dpi=220)
        plt.close()
        paths.append(p)
    return paths


def save_single_case_plots(results: Dict[str, Dict[str, Any]], outdir: Path):
    paths = []
    colors = {"both": "green", "A": "blue", "B": "orange"}

    # Cumulative plots
    for label, getter, target, filename in [
        ("Cumulative detected objects (A + B)", lambda h: h["cum_A"] + h["cum_B"], None, "singlecase_cumulative_total_comparison.png"),
        ("Cumulative detected objects - class A", lambda h: h["cum_A"], "A", "singlecase_cumulative_class_A_comparison.png"),
        ("Cumulative detected objects - class B", lambda h: h["cum_B"], "B", "singlecase_cumulative_class_B_comparison.png"),
    ]:
        plt.figure(figsize=(8, 5.5))
        for res in results.values():
            h = res["history"]
            plt.plot(h["scan"], getter(h), marker="o", label=res["strategy"])
        plt.title(label)
        plt.xlabel("Number of AFM images acquired")
        plt.ylabel("Cumulative detections")
        plt.legend()
        p = outdir / filename
        plt.tight_layout()
        plt.savefig(p, dpi=220)
        plt.close()
        paths.append(p)

    # Trajectories and accumulated class maps
    for res in results.values():
        hist = res["history"]
        name = res["strategy"]

        plt.figure(figsize=(7.5, 6.5))
        plt.imshow(res["measured_total"], origin="lower")
        plt.colorbar(label="Accumulated measured objects (A + B)")
        plt.title(f"Trajectory - {name}\nBackground = accumulated measured data")
        plt.xlabel("Grid x")
        plt.ylabel("Grid y")
        plt.plot(hist["x"], hist["y"], linewidth=1.0, color="gray", alpha=0.9)
        for focus, color in colors.items():
            sub = hist[hist["focus"] == focus]
            if len(sub) > 0:
                label = "Searching both classes" if focus == "both" else f"Prioritizing class {focus}"
                plt.scatter(sub["x"], sub["y"], color=color, s=52, label=label)
        for _, row in hist.iterrows():
            s = int(row["scan"])
            if s <= 10 or s > len(hist) - 3:
                plt.text(row["x"] + 0.18, row["y"] + 0.18, str(s), fontsize=8)
        plt.legend(loc="upper left")
        p = outdir / f"singlecase_trajectory_{name}.png"
        plt.tight_layout()
        plt.savefig(p, dpi=220)
        plt.close()
        paths.append(p)

        for cls, arr in [("A", res["measured_A"]), ("B", res["measured_B"])]:
            plt.figure(figsize=(7, 6))
            plt.imshow(arr, origin="lower")
            plt.colorbar(label=f"Accumulated measured class {cls} objects")
            plt.title(f"Accumulated measured map - class {cls} - {name}")
            plt.xlabel("Grid x")
            plt.ylabel("Grid y")
            p = outdir / f"singlecase_accumulated_class{cls}_{name}.png"
            plt.tight_layout()
            plt.savefig(p, dpi=220)
            plt.close()
            paths.append(p)

    return paths


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    def q25(x): return np.percentile(x, 25)
    def q75(x): return np.percentile(x, 75)

    summary = results.groupby("strategy").agg(
        success_rate=("success", "mean"),
        scans_mean=("scans_used", "mean"),
        scans_std=("scans_used", "std"),
        scans_median=("scans_used", "median"),
        scans_q25=("scans_used", q25),
        scans_q75=("scans_used", q75),
        found_A_mean=("found_A", "mean"),
        found_B_mean=("found_B", "mean"),
        fraction_grid_mean=("fraction_grid_scanned", "mean"),
        movement_mean=("total_stage_movement", "mean"),
        movement_std=("total_stage_movement", "std"),
        avg_move_mean=("average_move_per_scan", "mean"),
        focus_both_mean=("focus_both", "mean"),
        focus_A_mean=("focus_A", "mean"),
        focus_B_mean=("focus_B", "mean"),
        forced_unknown_mean=("forced_unknown_steps", "mean"),
    ).reset_index()

    order = ["baseline", "unknown_exploration", "lower_movement_penalty", "raster"]
    summary["strategy"] = pd.Categorical(summary["strategy"], categories=order, ordered=True)
    return summary.sort_values("strategy").reset_index(drop=True)


def save_monte_carlo_plots(results: pd.DataFrame, summary: pd.DataFrame, outdir: Path):
    paths = []
    order = ["baseline", "unknown_exploration", "lower_movement_penalty", "raster"]

    # Scans boxplot
    plt.figure(figsize=(8.8, 5.6))
    data = [results[results["strategy"] == st]["scans_used"].values for st in order]
    plt.boxplot(data, tick_labels=order, showmeans=True)
    plt.title("Monte Carlo: scans to reach both targets")
    plt.ylabel("AFM images acquired")
    plt.xticks(rotation=18, ha="right")
    p = outdir / "mc_scans_boxplot.png"
    plt.tight_layout(); plt.savefig(p, dpi=220); plt.close(); paths.append(p)

    # Movement boxplot
    plt.figure(figsize=(8.8, 5.6))
    data = [results[results["strategy"] == st]["total_stage_movement"].values for st in order]
    plt.boxplot(data, tick_labels=order, showmeans=True)
    plt.title("Monte Carlo: total movement")
    plt.ylabel("Movement in grid-cell units")
    plt.xticks(rotation=18, ha="right")
    p = outdir / "mc_movement_boxplot.png"
    plt.tight_layout(); plt.savefig(p, dpi=220); plt.close(); paths.append(p)

    # Relative cost vs baseline
    baseline_scans = float(summary.loc[summary["strategy"] == "baseline", "scans_mean"].iloc[0])
    baseline_move = float(summary.loc[summary["strategy"] == "baseline", "movement_mean"].iloc[0])
    rel = summary.copy()
    rel["relative_scans_cost"] = rel["scans_mean"] / baseline_scans
    rel["relative_movement_cost"] = rel["movement_mean"] / baseline_move
    xpos = np.arange(len(rel))
    width = 0.36
    plt.figure(figsize=(8.8, 5.4))
    plt.bar(xpos - width/2, rel["relative_scans_cost"], width, label="Relative scans cost")
    plt.bar(xpos + width/2, rel["relative_movement_cost"], width, label="Relative movement cost")
    plt.axhline(1.0, linestyle="--")
    plt.xticks(xpos, rel["strategy"].astype(str), rotation=18, ha="right")
    plt.ylabel("Relative to baseline (smaller is better)")
    plt.title("Relative cost vs baseline")
    plt.legend()
    p = outdir / "mc_relative_cost.png"
    plt.tight_layout(); plt.savefig(p, dpi=220); plt.close(); paths.append(p)

    # Scans vs movement scatter
    plt.figure(figsize=(7.8, 5.8))
    for st in order:
        sub = results[results["strategy"] == st]
        plt.scatter(sub["scans_used"], sub["total_stage_movement"], alpha=0.62, label=st)
    plt.xlabel("AFM images acquired")
    plt.ylabel("Total movement in grid-cell units")
    plt.title("Scans vs movement cost")
    plt.legend()
    p = outdir / "mc_scans_vs_movement.png"
    plt.tight_layout(); plt.savefig(p, dpi=220); plt.close(); paths.append(p)

    # Focus composition
    focus_data = summary[["strategy", "focus_both_mean", "focus_A_mean", "focus_B_mean"]].copy()
    plt.figure(figsize=(8.8, 5.6))
    bottom = np.zeros(len(focus_data))
    for col, lab in [("focus_both_mean", "both"), ("focus_A_mean", "A"), ("focus_B_mean", "B")]:
        vals = focus_data[col].to_numpy()
        plt.bar(focus_data["strategy"].astype(str), vals, bottom=bottom, label=lab)
        bottom += vals
    plt.title("Mean focus counts per strategy")
    plt.ylabel("Mean number of scans")
    plt.xticks(rotation=18, ha="right")
    plt.legend(title="Focus")
    p = outdir / "mc_focus_composition.png"
    plt.tight_layout(); plt.savefig(p, dpi=220); plt.close(); paths.append(p)

    return paths


def make_article_table(summary: pd.DataFrame, outdir: Path):
    method_names = {
        "baseline": "Baseline active search",
        "unknown_exploration": "Active search + 10% unknown exploration",
        "lower_movement_penalty": "Active search + lower movement penalty",
        "raster": "Raster scan",
    }
    table = pd.DataFrame({
        "Method": summary["strategy"].map(method_names),
        "Success rate (%)": (summary["success_rate"] * 100).round(0).astype(int),
        "Scans to target, mean ± SD": summary.apply(lambda r: f"{r['scans_mean']:.1f} ± {r['scans_std']:.1f}", axis=1),
        "Scans to target, median [Q1–Q3]": summary.apply(lambda r: f"{r['scans_median']:.1f} [{r['scans_q25']:.1f}–{r['scans_q75']:.1f}]", axis=1),
        "Grid sampled (%)": (summary["fraction_grid_mean"] * 100).map(lambda x: f"{x:.1f}"),
        "Movement, mean ± SD": summary.apply(lambda r: f"{r['movement_mean']:.1f} ± {r['movement_std']:.1f}", axis=1),
        "Mean move / scan": summary["avg_move_mean"].map(lambda x: f"{x:.2f}"),
        "Mean focus: both / A / B": summary.apply(lambda r: f"{r['focus_both_mean']:.1f} / {r['focus_A_mean']:.1f} / {r['focus_B_mean']:.1f}", axis=1),
    })
    raw = summary.copy()
    raw["rank_scans"] = raw["scans_mean"].rank(method="min")
    rank_map = dict(zip(raw["strategy"].astype(str), raw["rank_scans"].astype(int)))
    table["Rank by scans"] = summary["strategy"].astype(str).map(rank_map)

    csv_path = outdir / "article_table.csv"
    md_path = outdir / "article_table.md"
    table.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(table.columns) + " |\n")
        f.write("|" + "|".join(["---"] * len(table.columns)) + "|\n")
        for _, row in table.iterrows():
            f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
    return table, csv_path, md_path


# -----------------------------------------------------------------------------
# Experiment runners
# -----------------------------------------------------------------------------

def run_single_case(params: Params, outdir: Path, seed_sample: int = 44, seed_initial: int = 123):
    sim = ActiveSearchSimulator(params)
    rng_sample = np.random.default_rng(seed_sample)
    true_A, true_B, *_ = sim.generate_reachable_distribution(rng_sample)
    initial_cells = sim.make_initial_cells(np.random.default_rng(seed_initial))

    strategies = {
        "baseline": sim.run_active_strategy(true_A, true_B, initial_cells, "baseline", 0.0, params.baseline_lambda_move, np.random.default_rng(1)),
        "unknown_exploration": sim.run_active_strategy(true_A, true_B, initial_cells, "unknown_exploration", params.unknown_epsilon, params.baseline_lambda_move, np.random.default_rng(2)),
        "lower_movement_penalty": sim.run_active_strategy(true_A, true_B, initial_cells, "lower_movement_penalty", 0.0, params.lower_lambda_move, np.random.default_rng(3)),
        "raster": sim.run_raster(true_A, true_B),
    }

    rows = []
    histories = []
    for res in strategies.values():
        rows.append({k: res[k] for k in ["strategy", "scans_used", "found_A", "found_B", "success", "total_stage_movement", "average_move_per_scan", "fraction_grid_scanned"]})
        histories.append(res["history"])
    summary = pd.DataFrame(rows)
    history = pd.concat(histories, ignore_index=True)

    summary.to_csv(outdir / "singlecase_summary.csv", index=False)
    history.to_csv(outdir / "singlecase_history.csv", index=False)
    with open(outdir / "singlecase_parameters.json", "w", encoding="utf-8") as f:
        json.dump(asdict(params), f, indent=2)

    paths = []
    paths.extend(save_ground_truth(sim, true_A, true_B, outdir))
    paths.extend(save_single_case_plots(strategies, outdir))
    return summary, paths


def run_monte_carlo(params: Params, outdir: Path):
    sim = ActiveSearchSimulator(params)
    master_rng = np.random.default_rng(params.seed)
    records, distribution_records = [], []

    strategy_specs = [
        ("baseline", 0.0, params.baseline_lambda_move),
        ("unknown_exploration", params.unknown_epsilon, params.baseline_lambda_move),
        ("lower_movement_penalty", 0.0, params.lower_lambda_move),
    ]

    for rep in range(params.n_repeats):
        rng_dist = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
        true_A, true_B, rate_A, rate_B, centers_A, centers_B, attempts = sim.generate_reachable_distribution(rng_dist)
        rng_init = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
        initial_cells = sim.make_initial_cells(rng_init)

        distribution_records.append({
            "replicate": rep,
            "generation_attempts": attempts,
            "total_objects_A": int(true_A.sum()),
            "total_objects_B": int(true_B.sum()),
            "n_centers_A": len(centers_A),
            "n_centers_B": len(centers_B),
            "initial_cells_yx": str(initial_cells),
        })

        for s_name, eps, lam in strategy_specs:
            rng_strategy = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
            res = sim.run_active_strategy(true_A, true_B, initial_cells, s_name, eps, lam, rng_strategy, keep_maps=False)
            row = {k: res[k] for k in [
                "strategy", "found_A", "found_B", "success", "scans_used", "total_stage_movement",
                "average_move_per_scan", "fraction_grid_scanned", "focus_both", "focus_A", "focus_B",
                "forced_unknown_steps", "best_score_steps", "initial_steps"
            ]}
            row.update({"replicate": rep, "total_objects_A": int(true_A.sum()), "total_objects_B": int(true_B.sum()), "epsilon_unknown": eps, "lambda_move": lam})
            records.append(row)

        rast = sim.run_raster(true_A, true_B, keep_maps=False)
        row = {k: rast[k] for k in [
            "strategy", "found_A", "found_B", "success", "scans_used", "total_stage_movement",
            "average_move_per_scan", "fraction_grid_scanned", "focus_both", "focus_A", "focus_B",
            "forced_unknown_steps", "best_score_steps", "initial_steps"
        ]}
        row.update({"replicate": rep, "total_objects_A": int(true_A.sum()), "total_objects_B": int(true_B.sum()), "epsilon_unknown": np.nan, "lambda_move": np.nan})
        records.append(row)

    results = pd.DataFrame(records)
    distros = pd.DataFrame(distribution_records)
    summary = summarize_results(results)

    results.to_csv(outdir / "mc_all_results.csv", index=False)
    distros.to_csv(outdir / "mc_distributions_metadata.csv", index=False)
    summary.to_csv(outdir / "mc_summary.csv", index=False)
    with open(outdir / "mc_parameters.json", "w", encoding="utf-8") as f:
        json.dump(asdict(params), f, indent=2)

    paths = save_monte_carlo_plots(results, summary, outdir)
    table, table_csv, table_md = make_article_table(summary, outdir)
    return summary, table, paths + [table_csv, table_md]


def zip_outputs(outdir: Path, zip_name: str = "active_search_outputs.zip"):
    zip_path = outdir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(outdir.glob("*")):
            if p.is_file() and p.name != zip_name:
                zf.write(p, arcname=p.name)
    return zip_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Two-class probabilistic active-search simulations.")
    parser.add_argument("--mode", choices=["single", "mc", "all"], default="all", help="Which simulation to run.")
    parser.add_argument("--outdir", default="active_search_outputs", help="Output directory.")
    parser.add_argument("--n-repeats", type=int, default=100, help="Monte Carlo repetitions.")
    parser.add_argument("--seed", type=int, default=20260619, help="Random seed.")
    parser.add_argument("--max-scans", type=int, default=-1, help="Max scans cap; -1 means no cap.")
    parser.add_argument("--epsilon-unknown", type=float, default=0.10, help="Probability of forced unknown exploration.")
    parser.add_argument("--max-step-distance", type=float, default=6.0, help="Maximum movement per active step in grid-cell units.")
    return parser.parse_args()


def main():
    args = parse_args()
    params = Params(
        n_repeats=args.n_repeats,
        seed=args.seed,
        max_scans=None if args.max_scans < 0 else args.max_scans,
        unknown_epsilon=args.epsilon_unknown,
        max_step_distance=args.max_step_distance,
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("single", "all"):
        print("Running single-case simulation...")
        summary, paths = run_single_case(params, outdir)
        print(summary.to_string(index=False))

    if args.mode in ("mc", "all"):
        print("Running Monte Carlo simulation...")
        summary, article_table, paths = run_monte_carlo(params, outdir)
        print(summary.to_string(index=False))
        print("\nArticle-style table:")
        print(article_table.to_string(index=False))

    zip_path = zip_outputs(outdir)
    print(f"\nDone. Outputs saved in: {outdir.resolve()}")
    print(f"ZIP archive: {zip_path.resolve()}")


if __name__ == "__main__":
    main()
