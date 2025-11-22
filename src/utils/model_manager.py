"""
Model Management Utilities

Handles saving, loading, and checkpointing of specialist and CIO models.
Ensures proper directory structure and versioning.
"""

import os
import torch
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np


class ModelManager:
    """
    Manages model saving, loading, and checkpointing.
    Creates proper directory structure for each specialist.
    """

    def __init__(self, base_dir: str = "models"):
        """
        Initialize Model Manager.

        Args:
            base_dir: Base directory for all models
        """
        self.base_dir = Path(base_dir)
        self.specialists_dir = self.base_dir / "specialists"
        self.master_dir = self.base_dir / "master"
        self.checkpoints_dir = self.base_dir / "checkpoints"

        # Create directories
        self._create_directories()

    def _create_directories(self):
        """Create all necessary directories."""
        # Specialist strategies
        strategies = [
            "statistical_arbitrage",
            "market_making",
            "volatility_trading",
            "delta_hedging",
            "futures_spreads",
            "factor_tracking",
            "fx_arbitrage",
        ]

        for strategy in strategies:
            strategy_dir = self.specialists_dir / strategy
            strategy_dir.mkdir(parents=True, exist_ok=True)
            (strategy_dir / "checkpoints").mkdir(exist_ok=True)
            (strategy_dir / "logs").mkdir(exist_ok=True)

        # Master directory
        self.master_dir.mkdir(parents=True, exist_ok=True)
        (self.master_dir / "checkpoints").mkdir(exist_ok=True)
        (self.master_dir / "logs").mkdir(exist_ok=True)

        # Global checkpoints
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save_specialist(
        self,
        agent,
        strategy_name: str,
        metrics: Dict[str, float],
        is_best: bool = False,
        step: Optional[int] = None,
    ) -> str:
        """
        Save specialist model.

        Args:
            agent: Agent object with save() method
            strategy_name: Name of strategy
            metrics: Performance metrics
            is_best: Whether this is the best model
            step: Training step number

        Returns:
            Path where model was saved
        """
        strategy_dir = self.specialists_dir / strategy_name

        # Save best model
        if is_best:
            save_path = strategy_dir / f"{strategy_name}_best.pth"
            agent.save(str(save_path))

            # Save metrics
            metrics_path = strategy_dir / f"{strategy_name}_best_metrics.json"
            self._save_metrics(metrics, metrics_path)

            print(f"✓ Saved BEST model for {strategy_name} to {save_path}")
            return str(save_path)

        # Save checkpoint
        if step is not None:
            checkpoint_dir = strategy_dir / "checkpoints"
            save_path = checkpoint_dir / f"{strategy_name}_step_{step}.pth"
            agent.save(str(save_path))

            # Save metrics
            metrics_path = checkpoint_dir / f"{strategy_name}_step_{step}_metrics.json"
            self._save_metrics(metrics, metrics_path)

            print(f"✓ Saved checkpoint for {strategy_name} at step {step}")
            return str(save_path)

        # Save latest
        save_path = strategy_dir / f"{strategy_name}_latest.pth"
        agent.save(str(save_path))

        metrics_path = strategy_dir / f"{strategy_name}_latest_metrics.json"
        self._save_metrics(metrics, metrics_path)

        return str(save_path)

    def load_specialist(
        self, agent, strategy_name: str, best: bool = True, step: Optional[int] = None
    ):
        """
        Load specialist model.

        Args:
            agent: Agent object with load() method
            strategy_name: Name of strategy
            best: Whether to load best model
            step: Specific checkpoint step to load
        """
        strategy_dir = self.specialists_dir / strategy_name

        if best:
            load_path = strategy_dir / f"{strategy_name}_best.pth"
        elif step is not None:
            load_path = (
                strategy_dir / "checkpoints" / f"{strategy_name}_step_{step}.pth"
            )
        else:
            load_path = strategy_dir / f"{strategy_name}_latest.pth"

        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")

        agent.load(str(load_path))
        print(f"✓ Loaded model for {strategy_name} from {load_path}")

        # Load metrics if available
        metrics_path = load_path.parent / f"{load_path.stem}_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            return metrics

        return None

    def save_cio(
        self,
        agent,
        metrics: Dict[str, float],
        is_best: bool = False,
        step: Optional[int] = None,
    ) -> str:
        """
        Save CIO allocator model.

        Args:
            agent: Agent object
            metrics: Performance metrics
            is_best: Whether this is best model
            step: Training step

        Returns:
            Save path
        """
        if is_best:
            save_path = self.master_dir / "cio_allocator_best.pth"
            agent.save(str(save_path))

            metrics_path = self.master_dir / "cio_allocator_best_metrics.json"
            self._save_metrics(metrics, metrics_path)

            print(f"✓ Saved BEST CIO model to {save_path}")
            return str(save_path)

        if step is not None:
            checkpoint_dir = self.master_dir / "checkpoints"
            save_path = checkpoint_dir / f"cio_allocator_step_{step}.pth"
            agent.save(str(save_path))

            metrics_path = checkpoint_dir / f"cio_allocator_step_{step}_metrics.json"
            self._save_metrics(metrics, metrics_path)

            print(f"✓ Saved CIO checkpoint at step {step}")
            return str(save_path)

        save_path = self.master_dir / "cio_allocator_latest.pth"
        agent.save(str(save_path))

        return str(save_path)

    def load_cio(self, agent, best: bool = True, step: Optional[int] = None):
        """Load CIO allocator model."""
        if best:
            load_path = self.master_dir / "cio_allocator_best.pth"
        elif step is not None:
            load_path = (
                self.master_dir / "checkpoints" / f"cio_allocator_step_{step}.pth"
            )
        else:
            load_path = self.master_dir / "cio_allocator_latest.pth"

        if not load_path.exists():
            raise FileNotFoundError(f"CIO model not found: {load_path}")

        agent.load(str(load_path))
        print(f"✓ Loaded CIO model from {load_path}")

        metrics_path = load_path.parent / f"{load_path.stem}_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                return json.load(f)

        return None

    def _save_metrics(self, metrics: Dict[str, Any], path: Path):
        """Save metrics to JSON file."""
        # Convert numpy types to Python types
        metrics_clean = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                metrics_clean[key] = float(value)
            elif isinstance(value, np.ndarray):
                metrics_clean[key] = value.tolist()
            else:
                metrics_clean[key] = value

        # Add timestamp
        metrics_clean["save_timestamp"] = datetime.now().isoformat()

        with open(path, "w") as f:
            json.dump(metrics_clean, f, indent=4)

    def load_all_specialists(
        self, agents: Dict[str, Any], best: bool = True
    ) -> Dict[str, Dict]:
        """
        Load all specialist models.

        Args:
            agents: Dictionary of {strategy_name: agent_object}
            best: Whether to load best models

        Returns:
            Dictionary of {strategy_name: metrics}
        """
        all_metrics = {}

        for strategy_name, agent in agents.items():
            try:
                metrics = self.load_specialist(agent, strategy_name, best=best)
                all_metrics[strategy_name] = metrics
            except FileNotFoundError as e:
                print(f"Warning: Could not load {strategy_name}: {e}")

        return all_metrics

    def get_best_checkpoint(
        self, strategy_name: str, metric: str = "sharpe_ratio"
    ) -> Optional[Path]:
        """
        Find best checkpoint based on metric.

        Args:
            strategy_name: Name of strategy
            metric: Metric to optimize

        Returns:
            Path to best checkpoint
        """
        if strategy_name == "cio_allocator":
            checkpoint_dir = self.master_dir / "checkpoints"
        else:
            checkpoint_dir = self.specialists_dir / strategy_name / "checkpoints"

        if not checkpoint_dir.exists():
            return None

        best_value = -np.inf
        best_path = None

        for metrics_file in checkpoint_dir.glob("*_metrics.json"):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            if metric in metrics and metrics[metric] > best_value:
                best_value = metrics[metric]
                # Get corresponding model file
                model_file = (
                    metrics_file.parent
                    / f"{metrics_file.stem.replace('_metrics', '')}.pth"
                )
                if model_file.exists():
                    best_path = model_file

        return best_path

    def cleanup_old_checkpoints(self, strategy_name: str, keep_last_n: int = 5):
        """
        Remove old checkpoints, keeping only the last N.

        Args:
            strategy_name: Name of strategy
            keep_last_n: Number of recent checkpoints to keep
        """
        if strategy_name == "cio_allocator":
            checkpoint_dir = self.master_dir / "checkpoints"
        else:
            checkpoint_dir = self.specialists_dir / strategy_name / "checkpoints"

        if not checkpoint_dir.exists():
            return

        # Get all checkpoints sorted by step number
        checkpoints = sorted(
            checkpoint_dir.glob(f"{strategy_name}_step_*.pth"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        # Remove old ones
        for checkpoint in checkpoints[:-keep_last_n]:
            checkpoint.unlink()
            # Also remove metrics file
            metrics_file = checkpoint.parent / f"{checkpoint.stem}_metrics.json"
            if metrics_file.exists():
                metrics_file.unlink()

        if len(checkpoints) > keep_last_n:
            print(
                f"✓ Cleaned up {len(checkpoints) - keep_last_n} old checkpoints for {strategy_name}"
            )

    def export_for_deployment(self, output_dir: str = "deployment"):
        """
        Export all best models for deployment.

        Args:
            output_dir: Directory to export to
        """
        export_path = Path(output_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        # Export specialists
        specialists_export = export_path / "specialists"
        specialists_export.mkdir(exist_ok=True)

        for strategy_dir in self.specialists_dir.iterdir():
            if strategy_dir.is_dir():
                best_model = strategy_dir / f"{strategy_dir.name}_best.pth"
                if best_model.exists():
                    shutil.copy(best_model, specialists_export / best_model.name)

                best_metrics = strategy_dir / f"{strategy_dir.name}_best_metrics.json"
                if best_metrics.exists():
                    shutil.copy(best_metrics, specialists_export / best_metrics.name)

        # Export CIO
        cio_best = self.master_dir / "cio_allocator_best.pth"
        if cio_best.exists():
            shutil.copy(cio_best, export_path / cio_best.name)

        cio_metrics = self.master_dir / "cio_allocator_best_metrics.json"
        if cio_metrics.exists():
            shutil.copy(cio_metrics, export_path / cio_metrics.name)

        print(f"✓ Exported models to {export_path}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all saved models.

        Returns:
            Dictionary with model information
        """
        summary = {"specialists": {}, "cio": {}, "total_size_mb": 0}

        # Specialists
        for strategy_dir in self.specialists_dir.iterdir():
            if strategy_dir.is_dir():
                strategy_name = strategy_dir.name
                best_model = strategy_dir / f"{strategy_name}_best.pth"

                if best_model.exists():
                    size_mb = best_model.stat().st_size / (1024 * 1024)
                    summary["total_size_mb"] += size_mb

                    metrics_file = strategy_dir / f"{strategy_name}_best_metrics.json"
                    metrics = {}
                    if metrics_file.exists():
                        with open(metrics_file, "r") as f:
                            metrics = json.load(f)

                    summary["specialists"][strategy_name] = {
                        "exists": True,
                        "size_mb": size_mb,
                        "path": str(best_model),
                        "metrics": metrics,
                    }
                else:
                    summary["specialists"][strategy_name] = {"exists": False}

        # CIO
        cio_best = self.master_dir / "cio_allocator_best.pth"
        if cio_best.exists():
            size_mb = cio_best.stat().st_size / (1024 * 1024)
            summary["total_size_mb"] += size_mb

            metrics_file = self.master_dir / "cio_allocator_best_metrics.json"
            metrics = {}
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)

            summary["cio"] = {
                "exists": True,
                "size_mb": size_mb,
                "path": str(cio_best),
                "metrics": metrics,
            }
        else:
            summary["cio"] = {"exists": False}

        return summary

    def print_summary(self):
        """Print formatted summary of saved models."""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("MODEL SUMMARY")
        print("=" * 80)

        print("\nSPECIALIST MODELS:")
        print("-" * 80)
        for strategy, info in summary["specialists"].items():
            if info["exists"]:
                metrics = info.get("metrics", {})
                sharpe = metrics.get("sharpe_ratio", "N/A")
                print(
                    f"✓ {strategy:25s} | Size: {info['size_mb']:6.2f} MB | Sharpe: {sharpe}"
                )
            else:
                print(f"✗ {strategy:25s} | NOT FOUND")

        print("\nCIO ALLOCATOR:")
        print("-" * 80)
        if summary["cio"]["exists"]:
            metrics = summary["cio"].get("metrics", {})
            sharpe = metrics.get("sharpe_ratio", "N/A")
            print(
                f"✓ CIO Allocator | Size: {summary['cio']['size_mb']:6.2f} MB | Sharpe: {sharpe}"
            )
        else:
            print("✗ CIO Allocator | NOT FOUND")

        print("\n" + "=" * 80)
        print(f"Total Size: {summary['total_size_mb']:.2f} MB")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    # Test model manager
    manager = ModelManager()
    manager.print_summary()
