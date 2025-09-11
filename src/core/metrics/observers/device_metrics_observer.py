"""Device metrics observer for tracking system resource usage during metric computation."""

import time
import threading
import logging
from typing import List, Dict, Any, Optional
import psutil

from src.models.schemas import MetricResult
from src.core.metrics.metric_observer import MetricObserver

logger = logging.getLogger(__name__)


class DeviceMetricsObserver(MetricObserver):
    """Observer that tracks and reports device metrics during metric computation."""

    def __init__(self):
        """
        Initialize the device metrics observer.
        """

        # Tracking state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._current_metric = ""
        self._total_pairs = 0
        self._processed_pairs = 0

        # Metrics storage
        self._baseline_metrics: Dict[str, Any] = {}
        self._peak_metrics: Dict[str, Any] = {}
        self._final_metrics: Dict[str, Any] = {}
        self._snapshots: List[Dict[str, Any]] = []

    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics snapshot."""
        process = psutil.Process()

        # Get system-wide metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()

        # Get process-specific metrics
        with process.oneshot():
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()

        metrics = {
            "timestamp": time.time(),
            # System-wide metrics
            "system_cpu_percent": cpu_percent,
            "system_memory_used_gb": memory.used / (1024**3),
            "system_memory_total_gb": memory.total / (1024**3),
            "system_memory_percent": memory.percent,
            "system_memory_available_gb": memory.available / (1024**3),
            # Process-specific metrics
            "process_cpu_percent": process_cpu,
            "process_memory_rss_gb": process_memory.rss
            / (1024**3),  # Resident Set Size
            "process_memory_vms_gb": process_memory.vms
            / (1024**3),  # Virtual Memory Size
            # I/O metrics (system-wide)
            "disk_read_mb": disk_io.read_bytes / (1024**2) if disk_io else 0,
            "disk_write_mb": disk_io.write_bytes / (1024**2) if disk_io else 0,
            # Progress tracking
            "processed_pairs": self._processed_pairs,
            "total_pairs": self._total_pairs,
            "progress_percent": (
                (self._processed_pairs / self._total_pairs * 100)
                if self._total_pairs > 0
                else 0
            ),
        }

        return metrics

    def _update_peak_metrics(self, current: Dict[str, Any]) -> None:
        """Update peak metrics with current values where applicable."""
        if not self._peak_metrics:
            self._peak_metrics = current.copy()
            return

        # Update peaks for relevant metrics
        peak_keys = [
            "system_cpu_percent",
            "system_memory_percent",
            "system_memory_used_gb",
            "process_cpu_percent",
            "process_memory_rss_gb",
            "process_memory_vms_gb",
        ]

        for key in peak_keys:
            if key in current and current[key] > self._peak_metrics.get(key, 0):
                self._peak_metrics[key] = current[key]

    def _log_metrics_snapshot(
        self, metrics: Dict[str, Any], snapshot_type: str
    ) -> None:
        """Log a formatted metrics snapshot."""
        timestamp_str = time.strftime("%H:%M:%S", time.localtime(metrics["timestamp"]))

        elapsed_seconds = 0
        if self._baseline_metrics and "timestamp" in self._baseline_metrics:
            elapsed_seconds = metrics["timestamp"] - self._baseline_metrics["timestamp"]

        logger.info(
            "[%s] Device Metrics Snapshot - %s - %s (Elapsed: %.1fs)",
            snapshot_type,
            self._current_metric,
            timestamp_str,
            elapsed_seconds,
        )
        logger.info(
            "  System: CPU %.1f%%, RAM %.2fGB/%.2fGB (%.1f%%), Available %.2fGB",
            metrics["system_cpu_percent"],
            metrics["system_memory_used_gb"],
            metrics["system_memory_total_gb"],
            metrics["system_memory_percent"],
            metrics["system_memory_available_gb"],
        )
        logger.info(
            "  Process: CPU %.1f%%, RSS %.2fGB, VMS %.2fGB",
            metrics["process_cpu_percent"],
            metrics["process_memory_rss_gb"],
            metrics["process_memory_vms_gb"],
        )
        logger.info(
            "  Progress: %d/%d pairs (%.1f%%)",
            metrics["processed_pairs"],
            metrics["total_pairs"],
            metrics["progress_percent"],
        )

    def _continuous_monitor(self) -> None:
        """Continuously monitor system metrics in a separate thread."""
        logger.info(
            "Started continuous monitoring for %s (interval: %ss)",
            self._current_metric,
            2.0,
        )

        while self._monitoring:
            try:
                current = self._get_current_metrics()
                self._update_peak_metrics(current)
                self._snapshots.append(current.copy())

                self._log_metrics_snapshot(current, "CONTINUOUS")

                time.sleep(2.0)

            except Exception as e:
                logger.error("Error during continuous monitoring: %s", e)
                break

    def _log_summary(self) -> None:
        """Log summary of metrics for the completed metric computation."""
        if not self._baseline_metrics or not self._final_metrics:
            logger.warning(
                "Cannot generate summary - missing baseline or final metrics"
            )
            return

        duration = (
            self._final_metrics["timestamp"] - self._baseline_metrics["timestamp"]
            if "timestamp" in self._baseline_metrics
            and "timestamp" in self._final_metrics
            else 0
        )

        logger.info("=" * 60)
        logger.info("DEVICE METRICS SUMMARY: %s", self._current_metric)
        logger.info("=" * 60)
        logger.info("Duration: %.2f seconds", duration)
        logger.info("Total snapshots captured: %d", len(self._snapshots))

        # System metrics comparison
        logger.info("SYSTEM METRICS (Baseline → Peak → Final)")
        logger.info("-" * 40)

        cpu_baseline = self._baseline_metrics.get("system_cpu_percent", 0)
        cpu_peak = self._peak_metrics.get("system_cpu_percent", 0)
        cpu_final = self._final_metrics.get("system_cpu_percent", 0)
        logger.info("CPU: %.1f%% → %.1f%% → %.1f%%", cpu_baseline, cpu_peak, cpu_final)

        mem_baseline = self._baseline_metrics.get("system_memory_used_gb", 0)
        mem_peak = self._peak_metrics.get("system_memory_used_gb", 0)
        mem_final = self._final_metrics.get("system_memory_used_gb", 0)
        logger.info(
            "System RAM: %.2fGB → %.2fGB → %.2fGB", mem_baseline, mem_peak, mem_final
        )

        # Process metrics comparison
        logger.info("PROCESS METRICS (Baseline → Peak → Final)")
        logger.info("-" * 40)

        logger.info(
            "Process CPU: %.1f%% → %.1f%% → %.1f%%",
            self._baseline_metrics.get("process_cpu_percent", 0),
            self._peak_metrics.get("process_cpu_percent", 0),
            self._final_metrics.get("process_cpu_percent", 0),
        )

        proc_mem_baseline = self._baseline_metrics.get("process_memory_rss_gb", 0)
        proc_mem_final = self._final_metrics.get("process_memory_rss_gb", 0)
        proc_mem_change = proc_mem_final - proc_mem_baseline
        logger.info(
            "Process RSS: %.2fGB → %.2fGB → %.2fGB (%+.2fGB)",
            proc_mem_baseline,
            self._peak_metrics.get("process_memory_rss_gb", 0),
            proc_mem_final,
            proc_mem_change,
        )

        # Performance summary
        avg_pairs_per_second = self._total_pairs / duration if duration > 0 else 0
        logger.info("Processing rate: %.2f pairs/second", avg_pairs_per_second)

        logger.info("=" * 60)

    def on_metric_start(self, metric_name: str, total_pairs: int) -> None:
        """Called when metric computation starts."""
        self._current_metric = metric_name
        self._total_pairs = total_pairs
        self._processed_pairs = 0
        self._snapshots.clear()

        # Get baseline metrics
        self._baseline_metrics = self._get_current_metrics()
        self._peak_metrics = self._baseline_metrics.copy()

        logger.info("🚀 Starting metric computation: %s", metric_name)
        logger.info("📊 Total pairs to process: %d", total_pairs)

        # Log baseline snapshot
        self._log_metrics_snapshot(self._baseline_metrics, "START")

        # Start continuous monitoring
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._continuous_monitor, daemon=True
        )
        self._monitor_thread.start()

    def on_pair_processed(
        self, metric_name: str, pair_index: int, result: MetricResult
    ) -> None:
        """Called when a text pair is processed."""
        self._processed_pairs = pair_index + 1

        # Log progress at certain intervals
        if (
            self._processed_pairs % 50 == 0
            or self._processed_pairs == self._total_pairs
        ):
            progress_pct = (self._processed_pairs / self._total_pairs) * 100

            logger.info(
                "📈 Progress: %d/%d pairs (%.1f%%)",
                self._processed_pairs,
                self._total_pairs,
                progress_pct,
            )

    def on_metric_complete(self, metric_name: str, results: List[MetricResult]) -> None:
        """Called when metric computation completes."""
        # Stop continuous monitoring
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

        # Get final metrics snapshot
        self._final_metrics = self._get_current_metrics()
        self._update_peak_metrics(self._final_metrics)

        logger.info("✅ Completed metric computation: %s", metric_name)
        logger.info("📈 Processed %d results", len(results))

        # Log final snapshot
        self._log_metrics_snapshot(self._final_metrics, "END")

        # Log comprehensive summary
        self._log_summary()

    def on_metric_error(self, metric_name: str, error: Exception) -> None:
        """Called when metric computation fails."""
        # Stop continuous monitoring
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

        # Get final metrics even on error
        self._final_metrics = self._get_current_metrics()

        logger.error("❌ Error in metric computation: %s", metric_name)
        logger.error("🚨 Error details: %s", str(error))

        # Log final snapshot
        self._log_metrics_snapshot(self._final_metrics, "ERROR")

        # Log summary even on error
        self._log_summary()
