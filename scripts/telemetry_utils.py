"""Telemetry and benchmark reporting utilities."""
import os
import sys
import time
import json
import resource
from datetime import datetime
from pathlib import Path
import torch

def get_current_ram_mb() -> float:
    """Returns current process Resident Set Size (RSS) in Megabytes."""
    try:
        # On macOS ru_maxrss is in bytes; on Linux in kilobytes
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return usage / (1024 * 1024)
        else:
            return usage / 1024
    except Exception:
        return 0.0

def get_mps_ram_mb() -> float:
    """Returns MPS GPU allocated memory in Megabytes if available."""
    if torch.backends.mps.is_available():
        try:
            return torch.mps.current_allocated_memory() / (1024 * 1024)
        except Exception:
            return 0.0
    return 0.0

def save_benchmark_report(
    title: str,
    table_str: str,
    metrics_dict: dict,
    output_path: str | Path | None = None,
    default_filename_prefix: str = "benchmark",
) -> Path:
    """Saves benchmark results to both stdout and a structured markdown file."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if output_path is None:
        out_dir = Path("./data/benchmarks")
        out_dir.mkdir(parents=True, exist_ok=True)
        report_file = out_dir / f"{default_filename_prefix}_{timestamp}.md"
    else:
        report_file = Path(output_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)

    report_content = f"""# {title}

**Generated at**: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`
**PyTorch Version**: `{torch.__version__}`
**Apple Silicon MPS Available**: `{torch.backends.mps.is_available()}`

---

## Benchmark Results

```text
{table_str.strip()}
```

## Raw Telemetry Data (JSON)

```json
{json.dumps(metrics_dict, indent=2)}
```
"""
    report_file.write_text(report_content)
    print(f"\n📁 Benchmark report successfully written to: {report_file}")
    return report_file
