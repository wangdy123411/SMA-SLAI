"""Run the report's online workload from a configured SLAI checkout."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    meta = json.loads((Path(__file__).resolve().parents[1] / "provenance.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--qps", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--trace-file", type=Path, default=Path(meta["trace_path_in_upstream"]))
    parser.add_argument("--print-command", action="store_true", help="Show the benchmark command.")
    args = parser.parse_args()
    if not math.isfinite(args.qps) or args.qps <= 0:
        parser.error("--qps must be finite and positive")
    flags = dict(meta["common_benchmark_flags"])
    flags.update(model_name=str(args.model_dir.resolve()),
                 output_dir=str((args.out_dir or Path("results") / f"ema-q{args.qps:g}").resolve()),
                 poisson_request_interval_generator_qps=args.qps,
                 trace_request_length_generator_trace_file=str(args.trace_file.resolve()))
    command = [sys.executable, "-m", meta["benchmark_module"]]
    for name, value in sorted(flags.items()):
        command.extend([f"--{name}", str(value).lower() if isinstance(value, bool) else str(value)])
    env_name = meta["runtime_environment_variable"]
    alpha = str(meta["paper_alpha_previous_estimate_weight"])
    if args.print_command:
        print(shlex.join(["env", f"{env_name}={alpha}", *command]))
        return
    if not Path(meta["modified_file"]).is_file():
        parser.error("Run from the SLAI checkout containing the scheduler.")
    if not (args.model_dir / "config.json").is_file():
        parser.error("--model-dir must contain the pinned snapshot's config.json")
    if not args.trace_file.is_file() or hashlib.sha256(args.trace_file.read_bytes()).hexdigest() != meta["trace_sha256"]:
        parser.error("The length trace does not match the experiment. Use the LF file from the pinned upstream commit.")
    subprocess.run(command, env={**os.environ, env_name: alpha}, check=True)


if __name__ == "__main__":
    main()
