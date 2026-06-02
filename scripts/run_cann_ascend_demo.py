import argparse
import contextlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def run_command(cmd, timeout=30):
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except FileNotFoundError as exc:
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {"cmd": cmd, "returncode": None, "stdout": exc.stdout or "", "stderr": "timeout"}


def detect_mindspore():
    try:
        import mindspore as ms

        return {
            "available": True,
            "version": ms.__version__,
            "file": getattr(ms, "__file__", ""),
        }
    except Exception as exc:
        return {
            "available": False,
            "version": "",
            "file": "",
            "error": str(exc),
        }


def collect_environment(output_dir):
    ensure_dir(output_dir)

    env_keys = [
        "ASCEND_HOME_PATH",
        "ASCEND_OPP_PATH",
        "TOOLCHAIN_HOME",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "PATH",
    ]

    commands = {
        "npu_smi_info": ["npu-smi", "info"],
        "atc_version": ["atc", "--version"],
        "python_version": [sys.executable, "--version"],
        "pip_mindspore": [sys.executable, "-m", "pip", "show", "mindspore"],
    }

    command_results = {name: run_command(cmd) for name, cmd in commands.items()}
    mindspore_info = detect_mindspore()

    has_cann_tool = shutil.which("atc") is not None or os.environ.get("ASCEND_HOME_PATH")
    has_npu_smi = shutil.which("npu-smi") is not None
    npu_ok = command_results["npu_smi_info"]["returncode"] == 0

    report = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "mindspore": mindspore_info,
        "has_cann_tool_or_env": bool(has_cann_tool),
        "has_npu_smi": bool(has_npu_smi),
        "npu_smi_ok": bool(npu_ok),
        "env": {key: os.environ.get(key, "") for key in env_keys},
        "commands": command_results,
    }

    json_path = os.path.join(output_dir, "cann_env_report.json")
    md_path = os.path.join(output_dir, "cann_env_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("# SmartPlace CANN / Ascend Environment Report")
    lines.append("")
    lines.append(f"- Time: {report['time']}")
    lines.append(f"- Platform: {report['platform']}")
    lines.append(f"- Python: {sys.executable}")
    lines.append(f"- MindSpore available: {mindspore_info['available']}")
    lines.append(f"- MindSpore version: {mindspore_info.get('version', '')}")
    lines.append(f"- CANN env/tool detected: {report['has_cann_tool_or_env']}")
    lines.append(f"- npu-smi detected: {report['has_npu_smi']}")
    lines.append(f"- npu-smi runs successfully: {report['npu_smi_ok']}")
    lines.append("")
    lines.append("## Environment Variables")
    lines.append("")
    for key in env_keys:
        value = report["env"][key]
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Command Outputs")
    lines.append("")
    for name, result in command_results.items():
        lines.append(f"### {name}")
        lines.append(f"- Command: `{' '.join(result['cmd'])}`")
        lines.append(f"- Return code: `{result['returncode']}`")
        if result["stdout"]:
            lines.append("")
            lines.append("```text")
            lines.append(result["stdout"])
            lines.append("```")
        if result["stderr"]:
            lines.append("")
            lines.append("```text")
            lines.append(result["stderr"])
            lines.append("```")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report, json_path, md_path


def capture_to_log(func, log_path):
    buffer = io.StringIO()
    start = time.time()
    error = None

    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        try:
            func()
        except Exception as exc:
            error = exc

    elapsed = time.time() - start
    text = buffer.getvalue()

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")
        f.write(f"[SmartPlace-CANN] elapsed_sec={elapsed:.6f}\n")
        if error is not None:
            f.write(f"[SmartPlace-CANN] error={repr(error)}\n")

    if error is not None:
        raise error

    return log_path


def train_on_ascend(args):
    from scripts.run_mindspore_demo import train

    train_args = SimpleNamespace(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device_target="Ascend",
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        min_real_samples=args.min_real_samples,
        synthetic_count=args.synthetic_count,
    )

    log_path = os.path.join(args.output_dir, f"cann_ascend_train_{timestamp()}.log")
    return capture_to_log(lambda: train(train_args), log_path)


def export_mindir_on_ascend(args):
    ensure_dir(args.output_dir)

    def _run():
        import numpy as np
        import mindspore as ms
        from mindspore import Tensor, context, export, load_checkpoint, load_param_into_net
        from scripts.run_mindspore_demo import SmallCNN

        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args.device_id)

        ckpt_path = args.ckpt_path
        if not ckpt_path:
            ckpt_path = os.path.join(args.output_dir, "mindspore_aux_scorer.ckpt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = SmallCNN(num_classes=3)
        params = load_checkpoint(ckpt_path)
        load_param_into_net(model, params)
        model.set_train(False)

        dummy = Tensor(np.zeros((1, 3, args.image_size, args.image_size), dtype=np.float32), ms.float32)
        export_base = os.path.join(args.output_dir, "smartplace_aux_scorer_ascend")
        export(model, dummy, file_name=export_base, file_format="MINDIR")

        mindir_path = export_base + ".mindir"
        print("=" * 80)
        print("[CANN/Ascend] MindIR export finished")
        print(f"[MindSpore] version={ms.__version__}")
        print("[Device] device_target=Ascend")
        print(f"[Device] device_id={args.device_id}")
        print(f"[Input] dummy_shape={dummy.shape}")
        print(f"[Model] ckpt_path={ckpt_path}")
        print(f"[Output] mindir_path={mindir_path}")
        print("=" * 80)

    log_path = os.path.join(args.output_dir, f"cann_ascend_export_{timestamp()}.log")
    return capture_to_log(_run, log_path)


def infer_on_ascend(args):
    from scripts.run_mindspore_demo import infer

    image_path = args.image_path
    if not image_path:
        image_path = find_default_image(args.data_dir)
    if not image_path:
        raise FileNotFoundError("No image_path provided and no scored composite image was found.")

    ckpt_path = args.ckpt_path or os.path.join(args.output_dir, "mindspore_aux_scorer.ckpt")

    infer_args = SimpleNamespace(
        device_target="Ascend",
        ckpt_path=ckpt_path,
        image_path=image_path,
        image_size=args.image_size,
    )

    log_path = os.path.join(args.output_dir, f"cann_ascend_infer_{timestamp()}.log")
    return capture_to_log(lambda: infer(infer_args), log_path)


def find_default_image(data_dir):
    if not os.path.exists(data_dir):
        return ""
    for name in sorted(os.listdir(data_dir)):
        lower = name.lower()
        if "score_" in lower and lower.endswith((".png", ".jpg", ".jpeg")):
            return os.path.join(data_dir, name)
    return ""


def write_run_summary(args, artifacts, env_report):
    summary_path = os.path.join(args.output_dir, "cann_ascend_summary.md")
    lines = []
    lines.append("# SmartPlace CANN / Ascend Run Summary")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This run verifies the CANN / Ascend advanced item by executing the SmartPlace "
        "MindSpore auxiliary scorer on Ascend, exporting a MindIR model, and recording "
        "environment and runtime logs."
    )
    lines.append("")
    lines.append("## Inputs And Outputs")
    lines.append("")
    lines.append(f"- Data directory: `{args.data_dir}`")
    lines.append(f"- Output directory: `{args.output_dir}`")
    lines.append(f"- Image size: `{args.image_size}`")
    lines.append(f"- Device target: `Ascend`")
    lines.append(f"- Device id: `{args.device_id}`")
    lines.append("")
    lines.append("## Environment Check")
    lines.append("")
    lines.append(f"- MindSpore available: `{env_report['mindspore']['available']}`")
    lines.append(f"- MindSpore version: `{env_report['mindspore'].get('version', '')}`")
    lines.append(f"- CANN env/tool detected: `{env_report['has_cann_tool_or_env']}`")
    lines.append(f"- npu-smi runs successfully: `{env_report['npu_smi_ok']}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for key, path in artifacts.items():
        if path:
            lines.append(f"- {key}: `{path}`")
    lines.append("")
    lines.append("## Report Use")
    lines.append("")
    lines.append(
        "For the course report, use the environment report plus train/export/infer logs "
        "as evidence that SmartPlace has a CANN / Ascend execution path. If this script "
        "is run on a local PC without Ascend hardware, the environment report records "
        "the missing CANN/NPU state and should not be claimed as a completed CANN run."
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return summary_path


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="env", choices=["env", "train", "export", "infer", "all"])
    parser.add_argument("--data_dir", default="outputs/composites")
    parser.add_argument("--output_dir", default="outputs/cann")
    parser.add_argument("--ckpt_path", default="")
    parser.add_argument("--image_path", default="")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_real_samples", type=int, default=12)
    parser.add_argument("--synthetic_count", type=int, default=60)
    return parser


def main():
    args = build_argparser().parse_args()
    ensure_dir(args.output_dir)

    env_report, env_json, env_md = collect_environment(args.output_dir)
    artifacts = {
        "environment_json": env_json,
        "environment_markdown": env_md,
        "train_log": "",
        "export_log": "",
        "infer_log": "",
        "summary": "",
    }

    if args.mode in {"train", "all"}:
        artifacts["train_log"] = train_on_ascend(args)

    if args.mode in {"export", "all"}:
        artifacts["export_log"] = export_mindir_on_ascend(args)

    if args.mode in {"infer", "all"}:
        artifacts["infer_log"] = infer_on_ascend(args)

    artifacts["summary"] = write_run_summary(args, artifacts, env_report)

    print("=" * 80)
    print("[SmartPlace-CANN] artifacts")
    for key, value in artifacts.items():
        print(f"[{key}] {value}")
    print("=" * 80)


if __name__ == "__main__":
    main()
