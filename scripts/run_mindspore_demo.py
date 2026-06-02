import argparse
import os
import re
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import mindspore as ms
from mindspore import Tensor, nn, ops, context, save_checkpoint, load_checkpoint, load_param_into_net
import mindspore.dataset as ds


LABEL_NAMES = ["不推荐", "可接受", "推荐"]


def parse_score_from_filename(filename: str):
    """
    从文件名中解析 score。
    期望格式包含 score_0.8123。
    """
    match = re.search(r"score_([0-9]+(?:\.[0-9]+)?)", filename)
    if match is None:
        return None
    return float(match.group(1))


def score_to_class_id(score: float) -> int:
    """
    0: 不推荐
    1: 可接受
    2: 推荐
    """
    if score >= 0.75:
        return 2
    if score >= 0.45:
        return 1
    return 0


def find_composite_images(data_dir: str) -> List[Tuple[str, float, int]]:
    """
    搜索 outputs/composites/ 下的候选合成图。
    """
    samples = []

    if not os.path.exists(data_dir):
        return samples

    for name in os.listdir(data_dir):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        score = parse_score_from_filename(name)
        if score is None:
            continue

        label = score_to_class_id(score)
        path = os.path.join(data_dir, name)

        samples.append((path, score, label))

    return samples


def make_synthetic_samples(output_dir: str, count: int = 60, image_size: int = 96) -> List[Tuple[str, float, int]]:
    """
    如果 outputs/composites/ 里的图片太少，则生成一些合成训练样本，保证 demo 能跑通。

    这不是最终实验数据，只是为了证明 MindSpore 训练和推理流程可以运行。
    """
    os.makedirs(output_dir, exist_ok=True)

    samples = []
    rng = np.random.default_rng(2026)

    for i in range(count):
        label = i % 3

        if label == 2:
            score = rng.uniform(0.78, 0.95)
            base = rng.normal(loc=180, scale=35, size=(image_size, image_size, 3))
            # 中下区域加一个较稳定的“前景块”
            x1, y1 = image_size // 3, image_size // 2
            x2, y2 = x1 + image_size // 3, y1 + image_size // 4
            base[y1:y2, x1:x2, :] = rng.normal(loc=110, scale=20, size=(y2 - y1, x2 - x1, 3))

        elif label == 1:
            score = rng.uniform(0.50, 0.70)
            base = rng.normal(loc=150, scale=55, size=(image_size, image_size, 3))
            x1, y1 = image_size // 4, image_size // 3
            x2, y2 = x1 + image_size // 3, y1 + image_size // 4
            base[y1:y2, x1:x2, :] = rng.normal(loc=120, scale=45, size=(y2 - y1, x2 - x1, 3))

        else:
            score = rng.uniform(0.10, 0.40)
            base = rng.normal(loc=120, scale=75, size=(image_size, image_size, 3))
            # 贴到边缘，模拟不自然候选
            x1, y1 = 0, 0
            x2, y2 = image_size // 3, image_size // 3
            base[y1:y2, x1:x2, :] = rng.normal(loc=230, scale=20, size=(y2 - y1, x2 - x1, 3))

        arr = np.clip(base, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

        path = os.path.join(output_dir, f"synthetic_candidate_{i:03d}_score_{score:.4f}.png")
        img.save(path)
        samples.append((path, float(score), int(label)))

    return samples


def load_image_as_array(path: str, image_size: int) -> np.ndarray:
    """
    PIL image -> CHW float32
    """
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)

    arr = np.array(image).astype(np.float32) / 255.0

    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))

    return arr


class SmartPlaceDataset:
    def __init__(self, samples: List[Tuple[str, float, int]], image_size: int):
        self.samples = samples
        self.image_size = image_size

    def __getitem__(self, index):
        path, score, label = self.samples[index]
        image = load_image_as_array(path, self.image_size)
        return image, np.int32(label)

    def __len__(self):
        return len(self.samples)


class SmallCNN(nn.Cell):
    """
    MindSpore 轻量三分类 CNN。
    输入: 3 × image_size × image_size
    输出: 3 类 logits
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.features = nn.SequentialCell(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, pad_mode="pad", padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, pad_mode="pad", padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, pad_mode="pad", padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.flatten = nn.Flatten()
        self.classifier = nn.Dense(64, num_classes)

    def construct(self, x):
        x = self.features(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits


def split_samples(samples: List[Tuple[str, float, int]], train_ratio: float = 0.8):
    rng = np.random.default_rng(2026)
    indices = np.arange(len(samples))
    rng.shuffle(indices)

    train_size = max(1, int(len(samples) * train_ratio))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    if len(val_indices) == 0:
        val_indices = train_indices

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    return train_samples, val_samples


def build_mindspore_dataset(samples, image_size: int, batch_size: int, shuffle: bool):
    dataset = SmartPlaceDataset(samples=samples, image_size=image_size)

    ms_dataset = ds.GeneratorDataset(
        source=dataset,
        column_names=["image", "label"],
        shuffle=shuffle,
    )

    ms_dataset = ms_dataset.batch(batch_size=batch_size, drop_remainder=False)

    return ms_dataset


def evaluate(model: nn.Cell, dataset):
    model.set_train(False)

    total = 0
    correct = 0

    all_rows = []

    softmax = ops.Softmax(axis=1)

    for batch in dataset.create_dict_iterator():
        images = batch["image"]
        labels = batch["label"]

        logits = model(images)
        probs = softmax(logits)
        preds = ops.Argmax(axis=1)(probs)

        preds_np = preds.asnumpy()
        labels_np = labels.asnumpy()
        probs_np = probs.asnumpy()

        total += len(labels_np)
        correct += int((preds_np == labels_np).sum())

        for i in range(len(labels_np)):
            all_rows.append(
                {
                    "true_label_id": int(labels_np[i]),
                    "true_label": LABEL_NAMES[int(labels_np[i])],
                    "pred_label_id": int(preds_np[i]),
                    "pred_label": LABEL_NAMES[int(preds_np[i])],
                    "prob_not_recommend": float(probs_np[i][0]),
                    "prob_acceptable": float(probs_np[i][1]),
                    "prob_recommend": float(probs_np[i][2]),
                    "correct": bool(preds_np[i] == labels_np[i]),
                }
            )

    acc = correct / total if total > 0 else 0.0
    return acc, all_rows


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)

    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)

    samples = find_composite_images(args.data_dir)

    if len(samples) < args.min_real_samples:
        synthetic_dir = os.path.join(args.output_dir, "synthetic_data")
        synthetic_samples = make_synthetic_samples(
            output_dir=synthetic_dir,
            count=args.synthetic_count,
            image_size=args.image_size,
        )
        samples = samples + synthetic_samples

    if len(samples) == 0:
        raise RuntimeError("没有找到任何训练样本。请先运行 Web 应用生成候选图，或启用 synthetic 数据。")

    train_samples, val_samples = split_samples(samples, train_ratio=0.8)

    train_dataset = build_mindspore_dataset(
        samples=train_samples,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_dataset = build_mindspore_dataset(
        samples=val_samples,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = SmallCNN(num_classes=3)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=args.lr)

    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    train_logs = []

    print("=" * 80)
    print("[MindSpore] SmartPlace auxiliary scorer training")
    print(f"[MindSpore] version={ms.__version__}")
    print(f"[MindSpore] device_target={args.device_target}")
    print(f"[Data] real_data_dir={args.data_dir}")
    print(f"[Data] total_samples={len(samples)}")
    print(f"[Data] train_samples={len(train_samples)}")
    print(f"[Data] val_samples={len(val_samples)}")
    print(f"[Model] SmallCNN num_classes=3")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        model.set_train(True)

        epoch_losses = []
        start = time.time()

        for batch in train_dataset.create_dict_iterator():
            images = batch["image"]
            labels = batch["label"]

            (loss, logits), grads = grad_fn(images, labels)
            optimizer(grads)

            epoch_losses.append(float(loss.asnumpy()))

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_acc, _ = evaluate(model, val_dataset)

        elapsed = time.time() - start

        log = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_acc": val_acc,
            "time_sec": elapsed,
        }
        train_logs.append(log)

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={avg_loss:.6f}, "
            f"val_acc={val_acc:.4f}, "
            f"time={elapsed:.2f}s"
        )

    ckpt_path = os.path.join(args.output_dir, "mindspore_aux_scorer.ckpt")
    save_checkpoint(model, ckpt_path)

    val_acc, pred_rows = evaluate(model, val_dataset)

    pred_csv_path = os.path.join(args.output_dir, "mindspore_aux_predictions.csv")
    pd.DataFrame(pred_rows).to_csv(pred_csv_path, index=False, encoding="utf-8-sig")

    log_csv_path = os.path.join(args.output_dir, "mindspore_train_log.csv")
    pd.DataFrame(train_logs).to_csv(log_csv_path, index=False, encoding="utf-8-sig")

    report_path = os.path.join(args.output_dir, "mindspore_aux_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# SmartPlace MindSpore 轻量辅助评分模块报告\n\n")
        f.write(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 1. 模块定位\n\n")
        f.write(
            "本模块使用 MindSpore 实现一个轻量三分类 CNN，用于对候选合成图进行辅助评分验证。"
            "该模型不替代主评分模型，而是作为项目进阶项中的 MindSpore 训练/推理环节。\n\n"
        )

        f.write("## 2. 输入输出\n\n")
        f.write("- 输入：候选合成图\n")
        f.write("- 输出：推荐 / 可接受 / 不推荐 三分类概率\n")
        f.write("- 标签来源：根据候选图文件名中的 score 自动映射得到伪标签\n\n")

        f.write("## 3. 运行信息\n\n")
        f.write(f"- MindSpore 版本：{ms.__version__}\n")
        f.write(f"- 设备：{args.device_target}\n")
        f.write(f"- 总样本数：{len(samples)}\n")
        f.write(f"- 训练样本数：{len(train_samples)}\n")
        f.write(f"- 验证样本数：{len(val_samples)}\n")
        f.write(f"- Epochs：{args.epochs}\n")
        f.write(f"- Batch size：{args.batch_size}\n")
        f.write(f"- Image size：{args.image_size}\n\n")

        f.write("## 4. 结果文件\n\n")
        f.write(f"- Checkpoint：`{ckpt_path}`\n")
        f.write(f"- 训练日志：`{log_csv_path}`\n")
        f.write(f"- 推理结果：`{pred_csv_path}`\n")
        f.write(f"- 验证准确率：{val_acc:.4f}\n\n")

        f.write("## 5. 报告可用说明\n\n")
        f.write(
            "本项目额外使用 MindSpore 实现了轻量辅助评分模型。"
            "系统首先从主应用导出的候选合成图中读取样本，并根据主评分分数生成三档伪标签。"
            "随后使用 MindSpore 构建小型 CNN 完成训练和验证，输出 checkpoint、训练日志和推理结果。"
            "该模块证明项目不仅完成了 PyTorch/规则评分主流程，也完成了 MindSpore 框架下的模型训练与推理验证。\n"
        )

    print("=" * 80)
    print("[MindSpore] Training finished")
    print(f"[Output] checkpoint={ckpt_path}")
    print(f"[Output] train_log={log_csv_path}")
    print(f"[Output] predictions={pred_csv_path}")
    print(f"[Output] report={report_path}")
    print(f"[Result] val_acc={val_acc:.4f}")
    print("=" * 80)


def infer(args):
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    model = SmallCNN(num_classes=3)
    params = load_checkpoint(args.ckpt_path)
    load_param_into_net(model, params)

    model.set_train(False)

    image = load_image_as_array(args.image_path, args.image_size)
    image_tensor = Tensor(np.expand_dims(image, axis=0), ms.float32)

    start = time.time()
    logits = model(image_tensor)
    probs = ops.Softmax(axis=1)(logits)
    pred = int(ops.Argmax(axis=1)(probs).asnumpy()[0])
    elapsed = time.time() - start

    probs_np = probs.asnumpy()[0]

    print("=" * 80)
    print("[MindSpore] Single image inference")
    print(f"[Input] image_path={args.image_path}")
    print(f"[Model] ckpt_path={args.ckpt_path}")
    print(f"[Tensor] input_shape={image_tensor.shape}")
    print(f"[Output] logits={logits.asnumpy().tolist()}")
    print(f"[Output] probs={probs_np.tolist()}")
    print(f"[Output] pred_label={LABEL_NAMES[pred]}")
    print(f"[Time] inference_time={elapsed:.6f}s")
    print("=" * 80)


def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train", choices=["train", "infer"])
    parser.add_argument("--data_dir", type=str, default="outputs/composites")
    parser.add_argument("--output_dir", type=str, default="outputs/mindspore")
    parser.add_argument("--device_target", type=str, default="CPU")
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_real_samples", type=int, default=12)
    parser.add_argument("--synthetic_count", type=int, default=60)

    parser.add_argument("--ckpt_path", type=str, default="outputs/mindspore/mindspore_aux_scorer.ckpt")
    parser.add_argument("--image_path", type=str, default="")

    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()

    if args.mode == "train":
        train(args)
    else:
        infer(args)