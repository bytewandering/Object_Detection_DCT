# CoCo Person DataLoader

同一份 RGB 图像和 COCO 框标注用于：是否存在前景的分类、单类前景检测、图像重建。
依赖 PyTorch、NumPy、Pillow；加载本 Dataset 不需要导入 NanoDet 或安装 pycocotools。

当前“仅人作为前景”的 COCO 数据选择、下载方法与目录说明见
[COCO_PERSON.md](COCO_PERSON.md)。

## 代码结构

| 文件 | 职责 |
|---|---|
| `__init__.py` | 导出 `CoCoPersonDataLoader` |
| `dataloader.py` | 定义 `CoCoPersonDataLoader`，创建本目录 Dataset 并配置批量加载 |
| `dataset.py` | 定义 `FullCoderDataset`，读取和校验单前景类别 COCO 标注 |
| `transforms.py` | 定义 `FullCoderTransform`，同步变换图像、框和有效区域 |
| `collate.py` | 定义 `fullcoder_collate`，堆叠 Tensor 并保留变长标注列表 |

统一入口为 `Adaptive_Compression.dataloaders.build_dataloader`，参数规则见
[DataLoader 配置](../README.md)。

## 数据文件组织

```text
data/
  images/train/positive.png
  images/train/negative.png
  images/val/...
  annotations/instances_train.json
  annotations/instances_val.json
```

最小 COCO JSON 示例（这里两张图片的实际尺寸均应是 640×480）：

```json
{
  "images": [
    {"id": 1, "file_name": "positive.png", "width": 640, "height": 480},
    {"id": 2, "file_name": "negative.png", "width": 640, "height": 480}
  ],
  "categories": [{"id": 1, "name": "foreground"}],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1,
     "bbox": [100, 80, 60, 120], "area": 7200, "iscrowd": 0}
  ]
}
```

只声明一个前景类别，不声明背景类别。所有前景实例都画框，框格式是原图像素坐标
`[x, y, width, height]`。类别 ID 可以不是 1，Dataset 会统一映射为检测标签 0。
确认无前景的图片保留在 `images` 中，没有对应 annotation；未标注图片不能当作负样本。
图像 ID、标注 ID 分别唯一，图片尺寸必须与 JSON 一致。train/val/test 使用一致的类别定义。

## 使用

从 `/home/hzhao/project` 运行：

```python
from Adaptive_Compression.dataloaders import build_dataloader

loader = build_dataloader({
    "name": "CoCoPersonDataLoader",
    "img_path": "/path/to/data/images/train",
    "ann_path": "/path/to/data/annotations/instances_train.json",
    "input_size": [320, 320],
    "batch_size": 4,
    "num_workers": 0,
}, mode="train")
meta = next(iter(loader))
dataset = loader.dataset

```

也可以直接从 `Adaptive_Compression.dataloaders.CoCo_Person` 导入
`CoCoPersonDataLoader`，传入相同参数（去掉 `name`，直接传入 `mode`）。
统一入口支持的名称为 `CoCoPersonDataLoader`、`CoCo_Person`、`FullCoderDataset`。

`img_path` 和 `ann_path` 必填。此数据集不会自动划分 train/val/test，需分别传入对应的
图像目录和标注文件；`mode` 只控制增强及 DataLoader 的默认 shuffle。
`input_size` 默认 `(320, 320)`，按 `(H, W)` 解释；`flip_probability` 默认 `0.5`，
仅训练模式生效。DataLoader 不允许空数据集。

输入高宽必须是正的 32 倍数。处理顺序是 RGB 转换、等比例 resize、居中黑色补边、
可选水平翻转。框、有效区域掩码和变换矩阵同步更新；图像与重建目标均为 float32 RGB
`[0,1]`，不做均值/标准差归一化。val/test 禁用随机翻转。需要复现训练增强时设置
Python/PyTorch 随机种子；多进程加载也要保持 worker 的种子配置可复现。

## Batch 接口

| 字段 | 类型及形状 | 坐标/含义 |
|---|---|---|
| img | Tensor `[B,3,H,W]` | 模型输入 |
| reconstruction_target | Tensor `[B,3,H,W]` | 与输入对齐的重建目标 |
| valid_mask | Tensor `[B,1,H,W]` | 真实内容 1，补边 0 |
| classification_target | Tensor `[B,1]` | 存在有效前景框为 1，无则 0 |
| classification_weight | Tensor `[B,1]` | 仅有 ignore/crowd 框时为 0，其余为 1 |
| gt_bboxes | list of float32 NumPy `[Ni,4]` | 输入图像坐标，xyxy |
| gt_labels | list of int64 NumPy `[Ni]` | 前景类别统一为 0 |
| gt_bboxes_ignore | list of float32 NumPy `[Mi,4]` | 输入坐标的忽略框 |
| warp_matrix | list of NumPy `[3,3]` | 原图坐标 → 模型输入坐标 |
| img_info | dict of lists | id、file_name、原始 width/height |

移动上述五个 Tensor 字段到 GPU；框、warp_matrix、img_info 保留在 CPU，匹配现有
检测 loss 和 NumPy 后处理。Tensor 移动应在训练器中完成，Dataset worker 不操作 GPU。
忽略框不等于确认的背景：仅含此类框的图片分类权重为 0，但仍参与重建训练。

训练器需要使用分类权重和有效区域掩码，例如：

```python
import torch.nn.functional as F

weight = meta["classification_weight"]
cls_elementwise = F.binary_cross_entropy(
    outputs["classification"], meta["classification_target"], reduction="none")
loss_cls = (cls_elementwise * weight).sum() / weight.sum().clamp_min(1)

mask = meta["valid_mask"]
error = (outputs["reconstructed_image"] - meta["reconstruction_target"]).square()
loss_rec = (error * mask).sum() / (mask.sum() * error.shape[1]).clamp_min(1)

# 训练模式下 detection 是预测 Tensor：
loss_det, det_states = model.detector.head.loss(outputs["detection"], meta)
```

三个 loss 的权重由训练器配置。`gt_bboxes_ignore` 只提供忽略区域信息，检测 loss
是否使用它取决于检测头和 assigner 的实现。
本 Dataset 不提供 NanoDet evaluator 的 `coco_api` 属性；需要 COCO 指标时，可由评估器
使用 `dataset.ann_path` 加载标注。
