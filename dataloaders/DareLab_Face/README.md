# DareLab Face DataLoader

默认读取 `/home/hzhao/ZHN/datasets/DareLab_Face/label.json`，可通过 `img_path` 和
`ann_path` 修改位置。图片路径相对于数据根目录。
原始标注包含一个 `face` 类别，bbox 为 COCO 像素坐标 `[x,y,w,h]`。
标注由 YOLO-Face 自动生成；没有框表示没有检测到人脸，并不保证人工确认无人脸。
`score` 和 `has_face` 不参与训练标签计算，分类标签由有效框生成。

`DareLabFaceDataset` 直接继承 PyTorch `Dataset`，独立实现读取、校验及标签生成，
使用本目录的 `DareLabFaceTransform` 完成图像与框的同步变换，不依赖 FullCoderDataset 实现。
无需复制图片或转换 JSON。
输出字段、dtype、坐标和 batch 接口完全沿用
[CoCo Person 的 Batch 接口](../CoCo_Person/README.md#batch-接口)。检测标签统一为 0，
重建目标为缩放、补边及翻转后的 RGB `[0,1]` 图像。

## 代码与数据结构

| 文件 | 职责 |
|---|---|
| `__init__.py` | 导出 `DareLabFaceDataLoader`、`DareLabFaceDataset` 和 `darelab_face_collate` |
| `dataloader.py` | 创建本目录 Dataset，固定使用本目录 collate 并配置批量加载 |
| `dataset.py` | 读取和校验人脸标注，筛选录制目录并划分数据集 |
| `transforms.py` | 定义 `DareLabFaceTransform`，同步变换图像、框和有效区域 |
| `collate.py` | 堆叠 Tensor 并保留变长标注列表 |

```text
DareLab_Face/
  label.json
  recording_a/
    frame_001.jpg
    ...
  recording_b/
    ...
```

JSON 包含 COCO 格式的 `images`、`categories` 和 `annotations`，且唯一类别名必须为
`face`。例如 `file_name: recording_a/frame_001.jpg`；图片尺寸必须与标注一致。

## 配置与使用

配置示例（训练 YAML 中的 `dataloader` 部分）：

```yaml
dataloader:
  train:
    batch_size: 4
    num_workers: 4
    name: DareLabFaceDataLoader
    img_path: /home/hzhao/ZHN/datasets/DareLab_Face
    input_size: [320, 320]
    flip_probability: 0.5
  val:
    batch_size: 4
    num_workers: 4
    name: DareLabFaceDataLoader
    img_path: /home/hzhao/ZHN/datasets/DareLab_Face
    input_size: [320, 320]
```

统一入口也支持 `name: DareLab_Face` 和 `name: DareLabFaceDataset`。`ann_path` 默认是 `img_path/label.json`，
可显式指定其他标注文件。`sequences` 可以是一个目录名或目录名列表；不传时使用
全部录制目录，然后自动交叉划分：

- 按图片的父目录分组，每组内部按图像 ID 排序，每 5 张取第 5 张进入测试集，其他进入训练集。
- `mode="train"`：每组其余图片进入训练集，允许随机水平翻转。
- `mode="test"`：每组第 5、10、15……张图片进入测试集，禁用随机翻转。
- `mode="val"`：使用与 test 相同的图片，禁用随机翻转，不是独立的第三份数据。
- 无随机抽样，不依赖 seed；训练与测试的图像 ID 不重叠，合起来覆盖所选目录的全部图片；实际数量取决于标注内容。
- `sequences` 按 `file_name` 的第一级目录名先筛选录制目录，再应用上述规则。训练和测试应使用相同的目录配置。

这是加载时的逻辑划分，不移动原始图片、不改写 label.json。
连续视频的相邻帧可能分别进入训练和测试，因此结果不代表独立人员或场景的泛化表现。

```python
from Adaptive_Compression.dataloaders import build_dataloader

loader = build_dataloader({
    "name": "DareLabFaceDataLoader",
    "img_path": "/path/to/DareLab_Face",
    "input_size": [320, 320],
    "batch_size": 4,
    "num_workers": 0,
}, mode="train")
meta = next(iter(loader))
dataset = loader.dataset

```

`ann_path` 仍指向完整 JSON；若使用此数据集进行外部 COCO
评估，评估器需按 `dataset.img_ids` 限定图像范围。

本数据集的 batch 函数独立实现于 `collate.py`，训练入口会自动选择 `darelab_face_collate`。

也可以从 `Adaptive_Compression.dataloaders.DareLab_Face` 导入
`DareLabFaceDataLoader`，直接传入参数（去掉 `name`，直接传入 `mode`）。
DataLoader 的 `input_size` 默认 `(320, 320)`，按 `(H, W)` 解释，高宽均须为正的
32 倍数，也支持整数表示正方形尺寸；训练翻转概率 `flip_probability` 默认 `0.5`。
图像等比例缩放后居中黑色补边，不做均值/标准差归一化。
直接创建 `DareLabFaceDataset` 时应显式传入合法的 `input_size`：当前 Dataset 默认值
为 `(320, 240)`，其中 240 不满足 transform 的 32 倍数校验。
筛选或划分后数据集为空时，DataLoader 会报错。

`batch_size`、`num_workers`、`pin_memory` 等通用参数见
[DataLoader 配置](../README.md)。
