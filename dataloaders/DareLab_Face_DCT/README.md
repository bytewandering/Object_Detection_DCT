# DareLab Face DCT DataLoader

读取 `Adaptive_Compression/tools/generate_dct_dataset.py` 生成的 8×8 块 DCT 系数，
默认路径为 `/home/hzhao/ZHN/datasets/DareLab_Face_DCT`。标注、类别校验和分类标签与
[DareLab_Face](../DareLab_Face/README.md) 相同，数据划分不同（见下）。

## 数据划分

划分写在 COCO 标注文件里，Dataset 使用 `ann_path` 中的全部图片，不再自行筛选。
未指定 `ann_path` 时，train 读取 `img_path/annotations/train.json`，val 和 test 都读取
`img_path/annotations/val.json`。两个文件是从 `label.json` 按时间拆出的子集，
保留原图像/标注 ID 和像素坐标。

| 集合 | 内容 | 帧数 | 有人脸的帧 |
|---|---|---|---|
| train | 每段录像按帧顺序的前 60% | 1441 | 806 |
| val / test | 每段录像的后 40% | 964 | 555 |

| 录像 | 帧数 | train（前 60%） | val/test（后 40%） |
|---|---|---|---|
| rec_20260819_092243（无人脸） | 401 | 240 帧 | 161 帧 |
| rec_20260819_092256（无人脸） | 401 | 240 帧 | 161 帧 |
| rec_20260914_145600 | 801 | 480 帧（人脸 472） | 321 帧（人脸 321） |
| rec_20260914_150101 | 802 | 481 帧（人脸 334） | 321 帧（人脸 234） |

切分点为每段录像的 `int(帧数 × 0.6)`，约第 6 秒（无人脸录像）或第 12 秒处。四段录像都同时
出现在训练和验证中，验证衡量的是同一场景中后续时间段的表现；切分点两侧的相邻帧几乎相同，
但只占很小一部分。

rec_20260819_* 两段没有人脸，只提供负样本。只有两段有人脸的录像，且可能是同一人、
同一场景，指标不代表对新人员或新场景的泛化表现。修改划分时直接编辑这两个 JSON，
保持两个文件没有重复图片，并同步修改 DareLab_Face 下的同名文件，使 RGB 与 DCT 实验可比。

## 数据结构

```text
DareLab_Face_DCT/
  label.json                 # 复制自 DareLab_Face，全部图片，bbox 为原图像素坐标
  annotations/train.json     # 训练集划分（COCO 格式）
  annotations/val.json       # 验证/测试集划分（COCO 格式）
  dct_meta.json              # 生成参数、存储方式和量化表
  rec_xxx/f0000_xxx/         # 对应 label.json 中 file_name 去掉 .png 后缀
    Y_00.png … Y_63.png      # [H/8, W/8]，00 为直流，按 JPEG zigzag 顺序
    Cb_00.png … Cb_63.png    # [H/16, W/16]，4:2:0
    Cr_00.png … Cr_63.png
```

Dataset 根据 `dct_meta.json` 自动识别存储方式：

- 量化版（`quality` 非 0，当前数据）：uint8 PNG，值为量化后系数 + 128。
- 无损版（`--quality 0`）：uint16 PNG，值为 round(系数) + 32768。

## 输出

`img` 为 `[kY+kCb+kCr, H/8, W/8]`（320×240 原图对应 `[kY+kCb+kCr, 30, 40]`），依次是
Y 的前 kY 个、Cb 的前 kCb 个、Cr 的前 kCr 个 zigzag 系数。Cb/Cr 原生为 `[H/16, W/16]`，
每个块最近邻复制到对应的 2×2 个亮度块上（上采样到 Y 的网格）。其余字段沿用
[CoCo Person 的 Batch 接口](../CoCo_Person/README.md#batch-接口)，区别如下：

| 字段 | 说明 |
|---|---|
| img | 量化后的整数系数（不反量化）按通道对称归一化到 [-1, 1]；不做缩放或补边 |
| reconstruction_target | 与 img 相同的副本（DCT 域重建目标） |
| valid_mask | `[1, H/8, W/8]` 全 1 |
| gt_bboxes / gt_bboxes_ignore | 块网格坐标 xyxy，即原图像素坐标 / 8 |
| warp_matrix | 原图像素坐标 → 块网格坐标，`diag(1/8, 1/8, 1)` |

不做任何数据增强（包括翻转），train/val/test 的处理完全相同。

## 归一化

每个系数通道 c（Y/Cb/Cr 各 64 个 zigzag 系数，共 192 个）单独归一化：

```
s_c = max(|p1_c|, |p99_c|)        # 训练集上该通道量化后整数的 1% 与 99% 分位数
img_c = clip(q_c / s_c, -1, 1)
```

系数 0 仍映射为 0，补边填 0 即表示"无信号"，输出范围对称，适合对称 INT8 量化。
1%/99% 分位数都为 0 的高频通道改用该通道的最大绝对值作为 s_c，训练集中恒为 0 的通道
s_c = 1。分位数按通道在原生网格（Cb/Cr 为 15×20）上统计，由 `compute_normalization_stats`
生成；不传 `normalization_stats` 时不归一化，输出原始整数。

```python
from Object_Detection_DCT.dataloaders.DareLab_Face_DCT import compute_normalization_stats
compute_normalization_stats("/home/hzhao/ZHN/datasets/DareLab_Face_DCT/train_p01_p99_symmetric.npz")
```

当前文件按 `annotations/train.json`（1441 张）统计，`[16, 4, 4]` 所选 24 个通道的 s_c 为
Y 122、53、70、30 … 2，Cb 36、7、9、4，Cr 23、5、7、2；训练集与验证集各约 2% 的值达到 ±1。
修改划分后需重新生成。npz 中保存 lower、upper、max_abs、scale、percentiles、ann_path。

## 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `img_path` | `/home/hzhao/ZHN/datasets/DareLab_Face_DCT` | 数据根目录 |
| `ann_path` | train: `annotations/train.json`；val/test: `annotations/val.json` | 该划分的 COCO 标注文件 |
| `num_dct_coefficients` | `64` | 整数 k 表示 Y/Cb/Cr 各取前 k 个；`[kY, kCb, kCr]` 分别指定，每项 0–64、总和大于 0（如 `[16, 4, 4]`、`[8, 0, 0]` 只用亮度）。只读取需要的 PNG |
| `normalization_stats` | `None` | 归一化统计 npz（见上）；`None` 时输出未归一化的整数 |

## 配置示例

```yaml
dataloader:
  train:
    name: DareLabFaceDCTDataLoader
    img_path: /home/hzhao/ZHN/datasets/DareLab_Face_DCT
    ann_path: /home/hzhao/ZHN/datasets/DareLab_Face_DCT/annotations/train.json
    num_dct_coefficients: [16, 4, 4]   # Y 16 层，Cb/Cr 各 4 层
    normalization_stats: /home/hzhao/ZHN/datasets/DareLab_Face_DCT/train_p01_p99_symmetric.npz
    batch_size: 4
    num_workers: 4
  val:
    name: DareLabFaceDCTDataLoader
    img_path: /home/hzhao/ZHN/datasets/DareLab_Face_DCT
    ann_path: /home/hzhao/ZHN/datasets/DareLab_Face_DCT/annotations/val.json
    num_dct_coefficients: [16, 4, 4]
    normalization_stats: /home/hzhao/ZHN/datasets/DareLab_Face_DCT/train_p01_p99_symmetric.npz
    batch_size: 4
    num_workers: 4
```

统一入口也支持 `name: DareLab_Face_DCT` 和 `name: DareLabFaceDCTDataset`。
