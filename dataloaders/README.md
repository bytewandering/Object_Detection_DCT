# DataLoader 配置

每个子目录的 `dataloader.py` 定义一个 PyTorch DataLoader 子类，固定使用本目录的 Dataset、transform 和 collate。
顶层 `build_dataloader(config, mode="train")` 按 name 创建对应类。

```yaml
dataloader:
  train:
    name: DareLabFaceDataLoader
    img_path: /home/hzhao/ZHN/datasets/DareLab_Face
    input_size: [320, 320]
    batch_size: 4
    num_workers: 4
    persistent_workers: true
    prefetch_factor: 2
    pin_memory: true
    drop_last: false
  val:
    name: DareLabFaceDataLoader
    img_path: /home/hzhao/ZHN/datasets/DareLab_Face
    input_size: [320, 320]
    batch_size: 8
    num_workers: 0
```

```python
from Adaptive_Compression.dataloaders import build_dataloader
train_loader = build_dataloader(cfg.dataloader.train, mode="train")
val_loader = build_dataloader(cfg.dataloader.val, mode="val")
```

COCO 使用 `CoCoPersonDataLoader` 并指定 img_path、ann_path。
旧 name `FullCoderDataset` 和 `DareLabFaceDataset` 仍可用；workers 是 num_workers 的兼容别名。
shuffle 默认仅 train 开启，传入 sampler 或 batch_sampler 时默认关闭。
persistent_workers 默认为 false，启用时需要 num_workers > 0；prefetch_factor 也仅用于多进程。
使用 batch_sampler 时保持 batch_size=1、shuffle=None、drop_last=False，遵循 PyTorch 参数互斥规则。
sampler、batch_sampler、worker_init_fn、generator 等对象应通过 Python 传入。
完整参数见两个类的构造函数；数据划分仍由 Dataset 的 mode 决定。

## 目录与入口

| 目录 | DataLoader | Dataset | 说明 |
|---|---|---|---|
| `CoCo_Person/` | `CoCoPersonDataLoader` | `FullCoderDataset` | [单前景类别 COCO 数据](CoCo_Person/README.md) |
| `DareLab_Face/` | `DareLabFaceDataLoader` | `DareLabFaceDataset` | [人脸数据及自动划分规则](DareLab_Face/README.md) |
| `DareLab_Face_DCT/` | `DareLabFaceDCTDataLoader` | `DareLabFaceDCTDataset` | [人脸数据的 8×8 块 DCT 系数输入](DareLab_Face_DCT/README.md) |
| `Face_FOMO_DCT/` | `FaceFOMODCTDataLoader` | `FaceFOMODCTDataset` | [Edge Impulse 人脸数据，letterbox 后即时 DCT 编码](Face_FOMO_DCT/README.md) |

每个子目录包含 `__init__.py`、`dataloader.py`、`dataset.py`、`transforms.py`、
`collate.py` 和 `README.md`。Dataset、transform 和 collate 由各自 DataLoader 固定选择，
不通过配置注入。

完整统一入口为 `build_dataloader(config, mode="train", pin_memory=None)`：

- `config` 为参数映射，必须包含 `name`；入口深拷贝配置后取出名称，不修改原配置。
- COCO 名称支持 `CoCoPersonDataLoader`、`CoCo_Person`、`FullCoderDataset`。
- DareLab 名称支持 `DareLabFaceDataLoader`、`DareLab_Face`、`DareLabFaceDataset`。
- `mode` 支持 `train`、`val`、`test`，通过函数参数传入。当前入口不会移除配置中的
  `mode`，因此不要在 `config` 内再设置它，否则会重复传参。
- 显式传入 `pin_memory=True/False` 会覆盖配置中的同名值；传入 `None` 时保留配置值，
  配置也未指定时采用 DataLoader 默认值 `False`。

`pin_memory=True` 用于将加载得到的 Tensor 放入页锁定主机内存，以便向 GPU 传输；
它本身不将数据移动到 GPU。变长 NumPy 标注仍保留为列表。

## 通用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `batch_size` | `1` | 每批样本数 |
| `num_workers` | `0` | `0` 表示在主进程加载 |
| `workers` | `None` | `num_workers` 的兼容别名；与显式非零 `num_workers` 不一致时报错 |
| `shuffle` | `None` | 自动决定：仅 train 且未指定 sampler/batch_sampler 时开启 |
| `pin_memory` | `False` | 是否启用 Tensor 页锁定内存 |
| `drop_last` | `False` | 是否丢弃最后一个不完整批次 |
| `persistent_workers` | `False` | 是否保留 worker 进程，开启时要求 `num_workers > 0` |
| `prefetch_factor` | `None` | 传递给 PyTorch，仅多进程加载时可设置预取批次数 |

其他参数包括 `sampler`、`batch_sampler`、`timeout`、`worker_init_fn`、
`multiprocessing_context`、`generator`、`pin_memory_device` 和 `in_order`，
见各目录 `dataloader.py` 的构造函数。两个 DataLoader 都会拒绝空数据集。

COCO 通过传入不同 `img_path`/`ann_path` 选择数据划分，`mode` 不会自动筛选图片；
DareLab 则由 Dataset 根据 `mode` 自动划分，val 和 test 使用相同图片。
