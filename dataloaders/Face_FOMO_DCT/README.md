# Face FOMO DCT DataLoader

读取 Edge Impulse 导出的 "Face detection - FOMO" 数据集
（`/home/hzhao/ZHN/datasets/face-detection-fomo-export`），把原始 JPEG 即时编码为 8×8 块 DCT 系数。
输出字段、归一化方式和数据增强与 [DareLab_Face_DCT](../DareLab_Face_DCT/README.md) 相同，
可直接用于 FOMO / NanoDet 的 DCT 训练和测试任务。

## 准备

```bash
python Object_Detection_DCT/dataloaders/Face_FOMO_DCT/prepare_dataset.py                 # 浮点 JPEG Q75
python Object_Detection_DCT/dataloaders/Face_FOMO_DCT/prepare_dataset.py --hw --subsampling 422 --final-rounding floor
```

脚本把 `training/info.labels`、`testing/info.labels` 转成 COCO 格式的 `annotations/train.json`
（314 张、435 个框）和 `annotations/val.json`（99 张、152 个框），bbox 为原图像素坐标。
超出图像的框会裁剪到图像内；完全落在图像外的 2 个框（37013、37049）被丢弃。
同时按训练集统计归一化文件 `train_p01_p99_symmetric_<W>x<H>_<编码器>.npz`，只统计原图覆盖的块，
不统计补边区域。统计结果依赖 `input_size` 和编码器，修改任一项后都要重新生成。

## 数据划分

| 集合 | 来源 | 图片 | 有人脸 |
|---|---|---|---|
| train | `training/` | 314 | 312 |
| val / test | `testing/` | 99 | 98 |

## 预处理

原图为 2000–6000 px，横竖混合。由于 DCT 块无法缩放，先在像素域做 letterbox：保持长宽比缩放到
`input_size = [W, H]`，居中放置，四周填 128（编码后系数为 0，即"无信号"）。解码时利用 JPEG
draft 模式按 1/2–1/8 缩小以加快速度。默认 `cache_images: true`，初始化时缓存 letterbox 后的图片
（每张约 0.3 MB，约 7 秒），之后每次取样只做增强和编码。

编码器：

- 默认：与 `jpeg.py` 相同的浮点 DCT + libjpeg 量化表（`quality`，默认 75；0 表示不量化）。
- `hw_encoder: {final_rounding, subsampling}`：使用 `hw_jpeg.py` 的 FPGA 逐位模型。

## 输出

与 DareLab_Face_DCT 相同，区别如下：

| 字段 | 说明 |
|---|---|
| img | `[kY+kCb+kCr, H/8, W/8]`，320×320 时为 `[24, 40, 40]`（`[16, 4, 4]`） |
| valid_mask | `[1, H/8, W/8]`，原图覆盖的块为 1，补边块为 0 |
| gt_bboxes | 块网格坐标 xyxy |
| warp_matrix | 原图像素坐标 → 块网格坐标（letterbox、增强和 /8 的合成），预测框经其逆映射回原图 |
| img_info | `file_name` 为相对 `img_path` 的路径，如 `training/xxx.jpg` |

## 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `img_path` | 数据集根目录 | 包含 `training/`、`testing/`、`annotations/` |
| `ann_path` | train: `annotations/train.json`；val/test: `annotations/val.json` | COCO 标注文件 |
| `input_size` | `[320, 320]` | `[W, H]`，均须为 16 的倍数 |
| `num_dct_coefficients` | `64` | 同 DareLab_Face_DCT |
| `normalization_stats` | `None` | 由 `prepare_dataset.py` 生成 |
| `augmentation` | `None` | 仅 train 生效，键同 DareLab_Face_DCT |
| `quality` | `75` | 浮点编码量化质量；设置 `hw_encoder` 时忽略 |
| `hw_encoder` | `None` | `{final_rounding: half_away/floor, subsampling: "420"/"422"}` |
| `rgb_path` | `img_path` | 原图目录；测试任务画图也读取 `dataloader.train.rgb_path` |
| `cache_images` | `true` | 是否缓存 letterbox 后的图片 |

配置示例见 `config/Face_FOMO_DCT/FOMO_DCT_HW422.yml`。统一入口也支持
`name: Face_FOMO_DCT` 和 `name: FaceFOMODCTDataset`。
