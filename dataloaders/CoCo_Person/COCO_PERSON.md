# COCO 2017 人 / 背景数据

数据目录：`/home/hzhao/ZHN/datasets/coco2017_person_fullcoder`。
来源：[COCO 官方网站](https://cocodataset.org/#download)，下载自官方图片存储桶
`https://s3.amazonaws.com/images.cocodataset.org`。

| 划分 | 图像数 | 有人 | 无人（按原标注） | 人体框数 |
| --- | ---: | ---: | ---: | ---: |
| train | 20,000 | 10,000 | 10,000 | 32,644 |
| val | 4,773 | 2,466 | 2,307 | 7,846 |

训练图仅取自官方 train2017，按正负样本各半、随机种子 42 抽样；验证图仅取自
官方 val2017。因为当前检测器默认没有启用 ignore 区域处理，两组均排除含有人类
`iscrowd` / `ignore` 标注或无效人体框的整张图片；不会只删掉这些框后把图像当背景。
原验证集 5,000 张，排除 227 张后保留全部 4,773 张，不再抽样。

只保留 person 类别，JSON 中 category_id 为 1，Dataset 会映射成检测标签 0。
背景没有单独的检测类别或框。图像有人的有效框时 classification_target 为 1，
没有人的标注时为 0。车、动物、家具等均属于本任务的背景。
负样本依据 COCO 原标注判断，未逐张人工复核；原数据中的漏标也可能影响分类标签。

保留原 JPEG、分辨率和 RGB 内容，不提前裁剪人体、压缩或改变尺寸。
FullCoderDataset 在读取时进行 320×320 等比例缩放和补边，生成分类标签、检测框、
重建目标及有效区域掩码。训练阶段仍使用 YAML 指定的水平翻转。

```text
coco2017_person_fullcoder/
  images/
    train/*.jpg
    val/*.jpg
  annotations/
    instances_train.json
    instances_val.json
  raw/
    annotations_trainval2017.zip
    annotations/instances_train2017.json
    annotations/instances_val2017.json
  selection.json
  image_sha256.json
  report.json
  README.md
```

`selection.json` 记录下载选择，`report.json` 在全部下载及校验通过后生成，记录统计、
原始标注 SHA256 和校验结果。`image_sha256.json` 记录逐图 SHA256。
校验包含逐图完整解码、原始尺寸一致，以及 train/val 图像 ID 和文件 SHA256 无交集。
此验证集是经过筛选的单类任务验证集，结果不能当作官方 COCO 全类别指标。
没有将官方无公开标注的 test2017 当作负样本，也没有从训练集复制图像充当验证集。

## 重新准备

需要 Python 3.10+、Pillow、requests。从 `/home/hzhao/project` 执行：

```bash
python Adaptive_Compression/tools/prepare_coco_person.py \
  --root /home/hzhao/ZHN/datasets/coco2017_person_fullcoder \
  --train-images 20000 --seed 42 --workers 48
```

支持复用已经下载的完整文件；中断留下的 `.part` 会重下，不会当作完成文件。
修改抽样大小或 seed 时请使用新目录，脚本会拒绝混用不同抽样配置。
下载中止后不要开始正式训练，先重跑直至生成完整的 `report.json`。

`config/example_training.yml` 中四个数据路径指向上面的图片和标注文件。运行：

```bash
python Adaptive_Compression/tools/train.py Adaptive_Compression/config/example_training.yml
```

数据来源、图片原始 URL 和 license ID 保存在输出 JSON 中，licenses 表沿用原始文件。
图片及标注的使用条件见 [COCO Terms of Use](https://cocodataset.org/#termsofuse)。
