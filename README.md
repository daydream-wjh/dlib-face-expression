# dlib-face-expression

基于 **Dlib** 的人脸表情分析与识别：摄像头实时抓帧 → Dlib 检测人脸并取 68 个关键点 → 用关键点之间的几何比例算出 5–6 个表情特征 → **KNN** 分类成 7 类表情，结果直接叠加在视频画面上。

> 2019 年四人小组课程项目，共同完成，开发周期 7 天 + 结题完善报告 2 天，完成时间 2019 年 7 月 11 日，**已申请软件著作权**。
>
> 本仓库是 [smzwo/-Dlib-](https://github.com/smzwo/-Dlib-) 的 fork，沿用上游 MIT LICENSE（Copyright (c) 2020 喆喆有梦想）。

## 项目简介

不训练神经网络，也不做人脸图像的端到端分类，而是走"**人脸几何特征 + 传统机器学习**"的路线，好处是样本需求小、单帧推理快，能直接在笔记本摄像头上跑到实时；代价是特征由人工设计，对姿态和个体差异敏感。

运行流程（`face/` 下 5 个文件，一路串起来）：

1. `main.py` 实例化 `face_emotion.face_emotion()`，依次调用 `getdetector()` 与 `learningface()`；
2. `get_detector.py` 初始化三样东西：`dlib.get_frontal_face_detector()`（HOG 人脸检测器）、`dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")`（68 点关键点模型）、`cv2.VideoCapture(0)`（默认摄像头）；
3. `learning_face.py` 是主循环：每帧转灰度、检测人脸、画 68 点与连线，逐人计算特征 `a/b/c/dd/e`，把特征与当前人脸矩形交给 `face_pd.face_pd()`；
4. `face_pd.py` 先用"眼睛睁开程度"做一道疲劳判定，再 `joblib.load()` 载入 KNN 模型预测标签，把表情英文名 `cv2.putText` 到人脸框下方；
5. 画面右下角常驻 `Faces: n` / `No Face`、`S: screenshot`、`ESC: quit` 提示，按 `S` 截图存 `screenshootN.jpg`，按 `ESC` 退出并释放摄像头。

## 数据集

**本仓库不含数据集，也不含训练脚本**：1.0 版本只提交了推理侧代码与 68 点模型文件，KNN 的训练样本与训练过程没有一并入库（2.0 版本才补上 `数据集/` 与特征提取、分类器训练脚本）。

- `face/shape_predictor_68_face_landmarks.dat`：Dlib 官方预训练的 68 点人脸关键点模型（约 96 MB），负责把检测框细化成 68 个坐标点；
- KNN 模型文件名 `新最后最后最后五围2knn.m`（`face_pd.py` 中硬编码的 `joblib.load("新最后最后最后五围2knn.m")`）——**这个文件不在仓库里**，缺失时的表现见"已知问题"。

表情类别（`face_pd.py` 里的 `emo` 列表，索引 = 预测标签 − 1）：

```python
emo = ['angry', 'sadness', 'nature', 'nature', 'happy', 'normal', 'surprise']
```

## 模型 / 算法

### 人脸检测与关键点

`dlib.get_frontal_face_detector()`（HOG + 线性 SVM 的经典人脸检测器，`detector(img_gray, 0)` 中第二个参数 0 表示不做上采样）+ `shape_predictor_68_face_landmarks.dat`。视频流只取灰度帧，人脸框宽 `face_width = d.right() - d.left()` 作为所有长度量的归一化基准。

### 手工几何特征

全部用 68 点索引直接算，除以 `face_width` 做尺度归一化：

| 变量 | 含义 | 计算方式 |
| --- | --- | --- |
| `a` | 嘴巴咧开程度（嘴宽占比） | `(p54.x - p48.x) / face_width` |
| `b` | 嘴巴张开程度（嘴高占比） | `(p66.y - p62.y) / face_width` |
| `c` | 眉毛高度占比（挑眉程度） | `((Σ_{j=17..20}(p_j.y - top) + Σ(p_{j+5}.y - top)) / 10) / face_width` |
| `dd` | 眉毛间距占比（皱眉程度） | `(Σ_{j=17..20}(p_{j+5}.x - p_j.x) / 5) / face_width` |
| `e` | 眼睛睁开程度 | `((p41.y-p37.y) + (p40.y-p38.y) + (p47.y-p43.y) + (p46.y-p44.y)) / 4 / face_width` |
| `f` | 眉毛倾斜斜率 | 对 20 个眉毛点做 `np.polyfit(x, y, 1)`，取 `-斜率`（取反是因为图像 y 轴向下） |

`f`（`self.brow_k`）算了但**没有传给分类器**：`face_pd(im_rd, a, b, c, faces[k], dd, e)` 只传了 5 个特征 + 人脸框，分类输入是 `[[a, b, c, dd, e]]`。

### 分类与判定逻辑

`face_pd.py` 内部两步：

1. **疲劳/闭眼判定**：`if e < 0.02` → 在画面上打 `"tired"`，直接跳过分类（这一段是从"驾驶员疲劳检测"的代码沿用的，`music.runtired()` 播放提示音的调用被注释掉了）；
2. **KNN 分类**：`joblib.load("新最后最后最后五围2knn.m")` → `cld.predict([[a, b, c, dd, e]])` → `kll = int(...)` → `emo[kll-1]` 上屏。

### 评估指标

1.0 没有评估/混淆矩阵脚本，仓库内也没有准确率记录，故此处不写任何指标数字。

## 技术栈与依赖

按代码 import 推断：

- `dlib`（人脸检测 + 68 点预测）
- `opencv-python`（`cv2.VideoCapture`、绘制、`waitKey`、写图）
- `numpy`（`np.polyfit`、`np.array`）
- `joblib`（加载 KNN 模型）
- `music`（**项目内模块，本仓库缺失**，仅被注释调用）

```bash
pip install dlib opencv-python numpy joblib
```

无 `requirements.txt`；Python 版本按 2019 年的 `dlib`/`cv2` 轮子推断为 3.6–3.7 时代（`dlib` 需 `cmake` + C++ 编译器，或直接装 `dlib` 的预编译 wheel）。

## 目录结构

```
dlib-face-expression/
├── LICENSE                                    # MIT，Copyright (c) 2020 喆喆有梦想（上游）
├── README.md
├── .gitignore
└── face/
    ├── main.py                                # 入口：实例化并依次调用 getdetector / learningface
    ├── face_emotion.py                        # 空壳调度类 face_emotion，两个方法转调下面的模块
    ├── get_detector.py                        # 检测器、68 点模型、摄像头初始化
    ├── learning_face.py                       # 主循环：抓帧、检测、画点、算特征、调用分类
    ├── face_pd.py                             # 疲劳判定 + KNN 加载 + 表情文字上屏
    ├── shape_predictor_68_face_landmarks.dat  # Dlib 68 点模型（约 96 MB）
    ├── .idea/                                 # PyCharm 工程文件
    └── .DS_Store
```

## 快速开始

前置：有可用的摄像头；`shape_predictor_68_face_landmarks.dat` 与代码同目录（代码里是相对路径 `"shape_predictor_68_face_landmarks.dat"`），并且 **KNN 模型 `新最后最后最后五围2knn.m` 需要自己训练/放入 `face/`**（仓库未提供）。

```bash
cd face
python main.py
```

运行后：

- 红色矩形框出每张人脸，浅蓝色圆点标出 17–26、36–67 号关键点，并连线出眉毛/眼眶/内外嘴唇；
- 人脸框下方显示表情英文标签（`angry` / `sadness` / `nature` / `happy` / `normal` / `surprise`，闭眼时显示 `tired`）；
- 左上角显示 `Faces: n`，按 `S` 保存截图 `screenshoot1.jpg`、`screenshoot2.jpg`…，按 `ESC` 退出。

必须从 `face/` 目录启动：模型文件用的是相对当前工作目录的路径。

## 已知问题

- **KNN 模型文件缺失**：`face_pd.py` 里 `joblib.load("新最后最后最后五围2knn.m")` 硬编码了一个中文文件名的 `.m` 模型，仓库里没有该文件，也没有训练它的脚本与数据集 → 直接运行到第一帧人脸就会 `FileNotFoundError`。1.0 只能当作"算法流程 + 演示代码"阅读。
- **`import music` 缺失**：`face_pd.py` 顶部 `import music`，项目里没有 `music.py`，虽然 `music.runtired()` 被注释，但 import 本身就会 `ModuleNotFoundError`，需要自己删掉这行或补一个空模块。
- **代码里残留"驾驶员疲劳检测"逻辑**：`e < 0.02` 判闭眼并输出 `"tired"`，与"7 类表情识别"的项目目标不一致，是复用了疲劳检测 demo 的遗留分支。
- **分类器每帧重新加载模型**：`joblib.load()` 写在 `face_pd()` 里，主循环每个人脸每帧都从磁盘反序列化一次 KNN 模型，实时性被这部分 IO 拖累（应提到模块级只加载一次）。
- **`cv2.waitKey` 被调用多次**：循环内先 `cv2.waitKey(30)`，后面又分别用 `cv2.waitKey(100)` 判 `S` 和 `ESC`，同一次按键可能被前一个 `waitKey` 吃掉，导致截图/退出不灵。
- **`dd = round(brow_width, 310)`**：保留 310 位小数的笔误（应为 10），虽不影响数值但会让特征表看起来异常。
- **`.py` 与上游的差异无法追溯**：本仓库是 fork，只有一个 `Initial commit` 级别的提交历史（远程配置 `upstream = smzwo/-Dlib-`），无法从 git log 看出哪些改动是本地加的。
- **大文件入库**：`shape_predictor_68_face_landmarks.dat` 约 96 MB 直接提交，未用 Git LFS。
- **杂项入库**：`.idea/`、`face/.DS_Store` 都被提交。
- 摄像头索引固定为 `0`，分辨率先 `cap.set(3, 144)` 又在主循环里改成 480，实际生效值取决于后端。
