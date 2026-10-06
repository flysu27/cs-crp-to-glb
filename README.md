# crp2glb —— 城市天际线 .crp → .glb 转换工具

把《城市天际线》（Cities: Skylines）工坊资产 `.crp` 转成通用 3D 格式 **.glb**：几何、法线、UV、  
贴图全部带上，可直接拖进 Blender / three.js / Unity / 三维软件查看与二次利用。

**两种用法：**

- 🌐 **网页版（推荐）**：浏览器打开 `viewer.html`，把 `.crp` 拖进去 → 自动转换并预览，
  点「下载 GLB」保存。转换在浏览器内完成（Pyodide），文件不上传服务器。
- 💻 **命令行版**：见下文。

## 快速开始

```bash
# 1) 安装依赖（建议用虚拟环境）
pip install -r requirements.txt

# 2) 转换单个资产（输出到同目录）
python crp2glb.py "C:/Users/你/AppData/Local/Colossal Order/Cities_Skylines/Addons/Assets/你的建筑.crp"

# 3) 批量转换一个目录
python crp2glb.py "路径/Assets" -o out/

# 4) 只想看看文件结构（排查用）
python crp2glb.py 建筑.crp --inspect
```

Windows 上还可以直接把 `.crp` 文件拖到 `crp2glb.bat` 上转换。

查看结果：`viewer.html` 现在本身就是一个完整的转换器+预览器——直接把 `.crp` 拖进页面即可；
也可以把命令行生成的 `.glb` 拖进去检查模型与贴图。

## 命令行参数

| 参数                  | 说明                            |
| ------------------- | ----------------------------- |
| `-o, --out`         | 输出目录，默认与输入文件同目录               |
| `--inspect`         | 只打印容器分段与网格结构，不导出              |
| `--lod N`           | 资产含多级 LOD 时，只导出第 N 级（0 为主模型）  |
| `--no-tex`          | 不内嵌贴图（体积更小）                   |
| `--no-convert-hand` | 不做 Unity(左手) → glTF(右手) 转换    |
| `--flip-uv`         | 翻转 UV 的 V 轴（默认**不翻**：实测 CRP 网格 UV 已是 glTF 语义，1:1 原样贴图即正确） |
| `--json`            | 额外输出 JSON 报告                  |

## .crp 是什么 / 工具怎么做的

`.crp` 并不是普通的 Unity AssetBundle，而是 Colossal Order 自己的容器  
（"CRAP" = Colossal Raw Asset Package），内部按分段存放：

| 首字节    | 内容                                       |
| ------ | ---------------------------------------- |
| `0x53` | Assembly-CSharp（模组代码）                    |
| `0x54` | UnityEngine.Mesh                         |
| `0x58` | UnityEngine.Material                     |
| `0x59` | UnityEngine.Texture2D（内含 DDS）            |
| `0x5A` | UnityEngine.GameObject                   |
| `0x5B` | CustomAssetMetaData                      |
| `0x69` | ColossalFramework.Importers.Image（PNG 等） |

容器结构：`CRAP` 头 → 若干文件头（名称 / 偏移 / 长度）→ 数据区。  
本工具按该规范独立实现容器解析，并自动探测偏移基准（绝对 / 相对数据区起点）。

Mesh 分段里是**裸的 Unity 序列化 Mesh 数据**（不是 AssetBundle），所以 `crp2glb.py`  
内置了一个 Unity 5.x Mesh 反序列化器：

- 支持未压缩 `VertexData`（float32 / float16 / UNorm8 / SNorm16 等各顶点格式组合）；
- 用「索引缓冲合法性 + 顶点数据长度自洽 + AABB 合理 + 子网格顶点数吻合」多重校验；
- 因为不同 Unity 小版本的字段布局有细微差异（`m_Shapes` / `m_BindPose` / `CompressedMesh`  
  块数），工具会对同一段数据尝试多种布局并在候选起点中择优，取分数最高的解析结果；
- 索引位宽（16/32 位）不依赖版本标记，而是按「数据长度 + 索引值不超过顶点数」推断。

贴图：`0x59` 分段内定位 `DDS ` 魔数，优先用 Pillow 解码，失败则按 FourCC 走**纯 numpy 的  
BC1~BC5 解码器**（无需第三方扩展，浏览器版也能用），最后才是 texture2ddecoder 兜底（含 BC7）；  
`0x69` 分段按 PNG 处理。贴图以 PNG 形式内嵌进 GLB，无需外部文件。

**贴图归属规则（预制体引用链）**：真实 CRP 内含 GameObject → Transform / MeshFilter / MeshRenderer  
→ Material → Texture2D 的完整引用链（按 GUID 串接）。本工具解析该图后：每个渲染器导出  
其引用的网格与材质，贴图取材质的 `_MainTex`——**网格用哪张贴图由材质说了算**，不再按  
分段位置猜。LOD 网格默认剔除（`--lod N` 可选）、同网格变体去重（`_Data` 主实例优先）。  
多节车辆（铰接公交/挂车）按 GameObject 上的 `m_attachOffsetFront/Back` 计算接驳偏移，  
各节车身自动拼成一整车，不再堆叠穿模。无法建立引用链时回退到按分段位置的归组。  
预览图（SteamPreview/Snapshot）不参与材质。

坐标与 UV：Unity 为左手系，glTF 为右手系，因此默认做 `x = -x`、绕序翻转  
（可用 `--no-convert-hand` 关闭）。**UV 与贴图均 1:1 原样保留**——实测 CRP 网格 UV  
已是 glTF 语义（v=0 对应图像顶部），贴图解码后方向也正确，任何一方再做翻转都会  
上下颠倒（这是早期版本贴图错乱的根因）。

## 自测

```bash
python selftest.py
```

会生成一个样例 `.crp`（盒体 + 棋盘格 DXT1 贴图 + 材质/元数据分段），转换后逐项校验：  
GLB 容器结构、accessor/bufferView 越界、顶点数与三角面数、坐标与 UV 变换、内嵌 PNG 可解码、  
材质是否正确引用贴图。

最近一次自测输出：

```
== 1. 生成样例 CRP ==   3792 字节，顶点 24 / 索引 36
== 2. 转换 ==           ✓ test_building.crp → out/test_building.glb
== 3. 校验 GLB 结构 ==  JSON/BIN 块正常，4 个 accessor、1 个 mesh、1 个 image
== 4. 校验几何 ==       顶点 24、三角面 12，坐标已做 Unity→glTF 转换 ✓
== 5. 校验贴图与材质 == 内嵌 PNG 64x64 可解码 ✓ 材质已关联贴图 ✓
   法线、UV（1:1 原样保留，无需翻转）校验通过 ✓
全部自测通过 ✅
```

## 网页版（viewer.html）

浏览器内直接把 `.crp` 转成 `.glb` 并预览，无需安装任何东西：

1. 打开 `viewer.html`（双击即可，file:// 下也能用）；
2. 首次会自动加载转换引擎（Pyodide + numpy + Pillow，约 15~20 MB，之后有缓存）；
3. 拖入 `.crp` → 页面内完成解析/贴图解码/GLB 组装 → 自动预览；
4. 点「下载 GLB」保存结果；拖 `.glb` 则直接预览（老功能保留）。

`viewer.html` 由 `_viewer_template.html` + `crp2glb.py` 源码经 `build_viewer.py` 注入生成——  
**修改 `crp2glb.py` 后请运行 `python build_viewer.py` 重新生成**，页面内嵌的转换器才会更新。

浏览器版已知限制：BC7/BC6 贴图在无 texture2ddecoder 的 Pyodide 里暂不解码  
（会在报告中警告）；10 MB+ 的引擎首次加载依赖网络。

## 文件说明

| 文件                 | 说明                                       |
| ------------------ | ---------------------------------------- |
| `crp2glb.py`       | 主程序：容器解析 / Mesh 解析 / 贴图解码 / GLB 写出 / 命令行 |
| `viewer.html`      | 网页版：拖入 .crp 浏览器内转换并预览（含内嵌转换器）      |
| `_viewer_template.html` | 网页版模板（转换器源码占位）                      |
| `build_viewer.py`  | 重新生成 viewer.html（crp2glb.py 变更后运行）        |
| `make_test_crp.py` | 生成自测用样例 `.crp`（含 DXT1 编码器）               |
| `selftest.py`      | 端到端自测（生成 → 转换 → 校验 GLB）                  |
| `_test_pyodide.js` | 浏览器引擎链路回归测试（Node + Pyodide 模拟 viewer）    |
| `_v_glbcheck.py`   | GLB 产物验证：按 three.js 采样语义软件渲染成 PNG        |
| `crp2glb.bat`      | Windows 拖拽转换脚本                           |
| `requirements.txt` | 依赖清单                                     |

## 已知限制

- 只导出**静态网格**，不含 Unity 的节点层级/动画/骨骼蒙皮（CS 建筑资产基本用不到）；
- 贴图只取漫反射主贴图并内嵌，法线图（_XYSMap）/镜面图（_ACIMap）已解码但不进材质——  
  CS 用自定义着色器，标准 glTF 没有对应通道，如需可在 Blender 里手动指定；
- 浏览器版暂不支持 BC7/BC6 贴图兜底解码（Pyodide 无 texture2ddecoder，会警告）；
- 工具基于公开格式规范 + 真实资产逆向实现，已通过生成样例自测与  
  「福田 BJ6189SHEVCA-2 北京公交」真实资产验证（2 网格 / 1 贴图 / 28,972 三角面，  
  18.7 m 铰接公交自动拼装，GLB 产物经软件渲染逐像素核对）；  
  如遇其他资产失败请附上 `--inspect` 输出。

## 回归测试

```bash
python selftest.py        # 样例资产端到端自测
node _test_pyodide.js     # 浏览器引擎链路（Node + Pyodide）端到端
python _v_glbcheck.py out/xxx.glb   # 把 GLB 产物按 three.js 采样语义软件渲染成 PNG 人工核对
```

## 参考

- ArchiveTeam「Cities Skylines CRP」格式说明
- LiamBrandt/crp-extract（容器分段结构参考实现）
