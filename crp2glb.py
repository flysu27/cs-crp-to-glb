#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crp2glb —— 城市天际线（Cities: Skylines）资产 .crp → .glb 转换工具

功能
----
1. 解析 CRP 容器（"CRAP" = Colossal Raw Asset Package），按分段（section）拆分；
2. 识别分段类型：UnityEngine.Mesh / UnityEngine.Texture2D / ColossalFramework.Importers.Image
   / UnityEngine.Material / UnityEngine.GameObject / Assembly-CSharp …
3. 从 Mesh 段解析顶点（位置/法线/UV/顶点色）、子网格、索引，支持未压缩 VertexData；
4. 从贴图段解析 DDS（BC1/BC3/BC4/BC5/BC7 等）与 PNG，生成内嵌贴图；
5. 输出 glTF 2.0 二进制 .glb（贴图内嵌、材质自动关联、Unity→glTF 坐标系与 UV 自动转换）；
6. 顺带支持直接读 UnityFS AssetBundle（依赖可选的 UnityPy）。

用法
----
  python crp2glb.py 建筑.crp                    # 输出 建筑.glb（与输入同目录）
  python crp2glb.py 建筑.crp -o out/            # 指定输出目录
  python crp2glb.py 某个文件夹/ -o out/          # 批量转换目录下所有 .crp
  python crp2glb.py 建筑.crp --inspect          # 只打印结构信息，用于排查
  python crp2glb.py 建筑.crp --lod 0            # 多个 LOD 时只导出主模型
  python crp2glb.py 建筑.crp --no-tex           # 不内嵌贴图
  python crp2glb.py 建筑.crp --no-convert-hand  # 不做 Unity→glTF 坐标系转换

说明
----
* 容器格式参考：ArchiveTeam「Cities Skylines CRP」格式说明 + LiamBrandt/crp-extract 实现；
  本工具按该规范独立实现，并对 offset 的两种可能基准（绝对 / 相对 header 末尾）自动探测。
* Mesh 段在 CRP 中是裸的 Unity 序列化数据（非 AssetBundle），因此本工具内置了
  Unity 5.x Mesh 反序列化器，并用「索引合法性 / 数据长度自洽 / AABB 合理」三重校验
  在多个候选解析位置中择优，尽量兼容不同 Unity 版本导出的资产。
* 若某个 .crp 解析失败，请用 --inspect 查看分段结构，并把输出反馈给工具维护者。

依赖：numpy（必需）、Pillow（贴图，必需）、texture2ddecoder（BC 贴图兜底，可选）、UnityPy（可选）
"""

import argparse
import io
import json
import os
import struct
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.stderr.write("需要 numpy：pip install numpy\n")
    raise

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:  # 可选：BCn 贴图兜底解码
    import texture2ddecoder as _t2d
except Exception:  # pragma: no cover
    _t2d = None


# --------------------------------------------------------------------------- #
# 二进制读取
# --------------------------------------------------------------------------- #
class ParseError(Exception):
    pass


class Reader:
    """带边界的字节游标

    Unity 的对象序列化里，字符串/数组的对齐基准是「对象起始位置」而不是文件绝对
    位置；CRP 分段内的 Mesh 数据是从分段中某个偏移开始的，所以这里用 base 记录
    对象起点，align() 一律按 (pos - base) 对齐。
    """

    def __init__(self, data, pos=0, end=None, base=None):
        self.data = data
        self.pos = pos
        self.end = len(data) if end is None else end
        self.base = pos if base is None else base

    def __len__(self):
        return self.end - self.pos

    def need(self, n):
        if self.pos + n > self.end:
            raise ParseError("数据越界：需要 %d 字节，剩余 %d" % (n, self.end - self.pos))

    def u8(self):
        self.need(1)
        v = self.data[self.pos]
        self.pos += 1
        return v

    def u16(self):
        self.need(2)
        v = struct.unpack_from("<H", self.data, self.pos)[0]
        self.pos += 2
        return v

    def u32(self):
        self.need(4)
        v = struct.unpack_from("<I", self.data, self.pos)[0]
        self.pos += 4
        return v

    def i32(self):
        self.need(4)
        v = struct.unpack_from("<i", self.data, self.pos)[0]
        self.pos += 4
        return v

    def u64(self):
        self.need(8)
        v = struct.unpack_from("<Q", self.data, self.pos)[0]
        self.pos += 8
        return v

    def f32(self):
        self.need(4)
        v = struct.unpack_from("<f", self.data, self.pos)[0]
        self.pos += 4
        return v

    def raw(self, n):
        self.need(n)
        v = self.data[self.pos:self.pos + n]
        self.pos += n
        return v

    def align(self, n=4):
        pad = (-(self.pos - self.base)) % n
        if pad:
            self.need(pad)
            self.pos += pad

    def string(self, align=4):
        """Unity 字符串：i32 长度 + utf8 字节 + 对齐"""
        n = self.i32()
        if n < 0 or n > 4096:
            raise ParseError("字符串长度异常：%d" % n)
        s = self.raw(n).decode("utf-8", "ignore")
        if align:
            self.align(align)
        return s

    def float_array(self, n):
        self.need(4 * n)
        v = np.frombuffer(self.data, dtype="<f4", count=n, offset=self.pos).astype(np.float64)
        self.pos += 4 * n
        return v


# --------------------------------------------------------------------------- #
# CRP 容器解析
# --------------------------------------------------------------------------- #
PASCAL = lambda r: r.raw(r.u8()).decode("utf-8", "ignore")  # noqa: E731


def looks_like_section_start(data, offset):
    """分段起始处一般为 0x00 + 可读标识串"""
    if offset < 0 or offset + 4 > len(data):
        return False
    head = data[offset:offset + 64]
    if head[:1] == b"\x00" and _has_known_id(head):
        return True
    return _has_known_id(head)


def _has_known_id(head):
    low = head.lower()
    for key in (b"unityengine.mesh", b"unityengine.texture2d", b"unityengine.material",
                b"unityengine.gameobject", b"unityengine.animationclip",
                b"assembly-csharp", b"buildinginfogen", b"customassetmetadata",
                b"colossalframework.importers.image", b"locale"):
        if key in low:
            return True
    return False


def parse_crp(data):
    """解析 CRP 容器，返回 {version, steam_id, name, end_header_offset, sections:[...]}"""
    r = Reader(data)
    if data[:4] != b"CRAP":
        raise ParseError("不是 CRP 文件（缺少 CRAP 标识）")
    r.raw(4)
    version = r.u16()
    steam_id = PASCAL(r)
    unknown1 = PASCAL(r)
    unknown2 = r.u32()
    name = PASCAL(r)
    number_of_files = r.u32()
    end_header_offset = r.u64()

    if number_of_files > 4096:
        raise ParseError("分段数量异常：%d" % number_of_files)

    headers = []
    for _ in range(number_of_files):
        fname = PASCAL(r)
        unknown4 = PASCAL(r)
        unknown5 = r.u32()
        off = r.u64()
        size = r.u64()
        headers.append({"name": fname, "unknown4": unknown4, "unknown5": unknown5,
                        "offset": off, "size": size})

    sections = []
    for h in headers:
        body = None
        used = None
        for base, how in ((end_header_offset, "relative"), (0, "absolute")):
            start = h["offset"] + base
            if start < 0 or start + h["size"] > len(data) or h["size"] <= 0:
                continue
            chunk = data[start:start + h["size"]]
            if looks_like_section_start(data, start):
                body, used = chunk, how
                break
            if body is None:
                body, used = chunk, how + "(unverified)"
        if body is None:
            body = b""
            used = "missing"
        sections.append({
            "name": h["name"], "offset_offset": h["offset"], "size": h["size"],
            "origin": used, "raw": body, "unknown4": h["unknown4"], "unknown5": h["unknown5"],
        })

    return {"version": version, "steam_id": steam_id, "unknown1": unknown1,
            "unknown2": unknown2, "name": name, "end_header_offset": end_header_offset,
            "sections": sections}


# --------------------------------------------------------------------------- #
# 分段类型识别
# --------------------------------------------------------------------------- #
def classify_section(raw):
    """按 CRP 对象头（00 + 类型串）识别分段类型，比关键字搜索更准确。

    GameObject 段内可能嵌 PNG（缩略图），若按 PNG 魔数识别会把 GameObject
    误判成贴图，导致材质贴错 —— 这里优先用类型串判断。
    """
    tname, name, _ = _read_crap_strings(raw)
    if tname:
        low = tname.lower()
        if b"unityengine.mesh" in low:
            return "mesh"
        if b"unityengine.texture2d" in low:
            return "texture_dds"
        if b"colossalframework.importers.image" in low:
            return "image_png"
        if b"unityengine.material" in low:
            return "material"
        if b"unityengine.gameobject" in low:
            return "gameobject"
        if b"customassetmetadata" in low:
            return "metadata"
        if b"vehicleinfogen" in low or b"buildinginfogen" in low:
            return "assembly"
        if b"assembly-csharp" in low or b"assetdatawrapper" in low:
            return "assembly"
    # 兜底：老格式 / 非标准段
    low = raw[:96].lower()
    if b"unityengine.mesh" in low:
        return "mesh"
    if b"unityengine.texture2d" in low:
        return "texture_dds"
    if b"colossalframework.importers.image" in low:
        return "image_png"
    if b"unityengine.material" in low:
        return "material"
    if b"unityengine.gameobject" in low:
        return "gameobject"
    if b"assembly-csharp" in low:
        return "assembly"
    if b"buildinginfogen" in low:
        return "building_info"
    if b"customassetmetadata" in low:
        return "metadata"
    if raw.find(b"\x89PNG\r\n\x1a\n") >= 0:
        return "image_png"
    return "unknown"


# --------------------------------------------------------------------------- #
# Unity 5.x Mesh 反序列化
# --------------------------------------------------------------------------- #
FMT_SIZE = {0: 4, 1: 2, 2: 1, 3: 1, 4: 2, 5: 2, 6: 1, 7: 1, 8: 2, 9: 2, 10: 4, 11: 4}


def _parse_submeshes(r):
    n = r.i32()
    if n < 0 or n > 64:
        raise ParseError("submesh 数量异常")
    subs = []
    for _ in range(n):
        first_byte = r.u32()
        index_count = r.u32()
        topology = r.i32()
        base_vertex = r.u32()
        first_vertex = r.u32()
        vertex_count = r.u32()
        center = r.float_array(3)
        extent = r.float_array(3)
        subs.append({"first_byte": first_byte, "index_count": index_count, "topology": topology,
                     "base_vertex": base_vertex, "first_vertex": first_vertex,
                     "vertex_count": vertex_count, "center": center, "extent": extent})
    return subs


def _parse_vertex_data(r):
    vcount = r.u32()
    if vcount == 0 or vcount > 8_000_000:
        raise ParseError("顶点数异常：%d" % vcount)
    nch = r.i32()
    if nch < 1 or nch > 16:
        raise ParseError("通道数异常：%d" % nch)
    channels = []
    for _ in range(nch):
        stream = r.u8()
        offset = r.u8()
        fmt = r.u8()
        dim = r.u8()
        if fmt not in FMT_SIZE or dim == 0 or dim > 4:
            raise ParseError("通道格式异常")
        channels.append({"stream": stream, "offset": offset, "format": fmt, "dimension": dim})
    data_size = r.u32()
    stride = 0
    for c in channels:
        stride = max(stride, c["offset"] + FMT_SIZE[c["format"]] * c["dimension"])
    if stride == 0 or data_size == 0:
        raise ParseError("顶点数据长度异常")
    # 自洽校验：数据长度应约等于 步长×顶点数
    if not (stride * vcount <= data_size <= stride * vcount + 32):
        raise ParseError("顶点数据长度与顶点数不自洽")
    # 5.6+ 为「i32 长度 + 字节 + 对齐」，早期版本直接是 m_DataSize 字节
    body = None
    save = r.pos
    try:
        n = r.i32()
        if n == data_size:
            body = r.raw(n)
            r.align(4)
        else:
            r.pos = save
    except ParseError:
        r.pos = save
    if body is None:
        body = r.raw(data_size)
    if len(body) < stride * vcount:
        raise ParseError("顶点数据不足")
    return {"vertex_count": vcount, "channels": channels, "data_size": data_size,
            "stride": stride, "data": body}


def _skip_shapes(r):
    r.i32()          # m_VertexCount
    r.i32()          # m_Channels
    r.u8()           # m_FullWeights
    n = r.i32()
    if n < 0 or n > 1024:
        raise ParseError("blendShape 数量异常")
    for _ in range(n):
        r.u32(); r.u32(); r.u8(); r.u8()
        k = r.i32()
        if k < 0 or k > 1_000_000:
            raise ParseError("blendShape 顶点数量异常")
        r.pos += 40 * k          # 每个顶点：pos(12)+normal(12)+tangent(12)+index(4)


def _skip_bindpose(r):
    n = r.i32()
    if n < 0 or n > 4096:
        raise ParseError("bindpose 数量异常")
    r.pos += 64 * n


def _skip_compressed_mesh(r, blocks):
    """跳过 CompressedMesh（每块 PackedBitVector：u32 数量 + f32 range + f32 start
    + i32 长度 + 数据 + u8 位宽）。未压缩的网格该块全为 0，但依然占位。"""
    for _ in range(blocks):
        r.u32()
        r.f32()
        r.f32()
        n = r.i32()
        if n < 0 or n > 64 << 20:
            raise ParseError("压缩数据长度异常")
        r.raw(n)
        r.align(4)
        r.u8()
        r.align(4)


# 同一段里可能出现的 CompressedMesh 块数组合（不同 Unity 小版本差异）
COMP_BLOCK_CHOICES = (10, 11, 0)


def parse_mesh_at(data, start):
    """在 data[start:] 处尝试所有布局变体，返回 (best_mesh, best_score)"""
    best, best_score = None, -1.0
    for with_shapes in (True, False):
        for with_bindpose in (True, False):
            for blocks in COMP_BLOCK_CHOICES:
                try:
                    m = _parse_mesh_once(data, start, with_shapes, with_bindpose, blocks)
                except Exception:
                    continue
                sc = mesh_score(m)
                if sc > best_score:
                    m["layout"] = {"shapes": with_shapes, "bindpose": with_bindpose,
                                   "comp_blocks": blocks}
                    best, best_score = m, sc
    return best, best_score


def _parse_mesh_once(data, start, with_shapes, with_bindpose, comp_blocks):
    r = Reader(data, start)
    name = r.string()
    if len(name) > 256:
        raise ParseError("名字过长")
    subs = _parse_submeshes(r)
    if not subs:
        raise ParseError("无子网格")
    if with_shapes:
        _skip_shapes(r)
    if with_bindpose:
        _skip_bindpose(r)

    vd = _parse_vertex_data(r)

    aabb_pos = r.pos
    center = extent = None
    if comp_blocks:
        _skip_compressed_mesh(r, comp_blocks)
    try:
        center = r.float_array(3)
        extent = r.float_array(3)
    except ParseError:
        center = extent = None
        r.pos = aabb_pos

    total_index = sum(s["index_count"] for s in subs)
    idx, use16 = _locate_index_buffer(data, r.pos, total_index, vd["vertex_count"])
    if idx is None:
        raise ParseError("未找到合法索引缓冲")

    return {
        "name": name, "submeshes": subs, "vertex_data": vd,
        "aabb_center": center, "aabb_extent": extent,
        "indices": idx, "use16": use16,
        "tail_pos": r.pos, "total_indices": total_index,
    }


def parse_mesh_body(data, start, variant=0):
    """兼容旧接口：返回指定布局变体的解析结果"""
    combos = [(True, True, 10), (True, True, 11), (True, True, 0)]
    ws, wb, cb = combos[variant % len(combos)]
    return _parse_mesh_once(data, start, ws, wb, cb)


def _locate_index_buffer(data, pos, total_index, vertex_count):
    """在 data[pos:] 中定位索引缓冲，返回 (numpy 数组, 是否 16 位)

    不依赖版本差异，直接按「长度前缀 + 索引全在顶点范围内」来识别：
    先用 i32 长度前缀匹配（5.6 的 byte-array 写法），再尝试无前缀的裸索引。
    """
    tail = data[pos:]
    if not tail or total_index <= 0:
        return None, True
    scan = min(max(0, len(tail) - 8), 8192)

    def check(arr):
        sample = arr if arr.size <= 100000 else arr[:: max(1, arr.size // 100000)]
        return int(sample.max(initial=0)) < vertex_count and int(sample.min(initial=0)) >= 0

    # ① 带 i32 长度前缀
    for delta in range(0, scan + 1):
        ln = struct.unpack_from("<i", tail, delta)[0]
        if ln <= 0:
            continue
        for width, dtype in ((2, "<u2"), (4, "<u4")):
            if ln != total_index * width or delta + 4 + ln > len(tail):
                continue
            arr = np.frombuffer(tail, dtype=dtype, count=total_index, offset=delta + 4).astype(np.int64)
            if check(arr):
                return arr.astype(np.uint32), width == 2
    # ② 无长度前缀的裸索引
    for delta in range(0, min(scan, 256) + 1):
        for width, dtype in ((2, "<u2"), (4, "<u4")):
            need = total_index * width
            if delta + need > len(tail):
                continue
            arr = np.frombuffer(tail, dtype=dtype, count=total_index, offset=delta).astype(np.int64)
            if check(arr):
                return arr.astype(np.uint32), width == 2
    return None, True


# --------------------------------------------------------------------------- #
# CRP 原生 Mesh 段解析（新版：带类型串前缀 + 分属性数组布局）
# --------------------------------------------------------------------------- #
def _read_crap_strings(raw):
    """CRP 对象头：00 <u8 len> 类型名 <u8 len> 对象名，返回 (类型名, 对象名, 数据起点)"""
    try:
        if raw[0] != 0:
            return None, None, 0
        tlen = raw[1]
        tname = raw[2:2 + tlen]
        p = 2 + tlen
        nlen = raw[p]
        name = raw[p + 1:p + 1 + nlen].decode("utf-8", "ignore")
        return tname, name, p + 1 + nlen
    except Exception:
        return None, None, 0


def _solve_slots(data, vc, p, depth=0):
    """回溯解析「[count][数组] × N + 索引头16B + u32索引」布局。

    属性数组的常见步长（字节/顶点）：12=pos/tan3、16=nrm4/tan4、8=uv2、4=一维、20/32=其他。
    count==0 表示该槽位为空（该属性缺失）。返回 (槽位列表, 索引起点, 索引数) 或 None。
    """
    if depth > 10:
        return None
    rem = len(data) - p
    # 终止：索引区 = 16 字节头 + n×u32，且头尾自洽（头[3]==索引数）
    if rem >= 20 and rem % 4 == 0:
        n = (rem - 16) // 4
        if n > 0 and n % 3 == 0:
            icount = struct.unpack_from("<I", data, p + 12)[0]
            if icount == n:
                idx = np.frombuffer(data, dtype="<u4", count=n, offset=p + 16)
                if int(idx.max(initial=0)) < vc and int(idx.min(initial=0)) >= 0:
                    return [], p, n
    if rem < 4:
        return None
    c = struct.unpack_from("<I", data, p)[0]
    if c == 0:                                   # 空槽位
        r = _solve_slots(data, vc, p + 4, depth + 1)
        if r is not None:
            return [(p, 0, 0)] + r[0], r[1], r[2]
        return None
    if c != vc:
        return None
    for st in (12, 16, 8, 4, 20, 32):
        need = 4 + vc * st
        if rem < need:
            continue
        r = _solve_slots(data, vc, p + need, depth + 1)
        if r is not None:
            return [(p, vc, st)] + r[0], r[1], r[2]
    return None


def parse_mesh_crap(raw):
    """解析 CRP 原生 Unity Mesh 段（新版布局）。

    结构：00 <u8> "UnityEngine.Mesh, UnityEngine, Version=…" <u8> 对象名
          <u32 vertexCount>
          [count][pos float3×vc] [count][nrm float4×vc] [count][uv float2×vc]
          [count][扩展 float3×vc] [count][tangent float4×vc]      （count=0 的槽为空）
          <16B 索引头> <u32 索引×n>

    返回 {"format":"crap-v2", ...} 或 None。
    """
    tname, name, p = _read_crap_strings(raw)
    if tname is None or b"UnityEngine.Mesh" not in tname:
        return None
    if p + 4 > len(raw):
        return None
    vc = struct.unpack_from("<I", raw, p)[0]
    if not (0 < vc <= 8_000_000):
        return None
    r = _solve_slots(raw, vc, p)
    if r is None:
        return None
    slots, idx_off, n_idx = r
    if len(slots) < 2 or slots[0][1] != vc or slots[0][2] != 12:
        return None        # 第一个槽必须是位置数组

    def slot_array(i, dim):
        off, c, st = slots[i]
        if c == 0 or st != dim * 4:
            return None
        return np.frombuffer(raw, dtype="<f4", count=vc * dim, offset=off + 4).reshape(vc, dim).astype(np.float32)

    pos = slot_array(0, 3)
    if pos is None or not np.all(np.isfinite(pos)):
        return None
    nrm = slot_array(1, 4) if len(slots) > 1 else None
    if nrm is None and len(slots) > 3:
        nrm = slot_array(3, 3)               # 部分资产把法线放在第 4 槽（float3）
    if nrm is not None and not np.all(np.isfinite(nrm)):
        nrm = None
    uv = slot_array(2, 2) if len(slots) > 2 else None
    if uv is not None and not np.all(np.isfinite(uv)):
        uv = None
    indices = np.frombuffer(raw, dtype="<u4", count=n_idx, offset=idx_off + 16).astype(np.uint32)
    return {
        "format": "crap-v2", "name": name, "vertex_count": vc,
        "pos": pos, "nrm": nrm, "uv": uv,
        "indices": indices, "section": "",
    }


def find_mesh(raw, section_name=""):
    """在分段字节中寻找可解析的 Mesh，返回 (mesh dict, score)

    策略：先收集候选起点（标识串附近 + 「短可读字符串」模式），逐个尝试所有布局
    变体，用打分函数择优；得分足够高时提前结束，避免在大文件上做无谓扫描。
    """
    starts = set()
    for key in (b"UnityEngine.Mesh", b"Mesh"):
        i = raw.find(key)
        if i >= 0:
            starts.update(range(i, min(i + 64, len(raw) - 32)))
    # 全局扫描「i32 长度 + 可打印串」作为结构起点，只扫前 16KB
    for i in range(0, min(len(raw) - 8, 16384)):
        ln = struct.unpack_from("<i", raw, i)[0]
        if 1 <= ln <= 64:
            chunk = raw[i + 4:i + 4 + ln]
            if len(chunk) == ln and all(9 <= c < 127 or c == 0 for c in chunk):
                starts.add(i)

    best, best_score = None, -1.0
    for idx, st in enumerate(sorted(starts)):
        if st >= len(raw) - 32 or idx > 3000:
            break
        m, score = parse_mesh_at(raw, st)
        if m is None:
            continue
        if score > best_score:
            m["start"] = st
            m["section"] = section_name
            best, best_score = m, score
            if best_score >= 14.0:      # 结构完全自洽，无需继续
                break
    return best, max(best_score, 0.0)


def mesh_score(m):
    """给候选解析打分，用于在多个候选中择优"""
    score = 0.0
    vd = m["vertex_data"]
    vc = vd["vertex_count"]
    ch = vd["channels"]
    subs = m["submeshes"]
    total = m["total_indices"]

    score += 2.0                                   # 基础分：能完整解析
    # 通道排列合理：0=位置(3 维)、1=法线(3 维)、4=UV(2 维)
    if ch and ch[0]["dimension"] == 3:
        score += 2.0
    if len(ch) >= 2 and ch[1]["dimension"] == 3:
        score += 1.0
    if len(ch) >= 3 and ch[2]["dimension"] in (2, 4):
        score += 1.0
    if all(c["stream"] == 0 for c in ch):
        score += 0.5
    # 子网格顶点数合计与顶点数吻合
    vsum = sum(s["vertex_count"] for s in subs)
    if vsum == vc:
        score += 2.0
    elif 0 < vsum <= vc:
        score += 1.0
    if sum(s["index_count"] for s in subs) == total:
        score += 1.0
    # AABB 合理
    ext = m["aabb_extent"]
    if ext is not None and np.all(ext > 0) and np.all(ext < 1e5):
        score += 1.5
    # 索引最大值贴近顶点数（说明索引与顶点匹配同一网格）
    idx = m["indices"]
    if idx.size:
        mx = int(idx.max()) + 1
        if mx <= vc:
            score += 2.0 * (mx / max(1, vc))
    # 位置数据有限且量级合理（大网格跳过，避免每个候选都做全量解码）
    if vc <= 200_000:
        try:
            pos = decode_channel(vd, ch[0], vc)
            if np.all(np.isfinite(pos)) and np.abs(pos).max() < 1e5:
                score += 1.5
        except Exception:
            pass
    return score


# --------------------------------------------------------------------------- #
# 顶点属性解码
# --------------------------------------------------------------------------- #
def decode_channel(vd, channel, vertex_count):
    """把某个通道解码为 (N, dim) 的 float32 numpy 数组"""
    fmt, dim, off = channel["format"], channel["dimension"], channel["offset"]
    stride = vd["stride"]
    data = vd["data"]
    dtype_map = {0: "<f4", 1: "<f2", 2: "u1", 3: "i1", 4: "<u2", 5: "<i2",
                 6: "u1", 7: "i1", 8: "<u2", 9: "<i2", 10: "<u4", 11: "<i4"}
    dtype = dtype_map[fmt]
    size = FMT_SIZE[fmt]
    need = vertex_count * stride
    if len(data) < need:
        raise ParseError("顶点数据不足：%d < %d" % (len(data), need))
    buf = np.frombuffer(data, dtype=np.uint8, count=need)
    # 按 步长 切出每个顶点的字节，再取出该通道的 dim 个分量
    mat = np.lib.stride_tricks.as_strided(buf, shape=(vertex_count, stride), strides=(stride, 1))
    sub = np.ascontiguousarray(mat[:, off:off + dim * size])
    arr = sub.reshape(-1).view(dtype).reshape(vertex_count, dim).astype(np.float32)
    # 归一化整型格式
    if fmt in (2, 4, 6, 8):        # UNorm
        scale = 255.0 if size == 1 else 65535.0
        arr = arr / scale
    elif fmt in (3, 5, 7, 9):      # SNorm
        scale = 127.0 if size == 1 else 32767.0
        arr = np.maximum(arr / scale, -1.0)
    return arr


# --------------------------------------------------------------------------- #
# 贴图解码
# --------------------------------------------------------------------------- #
def _bc_decode_numpy(data, width, height, fourcc):
    """纯 numpy 的 BC1/BC2/BC3/BC4/BC5 解码（无需 texture2ddecoder，
    供 Pyodide 等无法装 C 扩展的环境使用）。返回 H×W×4 的 uint8 RGBA。"""
    if width % 4 or height % 4:      # 非整块尺寸：补齐到整块再裁剪
        pw, ph = (width + 3) // 4 * 4, (height + 3) // 4 * 4
    else:
        pw, ph = width, height
    bx, by = pw // 4, ph // 4
    nb = bx * by

    def blocks(size):
        need = nb * size
        if len(data) < need:
            raise ParseError("BC 数据不足")
        return np.frombuffer(data, dtype=np.uint8, count=need).reshape(nb, size)

    def expand565(c):
        r = ((c >> 11) & 31); g = ((c >> 5) & 63); b = (c & 31)
        return np.stack([r << 3 | r >> 2, g << 2 | g >> 4, b << 3 | b >> 2], axis=-1).astype(np.uint8)

    if fourcc in ("DXT1", "BC1"):
        blk = blocks(8)
        c0 = blk[:, 0].astype(np.uint16) | (blk[:, 1].astype(np.uint16) << 8)
        c1 = blk[:, 2].astype(np.uint16) | (blk[:, 3].astype(np.uint16) << 8)
        p0, p1 = expand565(c0).astype(np.int16), expand565(c1).astype(np.int16)
        p2 = ((p0 * 2 + p1) // 3).astype(np.uint8)
        p3 = ((p0 + p1 * 2) // 3).astype(np.uint8)
        pal = np.stack([p0.astype(np.uint8), p1.astype(np.uint8), p2, p3], axis=1)   # (nb,4,3)
        alpha = None
        has_alpha_key = (c0 <= c1)                  # 3 色 + 透明模式
        pal[:, 3, :] = 0
        bits = (blk[:, 4].astype(np.uint32) | (blk[:, 5].astype(np.uint32) << 8) |
                (blk[:, 6].astype(np.uint32) << 16) | (blk[:, 7].astype(np.uint32) << 24))
    elif fourcc in ("DXT3", "BC2"):
        raw16 = np.frombuffer(data, dtype=np.uint8, count=nb * 16).reshape(nb, 16)
        abl = raw16[:, :8]
        a = np.zeros((nb, 16), dtype=np.uint8)
        for i in range(8):
            a[:, i * 2 + 0] = (abl[:, i] & 15) | ((abl[:, i] & 15) << 4)
            a[:, i * 2 + 1] = (abl[:, i] >> 4) | (abl[:, i] & 0xF0)
        alpha = a.reshape(nb, 4, 4)
        blk = raw16[:, 8:]
        c0 = blk[:, 0].astype(np.uint16) | (blk[:, 1].astype(np.uint16) << 8)
        c1 = blk[:, 2].astype(np.uint16) | (blk[:, 3].astype(np.uint16) << 8)
        p0, p1 = expand565(c0).astype(np.int16), expand565(c1).astype(np.int16)
        pal = np.stack([p0.astype(np.uint8), p1.astype(np.uint8),
                        ((p0 * 2 + p1) // 3).astype(np.uint8),
                        ((p0 + p1 * 2) // 3).astype(np.uint8)], axis=1)
        has_alpha_key = np.zeros(nb, dtype=bool)
        bits = (blk[:, 4].astype(np.uint32) | (blk[:, 5].astype(np.uint32) << 8) |
                (blk[:, 6].astype(np.uint32) << 16) | (blk[:, 7].astype(np.uint32) << 24))
    elif fourcc in ("DXT5", "BC3"):
        raw16 = np.frombuffer(data, dtype=np.uint8, count=nb * 16).reshape(nb, 16)
        a0 = raw16[:, 0].astype(np.uint16); a1 = raw16[:, 1].astype(np.uint16)
        abits = (raw16[:, 2].astype(np.uint64) | (raw16[:, 3].astype(np.uint64) << 8) |
                 (raw16[:, 4].astype(np.uint64) << 16) | (raw16[:, 5].astype(np.uint64) << 24) |
                 (raw16[:, 6].astype(np.uint64) << 32) | (raw16[:, 7].astype(np.uint64) << 40))
        if a0 > a1:
            apal = np.stack([a0, a1,
                             ((a0 * 4 + a1) // 5), ((a0 * 3 + a1 * 2) // 5),
                             ((a0 * 2 + a1 * 3) // 5), ((a0 + a1 * 4) // 5),
                             np.full(nb, 255, np.uint16), np.zeros(nb, np.uint16)], axis=1)
        else:
            apal = np.stack([a0, a1, ((a0 + a1) // 2), np.zeros(nb, np.uint16),
                             np.zeros(nb, np.uint16), np.zeros(nb, np.uint16),
                             np.zeros(nb, np.uint16), np.full(nb, 255, np.uint16)], axis=1)
        aidx = np.zeros((nb, 16), dtype=np.uint8)
        for i in range(16):
            aidx[:, i] = ((abits >> (3 * i)) & 7).astype(np.uint8)
        alpha = apal[np.arange(nb)[:, None], aidx].reshape(nb, 4, 4).astype(np.uint8)
        blk = raw16[:, 8:]
        c0 = blk[:, 0].astype(np.uint16) | (blk[:, 1].astype(np.uint16) << 8)
        c1 = blk[:, 2].astype(np.uint16) | (blk[:, 3].astype(np.uint16) << 8)
        p0, p1 = expand565(c0).astype(np.int16), expand565(c1).astype(np.int16)
        pal = np.stack([p0.astype(np.uint8), p1.astype(np.uint8),
                        ((p0 * 2 + p1) // 3).astype(np.uint8),
                        ((p0 + p1 * 2) // 3).astype(np.uint8)], axis=1)
        has_alpha_key = np.zeros(nb, dtype=bool)
        bits = (blk[:, 4].astype(np.uint32) | (blk[:, 5].astype(np.uint32) << 8) |
                (blk[:, 6].astype(np.uint32) << 16) | (blk[:, 7].astype(np.uint32) << 24))
    elif fourcc in ("BC4", "ATI1"):
        raw8 = np.frombuffer(data, dtype=np.uint8, count=nb * 8).reshape(nb, 8)
        r0 = raw8[:, 0].astype(np.uint16); r1 = raw8[:, 1].astype(np.uint16)
        bits = (raw8[:, 2].astype(np.uint64) | (raw8[:, 3].astype(np.uint64) << 8) |
                (raw8[:, 4].astype(np.uint64) << 16) | (raw8[:, 5].astype(np.uint64) << 24) |
                (raw8[:, 6].astype(np.uint64) << 32) | (raw8[:, 7].astype(np.uint64) << 40))
        pal = np.stack([r0, r1,
                        ((r0 * 4 + r1) // 5), ((r0 * 3 + r1 * 2) // 5),
                        ((r0 * 2 + r1 * 3) // 5), ((r0 + r1 * 4) // 5),
                        np.zeros(nb, np.uint16), np.full(nb, 255, np.uint16)], axis=1)
        idx = np.zeros((nb, 16), dtype=np.uint8)
        for i in range(16):
            idx[:, i] = ((bits >> (3 * i)) & 7).astype(np.uint8)
        r = pal[np.arange(nb)[:, None], idx].reshape(nb, 4, 4).astype(np.uint8)
        rgba = np.zeros((nb, 4, 4), dtype=np.uint8)
        rgba[:, :, 0] = r
        rgba[:, :, 3] = 255
        return rgba.reshape(ph, pw)[:height, :width]
    elif fourcc in ("BC5", "ATI2"):
        raw16 = np.frombuffer(data, dtype=np.uint8, count=nb * 16).reshape(nb, 16)
        chans = []
        for o in (0, 8):
            r0 = raw16[:, o].astype(np.uint16); r1 = raw16[:, o + 1].astype(np.uint16)
            bits = (raw16[:, o + 2].astype(np.uint64) | (raw16[:, o + 3].astype(np.uint64) << 8) |
                    (raw16[:, o + 4].astype(np.uint64) << 16) | (raw16[:, o + 5].astype(np.uint64) << 24) |
                    (raw16[:, o + 6].astype(np.uint64) << 32) | (raw16[:, o + 7].astype(np.uint64) << 40))
            pal = np.stack([r0, r1,
                            ((r0 * 4 + r1) // 5), ((r0 * 3 + r1 * 2) // 5),
                            ((r0 * 2 + r1 * 3) // 5), ((r0 + r1 * 4) // 5),
                            np.zeros(nb, np.uint16), np.full(nb, 255, np.uint16)], axis=1)
            idx = np.zeros((nb, 16), dtype=np.uint8)
            for i in range(16):
                idx[:, i] = ((bits >> (3 * i)) & 7).astype(np.uint8)
            chans.append(pal[np.arange(nb)[:, None], idx].reshape(nb, 4, 4).astype(np.uint8))
        rgba = np.zeros((nb, 4, 4), dtype=np.uint8)
        rgba[:, :, 0] = chans[0]
        rgba[:, :, 1] = chans[1]
        rgba[:, :, 3] = 255
        return rgba.reshape(ph, pw)[:height, :width]
    else:
        raise ParseError("numpy 解码器不支持 %s" % fourcc)

    pidx = np.zeros((nb, 16), dtype=np.uint8)
    for i in range(16):
        pidx[:, i] = ((bits >> (2 * i)) & 3).astype(np.uint8)
    rgb = pal[np.arange(nb)[:, None], pidx].reshape(nb, 4, 4, 3)
    if alpha is None:
        a = np.full((nb, 4, 4), 255, dtype=np.uint8)
        # BC1 3 色模式：索引 3 = 透明
        if has_alpha_key.any():
            m = has_alpha_key[:, None, None] & (pidx.reshape(nb, 4, 4) == 3)
            a[m] = 0
    else:
        a = alpha
    rgba = np.concatenate([rgb, a[..., None]], axis=-1).reshape(ph, pw, 4)
    return rgba[:height, :width]


def _dds_to_image(data):
    """DDS → PIL.Image(RGBA)：先试 Pillow，再按 FourCC 走 BC 解码，最后处理未压缩格式"""
    if Image is not None:
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            return img.convert("RGBA")
        except Exception:
            pass
    if len(data) < 128 or data[:4] != b"DDS ":
        return None
    height, width = struct.unpack_from("<II", data, 12)
    if width == 0 or height == 0 or width > 16384 or height > 16384:
        return None
    fourcc = data[84:88].decode("latin1", "ignore")
    off = 128

    # 未压缩（RGB/RGBA/BGRA 掩码）
    if fourcc.strip("\x00") == "":
        bits = struct.unpack_from("<I", data, 88)[0]
        rmask, gmask, bmask, amask = struct.unpack_from("<IIII", data, 92)
        if bits in (24, 32):
            px = np.frombuffer(data, dtype=np.uint8, count=width * height * (bits // 8),
                               offset=off).reshape(height, width, bits // 8)
            if bits == 24:
                b, g, r = px[:, :, 0], px[:, :, 1], px[:, :, 2]
                a = np.full_like(r, 255)
            else:
                if bmask == 0x000000FF:            # BGRA
                    b, g, r, a = px[:, :, 0], px[:, :, 1], px[:, :, 2], px[:, :, 3]
                else:                              # RGBA
                    r, g, b, a = px[:, :, 0], px[:, :, 1], px[:, :, 2], px[:, :, 3]
            if amask == 0:
                a = np.full_like(r, 255)
            rgba = np.dstack([r, g, b, a]).astype(np.uint8)
            if Image is None:
                return None
            return Image.fromarray(np.ascontiguousarray(rgba), "RGBA")
        return None

    if fourcc == "DX10" and len(data) >= 148:
        dxgi = struct.unpack_from("<I", data, 128)[0]
        fourcc = {70: "BC1", 71: "BC1", 72: "BC2", 74: "BC3", 77: "BC4", 80: "BC5",
                  83: "BC6", 95: "BC7", 98: "BC7", 76: "BC7"}.get(dxgi, "BC7")
        off = 148
    blocks_x = max(1, (width + 3) // 4)
    blocks_y = max(1, (height + 3) // 4)

    # ① numpy 解码器（BC1/2/3/4/5）：不依赖第三方扩展，Pyodide 环境也可用
    np_fourcc = {"DXT1": "BC1", "BC1": "BC1", "DXT3": "BC2", "BC2": "BC2",
                 "DXT5": "BC3", "BC3": "BC3", "BC4": "BC4", "ATI1": "BC4",
                 "BC5": "BC5", "ATI2": "BC5"}.get(fourcc)
    if np_fourcc and np is not None:
        try:
            rgba = _bc_decode_numpy(data[off:], width, height, np_fourcc)
            return Image.fromarray(rgba, "RGBA")
        except Exception:
            pass

    # ② texture2ddecoder 兜底（含 BC6/BC7）
    if _t2d is None:
        return None
    try:
        if fourcc in ("DXT1", "BC1"):
            bgra = _t2d.decode_bc1(data[off:off + blocks_x * blocks_y * 8], width, height)
        elif fourcc in ("DXT3", "BC2"):
            bgra = _t2d.decode_bc2(data[off:off + blocks_x * blocks_y * 16], width, height)
        elif fourcc in ("DXT5", "BC3"):
            bgra = _t2d.decode_bc3(data[off:off + blocks_x * blocks_y * 16], width, height)
        elif fourcc in ("BC4", "ATI1"):
            bgra = _t2d.decode_bc4(data[off:off + blocks_x * blocks_y * 8], width, height)
        elif fourcc in ("BC5", "ATI2"):
            bgra = _t2d.decode_bc5(data[off:off + blocks_x * blocks_y * 16], width, height)
        elif fourcc in ("BC7", "BC7U"):
            bgra = _t2d.decode_bc7(data[off:off + blocks_x * blocks_y * 16], width, height)
        else:
            return None
        arr = np.frombuffer(bgra, dtype=np.uint8).reshape(height, width, 4)
        rgb = arr[:, :, [2, 1, 0, 3]]          # BGRA → RGBA
        return Image.fromarray(np.ascontiguousarray(rgb), "RGBA")
    except Exception:
        return None


def decode_section_image(raw):
    """从贴图分段里取出图像（DDS 优先，其次 PNG）"""
    d = raw.find(b"DDS ")
    if d >= 0:
        img = _dds_to_image(raw[d:])
        if img is not None:
            # 注意：不做垂直翻转。CRP 内嵌 DDS 行序自下而上，行 0 = 图像底部；
            # 配合导出时的标准 V 翻转（uv.y = 1 - v）恰好还原 Unity 采样语义
            # （v=0 → 图像底部）。此处再翻转会与 V 翻转叠加成双重翻转，
            # 导致贴图上下颠倒（实测福田公交车涂装/车轮错乱）。
            return img, "dds"
    p = raw.find(b"\x89PNG\r\n\x1a\n")
    if p >= 0:
        try:
            img = Image.open(io.BytesIO(raw[p:]))
            img.load()
            return img.convert("RGBA"), "png"
        except Exception:
            return None, "png(broken)"
    # 裸 RGBA 猜测：宽高 ×4 == 剩余长度
    if len(raw) > 64:
        for off in range(8, min(len(raw) - 16, 256)):
            for w in (256, 512, 1024, 2048, 128, 64, 4096):
                if (len(raw) - off) == w * w * 4:
                    try:
                        return Image.frombytes("RGBA", (w, w), raw[off:]), "raw"
                    except Exception:
                        pass
    return None, "unreadable"


# --------------------------------------------------------------------------- #
# glTF 2.0 / GLB 写出
# --------------------------------------------------------------------------- #
COMP = {"f4": 5126, "u4": 5125, "u2": 5123, "u1": 5121}
TYPES = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}


class Gltf:
    def __init__(self):
        self.bin = bytearray()
        self.buffer_views = []
        self.accessors = []
        self.meshes = []
        self.materials = []
        self.images = []
        self.textures = []
        self.samplers = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
        self.nodes = []

    # --- 底层 ---
    def view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        off = len(self.bin)
        self.bin += data
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        self.buffer_views.append(v)
        return len(self.buffer_views) - 1

    def accessor(self, arr, type_str, comp="f4", target=34962, minmax=False):
        view = self.view(arr.tobytes(), target)
        acc = {"bufferView": view, "componentType": COMP[comp],
               "count": int(len(arr)), "type": type_str}
        if minmax:
            acc["min"] = [float(x) for x in arr.min(axis=0)]
            acc["max"] = [float(x) for x in arr.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def add_image(self, img):
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        data = buf.getvalue()
        view = self.view(data)
        self.images.append({"bufferView": view, "mimeType": "image/png"})
        self.textures.append({"source": len(self.images) - 1, "sampler": 0})
        return len(self.textures) - 1

    def add_material(self, name, tex_index=None, base_color=(1, 1, 1, 1), alpha_blend=False):
        if tex_index is None:
            pbr = {"baseColorFactor": list(base_color), "metallicFactor": 0.0, "roughnessFactor": 0.85}
            mat = {"name": name, "pbrMetallicRoughness": pbr, "doubleSided": True}
        else:
            pbr = {"baseColorFactor": list(base_color), "metallicFactor": 0.0,
                   "roughnessFactor": 0.85, "baseColorTexture": {"index": tex_index}}
            mat = {"name": name, "pbrMetallicRoughness": pbr, "doubleSided": True}
        if alpha_blend:
            mat["alphaMode"] = "BLEND"
        self.materials.append(mat)
        return len(self.materials) - 1

    def add_mesh(self, name, primitives):
        self.meshes.append({"name": name, "primitives": primitives})
        return len(self.meshes) - 1

    def add_node(self, name, mesh_index=None, matrix=None):
        node = {"name": name}
        if mesh_index is not None:
            node["mesh"] = mesh_index
        if matrix:
            node["matrix"] = matrix
        self.nodes.append(node)
        return len(self.nodes) - 1

    def serialize(self):
        gltf = {
            "asset": {"version": "2.0", "generator": "crp2glb (Cities: Skylines CRP -> glTF)"},
            "scene": 0,
            "scenes": [{"nodes": list(range(len(self.nodes)))}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [{"byteLength": len(self.bin)}],
        }
        if self.materials:
            gltf["materials"] = self.materials
        if self.textures:
            gltf["textures"] = self.textures
            gltf["images"] = self.images
            gltf["samplers"] = self.samplers
        js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        js += b" " * ((-len(js)) % 4)
        bin_chunk = bytes(self.bin) + b"\x00" * ((-len(self.bin)) % 4)
        total = 12 + 8 + len(js) + 8 + len(bin_chunk)
        out = bytearray()
        out += b"glTF" + struct.pack("<II", 2, total)
        out += struct.pack("<I", len(js)) + b"JSON" + js
        out += struct.pack("<I", len(bin_chunk)) + b"BIN\x00" + bin_chunk
        return bytes(out), gltf


# --------------------------------------------------------------------------- #
# 转换流程
# --------------------------------------------------------------------------- #
def asset_key(name):
    """从分段名里提取「资产主名」，用于贴图与网格的关联"""
    base = os.path.basename(name)
    for suffix in ("_LOD0", "_LOD1", "_LOD2", "_LOD3", "_lod0", "_lod1", "_lod2", "_lod3"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    for tag in ("_MainTex", "_maintex", "_Color", "_Diffuse", "_d", "_n", "_s", "_c", " Diffuse"):
        if base.endswith(tag):
            base = base[:-len(tag)]
    return base.lower().strip("_ -")


def _is_diffuse_name(sec_name):
    """贴图段名是否为漫反射主贴图（_MainTex / 无后缀哈希名）"""
    low = (sec_name or "").lower()
    if "_maintex" in low:
        return True
    # 该资产类（车辆）中非 _XYSMap/_ACIMap 的贴图段视作漫反射候选
    return not any(t in low for t in ("_xysmap", "_acimap", "_normal", "_bump", "_spec"))


def _pick_diffuse(group_textures, all_textures):
    """从组内贴图挑漫反射；组内没有时回退到全局最大的漫反射贴图"""
    cand = [t for t in group_textures if _is_diffuse_name(t["name"])] or list(group_textures)
    if cand:
        return max(cand, key=lambda t: t["image"].size[0] * t["image"].size[1])
    if all_textures:
        pool = [t for t in all_textures if _is_diffuse_name(t["name"])] or list(all_textures)
        return max(pool, key=lambda t: t["image"].size[0] * t["image"].size[1])
    return None


# --------------------------------------------------------------------------- #
# 预制体图（GameObject / Material 引用链）
# --------------------------------------------------------------------------- #
import re as _re

_RE_GUID = _re.compile(rb"[0-9a-f]{32}")
_RE_PROP = _re.compile(rb"_[A-Za-z][A-Za-z0-9]{1,15}")


def _find_all(raw, needle):
    out, i = [], 0
    while True:
        i = raw.find(needle, i)
        if i < 0:
            return out
        out.append(i)
        i += 1


def _typestr_end(raw, off):
    """组件类型串 "…, PublicKeyToken=null" 的结束位置"""
    e = raw.find(b"PublicKeyToken=null", off)
    return e + len(b"PublicKeyToken=null") if e >= 0 else off


def _parse_transform_block(raw, start):
    """在[start, start+48)里定位 pos(3f)+quat(4f)+scale(3f)，返回 (pos, quat, scale) 或 None"""
    best = None
    for st in range(start, min(start + 48, len(raw) - 40)):
        f = struct.unpack_from("<10f", raw, st)
        px, py, pz, qx, qy, qz, qw, sx, sy, sz = f
        qn = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
        if (0.85 < qn < 1.15 and 1e-3 < sx < 1e4 and 1e-3 < sy < 1e4 and 1e-3 < sz < 1e4
                and all(abs(v) < 1e5 for v in (px, py, pz))):
            score = abs(qn - 1.0) + abs(sx - 1.0) + abs(sy - 1.0) + abs(sz - 1.0)
            if best is None or score < best[0]:
                best = (score, (px, py, pz), (qx, qy, qz, qw), (sx, sy, sz))
    return None if best is None else best[1:]


def _parse_attach_offset(raw, field):
    """_Data 预制体序列化字段：名字后紧跟 4 字节 float"""
    m = _re.search(_re.escape(field), raw)
    if not m:
        return None
    try:
        v = struct.unpack_from("<f", raw, m.end())[0]
    except struct.error:
        return None
    return v if abs(v) < 1e4 else 0.0


def parse_prefab_graph(crp):
    """解析 GameObject/Material 段，得到渲染器、材质→贴图引用、LOD 关系。

    返回 {
      "renderers": [{"go":段号, "name":GO名, "base":去_Data名, "mesh_guid":…,
                     "mat_guid":…, "is_lod":bool, "trs":(pos,quat,scale)|None}],
      "materials": {材质guid: {"_MainTex": 贴图guid, "_XYSMap": …, "_ACIMap": …}},
      "attach":    {GO段号: {"front": float, "back": float}},
    } 或 None（无可解析的引用链时）
    """
    secs = crp["sections"]
    guid2sec = {}
    for i, s in enumerate(secs):
        m = _re.search(rb"[0-9a-f]{32}", s["name"].encode("latin1", "ignore"))
        if m:
            guid2sec[m.group(0).decode()] = i
    sec2guid = {si: g for g, si in guid2sec.items()}

    renderers, materials, attach = [], {}, {}
    lod_gos = set()
    for i, s in enumerate(secs):
        raw = s["raw"]
        kind = classify_section(raw)
        if kind == "material":
            pairs, last = {}, None
            for m in _re.finditer(rb"_[A-Za-z][A-Za-z0-9]{1,15}|[0-9a-f]{32}", raw):
                t = m.group(0)
                if _RE_GUID.fullmatch(t):
                    if last:
                        pairs.setdefault(last, t.decode())
                        last = None
                else:
                    last = t.decode()
            if pairs:
                materials[sec2guid.get(i, "")] = pairs
        elif kind == "gameobject":
            nm = s["name"]
            if " - " in nm:
                nm = nm.rsplit(" - ", 1)[0].strip()
            base = nm[:-5] if nm.endswith("_Data") else nm
            for m in _re.finditer(rb"m_lodObject", raw):
                g = _RE_GUID.search(raw, m.end(), m.end() + 96)
                if g:
                    lod_gos.add(g.group(0).decode())
            mf = _find_all(raw, b"UnityEngine.MeshFilter,")
            mr_offs = _find_all(raw, b"UnityEngine.MeshRenderer,")
            if not mf:
                continue
            tr = None
            tf = _find_all(raw, b"UnityEngine.Transform,")
            if tf:
                tr = _parse_transform_block(raw, _typestr_end(raw, tf[0]))
            used_mr = set()
            for mo in mf:
                e = _typestr_end(raw, mo)
                g = _RE_GUID.search(raw, e, e + 96)
                if not g:
                    continue
                mesh_guid = g.group(0).decode()
                # 网格之后最近的、未消费的 MeshRenderer → 材质
                mat_guid = None
                for ro in mr_offs:
                    if ro <= mo or ro in used_mr:
                        continue
                    used_mr.add(ro)
                    e2 = _typestr_end(raw, ro)
                    gl = _RE_GUID.findall(raw, e2, e2 + 160)
                    if gl:
                        mat_guid = gl[0].decode()
                    break
                renderers.append({"go": i, "name": nm, "base": base,
                                  "mesh_guid": mesh_guid, "mat_guid": mat_guid,
                                  "is_lod": False, "trs": tr})
            fr = _parse_attach_offset(raw, b"m_attachOffsetFront")
            bk = _parse_attach_offset(raw, b"m_attachOffsetBack")
            if fr is not None or bk is not None:
                attach[i] = {"front": fr or 0.0, "back": bk or 0.0}
    if not renderers:
        return None
    for r in renderers:
        # 该 GO 是否是别人的 LOD 目标
        r["is_lod"] = sec2guid.get(r["go"], "") in lod_gos
    return {"renderers": renderers, "materials": materials, "attach": attach}


def _collect_prefab_entries(crp, graph, mesh_entries, textures, opts, report):
    """按预制体引用链生成导出条目：正确的材质贴图、变体/LOD 去重、多节车辆摆放。

    返回 [{"mesh", "texs"}] 或 None（引用链不完整时由调用方回退位置分组）。
    """
    mesh_by_guid = {e["guid"]: e for e in mesh_entries if e.get("guid")}
    tex_by_guid = {t["guid"]: t for t in textures if t.get("guid")}
    main_name = (crp.get("name") or "").strip()
    # 次级车辆信息（VehicleInfoGen）名：多节车辆的拖挂数据
    sub_infos = set()
    for s in crp["sections"]:
        if classify_section(s["raw"]) == "assembly":
            tn, nm, _ = _read_crap_strings(s["raw"])
            if nm and (b"VehicleInfoGen" in (tn or b"")):
                sub_infos.add(nm)

    want_lod = opts.lod is not None and opts.lod >= 1
    cands = [r for r in graph["renderers"]
             if r["mesh_guid"] and r["mat_guid"] and r["is_lod"] == want_lod]
    if not cands and want_lod:
        cands = [r for r in graph["renderers"]
                 if r["mesh_guid"] and r["mat_guid"] and not r["is_lod"]]
        report["warnings"].append("未找到 LOD 网格，已回退导出高模")
    if not cands:
        return None

    # 同一网格被多个变体 GO 引用（如颜色变体/ _Data 副本）：_Data 优先，只保留一个
    cands.sort(key=lambda r: (not r["name"].endswith("_Data"), r["go"]))
    kept, seen_guid, seen_sig = [], set(), set()
    for r in cands:
        me = mesh_by_guid.get(r["mesh_guid"])
        if me is None:
            continue
        m = me["mesh"]
        sig = (m["vertex_count"], int(m["indices"].size),
               (m["pos"].tobytes() if "pos" in m else b"")[:64])
        if r["mesh_guid"] in seen_guid or sig in seen_sig:
            continue
        seen_guid.add(r["mesh_guid"])
        seen_sig.add(sig)
        kept.append((r, me))

    if not kept:
        return None

    # ---- 贴图：材质的 _MainTex 优先，回退位置分组 ----
    out = []
    for r, me in kept:
        tex = None
        mat = graph["materials"].get(r["mat_guid"]) if r["mat_guid"] else None
        if mat and "_MainTex" in mat:
            tex = tex_by_guid.get(mat["_MainTex"])
        texs = [tex] if tex is not None else list(me["texs"])
        out.append({"mesh": me["mesh"], "texs": texs, "ref": r, "pos_entry": me})

    # ---- 多节车辆摆放：主车在原点，拖挂车按接驳偏移链到后方 ----
    def go_attach(r):
        return graph["attach"].get(r["go"], {"front": 0.0, "back": 0.0})

    def is_main(r):
        nm = r["name"]
        return nm == main_name or nm == main_name + "_Data" or \
            (nm.endswith("_Data") and nm[:-5] == main_name)

    def is_secondary(r):
        nm = r["name"]
        if is_main(r):
            return False
        core = nm[:-5] if nm.endswith("_Data") else nm
        return core in sub_infos or nm in sub_infos

    primary = [e for e in out if not is_secondary(e["ref"])]
    secondary = [e for e in out if is_secondary(e["ref"])]
    if not primary and out:
        primary, secondary = [out[0]], out[1:]
    if secondary:
        # 主车：接驳偏移最大者优先作原点基准，平局取顶点数最多
        primary.sort(key=lambda e: (-go_attach(e["ref"])["back"],
                                    -e["mesh"]["vertex_count"]))
        head = primary[0]
        movers = primary[1:] + secondary
        hz = head["mesh"]["pos"]
        hmin = float(hz[:, 2].min())
        hmax = float(hz[:, 2].max())
        hback = go_attach(head["ref"])["back"]
        cur_tail = hmin + hback          # 拖挂连接点（主车局部 Z）
        for e in movers:
            pz = e["mesh"]["pos"]
            zmin = float(pz[:, 2].min())
            zmax = float(pz[:, 2].max())
            zlen = zmax - zmin
            off = cur_tail - zmax
            # 判据：该车节当前与主车显著重叠（堆在原点）→ 游戏里是动态摆放，需要平移
            pre_overlap = min(hmax, zmax) - max(hmin, zmin)
            if zlen > 0.01 and pre_overlap > 0.35 * zlen:
                e["mesh"]["offset"] = (0.0, 0.0, off)
                cur_tail = (cur_tail - zlen) + go_attach(e["ref"])["back"]

    # ---- 命名 ----
    used = {}
    for e in out:
        nm = e["ref"]["name"]
        if nm.endswith("_Data"):
            nm = nm[:-5]
        if want_lod:
            nm += "_LOD"
        if nm in used:
            used[nm] += 1
            nm = "%s_%d" % (nm, used[nm])
        else:
            used[nm] = 0
        e["mesh"]["name"] = nm
    return [{"mesh": e["mesh"], "texs": e["texs"]} for e in out]


def convert_crp_bytes(data, opts, report=None, source_name="asset"):
    """CRP 字节 → (glb 字节, report)。不落盘，供 CLI 与网页端共用。"""
    if report is None:
        report = {"file": source_name, "size": len(data), "sections": [],
                  "outputs": [], "warnings": []}

    if data[:4] != b"CRAP":
        # 可能是纯 UnityFS bundle
        bundle_glbs = convert_bundle(data, source_name, "", opts, report)
        if bundle_glbs:
            return bundle_glbs[0], report
        raise ParseError("无法识别的文件：既不是 CRP，也不是可解析的 UnityFS 包")

    crp = parse_crp(data)
    report["version"] = crp["version"]
    report["name"] = crp["name"]
    report["steam_id"] = crp["steam_id"]

    # ------------------------------------------------------------------ #
    # 分组：真实 CRP 的分段顺序为「Mesh → 其后紧跟的贴图 → Material」
    # （贴图在本资产里位于网格之后）。因此按「向前归组」：
    #   网格 N 拥有从它之后直到下一个网格之间的所有贴图。
    # GameObject / Material / VehicleInfo 不打断分组。
    # ColossalFramework.Importers.Image（SteamPreview/Snapshot 缩略图）不参与材质。
    # ------------------------------------------------------------------ #
    textures = []
    pending = []          # 待归属的贴图（属于「上一个网格」）
    cur_owner = None      # 当前正在收集贴图的网格条目
    mesh_entries = []     # [{"mesh":m,"texs":[...],"sig":(vc,idx),"names":[...]}]

    def _flush():
        nonlocal pending, cur_owner
        if cur_owner is not None:
            cur_owner["texs"].extend(pending)
        pending = []

    for sec in crp["sections"]:
        kind = classify_section(sec["raw"])
        gm = _re.search(rb"[0-9a-f]{32}", sec["name"].encode("latin1", "ignore"))
        guid = gm.group(0).decode() if gm else None
        info = {"name": sec["name"], "size": sec["size"], "kind": kind,
                "origin": sec["origin"], "key": asset_key(sec["name"])}
        if kind == "mesh":
            _flush()
            m = parse_mesh_crap(sec["raw"])
            score = 99.0
            layout = "crap-v2"
            if m is None:
                m, score = find_mesh(sec["raw"], sec["name"])
                layout = "unity-vertexdata"
            if m is not None:
                m["key"] = asset_key(sec["name"])
                m["section"] = sec["name"]
                m["guid"] = guid
                mvc = (m["vertex_count"] if "vertex_count" in m
                       else m["vertex_data"]["vertex_count"])
                entry = {"mesh": m, "texs": [], "names": [], "guid": guid,
                         "sig": (mvc, int(m["indices"].size))}
                mesh_entries.append(entry)
                cur_owner = entry
                vc = mvc
                ic = int(m["indices"].size)
                info.update({"vertices": vc, "indices": ic,
                             "score": round(score, 2), "layout": layout})
            else:
                cur_owner = None
                info["error"] = "未能解析出网格"
                report["warnings"].append(
                    "网格分段 %s 解析失败，可尝试 --inspect 查看结构" % sec["name"])
        elif kind == "texture_dds":
            img, how = decode_section_image(sec["raw"])
            if img is not None:
                t = {"key": asset_key(sec["name"]), "image": img,
                     "name": sec["name"], "how": how, "guid": guid}
                textures.append(t)
                pending.append(t)
                info.update({"image": "%dx%d" % img.size, "decoded_by": how})
            else:
                info["error"] = "贴图解码失败"
        elif kind == "image_png":
            # 预览图/快照：仅记录，不参与材质
            img, how = decode_section_image(sec["raw"])
            if img is not None:
                info.update({"image": "%dx%d" % img.size, "decoded_by": how,
                             "note": "预览图，不用于材质"})
        elif kind in ("gameobject", "assembly"):
            # GameObject / VehicleInfoGen 段携带真实对象名，供网格命名用
            if cur_owner is not None:
                _, oname, _ = _read_crap_strings(sec["raw"])
                if oname and oname not in cur_owner["names"]:
                    cur_owner["names"].append(oname)
        report["sections"].append(info)
    _flush()

    # ---- 回退分组：同名/同几何的重复网格共享贴图 ----
    by_sig = {}
    for e in mesh_entries:
        by_sig.setdefault(e["sig"], []).append(e)
    for sig, group in by_sig.items():
        donor = next((e for e in group if e["texs"]), None)
        if donor:
            for e in group:
                if not e["texs"] and e is not donor:
                    e["texs"] = list(donor["texs"])

    # ---- 网格命名（回退路径）：优先用其后 GameObject 对象名 ----
    asset_name = crp.get("name") or os.path.splitext(os.path.basename(source_name))[0]
    used_names = {}
    for i, e in enumerate(mesh_entries):
        cand = e["names"][0] if e["names"] else None
        if not cand:
            cand = "%s_part%d" % (asset_name, i)
        if cand in used_names:
            used_names[cand] += 1
            cand = "%s_%d" % (cand, used_names[cand])
        else:
            used_names[cand] = 0
        e["mesh"]["name"] = cand

    # ------------------------------------------------------------------ #
    # 首选路径：预制体引用链（GameObject→MeshFilter/MeshRenderer→Material→贴图）。
    # 比位置分组准确：能拿到真实材质贴图、剔除 LOD、并按接驳偏移摆放多节车辆。
    # ------------------------------------------------------------------ #
    graph = parse_prefab_graph(crp)
    entries = mesh_entries
    if graph and any(r["mesh_guid"] and r["mat_guid"] for r in graph["renderers"]):
        pe = _collect_prefab_entries(crp, graph, mesh_entries, textures, opts, report)
        if pe:
            entries = pe
        else:
            report["warnings"].append("预制体引用链解析成功但未产出网格，回退位置分组")

    meshes = [(e["mesh"], e["texs"]) for e in entries]

    if not meshes:
        raise ParseError("该 .crp 中没有找到可导出的网格（可能只含贴图或模组代码）")

    # --lod：预制体路径已在上面按 LOD 选取；回退路径按段名过滤
    lod_filter = opts.lod
    if lod_filter is not None and entries is mesh_entries:
        keep = []
        for m, gt in meshes:
            nm = (m["section"] or m["name"] or "").lower()
            if ("lod%d" % lod_filter) in nm or (lod_filter == 0 and "lod" not in nm):
                keep.append((m, gt))
        if keep:
            meshes = keep

    glb = Gltf()
    # 贴图去重（按内容尺寸+名字）后写入，映射 (name, size) → texture index
    tex_index = {}

    def tex_id(t):
        k = (t["name"], t["image"].size)
        if k not in tex_index and not opts.no_tex:
            tex_index[k] = glb.add_image(t["image"])
        return tex_index.get(k)

    # ---- 第一步：每个网格用「自己相邻贴图组」的漫反射 ----
    # 重复几何（sig 相同）强制共享同一张贴图，保证左右/前后副本外观一致。
    def _mesh_vc(m):
        return m["vertex_count"] if "vertex_count" in m else m["vertex_data"]["vertex_count"]

    sig_tex = {}
    chosen_tex = [None] * len(meshes)
    for i, (m, gt) in enumerate(meshes):
        sig = (_mesh_vc(m), int(m["indices"].size),
               (m["pos"].tobytes() if "pos" in m
                else m["vertex_data"]["data"][:64] if "data" in m["vertex_data"]
                else b""))
        if sig in sig_tex:
            chosen_tex[i] = sig_tex[sig]
            continue
        t = _pick_diffuse(gt, None) if (gt and not opts.no_tex) else None
        chosen_tex[i] = t
        if t is not None:
            sig_tex[sig] = t

    # ---- 第二步：仍无贴图的网格，取「尚未被占用」的全局 atlas（面积优先）----
    if not opts.no_tex:
        used = {(t["name"], t["image"].size) for t in chosen_tex if t is not None}
        pool = [x for x in textures if _is_diffuse_name(x["name"])] or list(textures)
        free = sorted([x for x in pool if (x["name"], x["image"].size) not in used],
                      key=lambda x: -x["image"].size[0] * x["image"].size[1])
        if free:
            for i in range(len(meshes)):
                if chosen_tex[i] is None:
                    chosen_tex[i] = free[0]     # 未占用的最大 atlas 分给主模型
                    break
        # 其余仍未分到的，沿用全局最大漫反射
        fallback = _pick_diffuse(textures, None)
        for i in range(len(meshes)):
            if chosen_tex[i] is None:
                chosen_tex[i] = fallback

    for i, (m, gt) in enumerate(meshes):
        prim = build_primitive(glb, m, opts)
        if prim is None:
            continue
        chosen = tex_id(chosen_tex[i]) if chosen_tex[i] is not None else None
        mat = glb.add_material("Material_" + (m["name"] or "mesh"), chosen)
        prim["material"] = mat
        nm = m["name"] or os.path.basename(source_name)
        mi = glb.add_mesh(nm, [prim])
        glb.add_node(nm, mi)

    blob, gltf_json = glb.serialize()
    report["stats"] = {
        "meshes": len(glb.meshes),
        "textures": len(glb.images),
        "triangles": int(sum(a["count"] // 3 for a in gltf_json["accessors"]
                             if a["type"] == "SCALAR" and a["componentType"] in (5123, 5125))),
        "bytes": len(blob),
    }
    return blob, report


def convert_crp(path, out_dir, opts):
    with open(path, "rb") as f:
        data = f.read()
    report = {"file": path, "size": len(data), "sections": [], "outputs": [], "warnings": []}
    blob, report = convert_crp_bytes(data, opts, report, source_name=path)
    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, base + ".glb")
    with open(out_path, "wb") as f:
        f.write(blob)
    report["outputs"].append(out_path)
    return report


def build_primitive(glb, mesh, opts):
    """把解析出的 mesh 变成 glTF primitive"""
    if mesh.get("format") == "crap-v2":
        return build_primitive_arrays(glb, mesh, opts)
    vd = mesh["vertex_data"]
    vc = vd["vertex_count"]
    ch = vd["channels"]
    attrs = {}

    pos = decode_channel(vd, ch[0], vc)
    pos = np.ascontiguousarray(pos.astype(np.float32))
    if len(ch) > 1 and ch[1]["dimension"] == 3:
        nrm = decode_channel(vd, ch[1], vc).astype(np.float32)
    else:
        nrm = None
    # UV：位置型通道之后的第一个 2 维通道
    uv = None
    for c in ch[2:]:
        if c["dimension"] == 2:
            uv = decode_channel(vd, c, vc).astype(np.float32)
            break

    if not opts.no_convert_hand:
        pos = pos.copy()
        pos[:, 0] = -pos[:, 0]                  # Unity(左手) → glTF(右手)
        if nrm is not None:
            nrm = nrm.copy()
            nrm[:, 0] = -nrm[:, 0]

    attrs["POSITION"] = glb.accessor(pos, "VEC3", "f4", 34962, minmax=True)
    if nrm is not None:
        attrs["NORMAL"] = glb.accessor(np.ascontiguousarray(nrm), "VEC3", "f4")
    if uv is not None:
        uv = uv.copy()
        if opts.flip_uv:
            uv[:, 1] = 1.0 - uv[:, 1]           # 可选 V 翻转（默认关：原样 1:1 即正确）
        attrs["TEXCOORD_0"] = glb.accessor(np.ascontiguousarray(uv), "VEC2", "f4")

    idx = mesh["indices"].astype(np.uint32)
    if not opts.no_convert_hand:
        idx = idx.reshape(-1, 3)[:, ::-1].ravel().copy()   # 坐标镜像后翻转绕序
    comp = "u2" if int(idx.max(initial=0)) < 65536 else "u4"
    idx_arr = idx.astype(np.uint16 if comp == "u2" else np.uint32)
    attrs_idx = glb.accessor(np.ascontiguousarray(idx_arr), "SCALAR", comp, 34963)
    return {"attributes": attrs, "indices": attrs_idx, "mode": 4}


def build_primitive_arrays(glb, mesh, opts):
    """crap-v2（分属性数组）布局 → glTF primitive"""
    pos = np.ascontiguousarray(mesh["pos"], dtype=np.float32)
    off = mesh.get("offset")
    if off is not None and any(off):
        pos = pos + np.asarray(off, dtype=np.float32)     # 多节车辆接驳摆放
    nrm = mesh.get("nrm")
    uv = mesh.get("uv")
    if nrm is not None:
        nrm = np.ascontiguousarray(nrm, dtype=np.float32)
        if nrm.shape[1] == 4:
            nrm = np.ascontiguousarray(nrm[:, :3])     # float4 法线 → float3
    if uv is not None:
        uv = np.ascontiguousarray(uv, dtype=np.float32)

    if not opts.no_convert_hand:
        pos = pos.copy()
        pos[:, 0] = -pos[:, 0]                  # Unity(左手) → glTF(右手)
        if nrm is not None:
            nrm = nrm.copy()
            nrm[:, 0] = -nrm[:, 0]

    attrs = {"POSITION": glb.accessor(pos, "VEC3", "f4", 34962, minmax=True)}
    if nrm is not None:
        attrs["NORMAL"] = glb.accessor(nrm, "VEC3", "f4")
    if uv is not None:
        uv = uv.copy()
        if opts.flip_uv:
            uv[:, 1] = 1.0 - uv[:, 1]           # 可选 V 翻转（默认关：原样 1:1 即正确）
        attrs["TEXCOORD_0"] = glb.accessor(uv, "VEC2", "f4")

    idx = mesh["indices"].astype(np.uint32)
    if not opts.no_convert_hand:
        idx = idx.reshape(-1, 3)[:, ::-1].ravel().copy()   # 坐标镜像后翻转绕序
    comp = "u2" if int(idx.max(initial=0)) < 65536 else "u4"
    idx_arr = idx.astype(np.uint16 if comp == "u2" else np.uint32)
    attrs_idx = glb.accessor(np.ascontiguousarray(idx_arr), "SCALAR", comp, 34963)
    return {"attributes": attrs, "indices": attrs_idx, "mode": 4}


def convert_bundle(data, path, out_dir, opts, report):
    """可选：直接读 UnityFS AssetBundle（需要 UnityPy）"""
    try:
        import UnityPy
    except ImportError:
        report["warnings"].append("该文件不是 CRP；如需直读 UnityFS 包请先 pip install UnityPy")
        return []
    env = UnityPy.load(io.BytesIO(data))
    meshes = []
    for obj in env.objects:
        if obj.type.name != "Mesh":
            continue
        try:
            m = obj.read()
            vertices = np.array(m.m_Vertices, dtype=np.float32).reshape(-1, 3)
            normals = np.array(m.m_Normals, dtype=np.float32).reshape(-1, 3) if m.m_Normals else None
            uvs = np.array(m.m_UV0, dtype=np.float32).reshape(-1, 2) if m.m_UV0 else None
            tris = np.array(m.m_IndexBuffer, dtype=np.int64)
            meshes.append({"name": m.m_Name, "pos": vertices, "nrm": normals, "uv": uvs,
                           "indices": tris.reshape(-1, 3)[:, ::-1]})
        except Exception as e:
            report["warnings"].append("UnityPy 读取网格失败：%s" % e)
    if not meshes:
        return []
    glb = Gltf()
    for m in meshes:
        pos = m["pos"]
        if not opts.no_convert_hand:
            pos = pos.copy()
            pos[:, 0] = -pos[:, 0]
        attrs = {"POSITION": glb.accessor(pos, "VEC3", "f4", 34962, minmax=True)}
        if m["nrm"] is not None:
            nrm = m["nrm"].copy()
            if not opts.no_convert_hand:
                nrm[:, 0] = -nrm[:, 0]
            attrs["NORMAL"] = glb.accessor(nrm, "VEC3", "f4")
        if m["uv"] is not None:
            uv = m["uv"].copy()
            if opts.flip_uv:
                uv[:, 1] = 1.0 - uv[:, 1]
            attrs["TEXCOORD_0"] = glb.accessor(uv, "VEC2", "f4")
        idx = m["indices"].ravel().astype(np.uint32)
        comp = "u2" if int(idx.max(initial=0)) < 65536 else "u4"
        prim = {"attributes": attrs,
                "indices": glb.accessor(idx.astype(np.uint16 if comp == "u2" else np.uint32),
                                        "SCALAR", comp, 34963),
                "mode": 4}
        prim["material"] = glb.add_material("mat_" + m["name"])
        mi = glb.add_mesh(m["name"], [prim])
        glb.add_node(m["name"], mi)
    blob, gltf_json = glb.serialize()
    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, base + ".glb")
    with open(out_path, "wb") as f:
        f.write(blob)
    report["outputs"].append(out_path)
    report["stats"] = {"meshes": len(glb.meshes), "textures": 0, "bytes": len(blob)}
    return [out_path]


def inspect(path):
    with open(path, "rb") as f:
        data = f.read()
    print("文件：%s（%d 字节）" % (path, len(data)))
    if data[:4] != b"CRAP":
        print("前 4 字节不是 CRAP：%r" % data[:4])
        if data[:7] == b"UnityFS":
            print("这是一个 UnityFS AssetBundle（可用 UnityPy / AssetStudio 打开）")
        return
    crp = parse_crp(data)
    print("容器：version=%s steam_id=%s name=%s 分段数=%d end_header_offset=%d"
          % (crp["version"], crp["steam_id"], crp["name"], len(crp["sections"]),
             crp["end_header_offset"]))
    for sec in crp["sections"]:
        raw = sec["raw"]
        kind = classify_section(raw)
        head = raw[:48]
        printable = "".join(chr(c) if 32 <= c < 127 else "." for c in head)
        line = "  - %-42s %8d  %-14s origin=%-20s %s" % (
            sec["name"][:42], sec["size"], kind, sec["origin"], printable)
        print(line)
        if kind == "mesh":
            m = parse_mesh_crap(raw)
            if m is not None:
                print("      Mesh: name=%s 顶点=%d 索引=%d 布局=crap-v2（分离数组）"
                      % (m["name"] or "(无)", m["vertex_count"], m["indices"].size))
            else:
                m, score = find_mesh(raw, sec["name"])
                if m:
                    vd = m["vertex_data"]
                    print("      Mesh: name=%s 顶点=%d 索引=%d 通道=%s 起点=%d 评分=%.2f"
                          % (m["name"], vd["vertex_count"], m["indices"].size,
                             [(c["format"], c["dimension"]) for c in vd["channels"]],
                             m["start"], score))
                else:
                    print("      Mesh: 解析失败（可提供此文件给维护者分析）")
        elif kind in ("texture_dds", "image_png"):
            img, how = decode_section_image(raw)
            print("      贴图: %s" % ("%dx%d via %s" % (img.size[0], img.size[1], how)
                                     if img else "解码失败"))


def collect_inputs(target):
    if os.path.isdir(target):
        out = []
        for root, _, files in os.walk(target):
            for fn in files:
                if fn.lower().endswith(".crp"):
                    out.append(os.path.join(root, fn))
        return sorted(out)
    return [target]


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="crp2glb",
        description="城市天际线 Cities: Skylines 资产 .crp → .glb 转换工具")
    ap.add_argument("path", help=".crp 文件或包含 .crp 的目录")
    ap.add_argument("-o", "--out", default=None, help="输出目录（默认与输入同目录）")
    ap.add_argument("--inspect", action="store_true", help="只打印结构信息，不导出")
    ap.add_argument("--lod", type=int, default=None, help="只导出指定 LOD 序号（0/1/2…）")
    ap.add_argument("--no-tex", action="store_true", help="不内嵌贴图")
    ap.add_argument("--no-convert-hand", action="store_true",
                    help="不做 Unity(左手)→glTF(右手) 坐标转换")
    ap.add_argument("--flip-uv", dest="flip_uv", action="store_true",
                    help="翻转 UV 的 V 轴（默认不翻：实测 CRP 网格 UV 已是 glTF 语义，"
                         "v=0 对应图像顶部，1:1 原样贴图即正确）")
    ap.add_argument("--json", action="store_true", help="输出 JSON 报告")
    ap.set_defaults(flip_uv=False)
    opts = ap.parse_args(argv)

    inputs = collect_inputs(opts.path)
    if not inputs:
        print("没有找到 .crp 文件")
        return 1

    if opts.inspect:
        for p in inputs:
            inspect(p)
            print()
        return 0

    ok, fail = 0, 0
    reports = []
    for p in inputs:
        out_dir = opts.out or os.path.dirname(os.path.abspath(p))
        try:
            rep = convert_crp(p, out_dir, opts)
            reports.append(rep)
            ok += 1
            st = rep.get("stats", {})
            print("✓ %s → %s" % (os.path.basename(p), rep["outputs"][0]))
            print("  网格 %s 个 / 贴图 %s 张 / 三角面 %s / 输出 %.2f MB"
                  % (st.get("meshes", 0), st.get("textures", 0), st.get("triangles", 0),
                     st.get("bytes", 0) / 1048576.0))
            for w in rep.get("warnings", []):
                print("  ⚠ %s" % w)
        except Exception as e:
            fail += 1
            print("✗ %s 转换失败：%s" % (os.path.basename(p), e))
            reports.append({"file": p, "error": str(e)})

    if opts.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2, default=str))
    print("\n完成：成功 %d，失败 %d" % (ok, fail))
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
