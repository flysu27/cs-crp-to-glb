#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_test_crp.py —— 生成用于自测的 CRP 样例文件

按 City Skylines CRP（"CRAP" 容器）规范构造一个最小可用的资产包：
  * test_building           ：UnityEngine.Mesh 分段，内含一个带 UV 的盒体（Unity 5.x 布局）
  * test_building_MainTex   ：UnityEngine.Texture2D 分段，内含 DXT1 压缩的 DDS 贴图
  * test_building           ：UnityEngine.Material 分段（占位，模拟真实包结构）
  * metadata                ：CustomAssetMetaData 分段（占位）

用途：在没有真实 .crp 的情况下验证 crp2glb.py 的解析与导出链路。
"""

import argparse
import os
import struct
import sys

import numpy as np


def i32(v):
    return struct.pack("<i", int(v))


def u32(v):
    return struct.pack("<I", int(v))


def u64(v):
    return struct.pack("<Q", int(v))


def u16(v):
    return struct.pack("<H", int(v))


def f32(v):
    return struct.pack("<f", float(v))


def ustr(s):
    """Unity 字符串：i32 长度 + utf8 + 对齐到 4"""
    b = s.encode("utf-8")
    out = i32(len(b)) + b
    out += b"\x00" * ((-len(out)) % 4)
    return out


def pstring(s):
    """Pascal 短字符串：u8 长度 + utf8"""
    b = s.encode("utf-8")
    return bytes([len(b)]) + b


# --------------------------------------------------------------------------- #
# 几何：带 UV 的盒体
# --------------------------------------------------------------------------- #
def make_box(size=(6.0, 8.0, 6.0), uv_tile=(2.0, 2.0)):
    """返回 (positions, normals, uvs, indices)，单位：米（Y 轴向上）"""
    sx, sy, sz = [s / 2.0 for s in size]
    faces = [
        ((0, 0, 1), [(-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)]),
        ((0, 0, -1), [(sx, -sy, -sz), (-sx, -sy, -sz), (-sx, sy, -sz), (sx, sy, -sz)]),
        ((1, 0, 0), [(sx, -sy, sz), (sx, -sy, -sz), (sx, sy, -sz), (sx, sy, sz)]),
        ((-1, 0, 0), [(-sx, -sy, -sz), (-sx, -sy, sz), (-sx, sy, sz), (-sx, sy, -sz)]),
        ((0, 1, 0), [(-sx, sy, sz), (sx, sy, sz), (sx, sy, -sz), (-sx, sy, -sz)]),
        ((0, -1, 0), [(-sx, -sy, -sz), (sx, -sy, -sz), (sx, -sy, sz), (-sx, -sy, sz)]),
    ]
    pos, nrm, uv, idx = [], [], [], []
    for n, quad in faces:
        base = len(pos)
        for k, p in enumerate(quad):
            pos.append(p)
            nrm.append(n)
            u = 0.0 if k in (0, 3) else uv_tile[0]
            v = 0.0 if k in (0, 1) else uv_tile[1]
            uv.append((u, v))
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return (np.array(pos, dtype=np.float32), np.array(nrm, dtype=np.float32),
            np.array(uv, dtype=np.float32), np.array(idx, dtype=np.uint32))


# --------------------------------------------------------------------------- #
# Unity 5.x Mesh 序列化（未压缩 VertexData）
# --------------------------------------------------------------------------- #
def build_mesh_bytes(name, pos, nrm, uv, idx):
    n = len(pos)
    out = bytearray()
    out += ustr(name)

    # m_SubMeshes
    out += i32(1)
    out += u32(0)                  # firstByte
    out += u32(len(idx))           # indexCount
    out += i32(0)                  # topology = Triangles
    out += u32(0)                  # baseVertex
    out += u32(0)                  # firstVertex
    out += u32(n)                  # vertexCount
    out += b"".join(f32(v) for v in ((pos.min(axis=0) + pos.max(axis=0)) / 2.0))   # center
    out += b"".join(f32(v) for v in ((pos.max(axis=0) - pos.min(axis=0)) / 2.0))   # extent

    # m_Shapes（空）——注意 Unity 序列化中 bool 后面不做 4 字节对齐
    out += i32(n)      # m_VertexCount
    out += i32(0)      # m_Channels
    out += bytes([0])  # m_FullWeights (bool)
    out += i32(0)      # m_Shapes 数量

    # m_BindPose（空）
    out += i32(0)

    # m_VertexData：位置(0)、法线(12)、UV(24)，均为 float32
    channels = [(0, 0, 0, 3), (0, 12, 0, 3), (0, 24, 0, 2)]
    stride = 24 + 2 * 4
    data = bytearray()
    for i in range(n):
        data += pos[i].astype("<f4").tobytes()
        data += nrm[i].astype("<f4").tobytes()
        data += uv[i].astype("<f4").tobytes()
    assert len(data) == stride * n
    out += u32(n)                    # m_VertexCount
    out += i32(len(channels))        # m_Channels
    for st, off, fmt, dim in channels:
        out += bytes([st, off, fmt, dim])
    out += u32(len(data))            # m_DataSize
    out += i32(len(data)) + bytes(data)
    out += b"\x00" * ((-len(out)) % 4)     # 对齐

    # m_CompressedMesh：未压缩时占位（10 个空的 PackedBitVector）
    for _ in range(10):
        out += u32(0) + f32(0.0) + f32(0.0) + i32(0) + bytes([0, 0, 0, 0])

    # m_LocalAABB
    out += b"".join(f32(v) for v in ((pos.min(axis=0) + pos.max(axis=0)) / 2.0))
    out += b"".join(f32(v) for v in ((pos.max(axis=0) - pos.min(axis=0)) / 2.0))

    # m_MeshUsageFlags / m_IsReadable / m_KeepVertices / m_KeepIndices
    out += u32(0) + bytes([1, 1, 1]) + b"\x00" * 1

    # m_IndexBuffer：i32 长度 + 16 位索引 + 对齐
    idx16 = idx.astype("<u2")
    out += i32(len(idx16) * 2) + idx16.tobytes()
    out += b"\x00" * ((-len(out)) % 4)
    return bytes(out)


# --------------------------------------------------------------------------- #
# 贴图：棋盘格 → DXT1 DDS
# --------------------------------------------------------------------------- #
def make_checker(w=64, h=64, c1=(214, 92, 58), c2=(238, 236, 228)):
    img = np.zeros((h, w, 4), dtype=np.uint8)
    blk = 8
    for y in range(h):
        for x in range(w):
            c = c1 if ((x // blk) + (y // blk)) % 2 == 0 else c2
            img[y, x] = (c[0], c[1], c[2], 255)
    return img


def rgb888_to_565(c):
    r, g, b = int(c[0]), int(c[1]), int(c[2])
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def encode_dxt1(rgba):
    """极简 DXT1 编码：每个 4x4 块取最亮/最暗两色（range fit）"""
    h, w = rgba.shape[:2]
    assert w % 4 == 0 and h % 4 == 0
    out = bytearray()
    for by in range(0, h, 4):
        for bx in range(0, w, 4):
            block = rgba[by:by + 4, bx:bx + 4, :3].reshape(-1, 3).astype(np.int32)
            lum = block @ np.array([299, 587, 114])
            hi = block[int(np.argmax(lum))]
            lo = block[int(np.argmin(lum))]
            c0 = rgb888_to_565(hi)
            c1 = rgb888_to_565(lo)
            if c0 == c1:
                c1 = max(0, c0 - 1)
            if c0 < c1:                     # 需要 c0 > c1 才是 4 色模式
                c0, c1 = c1, c0
                hi, lo = lo, hi
            colors = np.array([[c0 >> 11 & 31, c0 >> 5 & 63, c0 & 31],
                               [c1 >> 11 & 31, c1 >> 5 & 63, c1 & 31]], dtype=np.int32)
            colors8 = np.stack([colors[:, 0] << 3, colors[:, 1] << 2, colors[:, 2] << 3], axis=1)
            d0 = np.sum((block - colors8[0]) ** 2, axis=1)
            d1 = np.sum((block - colors8[1]) ** 2, axis=1)
            sel = (d1 < d0).astype(np.uint16)
            bits = 0
            for i, s in enumerate(sel):
                bits |= int(s) << i
            out += struct.pack("<HHI", c0, c1, bits)
    return bytes(out)


def build_dds_dxt1(rgba):
    h, w = rgba.shape[:2]
    blocks = encode_dxt1(rgba)
    hdr = bytearray()
    hdr += b"DDS "
    hdr += u32(124)
    hdr += u32(0x1 | 0x2 | 0x4 | 0x1000 | 0x80000)   # CAPS|HEIGHT|WIDTH|PIXELFORMAT|LINEARSIZE
    hdr += u32(h) + u32(w)
    hdr += u32(len(blocks))                          # pitchOrLinearSize
    hdr += u32(0) + u32(1)                           # depth, mipmapCount
    hdr += b"\x00" * 44                              # reserved
    hdr += u32(32)                                   # DDS_PIXELFORMAT size
    hdr += u32(0x4)                                  # DDPF_FOURCC
    hdr += b"DXT1"
    hdr += u32(0) * 5                                # bitcount / masks
    hdr += u32(0x1000) + u32(0) * 4                  # caps
    assert len(hdr) == 128, len(hdr)
    return bytes(hdr) + blocks


# --------------------------------------------------------------------------- #
# CRP 容器组装
# --------------------------------------------------------------------------- #
def build_crp(sections, mod_name="crp2glb_test", steam_id="1234567890"):
    """sections: [(name, payload_bytes)]"""
    header = bytearray()
    header += b"CRAP"
    header += u16(1)
    header += pstring(steam_id)
    header += pstring("")            # unknown1
    header += u32(1)                 # unknown2
    header += pstring(mod_name)
    header += u32(len(sections))
    end_header_offset_pos = len(header)
    header += u64(0)                 # 占位，稍后回填

    # 计算数据区起点 = header 长度 + 所有文件头长度
    file_headers_size = 0
    for name, payload in sections:
        file_headers_size += (1 + len(name.encode()) + 1 + 0 + 4 + 8 + 8)
    payload_start = len(header) + file_headers_size
    header[end_header_offset_pos:end_header_offset_pos + 8] = u64(payload_start)

    offsets = []
    cur = 0
    for name, payload in sections:
        offsets.append((cur, len(payload)))
        cur += len(payload)

    for (name, payload), (off, size) in zip(sections, offsets):
        header += pstring(name)
        header += pstring("")        # unknown4
        header += u32(0)             # unknown5
        header += u64(off)           # 相对数据区起点的偏移
        header += u64(size)

    blob = bytearray(header)          # header 长度需与 payload_start 计算一致
    assert len(blob) == payload_start, (len(blob), payload_start)
    for name, payload in sections:
        blob += payload
    return bytes(blob)


def make_test_crp(out_path):
    pos, nrm, uv, idx = make_box()
    mesh_body = build_mesh_bytes("test_building", pos, nrm, uv, idx)
    mesh_section = (b"\x00" + b"UnityEngine.Mesh" + b"\x00\x01\x00\x00"
                    + b"colossal-metadata-placeholder" + b"\x00" * 3 + mesh_body)

    tex = make_checker(64, 64)
    dds = build_dds_dxt1(tex)
    tex_section = (b"\x00" + b"UnityEngine.Texture2D" + b"\x00\x02\x00\x00"
                   + b"t" + b"\x00" * 3 + dds)

    mat_section = (b"\x00" + b"UnityEngine.Material" + b"\x00" * 3
                   + b"{shader:Colossal/Base, mainTex:test_building_MainTex}"
                   + b"\x00" * 3)
    meta_section = (b"\x00" + b"CustomAssetMetaData" + b"\x00" * 3
                    + b"{\"name\":\"crp2glb_test\",\"type\":\"Building\"}" + b"\x00" * 3)

    crp = build_crp([("test_building", mesh_section),
                     ("test_building_MainTex", tex_section),
                     ("test_building", mat_section),
                     ("metadata", meta_section)])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(crp)
    return out_path, len(crp), (pos, nrm, uv, idx)


def main():
    ap = argparse.ArgumentParser(description="生成自测用 CRP 样例文件")
    ap.add_argument("-o", "--out", default="samples/test_building.crp")
    args = ap.parse_args()
    path, size, (pos, nrm, uv, idx) = make_test_crp(args.out)
    print("已生成 %s（%d 字节）：顶点 %d，索引 %d" % (path, size, len(pos), len(idx)))


if __name__ == "__main__":
    sys.exit(main())
