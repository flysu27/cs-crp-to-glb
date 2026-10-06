#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
selftest.py —— 端到端自测：生成样例 CRP → 转 GLB → 校验 GLB 结构与几何

校验内容：
  1. GLB 容器结构（magic/版本/块长度/JSON 可解析）
  2. accessors 与 bufferViews 的越界检查、必填字段
  3. 几何正确性：顶点数/三角面数与样例网格一致；坐标做了 Unity→glTF 转换（X 取反）
  4. 贴图：内嵌 PNG 可解码，尺寸与样例一致，材质引用了该贴图
  5. UV：1:1 原样保留（CRP 网格 UV 已是 glTF 语义）

用法：python selftest.py
"""

import io
import json
import os
import struct
import sys

import numpy as np

import crp2glb as C
from make_test_crp import make_test_crp

HERE = os.path.dirname(os.path.abspath(__file__))


def read_glb(path):
    blob = open(path, "rb").read()
    assert blob[:4] == b"glTF", "GLB magic 错误"
    version, total = struct.unpack_from("<II", blob, 4)
    assert version == 2, "GLB 版本应为 2"
    assert total == len(blob), "GLB 头部总长度与文件长度不一致：%d vs %d" % (total, len(blob))
    pos, js, bin_chunk = 12, None, None
    while pos < len(blob):
        clen, ctype = struct.unpack_from("<I4s", blob, pos)
        assert pos + 8 + clen <= len(blob), "块长度越界"
        chunk = blob[pos + 8:pos + 8 + clen]
        if ctype == b"JSON":
            js = json.loads(chunk.decode("utf-8"))
        elif ctype == b"BIN\x00":
            bin_chunk = chunk
        pos += 8 + clen
    assert js is not None and bin_chunk is not None, "缺少 JSON 或 BIN 块"
    return js, bin_chunk


def get_accessor(js, bin_chunk, index):
    acc = js["accessors"][index]
    view = js["bufferViews"][acc["bufferView"]]
    comp_fmt = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
    dtype = comp_fmt[acc["componentType"]]
    itemsize = np.dtype(dtype).itemsize
    stride = view.get("byteStride") or (itemsize * ncomp)
    base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    out = np.empty((acc["count"], ncomp), dtype=dtype)
    for i in range(acc["count"]):
        off = base + i * stride
        out[i] = np.frombuffer(bin_chunk, dtype=dtype, count=ncomp, offset=off)
    # 越界检查
    assert base + (acc["count"] - 1) * stride + itemsize * ncomp <= len(bin_chunk), \
        "accessor %d 越界" % index
    return out


def main():
    print("== 1. 生成样例 CRP ==")
    crp_path = os.path.join(HERE, "samples", "test_building.crp")
    _, size, (pos, nrm, uv, idx) = make_test_crp(crp_path)
    print("   %s（%d 字节，顶点 %d / 索引 %d）" % (crp_path, size, len(pos), len(idx)))

    print("== 2. 转换 ==")
    out_dir = os.path.join(HERE, "out")
    rc = C.main([crp_path, "-o", out_dir])
    assert rc == 0, "转换返回码 %d" % rc
    glb_path = os.path.join(out_dir, "test_building.glb")
    assert os.path.exists(glb_path), "未生成 GLB"

    print("== 3. 校验 GLB 结构 ==")
    js, bin_chunk = read_glb(glb_path)
    for i in range(len(js["accessors"])):
        get_accessor(js, bin_chunk, i)
    print("   JSON/BIN 块正常，%d 个 accessor、%d 个 mesh、%d 个 image"
          % (len(js["accessors"]), len(js["meshes"]), len(js.get("images", []))))

    print("== 4. 校验几何 ==")
    prim = js["meshes"][0]["primitives"][0]
    v = get_accessor(js, bin_chunk, prim["attributes"]["POSITION"]).astype(np.float32)
    idx_out = get_accessor(js, bin_chunk, prim["indices"]).astype(np.int64).ravel()
    assert v.shape[0] == len(pos), "顶点数不一致：%d vs %d" % (v.shape[0], len(pos))
    assert idx_out.size == len(idx), "索引数不一致：%d vs %d" % (idx_out.size, len(idx))
    expect = pos.copy()
    expect[:, 0] = -expect[:, 0]
    assert np.allclose(v, expect, atol=1e-5), "顶点坐标与预期（X 取反）不一致"
    print("   顶点 %d、三角面 %d，坐标已做 Unity→glTF 转换 ✓" % (len(v), idx_out.size // 3))

    n = get_accessor(js, bin_chunk, prim["attributes"]["NORMAL"]).astype(np.float32)
    exp_n = nrm.copy()
    exp_n[:, 0] = -exp_n[:, 0]
    assert np.allclose(n, exp_n, atol=1e-5), "法线与预期不一致"
    t = get_accessor(js, bin_chunk, prim["attributes"]["TEXCOORD_0"]).astype(np.float32)
    assert np.allclose(t, uv, atol=1e-5), "UV 应 1:1 原样保留（实测 CRP 网格 UV 已是 glTF 语义）"
    print("   法线、UV（1:1 原样保留，无需翻转）校验通过 ✓")

    print("== 5. 校验贴图与材质 ==")
    assert js.get("images"), "未内嵌贴图"
    img = js["images"][0]
    view = js["bufferViews"][img["bufferView"]]
    png = bin_chunk[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "内嵌贴图不是 PNG"
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(png))
        im.load()
        assert im.size == (64, 64), "贴图尺寸异常：%s" % (im.size,)
        print("   内嵌 PNG %dx%d 可解码 ✓" % im.size)
    except ImportError:
        print("   （未安装 Pillow，跳过解码检查）")
    mat = js["materials"][prim["material"]]
    tex_idx = mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    assert js["textures"][tex_idx]["source"] == 0, "材质未指向第一张贴图"
    print("   材质 %s 已关联贴图 ✓" % mat.get("name"))

    print("\n全部自测通过 ✅  输出：%s（%.1f KB）"
          % (glb_path, os.path.getsize(glb_path) / 1024.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
