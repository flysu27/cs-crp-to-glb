# -*- coding: utf-8 -*-
"""端到端验证：解析转换产物 GLB，按 three.js 采样语义软件渲染。
three.js/glTF 语义：图像行 0 = 顶部，pixel_row = uv.y * th，pixel_col = uv.x * tw。
渲染正确 => 浏览器里就正确。"""
import json, struct, sys
import numpy as np
from PIL import Image

def read_glb(path):
    blob = open(path, "rb").read()
    assert blob[:4] == b"glTF"
    pos, js, bin_chunk, images = 12, None, None, []
    while pos < len(blob):
        clen, ctype = struct.unpack_from("<I4s", blob, pos)
        chunk = blob[pos + 8:pos + 8 + clen]
        if ctype == b"JSON":
            js = json.loads(chunk.decode("utf-8"))
        elif ctype == b"BIN\x00":
            bin_chunk = chunk
        pos += 8 + clen
    for img in js.get("images", []):
        v = js["bufferViews"][img["bufferView"]]
        images.append(Image.open(__import__("io").BytesIO(
            bin_chunk[v["byteOffset"]:v["byteOffset"] + v["byteLength"]])).convert("RGB"))
    return js, bin_chunk, images

def acc(js, bin_chunk, index):
    a = js["accessors"][index]
    v = js["bufferViews"][a["bufferView"]]
    fmt = {5126: "<f4", 5123: "<u2", 5125: "<u4", 5121: "u1"}[a["componentType"]]
    n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
    stride = v.get("byteStride") or (np.dtype(fmt).itemsize * n)
    base = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    out = np.empty((a["count"], n), dtype=fmt)
    for i in range(a["count"]):
        out[i] = np.frombuffer(bin_chunk, dtype=fmt, count=n, offset=base + i * stride)
    return out

def render(prims, images, view, out_png, W=1600):
    """prims: [(pos, uv, idx, tex_img)]  glTF 坐标系（X 已镜像）"""
    sx_l, sy_l = [], []
    for pos, uv, idx, tex in prims:
        sx_l.append(pos[:, 2]); sy_l.append(pos[:, 1])
    sx_all = np.concatenate(sx_l); sy_all = np.concatenate(sy_l)
    x0, x1 = sx_all.min(), sx_all.max(); y0, y1 = sy_all.min(), sy_all.max()
    scale = W / (x1 - x0 + 1e-6)
    H = max(8, int((y1 - y0) * scale) + 2)
    imgb = np.full((H, W, 3), 38, dtype=np.uint8)
    zbuf = np.full((H, W), -1e9, dtype=np.float32)
    for pos, uv, idx, tex in prims:
        sx, sy, depth = pos[:, 2], pos[:, 1], -pos[:, 0]   # 侧视：屏幕x=Z, 纵=Y, 深度=-X
        X = (sx - x0) * scale; Y = (y1 - sy) * scale
        th, tw = tex.height, tex.width
        tarr = np.asarray(tex, dtype=np.uint8)
        for t in idx.reshape(-1, 3):
            ax, ay = X[t[0]], Y[t[0]]; bx, by = X[t[1]], Y[t[1]]; cx, cy = X[t[2]], Y[t[2]]
            minx = max(0, int(min(ax, bx, cx))); maxx = min(W - 1, int(max(ax, bx, cx)) + 1)
            miny = max(0, int(min(ay, by, cy))); maxy = min(H - 1, int(max(ay, by, cy)) + 1)
            if minx > maxx or miny > maxy: continue
            d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(d) < 1e-9: continue
            xs = np.arange(minx, maxx + 1) + 0.5; ys = np.arange(miny, maxy + 1) + 0.5
            gx, gy = np.meshgrid(xs, ys)
            w0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / d
            w1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / d
            w2 = 1.0 - w0 - w1
            mask = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not mask.any(): continue
            z = w0 * depth[t[0]] + w1 * depth[t[1]] + w2 * depth[t[2]]
            uu = w0 * uv[t[0], 0] + w1 * uv[t[1], 0] + w2 * uv[t[2], 0]
            vv = w0 * uv[t[0], 1] + w1 * uv[t[1], 1] + w2 * uv[t[2], 1]
            ti = np.clip((vv * th).astype(np.int32), 0, th - 1)   # three.js: row = v*th
            tj = np.clip((uu * tw).astype(np.int32), 0, tw - 1)
            sub = imgb[miny:maxy + 1, minx:maxx + 1]
            zsub = zbuf[miny:maxy + 1, minx:maxx + 1]
            upd = mask & (z > zsub)
            zsub[upd] = z[upd]
            sub[upd] = tarr[ti[upd], tj[upd]]
    Image.fromarray(imgb).save(out_png)
    print("saved", out_png, imgb.shape)

def main(glb_path):
    js, bin_chunk, images = read_glb(glb_path)
    prims = []
    for mesh in js["meshes"]:
        for p in mesh["primitives"]:
            pos = acc(js, bin_chunk, p["attributes"]["POSITION"]).astype(np.float64)
            uv = acc(js, bin_chunk, p["attributes"]["TEXCOORD_0"]).astype(np.float64)
            idx = acc(js, bin_chunk, p["indices"]).astype(np.int64).ravel()
            mat = js["materials"][p["material"]]
            tex_i = mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]
            img_i = js["textures"][tex_i]["source"]
            prims.append((pos, uv, idx, images[img_i]))
            print("mesh=%-24s verts=%-6d tris=%-6d tex=%dx%d" %
                  (mesh.get("name", "?"), len(pos), idx.size // 3, images[img_i].width, images[img_i].height))
    zall = np.concatenate([p[0][:, 2] for p in prims])
    print("整体 Z 范围: [%.2f, %.2f]  长度 %.2f m" % (zall.min(), zall.max(), zall.max() - zall.min()))
    render(prims, images, "side", "_dbg/glb_side.png")
    print("OK")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/福田BJ6189SHEVCA-2(北京).glb")
