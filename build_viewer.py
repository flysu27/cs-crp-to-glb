# -*- coding: utf-8 -*-
"""把 crp2glb.py 源码注入 viewer.html 模板，生成最终 viewer.html。

页面内嵌转换器源码后，file:// 直接打开也能在浏览器内完成 CRP→GLB 转换
（仅依赖 Pyodide/numpy/Pillow 的 CDN）。

用法：python build_viewer.py
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    with io.open(os.path.join(HERE, "crp2glb.py"), "r", encoding="utf-8") as f:
        src = f.read()
    # script 标签内不允许出现 </script
    assert "</script" not in src.lower(), "crp2glb.py 中含有 </script，需要转义"
    with io.open(os.path.join(HERE, "_viewer_template.html"), "r", encoding="utf-8") as f:
        tpl = f.read()
    assert "__CRP2GLB_SOURCE__" in tpl, "模板缺少占位符"
    out = tpl.replace("__CRP2GLB_SOURCE__", src)
    dst = os.path.join(HERE, "viewer.html")
    with io.open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    print("已生成 %s（%d 字节，含内嵌转换器）" % (dst, len(out.encode("utf-8"))))


if __name__ == "__main__":
    main()
