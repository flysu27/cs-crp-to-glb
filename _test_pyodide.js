// 模拟 viewer.html 在浏览器内的完整转换链路（Node + Pyodide）
const { loadPyodide } = require("pyodide");
const fs = require("fs");
const path = require("path");

const CRP = "C:/bus model/福田BJ6189SHEVCA-2(北京).crp";
const SRC = "C:/cs-crp-to-glb/crp2glb.py";
const OUT = "C:/cs-crp-to-glb/out/_pyodide_bus.glb";

async function main() {
  console.log("[1] loadPyodide…");
  const pyodide = await loadPyodide({ indexURL: path.dirname(require.resolve("pyodide/pyodide.mjs")) + "/" });
  console.log("[2] loadPackage numpy + Pillow…");
  await pyodide.loadPackage(["numpy", "Pillow"]);
  console.log("[3] 注入 crp2glb.py…");
  const src = fs.readFileSync(SRC, "utf8");
  pyodide.FS.writeFile("/crp2glb.py", src);
  await pyodide.runPythonAsync(`
import sys, json
sys.path.insert(0, "/")
import crp2glb

class _Opts:
    lod = None
    no_tex = False
    no_convert_hand = False
    flip_uv = False

def _convert(in_path, out_path):
    data = open(in_path, "rb").read()
    blob, rep = crp2glb.convert_crp_bytes(data, _Opts, None, in_path)
    with open(out_path, "wb") as f:
        f.write(blob)
    slim = {"name": rep.get("name"), "stats": rep.get("stats"),
            "warnings": rep.get("warnings", []),
            "sections": [s for s in rep.get("sections", []) if "error" in s]}
    return json.dumps(slim, ensure_ascii=False, default=str)
`);
  const convertFn = pyodide.globals.get("_convert");
  console.log("[4] 写入测试 crp…");
  pyodide.FS.writeFile("/input.crp", fs.readFileSync(CRP));
  console.log("[5] 转换…");
  const t0 = Date.now();
  const report = JSON.parse(convertFn("/input.crp", "/output.glb"));
  console.log("    耗时 %d ms", Date.now() - t0);
  console.log("    report:", JSON.stringify(report, null, 2));
  const glb = pyodide.FS.readFile("/output.glb");
  fs.writeFileSync(OUT, Buffer.from(glb));
  console.log("[6] GLB 已写出:", OUT, glb.length, "字节");

  // 校验 GLB 头
  const magic = Buffer.from(glb.slice(0, 4)).toString("latin1");
  if (magic !== "glTF") throw new Error("GLB magic 错误: " + magic);
  if (report.stats.meshes !== 2 || report.stats.textures !== 1)
    throw new Error("统计不符: " + JSON.stringify(report.stats));
  console.log("\n✅ Pyodide 链路端到端测试通过");
}

main().catch(e => { console.error("❌", e); process.exit(1); });
