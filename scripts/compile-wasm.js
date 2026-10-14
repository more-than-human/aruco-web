import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import wabt from "wabt";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function compileWat() {
  const rootDir = path.resolve(__dirname, "..");
  const watPath = path.resolve(rootDir, "src/lib-wasm-lite/aruco4x4.wat");
  const wasmPath = path.resolve(rootDir, "src/lib-wasm-lite/aruco4x4.wasm");

  const w = await wabt();
  const watSource = fs.readFileSync(watPath, "utf8");
  const features = {
    simd: true,
    relaxed_simd: true,
    bulk_memory: true,
    mutable_globals: true,
    sat_float_to_int: true,
    sign_extension: true,
    multi_value: true,
  };

  const parsedModule = w.parseWat(path.basename(watPath), watSource, features);
  const { buffer } = parsedModule.toBinary({
    log: false,
    write_debug_names: true,
  });

  fs.writeFileSync(wasmPath, Buffer.from(buffer));
  console.log(`Compiled ${watPath} -> ${wasmPath} (${buffer.length} bytes)`);
}

compileWat().catch((err) => {
  console.error("Failed to compile WAT:", err);
  process.exit(1);
});
