import { readFileSync } from "fs";
import { resolve } from "path";
import { defineConfig, type Plugin } from "vite";
import wabtFactory from "wabt";

function watPlugin(): Plugin {
  let wabtPromise: Promise<any> | null = null;
  const getWabt = () => {
    if (!wabtPromise) {
      wabtPromise = wabtFactory();
    }
    return wabtPromise;
  };

  return {
    name: "vite-plugin-wat-to-wasm",
    async transform(_code, id) {
      if (!id.endsWith(".wat")) return null;
      const wabt = await getWabt();
      const watSource = readFileSync(id, "utf-8");
      const parsed = wabt.parseWat(id, watSource, {
        simd: true,
        relaxed_simd: true,
        mutable_globals: true,
        multi_value: true,
        sat_float_to_int: true,
        sign_extension: true,
        bulk_memory: true,
      });
      const binary = parsed.toBinary({ log: false, write_debug_names: false });
      const base64 = Buffer.from(binary.buffer).toString("base64");
      return {
        code: `
const base64 = "${base64}";
const binaryString = atob(base64);
const len = binaryString.length;
const bytes = new Uint8Array(len);
for (let i = 0; i < len; i++) {
  bytes[i] = binaryString.charCodeAt(i);
}
export default bytes.buffer;
`,
        map: null,
      };
    },
  };
}

export default defineConfig({
  base: "/aruco-web/",
  plugins: [watPlugin()],
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, "index.html"),
        benchmark: resolve(import.meta.dirname, "benchmark.html"),
        draw: resolve(import.meta.dirname, "draw.html"),
      },
    },
  },
});
