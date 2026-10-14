Between the two, **implementing an ID-based Linear Kalman Filter from scratch** is by far the easiest, mathematically cleanest, and naturally fits SIMD hardware.

---

### Why the Kalman Filter is the best fit:

#### 1. It is extremely easy to implement from scratch

Unlike Lucas-Kanade (which requires bilinear subpixel interpolation, image spatial gradients $I_x, I_y$, image pyramid construction, and iterative warping), a standard Linear Kalman Filter is just **basic matrix arithmetic**:

- State prediction: $\mathbf{x}_{k|k-1} = \mathbf{F} \mathbf{x}_{k-1}$
- Covariance prediction: $\mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1} \mathbf{F}^T + \mathbf{Q}$
- Kalman Gain: $\mathbf{K} = \mathbf{P}_{k|k-1} \mathbf{H}^T (\mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + \mathbf{R})^{-1}$
- State update: $\mathbf{x}_k = \mathbf{x}_{k|k-1} + \mathbf{K}(\mathbf{z}_k - \mathbf{H} \mathbf{x}_{k|k-1})$
- Covariance update: $\mathbf{P}_k = (\mathbf{I} - \mathbf{K}\mathbf{H}) \mathbf{P}_{k|k-1}$

Because you already have the marker ID from the ArUco detector, you do **not** need complex data association like the Hungarian algorithm. You just maintain a dictionary of `MarkerID -> KalmanFilter`.

#### 2. How it maps to SIMD (AVX2 / AVX-512 / ARM NEON)

A standard 2D marker tracker usually maintains an 8-element or 16-element state:

- Tracking center + scale: $[x, y, w, h, \dot{x}, \dot{y}, \dot{w}, \dot{h}]^T$ (State dimension $N = 8$).
- Tracking all 4 corners directly: $[x_0, y_0, \dots, x_3, y_3, \dot{x}_0, \dot{y}_0, \dots, \dot{x}_3, \dot{y}_3]^T$ (State dimension $N = 16$).

These matrix operations heavily benefit from SIMD vector registers:

- **FMA (Fused Multiply-Add):** Matrix multiplications ($\mathbf{F}\mathbf{x}$, $\mathbf{F}\mathbf{P}\mathbf{F}^T$) map directly to 4-wide (`_mm_fmadd_ps` / NEON `vmlaq_f32`) or 8-wide (`_mm256_fmadd_ps`) SIMD instructions.
- **Batch processing across markers (SoA Layout):** If you are tracking multiple markers simultaneously, you can vectorize _across markers_ rather than inside a single small matrix:
  $$\text{Vector Register} = [\text{Marker}_0.x, \; \text{Marker}_1.x, \; \text{Marker}_2.x, \; \text{Marker}_3.x]$$
  This gives near-100% compute density and trivial auto-vectorization by compilers like GCC or Clang.

---

### Comparison: Implementation & SIMD

| Metric                  | Linear Kalman Filter                                          | Lucas-Kanade (LK Flow)                                                            |
| :---------------------- | :------------------------------------------------------------ | :-------------------------------------------------------------------------------- |
| **Lines of C/C++ Code** | ~100–150 lines (pure math)                                    | ~400–600 lines (requires image gradients, bilinear sampling, pyramids)            |
| **Dependencies**        | None (can use standard arrays, Eigen, or raw SIMD intrinsics) | Needs raw pixel buffer access                                                     |
| **SIMD Workload**       | Dense matrix-vector products, GEMM, dot products              | Image convolutions (Sobel) and windowed accumulation ($\sum I_x^2, \sum I_x I_y$) |
| **Memory Access**       | Fits entirely in L1 cache / CPU registers                     | Requires memory fetches across multiple image rows                                |
| **Failure Mode**        | Pure math; trivial to debug                                   | Corner drifting, lighting sensitivity, edge clipping                              |

---

### Recommended Implementation Strategy

1. **State Vector ($\mathbf{x}$):** Track the 4 corners in pixel space:
   $$\mathbf{x} = [x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, \; \dot{x}_1, \dot{y}_1, \dot{x}_2, \dot{y}_2, \dot{x}_3, \dot{y}_3, \dot{x}_4, \dot{y}_4]^T$$
2. **Measurement ($\mathbf{z}$):** The 4 corners returned by the ArUco detector:
   $$\mathbf{z} = [x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4]^T$$
3. **If detected:** Run the Kalman Measurement Update step.
4. **If detector drops frame:** Skip the Measurement Update, run only the Prediction step, and draw the predicted corners.
5. **SIMD Hook:** If writing in C++, compiling with `-O3 -march=native` on contiguous `float[16]` arrays will auto-vectorize most of the linear algebra, or you can drop in **Eigen** (which has built-in AVX/NEON vectorization for small matrices like `Matrix<float, 8, 8>`).
