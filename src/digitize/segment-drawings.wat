(module
  ;; ============================================================================
  ;; Memory & Allocator
  ;; ============================================================================
  ;; Initial 64 pages (4 MB). Dynamic heap managed by TS / bump allocator.
  (memory (export "memory") 64)

  (global $heap_ptr (mut i32) (i32.const 65536))

  (func $get_heap_ptr (export "get_heap_ptr") (result i32)
    (global.get $heap_ptr)
  )

  (func $set_heap_ptr (export "set_heap_ptr") (param $ptr i32)
    (global.set $heap_ptr (local.get $ptr))
  )

  (func $alloc (export "alloc") (param $size i32) (result i32)
    (local $res i32)
    (local $new_ptr i32)
    (local $needed_pages i32)
    (local.set $res (global.get $heap_ptr))
    ;; 16-byte align for SIMD safety
    (local.set $new_ptr
      (i32.and
        (i32.add (i32.add (local.get $res) (local.get $size)) (i32.const 15))
        (i32.const -16)
      )
    )
    (if (i32.gt_u (local.get $new_ptr) (i32.shl (memory.size) (i32.const 16)))
      (then
        (local.set $needed_pages
          (i32.sub
            (i32.shr_u (i32.add (local.get $new_ptr) (i32.const 65535)) (i32.const 16))
            (memory.size)
          )
        )
        (drop (memory.grow (local.get $needed_pages)))
      )
    )
    (global.set $heap_ptr (local.get $new_ptr))
    (local.get $res)
  )

  ;; ============================================================================
  ;; 1. Scan Group Bounding Box & Extreme Points with SIMD Acceleration
  ;; ============================================================================
  ;; Scans a region [rx, ry, rw, rh] in the binary inkMat (width $w).
  ;; Uses 16-lane vector comparison and bitmask testing to skip 0-pixel blocks in bulk.
  ;; Writes results to $outPtr (84 bytes, 21 x i32):
  ;;   [0] inkCount
  ;;   [1] minX, [2] minY, [3] maxX, [4] maxY
  ;;   [5,6] pMinX (x, y)
  ;;   [7,8] pMaxX (x, y)
  ;;   [9,10] pMinY (x, y)
  ;;   [11,12] pMaxY (x, y)
  ;;   [13,14] pMinSum (x, y)
  ;;   [15,16] pMaxSum (x, y)
  ;;   [17,18] pMinDiff (x, y)
  ;;   [19,20] pMaxDiff (x, y)
  (func $scan_group_bbox (export "scan_group_bbox")
    (param $inkPtr i32)
    (param $w i32)
    (param $rx i32)
    (param $ry i32)
    (param $rw i32)
    (param $rh i32)
    (param $outPtr i32)
    (result i32)

    (local $inkCount i32)
    (local $minX i32)
    (local $minY i32)
    (local $maxX i32)
    (local $maxY i32)
    (local $minSum i32)
    (local $maxSum i32)
    (local $minDiff i32)
    (local $maxDiff i32)

    (local $pMinX_x i32) (local $pMinX_y i32)
    (local $pMaxX_x i32) (local $pMaxX_y i32)
    (local $pMinY_x i32) (local $pMinY_y i32)
    (local $pMaxY_x i32) (local $pMaxY_y i32)
    (local $pMinSum_x i32) (local $pMinSum_y i32)
    (local $pMaxSum_x i32) (local $pMaxSum_y i32)
    (local $pMinDiff_x i32) (local $pMinDiff_y i32)
    (local $pMaxDiff_x i32) (local $pMaxDiff_y i32)

    (local $y i32)
    (local $ly i32)
    (local $rowPtr i32)
    (local $lx i32)
    (local $simdEnd i32)
    (local $vZero v128)
    (local $vPix v128)
    (local $mask i32)
    (local $bit i32)
    (local $curLx i32)
    (local $curX i32)
    (local $sum i32)
    (local $diff i32)
    (local $b i32)

    (local.set $inkCount (i32.const 0))
    (local.set $minX (local.get $rw))
    (local.set $minY (local.get $rh))
    (local.set $maxX (i32.const -1))
    (local.set $maxY (i32.const -1))
    (local.set $minSum (i32.const 2147483647))
    (local.set $maxSum (i32.const -2147483648))
    (local.set $minDiff (i32.const 2147483647))
    (local.set $maxDiff (i32.const -2147483648))

    (local.set $vZero (i8x16.splat (i32.const 0)))
    (local.set $simdEnd (i32.sub (local.get $rw) (i32.const 16)))

    (local.set $ly (i32.const 0))
    (block $y_done
      (loop $y_loop
        (br_if $y_done (i32.ge_s (local.get $ly) (local.get $rh)))
        (local.set $y (i32.add (local.get $ry) (local.get $ly)))
        (local.set $rowPtr
          (i32.add
            (local.get $inkPtr)
            (i32.add (i32.mul (local.get $y) (local.get $w)) (local.get $rx))
          )
        )

        (local.set $lx (i32.const 0))

        ;; SIMD loop (16 pixels per check)
        (block $simd_done
          (loop $simd_loop
            (br_if $simd_done (i32.gt_s (local.get $lx) (local.get $simdEnd)))
            (local.set $vPix (v128.load (i32.add (local.get $rowPtr) (local.get $lx))))
            (local.set $mask (i8x16.bitmask (i8x16.gt_u (local.get $vPix) (local.get $vZero))))

            (if (i32.ne (local.get $mask) (i32.const 0))
              (then
                ;; Non-zero ink pixels present in this 16-byte chunk
                (block $bit_done
                  (loop $bit_loop
                    (br_if $bit_done (i32.eqz (local.get $mask)))
                    (local.set $bit (i32.ctz (local.get $mask)))
                    (local.set $curLx (i32.add (local.get $lx) (local.get $bit)))
                    (local.set $curX (i32.add (local.get $rx) (local.get $curLx)))

                    ;; Update ink count
                    (local.set $inkCount (i32.add (local.get $inkCount) (i32.const 1)))

                    ;; Update minX / maxX
                    (if (i32.lt_s (local.get $curLx) (local.get $minX))
                      (then
                        (local.set $minX (local.get $curLx))
                        (local.set $pMinX_x (local.get $curX))
                        (local.set $pMinX_y (local.get $y))
                      )
                    )
                    (if (i32.gt_s (local.get $curLx) (local.get $maxX))
                      (then
                        (local.set $maxX (local.get $curLx))
                        (local.set $pMaxX_x (local.get $curX))
                        (local.set $pMaxX_y (local.get $y))
                      )
                    )

                    ;; Update minY / maxY
                    (if (i32.lt_s (local.get $ly) (local.get $minY))
                      (then
                        (local.set $minY (local.get $ly))
                        (local.set $pMinY_x (local.get $curX))
                        (local.set $pMinY_y (local.get $y))
                      )
                    )
                    (if (i32.gt_s (local.get $ly) (local.get $maxY))
                      (then
                        (local.set $maxY (local.get $ly))
                        (local.set $pMaxY_x (local.get $curX))
                        (local.set $pMaxY_y (local.get $y))
                      )
                    )

                    ;; Sum & Diff for 8-point convex hull extremes
                    (local.set $sum (i32.add (local.get $curX) (local.get $y)))
                    (local.set $diff (i32.sub (local.get $curX) (local.get $y)))

                    (if (i32.lt_s (local.get $sum) (local.get $minSum))
                      (then
                        (local.set $minSum (local.get $sum))
                        (local.set $pMinSum_x (local.get $curX))
                        (local.set $pMinSum_y (local.get $y))
                      )
                    )
                    (if (i32.gt_s (local.get $sum) (local.get $maxSum))
                      (then
                        (local.set $maxSum (local.get $sum))
                        (local.set $pMaxSum_x (local.get $curX))
                        (local.set $pMaxSum_y (local.get $y))
                      )
                    )
                    (if (i32.lt_s (local.get $diff) (local.get $minDiff))
                      (then
                        (local.set $minDiff (local.get $diff))
                        (local.set $pMinDiff_x (local.get $curX))
                        (local.set $pMinDiff_y (local.get $y))
                      )
                    )
                    (if (i32.gt_s (local.get $diff) (local.get $maxDiff))
                      (then
                        (local.set $maxDiff (local.get $diff))
                        (local.set $pMaxDiff_x (local.get $curX))
                        (local.set $pMaxDiff_y (local.get $y))
                      )
                    )

                    ;; Clear lowest set bit: mask = mask & (mask - 1)
                    (local.set $mask
                      (i32.and (local.get $mask) (i32.sub (local.get $mask) (i32.const 1)))
                    )
                    (br $bit_loop)
                  )
                )
              )
            )

            (local.set $lx (i32.add (local.get $lx) (i32.const 16)))
            (br $simd_loop)
          )
        )

        ;; Scalar tail loop for remaining pixels
        (block $tail_done
          (loop $tail_loop
            (br_if $tail_done (i32.ge_s (local.get $lx) (local.get $rw)))
            (local.set $b (i32.load8_u (i32.add (local.get $rowPtr) (local.get $lx))))

            (if (i32.gt_u (local.get $b) (i32.const 0))
              (then
                (local.set $curX (i32.add (local.get $rx) (local.get $lx)))
                (local.set $inkCount (i32.add (local.get $inkCount) (i32.const 1)))

                (if (i32.lt_s (local.get $lx) (local.get $minX))
                  (then
                    (local.set $minX (local.get $lx))
                    (local.set $pMinX_x (local.get $curX))
                    (local.set $pMinX_y (local.get $y))
                  )
                )
                (if (i32.gt_s (local.get $lx) (local.get $maxX))
                  (then
                    (local.set $maxX (local.get $lx))
                    (local.set $pMaxX_x (local.get $curX))
                    (local.set $pMaxX_y (local.get $y))
                  )
                )

                (if (i32.lt_s (local.get $ly) (local.get $minY))
                  (then
                    (local.set $minY (local.get $ly))
                    (local.set $pMinY_x (local.get $curX))
                    (local.set $pMinY_y (local.get $y))
                  )
                )
                (if (i32.gt_s (local.get $ly) (local.get $maxY))
                  (then
                    (local.set $maxY (local.get $ly))
                    (local.set $pMaxY_x (local.get $curX))
                    (local.set $pMaxY_y (local.get $y))
                  )
                )

                (local.set $sum (i32.add (local.get $curX) (local.get $y)))
                (local.set $diff (i32.sub (local.get $curX) (local.get $y)))

                (if (i32.lt_s (local.get $sum) (local.get $minSum))
                  (then
                    (local.set $minSum (local.get $sum))
                    (local.set $pMinSum_x (local.get $curX))
                    (local.set $pMinSum_y (local.get $y))
                  )
                )
                (if (i32.gt_s (local.get $sum) (local.get $maxSum))
                  (then
                    (local.set $maxSum (local.get $sum))
                    (local.set $pMaxSum_x (local.get $curX))
                    (local.set $pMaxSum_y (local.get $y))
                  )
                )
                (if (i32.lt_s (local.get $diff) (local.get $minDiff))
                  (then
                    (local.set $minDiff (local.get $diff))
                    (local.set $pMinDiff_x (local.get $curX))
                    (local.set $pMinDiff_y (local.get $y))
                  )
                )
                (if (i32.gt_s (local.get $diff) (local.get $maxDiff))
                  (then
                    (local.set $maxDiff (local.get $diff))
                    (local.set $pMaxDiff_x (local.get $curX))
                    (local.set $pMaxDiff_y (local.get $y))
                  )
                )
              )
            )

            (local.set $lx (i32.add (local.get $lx) (i32.const 1)))
            (br $tail_loop)
          )
        )

        (local.set $ly (i32.add (local.get $ly) (i32.const 1)))
        (br $y_loop)
      )
    )

    ;; Store results in $outPtr (21 x i32)
    (i32.store offset=0 (local.get $outPtr) (local.get $inkCount))
    (i32.store offset=4 (local.get $outPtr) (local.get $minX))
    (i32.store offset=8 (local.get $outPtr) (local.get $minY))
    (i32.store offset=12 (local.get $outPtr) (local.get $maxX))
    (i32.store offset=16 (local.get $outPtr) (local.get $maxY))

    (i32.store offset=20 (local.get $outPtr) (local.get $pMinX_x))
    (i32.store offset=24 (local.get $outPtr) (local.get $pMinX_y))
    (i32.store offset=28 (local.get $outPtr) (local.get $pMaxX_x))
    (i32.store offset=32 (local.get $outPtr) (local.get $pMaxX_y))

    (i32.store offset=36 (local.get $outPtr) (local.get $pMinY_x))
    (i32.store offset=40 (local.get $outPtr) (local.get $pMinY_y))
    (i32.store offset=44 (local.get $outPtr) (local.get $pMaxY_x))
    (i32.store offset=48 (local.get $outPtr) (local.get $pMaxY_y))

    (i32.store offset=52 (local.get $outPtr) (local.get $pMinSum_x))
    (i32.store offset=56 (local.get $outPtr) (local.get $pMinSum_y))
    (i32.store offset=60 (local.get $outPtr) (local.get $pMaxSum_x))
    (i32.store offset=64 (local.get $outPtr) (local.get $pMaxSum_y))

    (i32.store offset=68 (local.get $outPtr) (local.get $pMinDiff_x))
    (i32.store offset=72 (local.get $outPtr) (local.get $pMinDiff_y))
    (i32.store offset=76 (local.get $outPtr) (local.get $pMaxDiff_x))
    (i32.store offset=80 (local.get $outPtr) (local.get $pMaxDiff_y))

    (local.get $inkCount)
  )

  ;; ============================================================================
  ;; 2. Extract Raster Crop with SIMD Vectorization
  ;; ============================================================================
  ;; Extracts the tight bounding box crop from $inkPtr into destination buffers:
  ;;   1) $alphaMaskDst: 1-byte per pixel (255 for ink, 0 for bg)
  ;;   2) $defaultDataDst: 4-bytes per pixel ($inkPixel32 for ink, 0 for bg)
  ;;   3) $activeDataDst (optional, 0 to skip): 4-bytes per pixel ($activePixel32 for ink, 0 for bg)
  ;; Uses SIMD to process 16 pixels simultaneously across buffers in parallel.
  (func $extract_raster_crop (export "extract_raster_crop")
    (param $inkPtr i32)
    (param $w i32)
    (param $tightX i32)
    (param $tightY i32)
    (param $tightW i32)
    (param $tightH i32)
    (param $alphaMaskDst i32)
    (param $defaultDataDst i32)
    (param $activeDataDst i32)
    (param $inkPixel32 i32)
    (param $activePixel32 i32)

    (local $y i32)
    (local $srcRow i32)
    (local $dstAlphaRow i32)
    (local $dstDefaultRow i32)
    (local $dstActiveRow i32)
    (local $x i32)
    (local $simdEnd i32)

    (local $vZero v128)
    (local $vInkSplat v128)
    (local $vActiveSplat v128)
    (local $vSrc v128)
    (local $vMask8 v128)
    (local $vLo16 v128)
    (local $vHi16 v128)
    (local $vM0 v128)
    (local $vM1 v128)
    (local $vM2 v128)
    (local $vM3 v128)
    (local $b i32)

    (local.set $vZero (i8x16.splat (i32.const 0)))
    (local.set $vInkSplat (i32x4.splat (local.get $inkPixel32)))
    (local.set $simdEnd (i32.sub (local.get $tightW) (i32.const 16)))

    (if (i32.eqz (local.get $activeDataDst))
      (then
        ;; Fast path: Default & Alpha only (Active skipped)
        (local.set $y (i32.const 0))
        (block $y_done_def
          (loop $y_loop_def
            (br_if $y_done_def (i32.ge_s (local.get $y) (local.get $tightH)))

            (local.set $srcRow
              (i32.add
                (local.get $inkPtr)
                (i32.add
                  (i32.mul (i32.add (local.get $tightY) (local.get $y)) (local.get $w))
                  (local.get $tightX)
                )
              )
            )
            (local.set $dstAlphaRow
              (i32.add (local.get $alphaMaskDst) (i32.mul (local.get $y) (local.get $tightW)))
            )
            (local.set $dstDefaultRow
              (i32.add
                (local.get $defaultDataDst)
                (i32.shl (i32.mul (local.get $y) (local.get $tightW)) (i32.const 2))
              )
            )

            (local.set $x (i32.const 0))

            ;; SIMD loop (16 pixels per iteration)
            (block $simd_done_def
              (loop $simd_loop_def
                (br_if $simd_done_def (i32.gt_s (local.get $x) (local.get $simdEnd)))

                (local.set $vSrc (v128.load (i32.add (local.get $srcRow) (local.get $x))))
                (local.set $vMask8 (i8x16.gt_u (local.get $vSrc) (local.get $vZero)))

                (v128.store (i32.add (local.get $dstAlphaRow) (local.get $x)) (local.get $vMask8))

                (local.set $vLo16 (i16x8.extend_low_i8x16_s (local.get $vMask8)))
                (local.set $vHi16 (i16x8.extend_high_i8x16_s (local.get $vMask8)))

                (local.set $vM0 (i32x4.extend_low_i16x8_s (local.get $vLo16)))
                (v128.store
                  (i32.add (local.get $dstDefaultRow) (i32.shl (local.get $x) (i32.const 2)))
                  (v128.and (local.get $vM0) (local.get $vInkSplat))
                )

                (local.set $vM1 (i32x4.extend_high_i16x8_s (local.get $vLo16)))
                (v128.store
                  (i32.add
                    (local.get $dstDefaultRow)
                    (i32.shl (i32.add (local.get $x) (i32.const 4)) (i32.const 2))
                  )
                  (v128.and (local.get $vM1) (local.get $vInkSplat))
                )

                (local.set $vM2 (i32x4.extend_low_i16x8_s (local.get $vHi16)))
                (v128.store
                  (i32.add
                    (local.get $dstDefaultRow)
                    (i32.shl (i32.add (local.get $x) (i32.const 8)) (i32.const 2))
                  )
                  (v128.and (local.get $vM2) (local.get $vInkSplat))
                )

                (local.set $vM3 (i32x4.extend_high_i16x8_s (local.get $vHi16)))
                (v128.store
                  (i32.add
                    (local.get $dstDefaultRow)
                    (i32.shl (i32.add (local.get $x) (i32.const 12)) (i32.const 2))
                  )
                  (v128.and (local.get $vM3) (local.get $vInkSplat))
                )

                (local.set $x (i32.add (local.get $x) (i32.const 16)))
                (br $simd_loop_def)
              )
            )

            ;; Scalar tail loop
            (block $tail_done_def
              (loop $tail_loop_def
                (br_if $tail_done_def (i32.ge_s (local.get $x) (local.get $tightW)))
                (local.set $b (i32.load8_u (i32.add (local.get $srcRow) (local.get $x))))

                (if (i32.gt_u (local.get $b) (i32.const 0))
                  (then
                    (i32.store8
                      (i32.add (local.get $dstAlphaRow) (local.get $x))
                      (i32.const 255)
                    )
                    (i32.store
                      (i32.add (local.get $dstDefaultRow) (i32.shl (local.get $x) (i32.const 2)))
                      (local.get $inkPixel32)
                    )
                  )
                  (else
                    (i32.store8
                      (i32.add (local.get $dstAlphaRow) (local.get $x))
                      (i32.const 0)
                    )
                    (i32.store
                      (i32.add (local.get $dstDefaultRow) (i32.shl (local.get $x) (i32.const 2)))
                      (i32.const 0)
                    )
                  )
                )

                (local.set $x (i32.add (local.get $x) (i32.const 1)))
                (br $tail_loop_def)
              )
            )

            (local.set $y (i32.add (local.get $y) (i32.const 1)))
            (br $y_loop_def)
          )
        )
      )
      (else
        ;; Dual path: Both Default and Active
        (local.set $vActiveSplat (i32x4.splat (local.get $activePixel32)))
        (local.set $y (i32.const 0))
        (block $y_done
          (loop $y_loop
            (br_if $y_done (i32.ge_s (local.get $y) (local.get $tightH)))

            (local.set $srcRow
              (i32.add
                (local.get $inkPtr)
                (i32.add
                  (i32.mul (i32.add (local.get $tightY) (local.get $y)) (local.get $w))
                  (local.get $tightX)
                )
              )
            )
            (local.set $dstAlphaRow
              (i32.add (local.get $alphaMaskDst) (i32.mul (local.get $y) (local.get $tightW)))
            )
            (local.set $dstDefaultRow
              (i32.add
                (local.get $defaultDataDst)
                (i32.shl (i32.mul (local.get $y) (local.get $tightW)) (i32.const 2))
              )
            )
            (local.set $dstActiveRow
              (i32.add
                (local.get $activeDataDst)
                (i32.shl (i32.mul (local.get $y) (local.get $tightW)) (i32.const 2))
              )
            )

            (local.set $x (i32.const 0))

            ;; SIMD loop (16 pixels per iteration)
            (block $simd_done
              (loop $simd_loop
                (br_if $simd_done (i32.gt_s (local.get $x) (local.get $simdEnd)))

                ;; Load 16 bytes of ink mask
                (local.set $vSrc (v128.load (i32.add (local.get $srcRow) (local.get $x))))
                ;; Compare > 0: produces 0xFF for ink pixels, 0x00 for bg
                (local.set $vMask8 (i8x16.gt_u (local.get $vSrc) (local.get $vZero)))

                ;; 1. Store alpha mask (16 bytes in 1 instruction)
                (v128.store (i32.add (local.get $dstAlphaRow) (local.get $x)) (local.get $vMask8))

                ;; 2. Sign-extend 16 bytes (0xFF / 0x00) to 16-bit words (0xFFFF / 0x0000)
                (local.set $vLo16 (i16x8.extend_low_i8x16_s (local.get $vMask8)))
                (local.set $vHi16 (i16x8.extend_high_i8x16_s (local.get $vMask8)))

                ;; Pixels 0..3: sign-extend low 4 words to 32-bit (0xFFFFFFFF / 0x00000000)
                (local.set $vM0 (i32x4.extend_low_i16x8_s (local.get $vLo16)))
                (v128.store
                  (i32.add (local.get $dstDefaultRow) (i32.shl (local.get $x) (i32.const 2)))
                  (v128.and (local.get $vM0) (local.get $vInkSplat))
                )
                (v128.store
                  (i32.add (local.get $dstActiveRow) (i32.shl (local.get $x) (i32.const 2)))
                  (v128.and (local.get $vM0) (local.get $vActiveSplat))
                )

                ;; Pixels 4..7
                (local.set $vM1 (i32x4.extend_high_i16x8_s (local.get $vLo16)))
                (v128.store
                  (i32.add
                    (local.get $dstDefaultRow)
                    (i32.shl (i32.add (local.get $x) (i32.const 4)) (i32.const 2))
                  )
                  (v128.and (local.get $vM1) (local.get $vInkSplat))
                )
                (v128.store
                  (i32.add
                    (local.get $dstActiveRow)
                    (i32.shl (i32.add (local.get $x) (i32.const 4)) (i32.const 2))
                  )
                  (v128.and (local.get $vM1) (local.get $vActiveSplat))
                )

                ;; Pixels 8..11
                (local.set $vM2 (i32x4.extend_low_i16x8_s (local.get $vHi16)))
                (v128.store
                  (i32.add
                    (local.get $dstDefaultRow)
                    (i32.shl (i32.add (local.get $x) (i32.const 8)) (i32.const 2))
                  )
                  (v128.and (local.get $vM2) (local.get $vInkSplat))
                )
                (v128.store
                  (i32.add
                    (local.get $dstActiveRow)
                    (i32.shl (i32.add (local.get $x) (i32.const 8)) (i32.const 2))
                  )
                  (v128.and (local.get $vM2) (local.get $vActiveSplat))
                )

                ;; Pixels 12..15
                (local.set $vM3 (i32x4.extend_high_i16x8_s (local.get $vHi16)))
                (v128.store
                  (i32.add
                    (local.get $dstDefaultRow)
                    (i32.shl (i32.add (local.get $x) (i32.const 12)) (i32.const 2))
                  )
                  (v128.and (local.get $vM3) (local.get $vInkSplat))
                )
                (v128.store
                  (i32.add
                    (local.get $dstActiveRow)
                    (i32.shl (i32.add (local.get $x) (i32.const 12)) (i32.const 2))
                  )
                  (v128.and (local.get $vM3) (local.get $vActiveSplat))
                )

                (local.set $x (i32.add (local.get $x) (i32.const 16)))
                (br $simd_loop)
              )
            )

            ;; Scalar tail loop
            (block $tail_done
              (loop $tail_loop
                (br_if $tail_done (i32.ge_s (local.get $x) (local.get $tightW)))
                (local.set $b (i32.load8_u (i32.add (local.get $srcRow) (local.get $x))))

                (if (i32.gt_u (local.get $b) (i32.const 0))
                  (then
                    (i32.store8
                      (i32.add (local.get $dstAlphaRow) (local.get $x))
                      (i32.const 255)
                    )
                    (i32.store
                      (i32.add (local.get $dstDefaultRow) (i32.shl (local.get $x) (i32.const 2)))
                      (local.get $inkPixel32)
                    )
                    (i32.store
                      (i32.add (local.get $dstActiveRow) (i32.shl (local.get $x) (i32.const 2)))
                      (local.get $activePixel32)
                    )
                  )
                  (else
                    (i32.store8
                      (i32.add (local.get $dstAlphaRow) (local.get $x))
                      (i32.const 0)
                    )
                    (i32.store
                      (i32.add (local.get $dstDefaultRow) (i32.shl (local.get $x) (i32.const 2)))
                      (i32.const 0)
                    )
                    (i32.store
                      (i32.add (local.get $dstActiveRow) (i32.shl (local.get $x) (i32.const 2)))
                      (i32.const 0)
                    )
                  )
                )

                (local.set $x (i32.add (local.get $x) (i32.const 1)))
                (br $tail_loop)
              )
            )

            (local.set $y (i32.add (local.get $y) (i32.const 1)))
            (br $y_loop)
          )
        )
      )
    )
  )
)
