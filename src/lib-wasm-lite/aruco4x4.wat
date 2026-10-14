(module
  (memory (export "memory") 64)

  ;; DICT_4X4_50 (50 x u16 = 100 bytes at 0x0000)
  (data (i32.const 0)
    "\32\b5\9a\0f\2d\33\46\99\9e\54\cd\79\2e\9e\f2\c4\da\fe\56\cf"
    "\91\f9\a7\11\b7\0e\0f\2a\b1\24\3e\26\65\46\00\66\5e\6c\af\76"
    "\8b\86\2b\b0\d5\cc\82\dd\47\fe\71\94\e4\ac\54\a5\23\21\6f\34"
    "\15\44\b2\57\cf\9e\cb\f0\ae\08\29\09\75\18\ff\04\f6\0d\5a\1c"
    "\18\17\28\2a\8c\32\b2\38\e8\24\eb\2e\3f\2d\64\4b\2e\50\13\50"
  )

  ;; NEIGHBORHOOD dx (8 x i32 at 0x0080) and dy (8 x i32 at 0x00A0)
  (data (i32.const 128) "\01\00\00\00\01\00\00\00\00\00\00\00\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\00\00\00\00\01\00\00\00")
  (data (i32.const 160) "\00\00\00\00\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\00\00\00\00\01\00\00\00\01\00\00\00\01\00\00\00")

  ;; Fixed Memory Layout Constants:
  ;; 0x00C0 (192)   : neighborhoodDeltas (16 x i32 = 64 bytes)
  ;; 0x0100 (256)   : warpedData (48 * 48 = 2304 bytes)
  ;; 0x0A00 (2560)  : otsu hist (256 * 4 = 1024 bytes)
  ;; 0x0E00 (3584)  : perspective matrix (9 * f64 = 72 bytes)
  ;; 0x0F00 (3840)  : candidates (512 * 32 bytes = 16384 bytes) [8 x i32 per quad]
  ;; 0x4F00 (20224) : tooNear flags (512 bytes)
  ;; 0x5100 (20736) : approxPolyDP stack (2048 * 8 = 16384 bytes)
  ;; 0x9100 (37120) : approxPolyDP poly result (512 * 8 = 4096 bytes)
  ;; 0xA100 (41216) : detected markers output (256 * 40 = 10240 bytes)
  ;; 0xC900 (51456) : dynamic heap base

  (global $lastW (mut i32) (i32.const 0))
  (global $lastH (mut i32) (i32.const 0))
  (global $rgbaPtr (mut i32) (i32.const 51456))
  (global $greyPtr (mut i32) (i32.const 0))
  (global $thresPtr (mut i32) (i32.const 0))
  (global $binaryPtr (mut i32) (i32.const 0))
  (global $contourPtr (mut i32) (i32.const 0))
  (global $maxContourPts (mut i32) (i32.const 0))

  (func $ensureBuffers (export "ensureBuffers") (param $w i32) (param $h i32) (result i32)
    (local $len i32)
    (local $binLen i32)
    (local $totalBytes i32)
    (local $curBytes i32)
    (local $needPages i32)

    (if (i32.and (i32.eq (local.get $w) (global.get $lastW)) (i32.eq (local.get $h) (global.get $lastH)))
      (then (return (global.get $rgbaPtr)))
    )

    (global.set $lastW (local.get $w))
    (global.set $lastH (local.get $h))
    (local.set $len (i32.mul (local.get $w) (local.get $h)))
    (local.set $binLen (i32.mul (i32.add (local.get $w) (i32.const 2)) (i32.add (local.get $h) (i32.const 2))))

    ;; Align all pointers to 16 bytes
    (global.set $rgbaPtr (i32.const 51456))
    (global.set $greyPtr (i32.and (i32.add (i32.add (global.get $rgbaPtr) (i32.shl (local.get $len) (i32.const 2))) (i32.const 15)) (i32.const -16)))
    (global.set $thresPtr (i32.and (i32.add (i32.add (global.get $greyPtr) (local.get $len)) (i32.const 15)) (i32.const -16)))
    (global.set $binaryPtr (i32.and (i32.add (i32.add (global.get $thresPtr) (local.get $len)) (i32.const 15)) (i32.const -16)))
    (global.set $contourPtr (i32.and (i32.add (i32.add (global.get $binaryPtr) (i32.shl (local.get $binLen) (i32.const 2))) (i32.const 15)) (i32.const -16)))
    (global.set $maxContourPts (i32.shl (local.get $len) (i32.const 1)))

    (local.set $totalBytes (i32.add (global.get $contourPtr) (i32.shl (global.get $maxContourPts) (i32.const 3))))
    (local.set $curBytes (i32.shl (memory.size) (i32.const 16)))

    (if (i32.gt_u (local.get $totalBytes) (local.get $curBytes))
      (then
        (local.set $needPages (i32.shr_u (i32.add (i32.sub (local.get $totalBytes) (local.get $curBytes)) (i32.const 65535)) (i32.const 16)))
        (drop (memory.grow (local.get $needPages)))
      )
    )
    (global.get $rgbaPtr)
  )

  (func $getOutputPtr (export "getOutputPtr") (result i32)
    (i32.const 41216)
  )

  ;; SIMD Grayscale: dst[i] = (R*77 + G*150 + B*29) >> 8
  (func $grayscale (param $src i32) (param $dst i32) (param $len i32)
    (local $i i32)
    (local $len4 i32)
    (local $v v128)
    (local $r v128)
    (local $g v128)
    (local $b v128)
    (local $y v128)
    (local $maskFF v128)
    (local $c77 v128)
    (local $c150 v128)
    (local $c29 v128)

    (local.set $maskFF (i32x4.splat (i32.const 255)))
    (local.set $c77 (i32x4.splat (i32.const 77)))
    (local.set $c150 (i32x4.splat (i32.const 150)))
    (local.set $c29 (i32x4.splat (i32.const 29)))
    (local.set $len4 (i32.and (local.get $len) (i32.const -4)))

    (block $break
      (loop $loop
        (br_if $break (i32.ge_u (local.get $i) (local.get $len4)))
        (local.set $v (v128.load (local.get $src)))
        (local.set $r (v128.and (local.get $v) (local.get $maskFF)))
        (local.set $g (v128.and (i32x4.shr_u (local.get $v) (i32.const 8)) (local.get $maskFF)))
        (local.set $b (v128.and (i32x4.shr_u (local.get $v) (i32.const 16)) (local.get $maskFF)))

        (local.set $y
          (i32x4.shr_u
            (i32x4.add
              (i32x4.add (i32x4.mul (local.get $r) (local.get $c77)) (i32x4.mul (local.get $g) (local.get $c150)))
              (i32x4.mul (local.get $b) (local.get $c29))
            )
            (i32.const 8)
          )
        )

        (i32.store8 (local.get $dst) (i32x4.extract_lane 0 (local.get $y)))
        (i32.store8 offset=1 (local.get $dst) (i32x4.extract_lane 1 (local.get $y)))
        (i32.store8 offset=2 (local.get $dst) (i32x4.extract_lane 2 (local.get $y)))
        (i32.store8 offset=3 (local.get $dst) (i32x4.extract_lane 3 (local.get $y)))

        (local.set $src (i32.add (local.get $src) (i32.const 16)))
        (local.set $dst (i32.add (local.get $dst) (i32.const 4)))
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $loop)
      )
    )

    (block $tailBreak
      (loop $tail
        (br_if $tailBreak (i32.ge_u (local.get $i) (local.get $len)))
        (i32.store8 (local.get $dst)
          (i32.shr_u
            (i32.add
              (i32.add
                (i32.mul (i32.load8_u (local.get $src)) (i32.const 77))
                (i32.mul (i32.load8_u offset=1 (local.get $src)) (i32.const 150))
              )
              (i32.mul (i32.load8_u offset=2 (local.get $src)) (i32.const 29))
            )
            (i32.const 8)
          )
        )
        (local.set $src (i32.add (local.get $src) (i32.const 4)))
        (local.set $dst (i32.add (local.get $dst) (i32.const 1)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $tail)
      )
    )
  )

  ;; Stack Box Blur (kernelSize = 2, mult = 205, shift = 10) + SIMD Adaptive Threshold (threshold = 7)
  (func $adaptiveThreshold (param $src i32) (param $dst i32) (param $w i32) (param $h i32)
    (local $x i32) (local $y i32) (local $wm i32) (local $hm i32)
    (local $p i32) (local $pos i32) (local $sum i32)
    (local $s0 i32) (local $s1 i32) (local $s2 i32) (local $s3 i32) (local $s4 i32)
    (local $sp i32) (local $nextIdx i32) (local $addPx i32) (local $oldPx i32)
    (local $len i32) (local $len16 i32) (local $i i32)
    (local $vSrc v128) (local $vBlur v128) (local $vThresh v128)

    (local.set $wm (i32.sub (local.get $w) (i32.const 1)))
    (local.set $hm (i32.sub (local.get $h) (i32.const 1)))
    (local.set $pos (local.get $dst))

    ;; Horizontal pass (kernelSize = 2: 5-element circular buffer in locals s0..s4)
    (local.set $y (i32.const 0))
    (block $hBreak
      (loop $hLoop
        (br_if $hBreak (i32.ge_s (local.get $y) (local.get $h)))
        (local.set $p (i32.add (local.get $src) (i32.mul (local.get $y) (local.get $w))))
        (local.set $s0 (i32.load8_u (local.get $p)))
        (local.set $s1 (local.get $s0))
        (local.set $s2 (local.get $s0))
        (local.set $s3 (i32.load8_u (i32.add (local.get $p) (select (local.get $wm) (i32.const 1) (i32.gt_s (i32.const 1) (local.get $wm))))))
        (local.set $s4 (i32.load8_u (i32.add (local.get $p) (select (local.get $wm) (i32.const 2) (i32.gt_s (i32.const 2) (local.get $wm))))))
        (local.set $sum (i32.add (i32.add (i32.mul (local.get $s0) (i32.const 3)) (local.get $s3)) (local.get $s4)))
        (local.set $sp (i32.const 0))

        (local.set $x (i32.const 0))
        (block $xBreak
          (loop $xLoop
            (br_if $xBreak (i32.ge_s (local.get $x) (local.get $w)))
            (i32.store8 (local.get $pos) (i32.shr_u (i32.mul (local.get $sum) (i32.const 205)) (i32.const 10)))
            (local.set $pos (i32.add (local.get $pos) (i32.const 1)))

            (local.set $nextIdx (i32.add (local.get $x) (i32.const 3)))
            (if (i32.gt_s (local.get $nextIdx) (local.get $wm)) (then (local.set $nextIdx (local.get $wm))))
            (local.set $addPx (i32.load8_u (i32.add (local.get $p) (local.get $nextIdx))))

            (if (i32.eq (local.get $sp) (i32.const 0))
              (then (local.set $oldPx (local.get $s0)) (local.set $s0 (local.get $addPx)) (local.set $sp (i32.const 1)))
              (else (if (i32.eq (local.get $sp) (i32.const 1))
                (then (local.set $oldPx (local.get $s1)) (local.set $s1 (local.get $addPx)) (local.set $sp (i32.const 2)))
                (else (if (i32.eq (local.get $sp) (i32.const 2))
                  (then (local.set $oldPx (local.get $s2)) (local.set $s2 (local.get $addPx)) (local.set $sp (i32.const 3)))
                  (else (if (i32.eq (local.get $sp) (i32.const 3))
                    (then (local.set $oldPx (local.get $s3)) (local.set $s3 (local.get $addPx)) (local.set $sp (i32.const 4)))
                    (else (local.set $oldPx (local.get $s4)) (local.set $s4 (local.get $addPx)) (local.set $sp (i32.const 0)))
                  ))
                ))
              ))
            )
            (local.set $sum (i32.sub (i32.add (local.get $sum) (local.get $addPx)) (local.get $oldPx)))
            (local.set $x (i32.add (local.get $x) (i32.const 1)))
            (br $xLoop)
          )
        )
        (local.set $y (i32.add (local.get $y) (i32.const 1)))
        (br $hLoop)
      )
    )

    ;; Vertical pass in-place on dst (exact sequential semantics matching original stackBoxBlur)
    (local.set $x (i32.const 0))
    (block $vBreak
      (loop $vLoop
        (br_if $vBreak (i32.ge_s (local.get $x) (local.get $w)))
        (local.set $p (i32.add (local.get $dst) (local.get $x)))
        (local.set $s0 (i32.load8_u (local.get $p)))
        (local.set $s1 (local.get $s0))
        (local.set $s2 (local.get $s0))
        (local.set $s3 (i32.load8_u (i32.add (local.get $p) (i32.mul (select (local.get $hm) (i32.const 1) (i32.gt_s (i32.const 1) (local.get $hm))) (local.get $w)))))
        (local.set $s4 (i32.load8_u (i32.add (local.get $p) (i32.mul (select (local.get $hm) (i32.const 2) (i32.gt_s (i32.const 2) (local.get $hm))) (local.get $w)))))
        (local.set $sum (i32.add (i32.add (i32.mul (local.get $s0) (i32.const 3)) (local.get $s3)) (local.get $s4)))
        (local.set $sp (i32.const 0))

        (local.set $y (i32.const 0))
        (block $yBreak
          (loop $yLoop
            (br_if $yBreak (i32.ge_s (local.get $y) (local.get $h)))
            (i32.store8 (i32.add (local.get $p) (i32.mul (local.get $y) (local.get $w)))
              (i32.shr_u (i32.mul (local.get $sum) (i32.const 205)) (i32.const 10))
            )

            (local.set $nextIdx (i32.add (local.get $y) (i32.const 3)))
            (if (i32.gt_s (local.get $nextIdx) (local.get $hm)) (then (local.set $nextIdx (local.get $hm))))
            (local.set $addPx (i32.load8_u (i32.add (local.get $p) (i32.mul (local.get $nextIdx) (local.get $w)))))

            (if (i32.eq (local.get $sp) (i32.const 0))
              (then (local.set $oldPx (local.get $s0)) (local.set $s0 (local.get $addPx)) (local.set $sp (i32.const 1)))
              (else (if (i32.eq (local.get $sp) (i32.const 1))
                (then (local.set $oldPx (local.get $s1)) (local.set $s1 (local.get $addPx)) (local.set $sp (i32.const 2)))
                (else (if (i32.eq (local.get $sp) (i32.const 2))
                  (then (local.set $oldPx (local.get $s2)) (local.set $s2 (local.get $addPx)) (local.set $sp (i32.const 3)))
                  (else (if (i32.eq (local.get $sp) (i32.const 3))
                    (then (local.set $oldPx (local.get $s3)) (local.set $s3 (local.get $addPx)) (local.set $sp (i32.const 4)))
                    (else (local.set $oldPx (local.get $s4)) (local.set $s4 (local.get $addPx)) (local.set $sp (i32.const 0)))
                  ))
                ))
              ))
            )
            (local.set $sum (i32.sub (i32.add (local.get $sum) (local.get $addPx)) (local.get $oldPx)))
            (local.set $y (i32.add (local.get $y) (i32.const 1)))
            (br $yLoop)
          )
        )
        (local.set $x (i32.add (local.get $x) (i32.const 1)))
        (br $vLoop)
      )
    )

    ;; SIMD comparison: dst[i] = (src[i] - dst[i] <= -7) ? 255 : 0
    ;; Equivalent in unsigned bytes: (dst[i] -_sat_u src[i]) >= 7 ? 255 : 0
    (local.set $len (i32.mul (local.get $w) (local.get $h)))
    (local.set $len16 (i32.and (local.get $len) (i32.const -16)))
    (local.set $vThresh (i8x16.splat (i32.const 7)))
    (local.set $i (i32.const 0))

    (block $simdBreak
      (loop $simdLoop
        (br_if $simdBreak (i32.ge_u (local.get $i) (local.get $len16)))
        (local.set $vSrc (v128.load (i32.add (local.get $src) (local.get $i))))
        (local.set $vBlur (v128.load (i32.add (local.get $dst) (local.get $i))))
        (v128.store (i32.add (local.get $dst) (local.get $i))
          (i8x16.ge_u (i8x16.sub_sat_u (local.get $vBlur) (local.get $vSrc)) (local.get $vThresh))
        )
        (local.set $i (i32.add (local.get $i) (i32.const 16)))
        (br $simdLoop)
      )
    )

    (block $remBreak
      (loop $remLoop
        (br_if $remBreak (i32.ge_u (local.get $i) (local.get $len)))
        (i32.store8 (i32.add (local.get $dst) (local.get $i))
          (select (i32.const 255) (i32.const 0)
            (i32.le_s
              (i32.sub (i32.load8_u (i32.add (local.get $src) (local.get $i))) (i32.load8_u (i32.add (local.get $dst) (local.get $i))))
              (i32.const -7)
            )
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $remLoop)
      )
    )
  )

  ;; Initialize binary border and neighborhood deltas
  (func $initBinaryAndDeltas (param $src i32) (param $w i32) (param $h i32) (param $dst i32)
    (local $i i32) (local $j i32) (local $pSrc i32) (local $pDst i32) (local $stride i32) (local $val i32)
    (local.set $stride (i32.add (local.get $w) (i32.const 2)))

    ;; deltas at 0x00C0 (192)
    (local.set $i (i32.const 0))
    (loop $dLoop
      (local.set $val
        (i32.add
          (i32.load (i32.add (i32.const 128) (i32.shl (local.get $i) (i32.const 2))))
          (i32.mul (i32.load (i32.add (i32.const 160) (i32.shl (local.get $i) (i32.const 2)))) (local.get $stride))
        )
      )
      (i32.store (i32.add (i32.const 192) (i32.shl (local.get $i) (i32.const 2))) (local.get $val))
      (i32.store (i32.add (i32.const 224) (i32.shl (local.get $i) (i32.const 2))) (local.get $val))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br_if $dLoop (i32.lt_u (local.get $i) (i32.const 8)))
    )

    (local.set $pSrc (local.get $src))
    (local.set $pDst (local.get $dst))

    ;; Top zero row
    (memory.fill (local.get $pDst) (i32.const 0) (i32.shl (local.get $stride) (i32.const 2)))
    (local.set $pDst (i32.add (local.get $pDst) (i32.shl (local.get $stride) (i32.const 2))))

    (local.set $i (i32.const 0))
    (block $rowBreak
      (loop $rowLoop
        (br_if $rowBreak (i32.ge_u (local.get $i) (local.get $h)))
        (i32.store (local.get $pDst) (i32.const 0))
        (local.set $pDst (i32.add (local.get $pDst) (i32.const 4)))

        (local.set $j (i32.const 0))
        (block $colBreak
          (loop $colLoop
            (br_if $colBreak (i32.ge_u (local.get $j) (local.get $w)))
            (i32.store (local.get $pDst) (i32.ne (i32.load8_u (local.get $pSrc)) (i32.const 0)))
            (local.set $pSrc (i32.add (local.get $pSrc) (i32.const 1)))
            (local.set $pDst (i32.add (local.get $pDst) (i32.const 4)))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $colLoop)
          )
        )

        (i32.store (local.get $pDst) (i32.const 0))
        (local.set $pDst (i32.add (local.get $pDst) (i32.const 4)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $rowLoop)
      )
    )

    ;; Bottom zero row
    (memory.fill (local.get $pDst) (i32.const 0) (i32.shl (local.get $stride) (i32.const 2)))
  )

  ;; Border following: writes ContourPoints to $contourPtr, returns number of points
  (func $borderFollowing (param $bin i32) (param $pos i32) (param $nbd i32) (param $px i32) (param $py i32) (param $hole i32) (result i32)
    (local $s i32) (local $sEnd i32) (local $pos1 i32) (local $pos3 i32) (local $pos4 i32)
    (local $sStart i32) (local $ptX i32) (local $ptY i32) (local $count i32) (local $outPtr i32)
    (local $maxPts i32)

    (local.set $outPtr (global.get $contourPtr))
    (local.set $maxPts (global.get $maxContourPts))
    (local.set $s (select (i32.const 0) (i32.const 4) (local.get $hole)))
    (local.set $sEnd (local.get $s))

    (block $findFirst
      (loop $firstLoop
        (local.set $s (i32.and (i32.sub (local.get $s) (i32.const 1)) (i32.const 7)))
        (local.set $pos1 (i32.add (local.get $pos) (i32.load (i32.add (i32.const 192) (i32.shl (local.get $s) (i32.const 2))))))
        (br_if $findFirst (i32.ne (i32.load (i32.add (local.get $bin) (i32.shl (local.get $pos1) (i32.const 2)))) (i32.const 0)))
        (br_if $firstLoop (i32.ne (local.get $s) (local.get $sEnd)))
      )
      ;; Isolated pixel
      (i32.store (i32.add (local.get $bin) (i32.shl (local.get $pos) (i32.const 2))) (i32.sub (i32.const 0) (local.get $nbd)))
      (i32.store (local.get $outPtr) (local.get $px))
      (i32.store offset=4 (local.get $outPtr) (local.get $py))
      (return (i32.const 1))
    )

    (local.set $pos3 (local.get $pos))
    (local.set $ptX (local.get $px))
    (local.set $ptY (local.get $py))

    (block $followDone
      (loop $followLoop
        (local.set $sStart (local.get $s))
        (block $scanDone
          (loop $scanLoop
            (local.set $s (i32.and (i32.add (local.get $s) (i32.const 1)) (i32.const 15)))
            (local.set $pos4 (i32.add (local.get $pos3) (i32.load (i32.add (i32.const 192) (i32.shl (local.get $s) (i32.const 2))))))
            (br_if $scanDone (i32.ne (i32.load (i32.add (local.get $bin) (i32.shl (local.get $pos4) (i32.const 2)))) (i32.const 0)))
            (br $scanLoop)
          )
        )
        (local.set $s (i32.and (local.get $s) (i32.const 7)))

        (if (i32.lt_u (i32.sub (local.get $s) (i32.const 1)) (local.get $sStart))
          (then
            (i32.store (i32.add (local.get $bin) (i32.shl (local.get $pos3) (i32.const 2))) (i32.sub (i32.const 0) (local.get $nbd)))
          )
          (else
            (if (i32.eq (i32.load (i32.add (local.get $bin) (i32.shl (local.get $pos3) (i32.const 2)))) (i32.const 1))
              (then
                (i32.store (i32.add (local.get $bin) (i32.shl (local.get $pos3) (i32.const 2))) (local.get $nbd))
              )
            )
          )
        )

        (if (i32.lt_u (local.get $count) (local.get $maxPts))
          (then
            (i32.store (local.get $outPtr) (local.get $ptX))
            (i32.store offset=4 (local.get $outPtr) (local.get $ptY))
            (local.set $outPtr (i32.add (local.get $outPtr) (i32.const 8)))
            (local.set $count (i32.add (local.get $count) (i32.const 1)))
          )
        )

        (local.set $ptX (i32.add (local.get $ptX) (i32.load (i32.add (i32.const 128) (i32.shl (local.get $s) (i32.const 2))))))
        (local.set $ptY (i32.add (local.get $ptY) (i32.load (i32.add (i32.const 160) (i32.shl (local.get $s) (i32.const 2))))))

        (br_if $followDone (i32.and (i32.eq (local.get $pos4) (local.get $pos)) (i32.eq (local.get $pos3) (local.get $pos1))))
        (local.set $pos3 (local.get $pos4))
        (local.set $s (i32.and (i32.add (local.get $s) (i32.const 4)) (i32.const 7)))
        (br $followLoop)
      )
    )
    (local.get $count)
  )

  ;; approxPolyDP: reads contour of length $len from $contourPtr, writes poly to 0x9100 (37120), returns poly length
  (func $approxPolyDP (param $len i32) (param $eps f64) (result i32)
    (local $sliceStart i32) (local $sliceEnd i32) (local $rightStart i32) (local $rightEnd i32)
    (local $polyLen i32) (local $stackTop i32) (local $epsSq f64)
    (local $k i32) (local $i i32) (local $j i32)
    (local $startX i32) (local $startY i32) (local $endX i32) (local $endY i32)
    (local $ptX i32) (local $ptY i32) (local $dx i32) (local $dy i32)
    (local $dist i32) (local $maxDist i32) (local $leEps i32)
    (local $cPtr i32) (local $ptAddr i32)

    (local.set $cPtr (global.get $contourPtr))
    (local.set $epsSq (f64.mul (local.get $eps) (local.get $eps)))

    (local.set $i (i32.const 0))
    (loop $initLoop
      (local.set $maxDist (i32.const 0))
      (local.set $k (i32.rem_u (i32.add (local.get $k) (local.get $rightStart)) (local.get $len)))
      (local.set $ptAddr (i32.add (local.get $cPtr) (i32.shl (local.get $k) (i32.const 3))))
      (local.set $startX (i32.load (local.get $ptAddr)))
      (local.set $startY (i32.load offset=4 (local.get $ptAddr)))
      (local.set $k (i32.add (local.get $k) (i32.const 1)))
      (if (i32.eq (local.get $k) (local.get $len)) (then (local.set $k (i32.const 0))))

      (local.set $j (i32.const 1))
      (block $jBreak
        (loop $jLoop
          (br_if $jBreak (i32.ge_u (local.get $j) (local.get $len)))
          (local.set $ptAddr (i32.add (local.get $cPtr) (i32.shl (local.get $k) (i32.const 3))))
          (local.set $ptX (i32.load (local.get $ptAddr)))
          (local.set $ptY (i32.load offset=4 (local.get $ptAddr)))
          (local.set $k (i32.add (local.get $k) (i32.const 1)))
          (if (i32.eq (local.get $k) (local.get $len)) (then (local.set $k (i32.const 0))))

          (local.set $dx (i32.sub (local.get $ptX) (local.get $startX)))
          (local.set $dy (i32.sub (local.get $ptY) (local.get $startY)))
          (local.set $dist (i32.add (i32.mul (local.get $dx) (local.get $dx)) (i32.mul (local.get $dy) (local.get $dy))))
          (if (i32.gt_s (local.get $dist) (local.get $maxDist))
            (then
              (local.set $maxDist (local.get $dist))
              (local.set $rightStart (local.get $j))
            )
          )
          (local.set $j (i32.add (local.get $j) (i32.const 1)))
          (br $jLoop)
        )
      )
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br_if $initLoop (i32.lt_u (local.get $i) (i32.const 3)))
    )

    (if (f64.le (f64.convert_i32_s (local.get $maxDist)) (local.get $epsSq))
      (then
        (i32.store (i32.const 37120) (local.get $startX))
        (i32.store offset=4 (i32.const 37120) (local.get $startY))
        (local.set $polyLen (i32.const 1))
      )
      (else
        (local.set $sliceStart (local.get $k))
        (local.set $rightStart (i32.add (local.get $rightStart) (local.get $sliceStart)))
        (local.set $sliceEnd (local.get $rightStart))
        (if (i32.ge_s (local.get $rightStart) (local.get $len))
          (then (local.set $rightStart (i32.sub (local.get $rightStart) (local.get $len))))
        )
        (local.set $rightEnd (local.get $sliceStart))
        (if (i32.lt_s (local.get $rightEnd) (local.get $rightStart))
          (then (local.set $rightEnd (i32.add (local.get $rightEnd) (local.get $len))))
        )
        ;; Push right slice then left slice to stack at 0x5100 (20736)
        (i32.store (i32.const 20736) (local.get $rightStart))
        (i32.store offset=4 (i32.const 20736) (local.get $rightEnd))
        (i32.store offset=8 (i32.const 20736) (local.get $sliceStart))
        (i32.store offset=12 (i32.const 20736) (local.get $sliceEnd))
        (local.set $stackTop (i32.const 2))
      )
    )

    (block $stackBreak
      (loop $stackLoop
        (br_if $stackBreak (i32.eqz (local.get $stackTop)))
        (local.set $stackTop (i32.sub (local.get $stackTop) (i32.const 1)))
        (local.set $ptAddr (i32.add (i32.const 20736) (i32.shl (local.get $stackTop) (i32.const 3))))
        (local.set $sliceStart (i32.load (local.get $ptAddr)))
        (local.set $sliceEnd (i32.load offset=4 (local.get $ptAddr)))

        (local.set $ptAddr (i32.add (local.get $cPtr) (i32.shl (i32.rem_u (local.get $sliceEnd) (local.get $len)) (i32.const 3))))
        (local.set $endX (i32.load (local.get $ptAddr)))
        (local.set $endY (i32.load offset=4 (local.get $ptAddr)))

        (local.set $k (i32.rem_u (local.get $sliceStart) (local.get $len)))
        (local.set $ptAddr (i32.add (local.get $cPtr) (i32.shl (local.get $k) (i32.const 3))))
        (local.set $startX (i32.load (local.get $ptAddr)))
        (local.set $startY (i32.load offset=4 (local.get $ptAddr)))
        (local.set $k (i32.add (local.get $k) (i32.const 1)))
        (if (i32.eq (local.get $k) (local.get $len)) (then (local.set $k (i32.const 0))))

        (if (i32.le_s (local.get $sliceEnd) (i32.add (local.get $sliceStart) (i32.const 1)))
          (then (local.set $leEps (i32.const 1)))
          (else
            (local.set $maxDist (i32.const 0))
            (local.set $dx (i32.sub (local.get $endX) (local.get $startX)))
            (local.set $dy (i32.sub (local.get $endY) (local.get $startY)))
            (local.set $i (i32.add (local.get $sliceStart) (i32.const 1)))
            (block $sBreak
              (loop $sLoop
                (br_if $sBreak (i32.ge_s (local.get $i) (local.get $sliceEnd)))
                (local.set $ptAddr (i32.add (local.get $cPtr) (i32.shl (local.get $k) (i32.const 3))))
                (local.set $ptX (i32.load (local.get $ptAddr)))
                (local.set $ptY (i32.load offset=4 (local.get $ptAddr)))
                (local.set $k (i32.add (local.get $k) (i32.const 1)))
                (if (i32.eq (local.get $k) (local.get $len)) (then (local.set $k (i32.const 0))))

                (local.set $dist
                  (i32.sub
                    (i32.mul (i32.sub (local.get $ptY) (local.get $startY)) (local.get $dx))
                    (i32.mul (i32.sub (local.get $ptX) (local.get $startX)) (local.get $dy))
                  )
                )
                (if (i32.lt_s (local.get $dist) (i32.const 0))
                  (then (local.set $dist (i32.sub (i32.const 0) (local.get $dist))))
                )
                (if (i32.gt_s (local.get $dist) (local.get $maxDist))
                  (then
                    (local.set $maxDist (local.get $dist))
                    (local.set $rightStart (local.get $i))
                  )
                )
                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                (br $sLoop)
              )
            )
            (local.set $leEps
              (f64.le
                (f64.mul (f64.convert_i32_s (local.get $maxDist)) (f64.convert_i32_s (local.get $maxDist)))
                (f64.mul
                  (local.get $epsSq)
                  (f64.convert_i32_s (i32.add (i32.mul (local.get $dx) (local.get $dx)) (i32.mul (local.get $dy) (local.get $dy))))
                )
              )
            )
          )
        )

        (if (local.get $leEps)
          (then
            (if (i32.lt_u (local.get $polyLen) (i32.const 512))
              (then
                (local.set $ptAddr (i32.add (i32.const 37120) (i32.shl (local.get $polyLen) (i32.const 3))))
                (i32.store (local.get $ptAddr) (local.get $startX))
                (i32.store offset=4 (local.get $ptAddr) (local.get $startY))
                (local.set $polyLen (i32.add (local.get $polyLen) (i32.const 1)))
              )
            )
          )
          (else
            (local.set $rightEnd (local.get $sliceEnd))
            (local.set $sliceEnd (local.get $rightStart))
            (if (i32.lt_u (local.get $stackTop) (i32.const 2046))
              (then
                (local.set $ptAddr (i32.add (i32.const 20736) (i32.shl (local.get $stackTop) (i32.const 3))))
                (i32.store (local.get $ptAddr) (local.get $rightStart))
                (i32.store offset=4 (local.get $ptAddr) (local.get $rightEnd))
                (i32.store offset=8 (local.get $ptAddr) (local.get $sliceStart))
                (i32.store offset=12 (local.get $ptAddr) (local.get $sliceEnd))
                (local.set $stackTop (i32.add (local.get $stackTop) (i32.const 2)))
              )
            )
          )
        )
        (br $stackLoop)
      )
    )
    (local.get $polyLen)
  )

  ;; Check if 4-point polygon at 0x9100 (37120) is convex and has minEdgeLength >= 10
  (func $isQuadValid (result i32)
    (local $i i32) (local $sign i32) (local $curSign i32) (local $cross i32)
    (local $p1x i32) (local $p1y i32) (local $p2x i32) (local $p2y i32) (local $p3x i32) (local $p3y i32)
    (local $dx i32) (local $dy i32) (local $addr i32)

    (local.set $i (i32.const 0))
    (loop $cLoop
      (local.set $addr (i32.add (i32.const 37120) (i32.shl (local.get $i) (i32.const 3))))
      (local.set $p1x (i32.load (local.get $addr)))
      (local.set $p1y (i32.load offset=4 (local.get $addr)))

      (local.set $addr (i32.add (i32.const 37120) (i32.shl (i32.and (i32.add (local.get $i) (i32.const 1)) (i32.const 3)) (i32.const 3))))
      (local.set $p2x (i32.load (local.get $addr)))
      (local.set $p2y (i32.load offset=4 (local.get $addr)))

      ;; Edge length check: dx*dx + dy*dy < 100 -> false
      (local.set $dx (i32.sub (local.get $p2x) (local.get $p1x)))
      (local.set $dy (i32.sub (local.get $p2y) (local.get $p1y)))
      (if (i32.lt_s (i32.add (i32.mul (local.get $dx) (local.get $dx)) (i32.mul (local.get $dy) (local.get $dy))) (i32.const 100))
        (then (return (i32.const 0)))
      )

      (local.set $addr (i32.add (i32.const 37120) (i32.shl (i32.and (i32.add (local.get $i) (i32.const 2)) (i32.const 3)) (i32.const 3))))
      (local.set $p3x (i32.load (local.get $addr)))
      (local.set $p3y (i32.load offset=4 (local.get $addr)))

      (local.set $cross
        (i32.sub
          (i32.mul (local.get $dx) (i32.sub (local.get $p3y) (local.get $p2y)))
          (i32.mul (local.get $dy) (i32.sub (local.get $p3x) (local.get $p2x)))
        )
      )
      (if (i32.ne (local.get $cross) (i32.const 0))
        (then
          (local.set $curSign (select (i32.const 1) (i32.const -1) (i32.gt_s (local.get $cross) (i32.const 0))))
          (if (i32.eqz (local.get $sign))
            (then (local.set $sign (local.get $curSign)))
            (else (if (i32.ne (local.get $curSign) (local.get $sign)) (then (return (i32.const 0)))))
          )
        )
      )
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br_if $cLoop (i32.lt_u (local.get $i) (i32.const 4)))
    )
    (i32.const 1)
  )

  ;; Perimeter of quad at $quadAddr (32 bytes = 4 x (x:i32, y:i32))
  (func $perimeter (param $quadAddr i32) (result f64)
    (local $i i32) (local $p f64) (local $dx f64) (local $dy f64) (local $a1 i32) (local $a2 i32)
    (loop $pLoop
      (local.set $a1 (i32.add (local.get $quadAddr) (i32.shl (local.get $i) (i32.const 3))))
      (local.set $a2 (i32.add (local.get $quadAddr) (i32.shl (i32.and (i32.add (local.get $i) (i32.const 1)) (i32.const 3)) (i32.const 3))))
      (local.set $dx (f64.convert_i32_s (i32.sub (i32.load (local.get $a2)) (i32.load (local.get $a1)))))
      (local.set $dy (f64.convert_i32_s (i32.sub (i32.load offset=4 (local.get $a2)) (i32.load offset=4 (local.get $a1)))))
      (local.set $p (f64.add (local.get $p) (f64.sqrt (f64.add (f64.mul (local.get $dx) (local.get $dx)) (f64.mul (local.get $dy) (local.get $dy))))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br_if $pLoop (i32.lt_u (local.get $i) (i32.const 4)))
    )
    (local.get $p)
  )

  ;; Warp perspective of 48x48 patch into 0x0100 (256)
  (func $warpPerspective (param $srcGray i32) (param $srcW i32) (param $srcH i32) (param $quadAddr i32)
    (local $x0 f64) (local $y0 f64) (local $x1 f64) (local $y1 f64)
    (local $x2 f64) (local $y2 f64) (local $x3 f64) (local $y3 f64)
    (local $px f64) (local $py f64) (local $dx1 f64) (local $dx2 f64) (local $dy1 f64) (local $dy2 f64) (local $den f64)
    (local $m0 f64) (local $m1 f64) (local $m2 f64) (local $m3 f64) (local $m4 f64) (local $m5 f64) (local $m6 f64) (local $m7 f64) (local $m8 f64)
    (local $r f64) (local $s f64) (local $t f64) (local $u f64) (local $v f64) (local $w f64)
    (local $x f64) (local $y f64) (local $i i32) (local $j i32) (local $pos i32)
    (local $sx1 i32) (local $sx2 i32) (local $sy1 i32) (local $sy2 i32)
    (local $fx1 f64) (local $fx2 f64) (local $fy1 f64) (local $fy2 f64)
    (local $p1 i32) (local $p3 i32) (local $wm i32) (local $hm i32)

    (local.set $x0 (f64.convert_i32_s (i32.load (local.get $quadAddr))))
    (local.set $y0 (f64.convert_i32_s (i32.load offset=4 (local.get $quadAddr))))
    (local.set $x1 (f64.convert_i32_s (i32.load offset=8 (local.get $quadAddr))))
    (local.set $y1 (f64.convert_i32_s (i32.load offset=12 (local.get $quadAddr))))
    (local.set $x2 (f64.convert_i32_s (i32.load offset=16 (local.get $quadAddr))))
    (local.set $y2 (f64.convert_i32_s (i32.load offset=20 (local.get $quadAddr))))
    (local.set $x3 (f64.convert_i32_s (i32.load offset=24 (local.get $quadAddr))))
    (local.set $y3 (f64.convert_i32_s (i32.load offset=28 (local.get $quadAddr))))

    (local.set $px (f64.sub (f64.add (f64.sub (local.get $x0) (local.get $x1)) (local.get $x2)) (local.get $x3)))
    (local.set $py (f64.sub (f64.add (f64.sub (local.get $y0) (local.get $y1)) (local.get $y2)) (local.get $y3)))

    (if (i32.and (f64.eq (local.get $px) (f64.const 0)) (f64.eq (local.get $py) (f64.const 0)))
      (then
        (local.set $m0 (f64.div (f64.sub (local.get $x1) (local.get $x0)) (f64.const 47)))
        (local.set $m1 (f64.div (f64.sub (local.get $x2) (local.get $x1)) (f64.const 47)))
        (local.set $m2 (local.get $x0))
        (local.set $m3 (f64.div (f64.sub (local.get $y1) (local.get $y0)) (f64.const 47)))
        (local.set $m4 (f64.div (f64.sub (local.get $y2) (local.get $y1)) (f64.const 47)))
        (local.set $m5 (local.get $y0))
        (local.set $m6 (f64.const 0))
        (local.set $m7 (f64.const 0))
        (local.set $m8 (f64.const 1))
      )
      (else
        (local.set $dx1 (f64.sub (local.get $x1) (local.get $x2)))
        (local.set $dx2 (f64.sub (local.get $x3) (local.get $x2)))
        (local.set $dy1 (f64.sub (local.get $y1) (local.get $y2)))
        (local.set $dy2 (f64.sub (local.get $y3) (local.get $y2)))
        (local.set $den (f64.sub (f64.mul (local.get $dx1) (local.get $dy2)) (f64.mul (local.get $dx2) (local.get $dy1))))
        (local.set $m6 (f64.div (f64.sub (f64.mul (local.get $px) (local.get $dy2)) (f64.mul (local.get $dx2) (local.get $py))) (local.get $den)))
        (local.set $m7 (f64.div (f64.sub (f64.mul (local.get $dx1) (local.get $py)) (f64.mul (local.get $px) (local.get $dy1))) (local.get $den)))
        (local.set $m8 (f64.const 1))
        (local.set $m0 (f64.div (f64.add (f64.sub (local.get $x1) (local.get $x0)) (f64.mul (local.get $m6) (local.get $x1))) (f64.const 47)))
        (local.set $m1 (f64.div (f64.add (f64.sub (local.get $x3) (local.get $x0)) (f64.mul (local.get $m7) (local.get $x3))) (f64.const 47)))
        (local.set $m2 (local.get $x0))
        (local.set $m3 (f64.div (f64.add (f64.sub (local.get $y1) (local.get $y0)) (f64.mul (local.get $m6) (local.get $y1))) (f64.const 47)))
        (local.set $m4 (f64.div (f64.add (f64.sub (local.get $y3) (local.get $y0)) (f64.mul (local.get $m7) (local.get $y3))) (f64.const 47)))
        (local.set $m5 (local.get $y0))
        (local.set $m6 (f64.div (local.get $m6) (f64.const 47)))
        (local.set $m7 (f64.div (local.get $m7) (f64.const 47)))
      )
    )

    (local.set $pos (i32.const 256))
    (local.set $r (local.get $m8))
    (local.set $s (local.get $m2))
    (local.set $t (local.get $m5))
    (local.set $wm (i32.sub (local.get $srcW) (i32.const 1)))
    (local.set $hm (i32.sub (local.get $srcH) (i32.const 1)))

    (local.set $i (i32.const 0))
    (loop $iLoop
      (local.set $r (f64.add (local.get $r) (local.get $m7)))
      (local.set $s (f64.add (local.get $s) (local.get $m1)))
      (local.set $t (f64.add (local.get $t) (local.get $m4)))
      (local.set $u (local.get $r))
      (local.set $v (local.get $s))
      (local.set $w (local.get $t))

      (local.set $j (i32.const 0))
      (loop $jLoop
        (local.set $u (f64.add (local.get $u) (local.get $m6)))
        (local.set $v (f64.add (local.get $v) (local.get $m0)))
        (local.set $w (f64.add (local.get $w) (local.get $m3)))

        (local.set $x (f64.div (local.get $v) (local.get $u)))
        (local.set $y (f64.div (local.get $w) (local.get $u)))

        (local.set $sx1 (i32.trunc_sat_f64_s (local.get $x)))
        (if (i32.lt_s (local.get $sx1) (i32.const 0)) (then (local.set $sx1 (i32.const 0))))
        (if (i32.gt_s (local.get $sx1) (local.get $wm)) (then (local.set $sx1 (local.get $wm))))
        (local.set $sx2 (select (local.get $sx1) (i32.add (local.get $sx1) (i32.const 1)) (i32.eq (local.get $sx1) (local.get $wm))))
        (local.set $fx1 (f64.sub (local.get $x) (f64.convert_i32_s (local.get $sx1))))
        (local.set $fx2 (f64.sub (f64.const 1) (local.get $fx1)))

        (local.set $sy1 (i32.trunc_sat_f64_s (local.get $y)))
        (if (i32.lt_s (local.get $sy1) (i32.const 0)) (then (local.set $sy1 (i32.const 0))))
        (if (i32.gt_s (local.get $sy1) (local.get $hm)) (then (local.set $sy1 (local.get $hm))))
        (local.set $sy2 (select (local.get $sy1) (i32.add (local.get $sy1) (i32.const 1)) (i32.eq (local.get $sy1) (local.get $hm))))
        (local.set $fy1 (f64.sub (local.get $y) (f64.convert_i32_s (local.get $sy1))))
        (local.set $fy2 (f64.sub (f64.const 1) (local.get $fy1)))

        (local.set $p1 (i32.add (local.get $srcGray) (i32.mul (local.get $sy1) (local.get $srcW))))
        (local.set $p3 (i32.add (local.get $srcGray) (i32.mul (local.get $sy2) (local.get $srcW))))

        (i32.store8 (local.get $pos)
          (i32.trunc_sat_f64_s
            (f64.add
              (f64.mul (local.get $fy2)
                (f64.add
                  (f64.mul (local.get $fx2) (f64.convert_i32_u (i32.load8_u (i32.add (local.get $p1) (local.get $sx1)))))
                  (f64.mul (local.get $fx1) (f64.convert_i32_u (i32.load8_u (i32.add (local.get $p1) (local.get $sx2)))))
                )
              )
              (f64.mul (local.get $fy1)
                (f64.add
                  (f64.mul (local.get $fx2) (f64.convert_i32_u (i32.load8_u (i32.add (local.get $p3) (local.get $sx1)))))
                  (f64.mul (local.get $fx1) (f64.convert_i32_u (i32.load8_u (i32.add (local.get $p3) (local.get $sx2)))))
                )
              )
            )
          )
        )
        (local.set $pos (i32.add (local.get $pos) (i32.const 1)))
        (local.set $j (i32.add (local.get $j) (i32.const 1)))
        (br_if $jLoop (i32.lt_u (local.get $j) (i32.const 48)))
      )
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br_if $iLoop (i32.lt_u (local.get $i) (i32.const 48)))
    )
  )

  ;; Otsu + SIMD Binary Threshold on warpedData (2304 bytes at 0x0100)
  (func $otsuAndThreshold
    (local $i i32) (local $px i32) (local $hAddr i32) (local $sum i32)
    (local $sumB i32) (local $wB i32) (local $wF i32) (local $thresh i32)
    (local $mB f64) (local $mF f64) (local $between f64) (local $maxBetween f64)
    (local $vThresh v128) (local $vZero v128)

    ;; Clear histogram at 0x0A00 (2560)
    (memory.fill (i32.const 2560) (i32.const 0) (i32.const 1024))

    (local.set $i (i32.const 0))
    (loop $histLoop
      (local.set $px (i32.load8_u (i32.add (i32.const 256) (local.get $i))))
      (local.set $hAddr (i32.add (i32.const 2560) (i32.shl (local.get $px) (i32.const 2))))
      (i32.store (local.get $hAddr) (i32.add (i32.load (local.get $hAddr)) (i32.const 1)))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br_if $histLoop (i32.lt_u (local.get $i) (i32.const 2304)))
    )

    (local.set $i (i32.const 0))
    (loop $sumLoop
      (local.set $sum (i32.add (local.get $sum) (i32.mul (i32.load (i32.add (i32.const 2560) (i32.shl (local.get $i) (i32.const 2)))) (local.get $i))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br_if $sumLoop (i32.lt_u (local.get $i) (i32.const 256)))
    )

    (local.set $thresh (i32.const 128))
    (local.set $i (i32.const 0))
    (block $otsuBreak
      (loop $otsuLoop
        (br_if $otsuBreak (i32.ge_u (local.get $i) (i32.const 256)))
        (local.set $px (i32.load (i32.add (i32.const 2560) (i32.shl (local.get $i) (i32.const 2)))))
        (local.set $wB (i32.add (local.get $wB) (local.get $px)))
        (if (i32.ne (local.get $wB) (i32.const 0))
          (then
            (local.set $wF (i32.sub (i32.const 2304) (local.get $wB)))
            (br_if $otsuBreak (i32.eqz (local.get $wF)))
            (local.set $sumB (i32.add (local.get $sumB) (i32.mul (local.get $px) (local.get $i))))
            (local.set $mB (f64.div (f64.convert_i32_s (local.get $sumB)) (f64.convert_i32_s (local.get $wB))))
            (local.set $mF (f64.div (f64.convert_i32_s (i32.sub (local.get $sum) (local.get $sumB))) (f64.convert_i32_s (local.get $wF))))
            (local.set $between
              (f64.mul
                (f64.mul (f64.convert_i32_s (local.get $wB)) (f64.convert_i32_s (local.get $wF)))
                (f64.mul (f64.sub (local.get $mB) (local.get $mF)) (f64.sub (local.get $mB) (local.get $mF)))
              )
            )
            (if (f64.gt (local.get $between) (local.get $maxBetween))
              (then
                (local.set $maxBetween (local.get $between))
                (local.set $thresh (local.get $i))
              )
            )
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $otsuLoop)
      )
    )

    ;; SIMD binary threshold on 2304 bytes (144 x 16-byte vectors)
    (local.set $vThresh (i8x16.splat (local.get $thresh)))
    (local.set $vZero (i8x16.splat (i32.const 0)))
    (local.set $i (i32.const 0))
    (loop $tLoop
      (v128.store (i32.add (i32.const 256) (local.get $i))
        (i8x16.gt_u
          (i8x16.sub_sat_u (v128.load (i32.add (i32.const 256) (local.get $i))) (local.get $vThresh))
          (local.get $vZero)
        )
      )
      (local.set $i (i32.add (local.get $i) (i32.const 16)))
      (br_if $tLoop (i32.lt_u (local.get $i) (i32.const 2304)))
    )
  )

  ;; Count non-zero pixels in 8x8 cell at (cellX*8, cellY*8) in warpedData (0x0100, stride 48)
  (func $countCellNonZero (param $cellX i32) (param $cellY i32) (result i32)
    (local $r i32) (local $c i32) (local $pos i32) (local $nz i32)
    (local.set $pos (i32.add (i32.const 256) (i32.add (i32.mul (i32.shl (local.get $cellY) (i32.const 3)) (i32.const 48)) (i32.shl (local.get $cellX) (i32.const 3)))))
    (loop $rLoop
      (local.set $c (i32.const 0))
      (loop $cLoop
        (local.set $nz (i32.add (local.get $nz) (i32.ne (i32.load8_u (i32.add (local.get $pos) (local.get $c))) (i32.const 0))))
        (local.set $c (i32.add (local.get $c) (i32.const 1)))
        (br_if $cLoop (i32.lt_u (local.get $c) (i32.const 8)))
      )
      (local.set $pos (i32.add (local.get $pos) (i32.const 48)))
      (local.set $r (i32.add (local.get $r) (i32.const 1)))
      (br_if $rLoop (i32.lt_u (local.get $r) (i32.const 8)))
    )
    (local.get $nz)
  )

  ;; Rotate 16-bit 4x4 marker code 90 degrees clockwise
  (func $rotateCode4x4 (param $code i32) (result i32)
    (local $r i32) (local $c i32) (local $srcIdx i32) (local $bit i32) (local $out i32)
    (loop $rLoop
      (local.set $c (i32.const 0))
      (loop $cLoop
        (local.set $srcIdx (i32.add (i32.shl (i32.sub (i32.const 3) (local.get $c)) (i32.const 2)) (local.get $r)))
        (local.set $bit (i32.and (i32.shr_u (local.get $code) (i32.sub (i32.const 15) (local.get $srcIdx))) (i32.const 1)))
        (local.set $out (i32.or (i32.shl (local.get $out) (i32.const 1)) (local.get $bit)))
        (local.set $c (i32.add (local.get $c) (i32.const 1)))
        (br_if $cLoop (i32.lt_u (local.get $c) (i32.const 4)))
      )
      (local.set $r (i32.add (local.get $r) (i32.const 1)))
      (br_if $rLoop (i32.lt_u (local.get $r) (i32.const 4)))
    )
    (local.get $out)
  )

  ;; Main Pipeline Entrypoint: returns number of detected markers
  (func $detect (export "detect") (param $w i32) (param $h i32) (param $maxHamming i32) (result i32)
    (local $len i32) (local $pos i32) (local $nbd i32) (local $i i32) (local $j i32) (local $k i32)
    (local $pix i32) (local $outer i32) (local $hole i32) (local $cLen i32) (local $minSize f64)
    (local $candCount i32) (local $candAddr i32) (local $candAddr2 i32)
    (local $dx1 i32) (local $dy1 i32) (local $dx2 i32) (local $dy2 i32) (local $tmpX i32) (local $tmpY i32)
    (local $distSq i32) (local $borderOk i32) (local $inc i32) (local $code i32)
    (local $rot i32) (local $bestId i32) (local $bestDist i32) (local $bestRot i32)
    (local $id i32) (local $hamming i32) (local $detCount i32) (local $outAddr i32) (local $cornerIdx i32)

    (drop (call $ensureBuffers (local.get $w) (local.get $h)))
    (local.set $len (i32.mul (local.get $w) (local.get $h)))

    ;; 1. Grayscale & Adaptive Threshold
    (call $grayscale (global.get $rgbaPtr) (global.get $greyPtr) (local.get $len))
    (call $adaptiveThreshold (global.get $greyPtr) (global.get $thresPtr) (local.get $w) (local.get $h))

    ;; 2. Find Contours + Filter Candidates fused
    (call $initBinaryAndDeltas (global.get $thresPtr) (local.get $w) (local.get $h) (global.get $binaryPtr))
    (local.set $minSize (f64.mul (f64.convert_i32_s (local.get $w)) (f64.const 0.01)))
    (local.set $pos (i32.add (local.get $w) (i32.const 3)))
    (local.set $nbd (i32.const 1))

    (local.set $i (i32.const 0))
    (block $rBreak
      (loop $rLoop
        (br_if $rBreak (i32.ge_u (local.get $i) (local.get $h)))
        (local.set $j (i32.const 0))
        (block $cBreak
          (loop $cLoop
            (br_if $cBreak (i32.ge_u (local.get $j) (local.get $w)))
            (local.set $pix (i32.load (i32.add (global.get $binaryPtr) (i32.shl (local.get $pos) (i32.const 2)))))
            (if (i32.ne (local.get $pix) (i32.const 0))
              (then
                (local.set $outer (i32.const 0))
                (local.set $hole (i32.const 0))
                (if (i32.and (i32.eq (local.get $pix) (i32.const 1)) (i32.eqz (i32.load (i32.add (global.get $binaryPtr) (i32.shl (i32.sub (local.get $pos) (i32.const 1)) (i32.const 2))))))
                  (then (local.set $outer (i32.const 1)))
                  (else
                    (if (i32.and (i32.ge_s (local.get $pix) (i32.const 1)) (i32.eqz (i32.load (i32.add (global.get $binaryPtr) (i32.shl (i32.add (local.get $pos) (i32.const 1)) (i32.const 2))))))
                      (then (local.set $hole (i32.const 1)))
                    )
                  )
                )
                (if (i32.or (local.get $outer) (local.get $hole))
                  (then
                    (local.set $nbd (i32.add (local.get $nbd) (i32.const 1)))
                    (local.set $cLen (call $borderFollowing (global.get $binaryPtr) (local.get $pos) (local.get $nbd) (local.get $j) (local.get $i) (local.get $hole)))
                    (if (i32.and (f64.ge (f64.convert_i32_s (local.get $cLen)) (local.get $minSize)) (i32.lt_u (local.get $candCount) (i32.const 512)))
                      (then
                        (if (i32.eq (call $approxPolyDP (local.get $cLen) (f64.mul (f64.convert_i32_s (local.get $cLen)) (f64.const 0.05))) (i32.const 4))
                          (then
                            (if (call $isQuadValid)
                              (then
                                (local.set $candAddr (i32.add (i32.const 3840) (i32.shl (local.get $candCount) (i32.const 5))))
                                (memory.copy (local.get $candAddr) (i32.const 37120) (i32.const 32))
                                ;; clockwiseCorners check
                                (local.set $dx1 (i32.sub (i32.load offset=8 (local.get $candAddr)) (i32.load (local.get $candAddr))))
                                (local.set $dy1 (i32.sub (i32.load offset=12 (local.get $candAddr)) (i32.load offset=4 (local.get $candAddr))))
                                (local.set $dx2 (i32.sub (i32.load offset=16 (local.get $candAddr)) (i32.load (local.get $candAddr))))
                                (local.set $dy2 (i32.sub (i32.load offset=20 (local.get $candAddr)) (i32.load offset=4 (local.get $candAddr))))
                                (if (i32.lt_s (i32.sub (i32.mul (local.get $dx1) (local.get $dy2)) (i32.mul (local.get $dy1) (local.get $dx2))) (i32.const 0))
                                  (then
                                    (local.set $tmpX (i32.load offset=8 (local.get $candAddr)))
                                    (local.set $tmpY (i32.load offset=12 (local.get $candAddr)))
                                    (i32.store offset=8 (local.get $candAddr) (i32.load offset=24 (local.get $candAddr)))
                                    (i32.store offset=12 (local.get $candAddr) (i32.load offset=28 (local.get $candAddr)))
                                    (i32.store offset=24 (local.get $candAddr) (local.get $tmpX))
                                    (i32.store offset=28 (local.get $candAddr) (local.get $tmpY))
                                  )
                                )
                                (local.set $candCount (i32.add (local.get $candCount) (i32.const 1)))
                              )
                            )
                          )
                        )
                      )
                    )
                  )
                )
              )
            )
            (local.set $pos (i32.add (local.get $pos) (i32.const 1)))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $cLoop)
          )
        )
        (local.set $pos (i32.add (local.get $pos) (i32.const 2)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $rLoop)
      )
    )

    ;; 3. notTooNear filtering (minDist = 10 -> dist < 400)
    (memory.fill (i32.const 20224) (i32.const 0) (local.get $candCount))
    (local.set $i (i32.const 0))
    (block $ntnBreakI
      (loop $ntnLoopI
        (br_if $ntnBreakI (i32.ge_u (local.get $i) (local.get $candCount)))
        (local.set $candAddr (i32.add (i32.const 3840) (i32.shl (local.get $i) (i32.const 5))))
        (local.set $j (i32.add (local.get $i) (i32.const 1)))
        (block $ntnBreakJ
          (loop $ntnLoopJ
            (br_if $ntnBreakJ (i32.ge_u (local.get $j) (local.get $candCount)))
            (local.set $candAddr2 (i32.add (i32.const 3840) (i32.shl (local.get $j) (i32.const 5))))
            (local.set $distSq (i32.const 0))
            (local.set $k (i32.const 0))
            (loop $kLoop
              (local.set $dx1 (i32.sub (i32.load (i32.add (local.get $candAddr) (i32.shl (local.get $k) (i32.const 3)))) (i32.load (i32.add (local.get $candAddr2) (i32.shl (local.get $k) (i32.const 3))))))
              (local.set $dy1 (i32.sub (i32.load offset=4 (i32.add (local.get $candAddr) (i32.shl (local.get $k) (i32.const 3)))) (i32.load offset=4 (i32.add (local.get $candAddr2) (i32.shl (local.get $k) (i32.const 3))))))
              (local.set $distSq (i32.add (local.get $distSq) (i32.add (i32.mul (local.get $dx1) (local.get $dx1)) (i32.mul (local.get $dy1) (local.get $dy1)))))
              (local.set $k (i32.add (local.get $k) (i32.const 1)))
              (br_if $kLoop (i32.lt_u (local.get $k) (i32.const 4)))
            )
            (if (i32.lt_s (i32.shr_s (local.get $distSq) (i32.const 2)) (i32.const 100))
              (then
                (if (f64.lt (call $perimeter (local.get $candAddr)) (call $perimeter (local.get $candAddr2)))
                  (then (i32.store8 (i32.add (i32.const 20224) (local.get $i)) (i32.const 1)))
                  (else (i32.store8 (i32.add (i32.const 20224) (local.get $j)) (i32.const 1)))
                )
              )
            )
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $ntnLoopJ)
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $ntnLoopI)
      )
    )

    ;; 4. Warp, Otsu threshold, verify border, decode 4x4 bits & match DICT_4X4_50
    (local.set $i (i32.const 0))
    (block $candBreak
      (loop $candLoop
        (br_if $candBreak (i32.ge_u (local.get $i) (local.get $candCount)))
        (if (i32.eqz (i32.load8_u (i32.add (i32.const 20224) (local.get $i))))
          (then
            (local.set $candAddr (i32.add (i32.const 3840) (i32.shl (local.get $i) (i32.const 5))))
            (call $warpPerspective (global.get $greyPtr) (local.get $w) (local.get $h) (local.get $candAddr))
            (call $otsuAndThreshold)

            ;; Check 1-cell black border (nz <= 32)
            (local.set $borderOk (i32.const 1))
            (local.set $j (i32.const 0))
            (block $bBreak
              (loop $bLoop
                (br_if $bBreak (i32.ge_u (local.get $j) (i32.const 6)))
                (local.set $inc (select (i32.const 1) (i32.const 5) (i32.or (i32.eqz (local.get $j)) (i32.eq (local.get $j) (i32.const 5)))))
                (local.set $k (i32.const 0))
                (loop $bkLoop
                  (if (i32.gt_u (call $countCellNonZero (local.get $k) (local.get $j)) (i32.const 32))
                    (then (local.set $borderOk (i32.const 0)) (br $bBreak))
                  )
                  (local.set $k (i32.add (local.get $k) (local.get $inc)))
                  (br_if $bkLoop (i32.lt_u (local.get $k) (i32.const 6)))
                )
                (local.set $j (i32.add (local.get $j) (i32.const 1)))
                (br $bLoop)
              )
            )

            (if (local.get $borderOk)
              (then
                ;; Extract 16-bit code from inner 4x4 grid
                (local.set $code (i32.const 0))
                (local.set $j (i32.const 1))
                (loop $bitR
                  (local.set $k (i32.const 1))
                  (loop $bitC
                    (local.set $code
                      (i32.or
                        (i32.shl (local.get $code) (i32.const 1))
                        (i32.gt_u (call $countCellNonZero (local.get $k) (local.get $j)) (i32.const 32))
                      )
                    )
                    (local.set $k (i32.add (local.get $k) (i32.const 1)))
                    (br_if $bitC (i32.le_u (local.get $k) (i32.const 4)))
                  )
                  (local.set $j (i32.add (local.get $j) (i32.const 1)))
                  (br_if $bitR (i32.le_u (local.get $j) (i32.const 4)))
                )

                ;; Match across 4 rotations using hardware i32.popcnt
                (local.set $bestId (i32.const -1))
                (local.set $bestDist (i32.const 99))
                (local.set $rot (i32.const 0))
                (block $rotBreak
                  (loop $rotLoop
                    (br_if $rotBreak (i32.ge_u (local.get $rot) (i32.const 4)))
                    (local.set $id (i32.const 0))
                    (loop $dictLoop
                      (local.set $hamming
                        (i32.popcnt
                          (i32.xor (local.get $code) (i32.load16_u (i32.shl (local.get $id) (i32.const 1))))
                        )
                      )
                      (if (i32.and (i32.le_s (local.get $hamming) (local.get $maxHamming)) (i32.lt_s (local.get $hamming) (local.get $bestDist)))
                        (then
                          (local.set $bestId (local.get $id))
                          (local.set $bestDist (local.get $hamming))
                          (local.set $bestRot (local.get $rot))
                        )
                      )
                      (local.set $id (i32.add (local.get $id) (i32.const 1)))
                      (br_if $dictLoop (i32.lt_u (local.get $id) (i32.const 50)))
                    )
                    (br_if $rotBreak (i32.eqz (local.get $bestDist)))
                    (local.set $code (call $rotateCode4x4 (local.get $code)))
                    (local.set $rot (i32.add (local.get $rot) (i32.const 1)))
                    (br $rotLoop)
                  )
                )

                (if (i32.and (i32.ge_s (local.get $bestId) (i32.const 0)) (i32.lt_u (local.get $detCount) (i32.const 256)))
                  (then
                    (local.set $outAddr (i32.add (i32.const 41216) (i32.mul (local.get $detCount) (i32.const 40))))
                    (i32.store (local.get $outAddr) (local.get $bestId))
                    (i32.store offset=4 (local.get $outAddr) (local.get $bestDist))
                    (local.set $k (i32.const 0))
                    (loop $storeCorners
                      (local.set $cornerIdx (i32.and (i32.add (local.get $k) (local.get $bestRot)) (i32.const 3)))
                      (f32.store (i32.add (local.get $outAddr) (i32.add (i32.const 8) (i32.shl (local.get $k) (i32.const 3))))
                        (f32.convert_i32_s (i32.load (i32.add (local.get $candAddr) (i32.shl (local.get $cornerIdx) (i32.const 3)))))
                      )
                      (f32.store (i32.add (local.get $outAddr) (i32.add (i32.const 12) (i32.shl (local.get $k) (i32.const 3))))
                        (f32.convert_i32_s (i32.load offset=4 (i32.add (local.get $candAddr) (i32.shl (local.get $cornerIdx) (i32.const 3)))))
                      )
                      (local.set $k (i32.add (local.get $k) (i32.const 1)))
                      (br_if $storeCorners (i32.lt_u (local.get $k) (i32.const 4)))
                    )
                    (local.set $detCount (i32.add (local.get $detCount) (i32.const 1)))
                  )
                )
              )
            )
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $candLoop)
      )
    )

    (local.get $detCount)
  )
)