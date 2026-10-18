(module
  ;; ============================================================================
  ;; Memory & Static Data Layout
  ;; ============================================================================
  ;; 512 pages = 32 MB linear memory
  ;; 0x000000 - 0x00FFFF : Static Lookup Tables & Scratch Registers
  ;; 0x010000 - 0x01FFFF : Dictionary Bitmasks
  ;; 0x020000 - 0x1FFFFF : Dynamic Heap & Detector Buffers (Managed by TS/Bump Allocator)
  (memory (export "memory") 512)

  (global $heap_ptr (mut i32) (i32.const 131072))

  ;; CV.stackBoxBlurMult (16 x i32 at 0x0100 = 256)
  (data (i32.const 256)
    "\01\00\00\00\ab\00\00\00\cd\00\00\00\25\01\00\00"
    "\39\00\00\00\75\01\00\00\4f\00\00\00\89\00\00\00"
    "\f1\00\00\00\1b\00\00\00\87\01\00\00\65\01\00\00"
    "\29\00\00\00\13\00\00\00\1b\01\00\00\09\01\00\00"
  )

  ;; CV.stackBoxBlurShift (16 x i32 at 0x0140 = 320)
  (data (i32.const 320)
    "\00\00\00\00\09\00\00\00\0a\00\00\00\0b\00\00\00"
    "\09\00\00\00\0c\00\00\00\0a\00\00\00\0b\00\00\00"
    "\0c\00\00\00\09\00\00\00\0d\00\00\00\0d\00\00\00"
    "\0a\00\00\00\09\00\00\00\0d\00\00\00\0d\00\00\00"
  )

  ;; CV.neighborhood (8 x 2 x i32 at 0x0180 = 384)
  ;; [1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1], [0,1], [1,1]
  (data (i32.const 384)
    "\01\00\00\00\00\00\00\00"
    "\01\00\00\00\ff\ff\ff\ff"
    "\00\00\00\00\ff\ff\ff\ff"
    "\ff\ff\ff\ff\ff\ff\ff\ff"
    "\ff\ff\ff\ff\00\00\00\00"
    "\ff\ff\ff\ff\01\00\00\00"
    "\00\00\00\00\01\00\00\00"
    "\01\00\00\00\01\00\00\00"
  )

  ;; Memory Allocator Helpers
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
  ;; Section 1: Computer Vision (CV) with SIMD Acceleration
  ;; ============================================================================

  ;; CV.grayscale: SIMD RGBA (4 bytes/pixel) -> Grayscale (1 byte/pixel)
  ;; dst[i] = (R*77 + G*150 + B*29 + 128) >> 8
  (func $cv_grayscale (export "cv_grayscale") (param $src i32) (param $dst i32) (param $width i32) (param $height i32)
    (local $len i32) (local $len4 i32) (local $i i32)
    (local $v v128) (local $r v128) (local $g v128) (local $b v128) (local $y v128)
    (local $maskFF v128) (local $c77 v128) (local $c150 v128) (local $c29 v128) (local $c128 v128)

    (local.set $len (i32.mul (local.get $width) (local.get $height)))
    (local.set $len4 (i32.and (local.get $len) (i32.const -4)))
    (local.set $maskFF (i32x4.splat (i32.const 255)))
    (local.set $c77 (i32x4.splat (i32.const 77)))
    (local.set $c150 (i32x4.splat (i32.const 150)))
    (local.set $c29 (i32x4.splat (i32.const 29)))
    (local.set $c128 (i32x4.splat (i32.const 128)))
    (local.set $i (i32.const 0))

    ;; SIMD loop (4 pixels / 16 bytes per iteration)
    (block $simd_done
      (loop $simd_loop
        (br_if $simd_done (i32.ge_u (local.get $i) (local.get $len4)))
        (local.set $v (v128.load (local.get $src)))
        (local.set $r (v128.and (local.get $v) (local.get $maskFF)))
        (local.set $g (v128.and (i32x4.shr_u (local.get $v) (i32.const 8)) (local.get $maskFF)))
        (local.set $b (v128.and (i32x4.shr_u (local.get $v) (i32.const 16)) (local.get $maskFF)))

        (local.set $y
          (i32x4.shr_u
            (i32x4.add
              (i32x4.add (i32x4.mul (local.get $r) (local.get $c77)) (i32x4.mul (local.get $g) (local.get $c150)))
              (i32x4.add (i32x4.mul (local.get $b) (local.get $c29)) (local.get $c128))
            )
            (i32.const 8)
          )
        )
        (local.set $y (i8x16.shuffle 0 4 8 12 0 0 0 0 0 0 0 0 0 0 0 0 (local.get $y) (local.get $y)))
        (i32.store (local.get $dst) (i32x4.extract_lane 0 (local.get $y)))

        (local.set $src (i32.add (local.get $src) (i32.const 16)))
        (local.set $dst (i32.add (local.get $dst) (i32.const 4)))
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $simd_loop)
      )
    )

    ;; Scalar tail loop
    (block $tail_done
      (loop $tail_loop
        (br_if $tail_done (i32.ge_u (local.get $i) (local.get $len)))
        (i32.store8
          (local.get $dst)
          (i32.shr_u
            (i32.add
              (i32.add
                (i32.mul (i32.load8_u (local.get $src)) (i32.const 77))
                (i32.mul (i32.load8_u offset=1 (local.get $src)) (i32.const 150))
              )
              (i32.add
                (i32.mul (i32.load8_u offset=2 (local.get $src)) (i32.const 29))
                (i32.const 128)
              )
            )
            (i32.const 8)
          )
        )
        (local.set $src (i32.add (local.get $src) (i32.const 4)))
        (local.set $dst (i32.add (local.get $dst) (i32.const 1)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $tail_loop)
      )
    )
  )

  ;; CV.threshold: SIMD 16-pixel parallel thresholding
  (func $cv_threshold (export "cv_threshold") (param $src i32) (param $dst i32) (param $len i32) (param $thres i32)
    (local $i i32)
    (local $simd_end i32)
    (local $v_thres v128)
    (local $v_all_ones v128)
    (local $v_pix v128)
    (local $v_le v128)

    (local.set $v_thres (i8x16.splat (local.get $thres)))
    (local.set $v_all_ones (i8x16.splat (i32.const -1)))
    (local.set $simd_end (i32.and (local.get $len) (i32.const -16)))
    (local.set $i (i32.const 0))

    ;; SIMD loop (16 bytes per iteration)
    (block $simd_done
      (loop $simd_loop
        (br_if $simd_done (i32.ge_u (local.get $i) (local.get $simd_end)))
        (local.set $v_pix (v128.load (i32.add (local.get $src) (local.get $i))))
        ;; i <= threshold ? 0 : 255  ==>  andnot(all_ones, i <= threshold)
        (local.set $v_le (i8x16.le_u (local.get $v_pix) (local.get $v_thres)))
        (v128.store (i32.add (local.get $dst) (local.get $i)) (v128.andnot (local.get $v_all_ones) (local.get $v_le)))
        (local.set $i (i32.add (local.get $i) (i32.const 16)))
        (br $simd_loop)
      )
    )

    ;; Scalar tail loop
    (block $tail_done
      (loop $tail_loop
        (br_if $tail_done (i32.ge_u (local.get $i) (local.get $len)))
        (i32.store8
          (i32.add (local.get $dst) (local.get $i))
          (select
            (i32.const 0)
            (i32.const 255)
            (i32.le_u (i32.load8_u (i32.add (local.get $src) (local.get $i))) (local.get $thres))
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $tail_loop)
      )
    )
  )

  ;; CV.otsu: Histogram & between-class variance maximization
  (func $cv_otsu (export "cv_otsu") (param $src i32) (param $len i32) (result i32)
    (local $hist i32)
    (local $i i32)
    (local $val i32)
    (local $threshold i32)
    (local $wB i32)
    (local $wF i32)
    (local $count i32)
    (local $sum f64)
    (local $sumB f64)
    (local $max f64)
    (local $mu f64)
    (local $between f64)

    ;; Use scratch 0x0400 (1024 bytes) for 256 x u32 histogram
    (local.set $hist (i32.const 1024))
    (memory.fill (local.get $hist) (i32.const 0) (i32.const 1024))

    (local.set $i (i32.const 0))
    (block $h_done
      (loop $h_loop
        (br_if $h_done (i32.ge_u (local.get $i) (local.get $len)))
        (local.set $val (i32.shl (i32.load8_u (i32.add (local.get $src) (local.get $i))) (i32.const 2)))
        (i32.store (i32.add (local.get $hist) (local.get $val))
          (i32.add (i32.load (i32.add (local.get $hist) (local.get $val))) (i32.const 1))
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $h_loop)
      )
    )

    (local.set $sum (f64.const 0.0))
    (local.set $i (i32.const 0))
    (block $s_done
      (loop $s_loop
        (br_if $s_done (i32.ge_u (local.get $i) (i32.const 256)))
        (local.set $count (i32.load (i32.add (local.get $hist) (i32.shl (local.get $i) (i32.const 2)))))
        (local.set $sum (f64.add (local.get $sum) (f64.convert_i32_u (i32.mul (local.get $count) (local.get $i)))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $s_loop)
      )
    )

    (local.set $sumB (f64.const 0.0))
    (local.set $wB (i32.const 0))
    (local.set $max (f64.const 0.0))
    (local.set $threshold (i32.const 0))
    (local.set $i (i32.const 0))

    (block $o_done
      (loop $o_loop
        (br_if $o_done (i32.ge_u (local.get $i) (i32.const 256)))
        (local.set $count (i32.load (i32.add (local.get $hist) (i32.shl (local.get $i) (i32.const 2)))))
        (local.set $wB (i32.add (local.get $wB) (local.get $count)))
        (if (i32.ne (local.get $wB) (i32.const 0))
          (then
            (local.set $wF (i32.sub (local.get $len) (local.get $wB)))
            (br_if $o_done (i32.eq (local.get $wF) (i32.const 0)))
            (local.set $sumB (f64.add (local.get $sumB) (f64.convert_i32_u (i32.mul (local.get $count) (local.get $i)))))
            (local.set $mu
              (f64.sub
                (f64.div (local.get $sumB) (f64.convert_i32_u (local.get $wB)))
                (f64.div (f64.sub (local.get $sum) (local.get $sumB)) (f64.convert_i32_u (local.get $wF)))
              )
            )
            (local.set $between
              (f64.mul
                (f64.mul (f64.convert_i32_u (local.get $wB)) (f64.convert_i32_u (local.get $wF)))
                (f64.mul (local.get $mu) (local.get $mu))
              )
            )
            (if (f64.gt (local.get $between) (local.get $max))
              (then
                (local.set $max (local.get $between))
                (local.set $threshold (local.get $i))
              )
            )
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $o_loop)
      )
    )
    (local.get $threshold)
  )

  ;; CV.stackBoxBlur: Circular ring buffer box blur (horizontal + vertical passes)
  (func $cv_stack_box_blur (export "cv_stack_box_blur")
    (param $src i32) (param $dst i32) (param $width i32) (param $height i32) (param $kernel_size i32)
    (local $height_m1 i32) (local $width_m1 i32) (local $size i32) (local $radius i32)
    (local $mult i32) (local $shift i32) (local $ring i32) (local $head i32)
    (local $pos i32) (local $y i32) (local $x i32) (local $i i32)
    (local $start i32) (local $color i32) (local $sum i32) (local $p i32)

    (local.set $height_m1 (i32.sub (local.get $height) (i32.const 1)))
    (local.set $width_m1 (i32.sub (local.get $width) (i32.const 1)))
    (local.set $size (i32.add (i32.shl (local.get $kernel_size) (i32.const 1)) (i32.const 1)))
    (local.set $radius (i32.add (local.get $kernel_size) (i32.const 1)))
    (local.set $mult (i32.load (i32.add (i32.const 256) (i32.shl (local.get $kernel_size) (i32.const 2)))))
    (local.set $shift (i32.load (i32.add (i32.const 320) (i32.shl (local.get $kernel_size) (i32.const 2)))))
    ;; Ring buffer at scratch 0x0800 (2048)
    (local.set $ring (i32.const 2048))

    (local.set $pos (i32.const 0))
    (local.set $y (i32.const 0))
    (block $y_done
      (loop $y_loop
        (br_if $y_done (i32.ge_s (local.get $y) (local.get $height)))
        (local.set $start (local.get $pos))
        (local.set $color (i32.load8_u (i32.add (local.get $src) (local.get $pos))))
        (local.set $sum (i32.mul (local.get $radius) (local.get $color)))

        (local.set $head (i32.const 0))
        (local.set $i (i32.const 0))
        (block $r1_done
          (loop $r1_loop
            (br_if $r1_done (i32.ge_s (local.get $i) (local.get $radius)))
            (i32.store (i32.add (local.get $ring) (i32.shl (local.get $head) (i32.const 2))) (local.get $color))
            (local.set $head (i32.add (local.get $head) (i32.const 1)))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $r1_loop)
          )
        )

        (local.set $i (i32.const 1))
        (block $r2_done
          (loop $r2_loop
            (br_if $r2_done (i32.ge_s (local.get $i) (local.get $radius)))
            (local.set $color (i32.load8_u (i32.add (local.get $src) (i32.add (local.get $pos) (local.get $i)))))
            (i32.store (i32.add (local.get $ring) (i32.shl (local.get $head) (i32.const 2))) (local.get $color))
            (local.set $sum (i32.add (local.get $sum) (local.get $color)))
            (local.set $head (i32.add (local.get $head) (i32.const 1)))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $r2_loop)
          )
        )

        (local.set $head (i32.const 0))
        (local.set $x (i32.const 0))
        (block $x_done
          (loop $x_loop
            (br_if $x_done (i32.ge_s (local.get $x) (local.get $width)))
            (i32.store8
              (i32.add (local.get $dst) (local.get $pos))
              (i32.shr_u (i32.mul (local.get $sum) (local.get $mult)) (local.get $shift))
            )
            (local.set $pos (i32.add (local.get $pos) (i32.const 1)))
            (local.set $p (i32.add (local.get $x) (local.get $radius)))
            (local.set $p
              (i32.add
                (local.get $start)
                (select (local.get $p) (local.get $width_m1) (i32.lt_s (local.get $p) (local.get $width_m1)))
              )
            )
            (local.set $color (i32.load8_u (i32.add (local.get $src) (local.get $p))))
            (local.set $sum
              (i32.sub
                (local.get $sum)
                (i32.sub (i32.load (i32.add (local.get $ring) (i32.shl (local.get $head) (i32.const 2)))) (local.get $color))
              )
            )
            (i32.store (i32.add (local.get $ring) (i32.shl (local.get $head) (i32.const 2))) (local.get $color))
            (local.set $head (i32.add (local.get $head) (i32.const 1)))
            (if (i32.eq (local.get $head) (local.get $size)) (then (local.set $head (i32.const 0))))
            (local.set $x (i32.add (local.get $x) (i32.const 1)))
            (br $x_loop)
          )
        )
        (local.set $y (i32.add (local.get $y) (i32.const 1)))
        (br $y_loop)
      )
    )

    ;; Vertical pass
    (local.set $x (i32.const 0))
    (block $vx_done
      (loop $vx_loop
        (br_if $vx_done (i32.ge_s (local.get $x) (local.get $width)))
        (local.set $pos (local.get $x))
        (local.set $start (i32.add (local.get $pos) (local.get $width)))
        (local.set $color (i32.load8_u (i32.add (local.get $dst) (local.get $pos))))
        (local.set $sum (i32.mul (local.get $radius) (local.get $color)))

        (local.set $head (i32.const 0))
        (local.set $i (i32.const 0))
        (block $vr1_done
          (loop $vr1_loop
            (br_if $vr1_done (i32.ge_s (local.get $i) (local.get $radius)))
            (i32.store (i32.add (local.get $ring) (i32.shl (local.get $head) (i32.const 2))) (local.get $color))
            (local.set $head (i32.add (local.get $head) (i32.const 1)))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $vr1_loop)
          )
        )

        (local.set $i (i32.const 1))
        (block $vr2_done
          (loop $vr2_loop
            (br_if $vr2_done (i32.ge_s (local.get $i) (local.get $radius)))
            (local.set $color (i32.load8_u (i32.add (local.get $dst) (local.get $start))))
            (i32.store (i32.add (local.get $ring) (i32.shl (local.get $head) (i32.const 2))) (local.get $color))
            (local.set $sum (i32.add (local.get $sum) (local.get $color)))
            (local.set $head (i32.add (local.get $head) (i32.const 1)))
            (local.set $start (i32.add (local.get $start) (local.get $width)))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $vr2_loop)
          )
        )

        (local.set $head (i32.const 0))
        (local.set $y (i32.const 0))
        (block $vy_done
          (loop $vy_loop
            (br_if $vy_done (i32.ge_s (local.get $y) (local.get $height)))
            (i32.store8
              (i32.add (local.get $dst) (local.get $pos))
              (i32.shr_u (i32.mul (local.get $sum) (local.get $mult)) (local.get $shift))
            )
            (local.set $p (i32.add (local.get $y) (local.get $radius)))
            (local.set $p
              (i32.add
                (local.get $x)
                (i32.mul
                  (select (local.get $p) (local.get $height_m1) (i32.lt_s (local.get $p) (local.get $height_m1)))
                  (local.get $width)
                )
              )
            )
            (local.set $color (i32.load8_u (i32.add (local.get $dst) (local.get $p))))
            (local.set $sum
              (i32.sub
                (local.get $sum)
                (i32.sub (i32.load (i32.add (local.get $ring) (i32.shl (local.get $head) (i32.const 2)))) (local.get $color))
              )
            )
            (i32.store (i32.add (local.get $ring) (i32.shl (local.get $head) (i32.const 2))) (local.get $color))
            (local.set $head (i32.add (local.get $head) (i32.const 1)))
            (if (i32.eq (local.get $head) (local.get $size)) (then (local.set $head (i32.const 0))))
            (local.set $pos (i32.add (local.get $pos) (local.get $width)))
            (local.set $y (i32.add (local.get $y) (i32.const 1)))
            (br $vy_loop)
          )
        )
        (local.set $x (i32.add (local.get $x) (i32.const 1)))
        (br $vx_loop)
      )
    )
  )

  ;; CV.adaptiveThreshold: BoxBlur + SIMD 16-pixel parallel comparison
  (func $cv_adaptive_threshold (export "cv_adaptive_threshold")
    (param $src i32) (param $dst i32) (param $width i32) (param $height i32) (param $kernel_size i32) (param $threshold i32)
    (local $len i32) (local $i i32) (local $simd_end i32)
    (local $v_src v128) (local $v_dst v128)
    (local $v_diff_lo v128) (local $v_diff_hi v128)
    (local $v_neg_thres v128)

    (call $cv_stack_box_blur (local.get $src) (local.get $dst) (local.get $width) (local.get $height) (local.get $kernel_size))

    (local.set $len (i32.mul (local.get $width) (local.get $height)))
    (local.set $simd_end (i32.and (local.get $len) (i32.const -16)))
    (local.set $v_neg_thres (i16x8.splat (i32.sub (i32.const 0) (local.get $threshold))))
    (local.set $i (i32.const 0))

    ;; SIMD loop: 16 pixels at a time via i16x8 widening
    (block $simd_done
      (loop $simd_loop
        (br_if $simd_done (i32.ge_u (local.get $i) (local.get $simd_end)))
        (local.set $v_src (v128.load (i32.add (local.get $src) (local.get $i))))
        (local.set $v_dst (v128.load (i32.add (local.get $dst) (local.get $i))))

        (local.set $v_diff_lo
          (i16x8.le_s
            (i16x8.sub (i16x8.extend_low_i8x16_u (local.get $v_src)) (i16x8.extend_low_i8x16_u (local.get $v_dst)))
            (local.get $v_neg_thres)
          )
        )
        (local.set $v_diff_hi
          (i16x8.le_s
            (i16x8.sub (i16x8.extend_high_i8x16_u (local.get $v_src)) (i16x8.extend_high_i8x16_u (local.get $v_dst)))
            (local.get $v_neg_thres)
          )
        )
        ;; Narrow i16 masks (0xFFFF -> -1 = 0xFF, 0x0000 -> 0)
        (v128.store
          (i32.add (local.get $dst) (local.get $i))
          (i8x16.narrow_i16x8_s (local.get $v_diff_lo) (local.get $v_diff_hi))
        )
        (local.set $i (i32.add (local.get $i) (i32.const 16)))
        (br $simd_loop)
      )
    )

    ;; Scalar tail
    (block $tail_done
      (loop $tail_loop
        (br_if $tail_done (i32.ge_u (local.get $i) (local.get $len)))
        (i32.store8
          (i32.add (local.get $dst) (local.get $i))
          (select
            (i32.const 255)
            (i32.const 0)
            (i32.le_s
              (i32.sub
                (i32.load8_u (i32.add (local.get $src) (local.get $i)))
                (i32.load8_u (i32.add (local.get $dst) (local.get $i)))
              )
              (i32.sub (i32.const 0) (local.get $threshold))
            )
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $tail_loop)
      )
    )
  )

  ;; CV.binaryBorder: Pads image with 0-border and converts non-zero to 1 (i32 array)
  ;; Uses SIMD to convert 4 pixels at a time into 32-bit integers
  (func $cv_binary_border (export "cv_binary_border")
    (param $src i32) (param $dst i32) (param $width i32) (param $height i32)
    (local $pos_src i32) (local $pos_dst i32) (local $i i32) (local $j i32) (local $simd_w i32)
    (local $v_zero v128) (local $v_one v128) (local $v_bytes v128) (local $v_mask v128)

    (local.set $pos_src (i32.const 0))
    (local.set $pos_dst (i32.const 0))
    (local.set $v_zero (i8x16.splat (i32.const 0)))
    (local.set $v_one (i8x16.splat (i32.const 1)))
    (local.set $simd_w (i32.and (local.get $width) (i32.const -16)))

    ;; Top border (width + 2 zeros)
    (memory.fill (local.get $dst) (i32.const 0) (i32.shl (i32.add (local.get $width) (i32.const 2)) (i32.const 2)))
    (local.set $pos_dst (i32.add (local.get $width) (i32.const 2)))

    (local.set $i (i32.const 0))
    (block $row_done
      (loop $row_loop
        (br_if $row_done (i32.ge_s (local.get $i) (local.get $height)))
        ;; Left border
        (i32.store (i32.add (local.get $dst) (i32.shl (local.get $pos_dst) (i32.const 2))) (i32.const 0))
        (local.set $pos_dst (i32.add (local.get $pos_dst) (i32.const 1)))

        (local.set $j (i32.const 0))
        ;; SIMD 16-pixel conversion
        (block $col_simd_done
          (loop $col_simd_loop
            (br_if $col_simd_done (i32.ge_s (local.get $j) (local.get $simd_w)))
            (local.set $v_bytes (v128.load (i32.add (local.get $src) (local.get $pos_src))))
            ;; (bytes != 0) & 1
            (local.set $v_mask (v128.andnot (local.get $v_one) (i8x16.eq (local.get $v_bytes) (local.get $v_zero))))
            ;; Store as 16 x i32
            (v128.store (i32.add (local.get $dst) (i32.shl (local.get $pos_dst) (i32.const 2)))
              (i32x4.extend_low_i16x8_u (i16x8.extend_low_i8x16_u (local.get $v_mask))))
            (v128.store offset=16 (i32.add (local.get $dst) (i32.shl (local.get $pos_dst) (i32.const 2)))
              (i32x4.extend_high_i16x8_u (i16x8.extend_low_i8x16_u (local.get $v_mask))))
            (v128.store offset=32 (i32.add (local.get $dst) (i32.shl (local.get $pos_dst) (i32.const 2)))
              (i32x4.extend_low_i16x8_u (i16x8.extend_high_i8x16_u (local.get $v_mask))))
            (v128.store offset=48 (i32.add (local.get $dst) (i32.shl (local.get $pos_dst) (i32.const 2)))
              (i32x4.extend_high_i16x8_u (i16x8.extend_high_i8x16_u (local.get $v_mask))))

            (local.set $pos_src (i32.add (local.get $pos_src) (i32.const 16)))
            (local.set $pos_dst (i32.add (local.get $pos_dst) (i32.const 16)))
            (local.set $j (i32.add (local.get $j) (i32.const 16)))
            (br $col_simd_loop)
          )
        )

        ;; Scalar tail
        (block $col_tail_done
          (loop $col_tail_loop
            (br_if $col_tail_done (i32.ge_s (local.get $j) (local.get $width)))
            (i32.store
              (i32.add (local.get $dst) (i32.shl (local.get $pos_dst) (i32.const 2)))
              (select (i32.const 0) (i32.const 1) (i32.eq (i32.load8_u (i32.add (local.get $src) (local.get $pos_src))) (i32.const 0)))
            )
            (local.set $pos_src (i32.add (local.get $pos_src) (i32.const 1)))
            (local.set $pos_dst (i32.add (local.get $pos_dst) (i32.const 1)))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $col_tail_loop)
          )
        )

        ;; Right border
        (i32.store (i32.add (local.get $dst) (i32.shl (local.get $pos_dst) (i32.const 2))) (i32.const 0))
        (local.set $pos_dst (i32.add (local.get $pos_dst) (i32.const 1)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $row_loop)
      )
    )

    ;; Bottom border (width + 2 zeros)
    (memory.fill (i32.add (local.get $dst) (i32.shl (local.get $pos_dst) (i32.const 2))) (i32.const 0) (i32.shl (i32.add (local.get $width) (i32.const 2)) (i32.const 2)))
  )

  ;; CV.neighborhoodDeltas: Writes 16 x i32 deltas to $deltas_ptr
  (func $cv_neighborhood_deltas (export "cv_neighborhood_deltas") (param $width i32) (param $deltas_ptr i32)
    (local $i i32) (local $d i32)
    (local.set $i (i32.const 0))
    (block $done
      (loop $loop
        (br_if $done (i32.ge_s (local.get $i) (i32.const 8)))
        (local.set $d
          (i32.add
            (i32.load (i32.add (i32.const 384) (i32.shl (local.get $i) (i32.const 3))))
            (i32.mul
              (i32.load offset=4 (i32.add (i32.const 384) (i32.shl (local.get $i) (i32.const 3))))
              (local.get $width)
            )
          )
        )
        (i32.store (i32.add (local.get $deltas_ptr) (i32.shl (local.get $i) (i32.const 2))) (local.get $d))
        (i32.store (i32.add (local.get $deltas_ptr) (i32.shl (i32.add (local.get $i) (i32.const 8)) (i32.const 2))) (local.get $d))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
  )

  ;; CV.borderFollowing: Writes points [x, y] (2 x i32 per point) to $out_pts, returns point count
  (func $cv_border_following (export "cv_border_following")
    (param $src i32) (param $pos i32) (param $nbd i32) (param $pt_x i32) (param $pt_y i32) (param $hole i32) (param $deltas i32) (param $out_pts i32)
    (result i32)
    (local $count i32) (local $pos1 i32) (local $pos3 i32) (local $pos4 i32) (local $s i32) (local $s_end i32)

    (local.set $count (i32.const 0))
    (local.set $s (select (i32.const 0) (i32.const 4) (local.get $hole)))
    (local.set $s_end (local.get $s))

    (block $scan_done
      (loop $scan_loop
        (local.set $s (i32.and (i32.sub (local.get $s) (i32.const 1)) (i32.const 7)))
        (local.set $pos1 (i32.add (local.get $pos) (i32.load (i32.add (local.get $deltas) (i32.shl (local.get $s) (i32.const 2))))))
        (br_if $scan_done (i32.ne (i32.load (i32.add (local.get $src) (i32.shl (local.get $pos1) (i32.const 2)))) (i32.const 0)))
        (br_if $scan_loop (i32.ne (local.get $s) (local.get $s_end)))
      )
    )

    (if (i32.eq (local.get $s) (local.get $s_end))
      (then
        (i32.store (i32.add (local.get $src) (i32.shl (local.get $pos) (i32.const 2))) (i32.sub (i32.const 0) (local.get $nbd)))
        (i32.store (local.get $out_pts) (local.get $pt_x))
        (i32.store offset=4 (local.get $out_pts) (local.get $pt_y))
        (local.set $count (i32.const 1))
      )
      (else
        (local.set $pos3 (local.get $pos))
        (block $follow_done
          (loop $follow_loop
            (local.set $s_end (local.get $s))
            (loop $inner_loop
              (local.set $s (i32.add (local.get $s) (i32.const 1)))
              (local.set $pos4 (i32.add (local.get $pos3) (i32.load (i32.add (local.get $deltas) (i32.shl (local.get $s) (i32.const 2))))))
              (br_if $inner_loop (i32.eq (i32.load (i32.add (local.get $src) (i32.shl (local.get $pos4) (i32.const 2)))) (i32.const 0)))
            )
            (local.set $s (i32.and (local.get $s) (i32.const 7)))

            (if (i32.lt_u (i32.sub (local.get $s) (i32.const 1)) (local.get $s_end))
              (then
                (i32.store (i32.add (local.get $src) (i32.shl (local.get $pos3) (i32.const 2))) (i32.sub (i32.const 0) (local.get $nbd)))
              )
              (else
                (if (i32.eq (i32.load (i32.add (local.get $src) (i32.shl (local.get $pos3) (i32.const 2)))) (i32.const 1))
                  (then
                    (i32.store (i32.add (local.get $src) (i32.shl (local.get $pos3) (i32.const 2))) (local.get $nbd))
                  )
                )
              )
            )

            (i32.store (i32.add (local.get $out_pts) (i32.shl (local.get $count) (i32.const 3))) (local.get $pt_x))
            (i32.store offset=4 (i32.add (local.get $out_pts) (i32.shl (local.get $count) (i32.const 3))) (local.get $pt_y))
            (local.set $count (i32.add (local.get $count) (i32.const 1)))

            (local.set $pt_x (i32.add (local.get $pt_x) (i32.load (i32.add (i32.const 384) (i32.shl (local.get $s) (i32.const 3))))))
            (local.set $pt_y (i32.add (local.get $pt_y) (i32.load offset=4 (i32.add (i32.const 384) (i32.shl (local.get $s) (i32.const 3))))))

            (br_if $follow_done (i32.and (i32.eq (local.get $pos4) (local.get $pos)) (i32.eq (local.get $pos3) (local.get $pos1))))

            (local.set $pos3 (local.get $pos4))
            (local.set $s (i32.and (i32.add (local.get $s) (i32.const 4)) (i32.const 7)))
            (br $follow_loop)
          )
        )
      )
    )
    (local.get $count)
  )

  ;; CV.findContours:
  ;; Stores contour headers at $contours_headers: [pts_offset, length, hole] (12 bytes per contour)
  ;; Stores contour points at $contours_pts: [x, y] (8 bytes per point)
  ;; Returns number of contours found
  (func $cv_find_contours (export "cv_find_contours")
    (param $src i32) (param $width i32) (param $height i32) (param $binary_buf i32) (param $contours_headers i32) (param $contours_pts i32)
    (result i32)
    (local $deltas i32) (local $pos i32) (local $nbd i32) (local $i i32) (local $j i32)
    (local $pix i32) (local $outer i32) (local $hole i32)
    (local $num_contours i32) (local $pts_offset i32) (local $c_len i32) (local $hdr_ptr i32)

    (call $cv_binary_border (local.get $src) (local.get $binary_buf) (local.get $width) (local.get $height))
    (local.set $deltas (i32.const 3584)) ;; Scratch 0x0E00
    (call $cv_neighborhood_deltas (i32.add (local.get $width) (i32.const 2)) (local.get $deltas))

    (local.set $pos (i32.add (local.get $width) (i32.const 3)))
    (local.set $nbd (i32.const 1))
    (local.set $num_contours (i32.const 0))
    (local.set $pts_offset (local.get $contours_pts))

    (local.set $i (i32.const 0))
    (block $i_done
      (loop $i_loop
        (br_if $i_done (i32.ge_s (local.get $i) (local.get $height)))
        (local.set $j (i32.const 0))
        (block $j_done
          (loop $j_loop
            (br_if $j_done (i32.ge_s (local.get $j) (local.get $width)))
            (local.set $pix (i32.load (i32.add (local.get $binary_buf) (i32.shl (local.get $pos) (i32.const 2)))))
            (if (i32.ne (local.get $pix) (i32.const 0))
              (then
                (local.set $outer (i32.const 0))
                (local.set $hole (i32.const 0))
                (if (i32.and (i32.eq (local.get $pix) (i32.const 1)) (i32.eq (i32.load (i32.add (local.get $binary_buf) (i32.shl (i32.sub (local.get $pos) (i32.const 1)) (i32.const 2)))) (i32.const 0)))
                  (then (local.set $outer (i32.const 1)))
                  (else
                    (if (i32.and (i32.ge_s (local.get $pix) (i32.const 1)) (i32.eq (i32.load (i32.add (local.get $binary_buf) (i32.shl (i32.add (local.get $pos) (i32.const 1)) (i32.const 2)))) (i32.const 0)))
                      (then (local.set $hole (i32.const 1)))
                    )
                  )
                )
                (if (i32.or (local.get $outer) (local.get $hole))
                  (then
                    (local.set $nbd (i32.add (local.get $nbd) (i32.const 1)))
                    (local.set $c_len
                      (call $cv_border_following
                        (local.get $binary_buf) (local.get $pos) (local.get $nbd)
                        (local.get $j) (local.get $i) (local.get $hole)
                        (local.get $deltas) (local.get $pts_offset)
                      )
                    )
                    (local.set $hdr_ptr (i32.add (local.get $contours_headers) (i32.mul (local.get $num_contours) (i32.const 12))))
                    (i32.store (local.get $hdr_ptr) (local.get $pts_offset))
                    (i32.store offset=4 (local.get $hdr_ptr) (local.get $c_len))
                    (i32.store offset=8 (local.get $hdr_ptr) (local.get $hole))
                    (local.set $pts_offset (i32.add (local.get $pts_offset) (i32.shl (local.get $c_len) (i32.const 3))))
                    (local.set $num_contours (i32.add (local.get $num_contours) (i32.const 1)))
                  )
                )
              )
            )
            (local.set $pos (i32.add (local.get $pos) (i32.const 1)))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $j_loop)
          )
        )
        (local.set $pos (i32.add (local.get $pos) (i32.const 2)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $i_loop)
      )
    )
    (local.get $num_contours)
  )

  ;; CV.approxPolyDP: Ramer-Douglas-Peucker polygon approximation
  ;; $contour_pts: array of [x, y] (i32), $len: point count, $epsilon: f64
  ;; Writes output points [x, y] (i32) to $out_poly, returns output point count
  (func $cv_approx_poly_dp (export "cv_approx_poly_dp")
    (param $contour_pts i32) (param $len i32) (param $epsilon f64) (param $out_poly i32) (param $stack_buf i32)
    (result i32)
    (local $slice_start i32) (local $slice_end i32)
    (local $r_start i32) (local $r_end i32)
    (local $poly_len i32) (local $stack_top i32)
    (local $start_x i32) (local $start_y i32)
    (local $end_x i32) (local $end_y i32)
    (local $pt_x i32) (local $pt_y i32)
    (local $dx f64) (local $dy f64) (local $dist f64) (local $max_dist f64)
    (local $le_eps i32) (local $i i32) (local $j i32) (local $k i32) (local $pt_ptr i32)

    (local.set $epsilon (f64.mul (local.get $epsilon) (local.get $epsilon)))
    (local.set $k (i32.const 0))
    (local.set $r_start (i32.const 0))
    (local.set $poly_len (i32.const 0))
    (local.set $stack_top (i32.const 0))

    (local.set $i (i32.const 0))
    (block $init_done
      (loop $init_loop
        (br_if $init_done (i32.ge_s (local.get $i) (i32.const 3)))
        (local.set $max_dist (f64.const 0.0))
        (local.set $k (i32.rem_u (i32.add (local.get $k) (local.get $r_start)) (local.get $len)))
        (local.set $pt_ptr (i32.add (local.get $contour_pts) (i32.shl (local.get $k) (i32.const 3))))
        (local.set $start_x (i32.load (local.get $pt_ptr)))
        (local.set $start_y (i32.load offset=4 (local.get $pt_ptr)))
        (local.set $k (i32.add (local.get $k) (i32.const 1)))
        (if (i32.eq (local.get $k) (local.get $len)) (then (local.set $k (i32.const 0))))

        (local.set $j (i32.const 1))
        (block $j_done
          (loop $j_loop
            (br_if $j_done (i32.ge_s (local.get $j) (local.get $len)))
            (local.set $pt_ptr (i32.add (local.get $contour_pts) (i32.shl (local.get $k) (i32.const 3))))
            (local.set $pt_x (i32.load (local.get $pt_ptr)))
            (local.set $pt_y (i32.load offset=4 (local.get $pt_ptr)))
            (local.set $k (i32.add (local.get $k) (i32.const 1)))
            (if (i32.eq (local.get $k) (local.get $len)) (then (local.set $k (i32.const 0))))

            (local.set $dx (f64.convert_i32_s (i32.sub (local.get $pt_x) (local.get $start_x))))
            (local.set $dy (f64.convert_i32_s (i32.sub (local.get $pt_y) (local.get $start_y))))
            (local.set $dist (f64.add (f64.mul (local.get $dx) (local.get $dx)) (f64.mul (local.get $dy) (local.get $dy))))

            (if (f64.gt (local.get $dist) (local.get $max_dist))
              (then
                (local.set $max_dist (local.get $dist))
                (local.set $r_start (local.get $j))
              )
            )
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $j_loop)
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $init_loop)
      )
    )

    (if (f64.le (local.get $max_dist) (local.get $epsilon))
      (then
        (i32.store (local.get $out_poly) (local.get $start_x))
        (i32.store offset=4 (local.get $out_poly) (local.get $start_y))
        (local.set $poly_len (i32.const 1))
      )
      (else
        (local.set $slice_start (local.get $k))
        (local.set $r_start (i32.add (local.get $r_start) (local.get $slice_start)))
        (local.set $slice_end (local.get $r_start))
        (if (i32.ge_s (local.get $r_start) (local.get $len))
          (then (local.set $r_start (i32.sub (local.get $r_start) (local.get $len))))
        )
        (local.set $r_end (local.get $slice_start))
        (if (i32.lt_s (local.get $r_end) (local.get $r_start))
          (then (local.set $r_end (i32.add (local.get $r_end) (local.get $len))))
        )
        ;; Push right_slice
        (i32.store (i32.add (local.get $stack_buf) (i32.shl (local.get $stack_top) (i32.const 3))) (local.get $r_start))
        (i32.store offset=4 (i32.add (local.get $stack_buf) (i32.shl (local.get $stack_top) (i32.const 3))) (local.get $r_end))
        (local.set $stack_top (i32.add (local.get $stack_top) (i32.const 1)))
        ;; Push slice
        (i32.store (i32.add (local.get $stack_buf) (i32.shl (local.get $stack_top) (i32.const 3))) (local.get $slice_start))
        (i32.store offset=4 (i32.add (local.get $stack_buf) (i32.shl (local.get $stack_top) (i32.const 3))) (local.get $slice_end))
        (local.set $stack_top (i32.add (local.get $stack_top) (i32.const 1)))
      )
    )

    (block $stack_done
      (loop $stack_loop
        (br_if $stack_done (i32.eq (local.get $stack_top) (i32.const 0)))
        (local.set $stack_top (i32.sub (local.get $stack_top) (i32.const 1)))
        (local.set $slice_start (i32.load (i32.add (local.get $stack_buf) (i32.shl (local.get $stack_top) (i32.const 3)))))
        (local.set $slice_end (i32.load offset=4 (i32.add (local.get $stack_buf) (i32.shl (local.get $stack_top) (i32.const 3)))))

        (local.set $pt_ptr (i32.add (local.get $contour_pts) (i32.shl (i32.rem_u (local.get $slice_end) (local.get $len)) (i32.const 3))))
        (local.set $end_x (i32.load (local.get $pt_ptr)))
        (local.set $end_y (i32.load offset=4 (local.get $pt_ptr)))

        (local.set $k (i32.rem_u (local.get $slice_start) (local.get $len)))
        (local.set $pt_ptr (i32.add (local.get $contour_pts) (i32.shl (local.get $k) (i32.const 3))))
        (local.set $start_x (i32.load (local.get $pt_ptr)))
        (local.set $start_y (i32.load offset=4 (local.get $pt_ptr)))
        (local.set $k (i32.add (local.get $k) (i32.const 1)))
        (if (i32.eq (local.get $k) (local.get $len)) (then (local.set $k (i32.const 0))))

        (if (i32.le_s (local.get $slice_end) (i32.add (local.get $slice_start) (i32.const 1)))
          (then (local.set $le_eps (i32.const 1)))
          (else
            (local.set $max_dist (f64.const 0.0))
            (local.set $dx (f64.convert_i32_s (i32.sub (local.get $end_x) (local.get $start_x))))
            (local.set $dy (f64.convert_i32_s (i32.sub (local.get $end_y) (local.get $start_y))))
            (local.set $i (i32.add (local.get $slice_start) (i32.const 1)))
            (block $sub_done
              (loop $sub_loop
                (br_if $sub_done (i32.ge_s (local.get $i) (local.get $slice_end)))
                (local.set $pt_ptr (i32.add (local.get $contour_pts) (i32.shl (local.get $k) (i32.const 3))))
                (local.set $pt_x (i32.load (local.get $pt_ptr)))
                (local.set $pt_y (i32.load offset=4 (local.get $pt_ptr)))
                (local.set $k (i32.add (local.get $k) (i32.const 1)))
                (if (i32.eq (local.get $k) (local.get $len)) (then (local.set $k (i32.const 0))))

                (local.set $dist
                  (f64.abs
                    (f64.sub
                      (f64.mul (f64.convert_i32_s (i32.sub (local.get $pt_y) (local.get $start_y))) (local.get $dx))
                      (f64.mul (f64.convert_i32_s (i32.sub (local.get $pt_x) (local.get $start_x))) (local.get $dy))
                    )
                  )
                )
                (if (f64.gt (local.get $dist) (local.get $max_dist))
                  (then
                    (local.set $max_dist (local.get $dist))
                    (local.set $r_start (local.get $i))
                  )
                )
                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                (br $sub_loop)
              )
            )
            (local.set $le_eps
              (f64.le
                (f64.mul (local.get $max_dist) (local.get $max_dist))
                (f64.mul (local.get $epsilon) (f64.add (f64.mul (local.get $dx) (local.get $dx)) (f64.mul (local.get $dy) (local.get $dy))))
              )
            )
          )
        )

        (if (local.get $le_eps)
          (then
            (i32.store (i32.add (local.get $out_poly) (i32.shl (local.get $poly_len) (i32.const 3))) (local.get $start_x))
            (i32.store offset=4 (i32.add (local.get $out_poly) (i32.shl (local.get $poly_len) (i32.const 3))) (local.get $start_y))
            (local.set $poly_len (i32.add (local.get $poly_len) (i32.const 1)))
          )
          (else
            (local.set $r_end (local.get $slice_end))
            (local.set $slice_end (local.get $r_start))
            (i32.store (i32.add (local.get $stack_buf) (i32.shl (local.get $stack_top) (i32.const 3))) (local.get $r_start))
            (i32.store offset=4 (i32.add (local.get $stack_buf) (i32.shl (local.get $stack_top) (i32.const 3))) (local.get $r_end))
            (local.set $stack_top (i32.add (local.get $stack_top) (i32.const 1)))

            (i32.store (i32.add (local.get $stack_buf) (i32.shl (local.get $stack_top) (i32.const 3))) (local.get $slice_start))
            (i32.store offset=4 (i32.add (local.get $stack_buf) (i32.shl (local.get $stack_top) (i32.const 3))) (local.get $slice_end))
            (local.set $stack_top (i32.add (local.get $stack_top) (i32.const 1)))
          )
        )
        (br $stack_loop)
      )
    )
    (local.get $poly_len)
  )

  ;; CV.square2quad: $src_pts is 4 x [f64 x, f64 y] (64 bytes), $out_sq is 9 x f64 (72 bytes)
  (func $cv_square2quad (export "cv_square2quad") (param $src_pts i32) (param $out_sq i32)
    (local $x0 f64) (local $y0 f64) (local $x1 f64) (local $y1 f64)
    (local $x2 f64) (local $y2 f64) (local $x3 f64) (local $y3 f64)
    (local $px f64) (local $py f64) (local $dx1 f64) (local $dx2 f64)
    (local $dy1 f64) (local $dy2 f64) (local $den f64) (local $sq6 f64) (local $sq7 f64)

    (local.set $x0 (f64.load (local.get $src_pts)))
    (local.set $y0 (f64.load offset=8 (local.get $src_pts)))
    (local.set $x1 (f64.load offset=16 (local.get $src_pts)))
    (local.set $y1 (f64.load offset=24 (local.get $src_pts)))
    (local.set $x2 (f64.load offset=32 (local.get $src_pts)))
    (local.set $y2 (f64.load offset=40 (local.get $src_pts)))
    (local.set $x3 (f64.load offset=48 (local.get $src_pts)))
    (local.set $y3 (f64.load offset=56 (local.get $src_pts)))

    (local.set $px (f64.sub (f64.add (f64.sub (local.get $x0) (local.get $x1)) (local.get $x2)) (local.get $x3)))
    (local.set $py (f64.sub (f64.add (f64.sub (local.get $y0) (local.get $y1)) (local.get $y2)) (local.get $y3)))

    (if (i32.and (f64.eq (local.get $px) (f64.const 0.0)) (f64.eq (local.get $py) (f64.const 0.0)))
      (then
        (f64.store (local.get $out_sq) (f64.sub (local.get $x1) (local.get $x0)))
        (f64.store offset=8 (local.get $out_sq) (f64.sub (local.get $x2) (local.get $x1)))
        (f64.store offset=16 (local.get $out_sq) (local.get $x0))
        (f64.store offset=24 (local.get $out_sq) (f64.sub (local.get $y1) (local.get $y0)))
        (f64.store offset=32 (local.get $out_sq) (f64.sub (local.get $y2) (local.get $y1)))
        (f64.store offset=40 (local.get $out_sq) (local.get $y0))
        (f64.store offset=48 (local.get $out_sq) (f64.const 0.0))
        (f64.store offset=56 (local.get $out_sq) (f64.const 0.0))
        (f64.store offset=64 (local.get $out_sq) (f64.const 1.0))
      )
      (else
        (local.set $dx1 (f64.sub (local.get $x1) (local.get $x2)))
        (local.set $dx2 (f64.sub (local.get $x3) (local.get $x2)))
        (local.set $dy1 (f64.sub (local.get $y1) (local.get $y2)))
        (local.set $dy2 (f64.sub (local.get $y3) (local.get $y2)))
        (local.set $den (f64.sub (f64.mul (local.get $dx1) (local.get $dy2)) (f64.mul (local.get $dx2) (local.get $dy1))))

        (local.set $sq6 (f64.div (f64.sub (f64.mul (local.get $px) (local.get $dy2)) (f64.mul (local.get $dx2) (local.get $py))) (local.get $den)))
        (local.set $sq7 (f64.div (f64.sub (f64.mul (local.get $dx1) (local.get $py)) (f64.mul (local.get $px) (local.get $dy1))) (local.get $den)))

        (f64.store offset=48 (local.get $out_sq) (local.get $sq6))
        (f64.store offset=56 (local.get $out_sq) (local.get $sq7))
        (f64.store offset=64 (local.get $out_sq) (f64.const 1.0))
        (f64.store (local.get $out_sq) (f64.add (f64.sub (local.get $x1) (local.get $x0)) (f64.mul (local.get $sq6) (local.get $x1))))
        (f64.store offset=8 (local.get $out_sq) (f64.add (f64.sub (local.get $x3) (local.get $x0)) (f64.mul (local.get $sq7) (local.get $x3))))
        (f64.store offset=16 (local.get $out_sq) (local.get $x0))
        (f64.store offset=24 (local.get $out_sq) (f64.add (f64.sub (local.get $y1) (local.get $y0)) (f64.mul (local.get $sq6) (local.get $y1))))
        (f64.store offset=32 (local.get $out_sq) (f64.add (f64.sub (local.get $y3) (local.get $y0)) (f64.mul (local.get $sq7) (local.get $y3))))
        (f64.store offset=40 (local.get $out_sq) (local.get $y0))
      )
    )
  )

  ;; CV.getPerspectiveTransform
  (func $cv_get_perspective_transform (export "cv_get_perspective_transform") (param $src_pts i32) (param $size f64) (param $out_rq i32)
    (call $cv_square2quad (local.get $src_pts) (local.get $out_rq))
    (f64.store (local.get $out_rq) (f64.div (f64.load (local.get $out_rq)) (local.get $size)))
    (f64.store offset=8 (local.get $out_rq) (f64.div (f64.load offset=8 (local.get $out_rq)) (local.get $size)))
    (f64.store offset=24 (local.get $out_rq) (f64.div (f64.load offset=24 (local.get $out_rq)) (local.get $size)))
    (f64.store offset=32 (local.get $out_rq) (f64.div (f64.load offset=32 (local.get $out_rq)) (local.get $size)))
    (f64.store offset=48 (local.get $out_rq) (f64.div (f64.load offset=48 (local.get $out_rq)) (local.get $size)))
    (f64.store offset=56 (local.get $out_rq) (f64.div (f64.load offset=56 (local.get $out_rq)) (local.get $size)))
  )

  ;; CV.warp: Bilinear perspective warp ($contour_pts: 4 x [f64 x, f64 y])
  (func $cv_warp (export "cv_warp")
    (param $src i32) (param $dst i32) (param $width i32) (param $height i32) (param $contour_pts i32) (param $warp_size i32)
    (local $pos i32) (local $m i32)
    (local $m0 f64) (local $m1 f64) (local $m2 f64) (local $m3 f64) (local $m4 f64) (local $m5 f64) (local $m6 f64) (local $m7 f64) (local $m8 f64)
    (local $r f64) (local $s f64) (local $t f64) (local $u f64) (local $v f64) (local $w f64)
    (local $x f64) (local $y f64) (local $dx1 f64) (local $dx2 f64) (local $dy1 f64) (local $dy2 f64)
    (local $sx1 i32) (local $sx2 i32) (local $sy1 i32) (local $sy2 i32)
    (local $p1 i32) (local $p3 i32) (local $i i32) (local $j i32) (local $wm1 i32) (local $hm1 i32)

    (local.set $m (i32.const 3840)) ;; Scratch 0x0F00
    (call $cv_get_perspective_transform (local.get $contour_pts) (f64.convert_i32_s (i32.sub (local.get $warp_size) (i32.const 1))) (local.get $m))

    (local.set $m0 (f64.load (local.get $m)))
    (local.set $m1 (f64.load offset=8 (local.get $m)))
    (local.set $m2 (f64.load offset=16 (local.get $m)))
    (local.set $m3 (f64.load offset=24 (local.get $m)))
    (local.set $m4 (f64.load offset=32 (local.get $m)))
    (local.set $m5 (f64.load offset=40 (local.get $m)))
    (local.set $m6 (f64.load offset=48 (local.get $m)))
    (local.set $m7 (f64.load offset=56 (local.get $m)))
    (local.set $m8 (f64.load offset=64 (local.get $m)))

    (local.set $r (local.get $m8))
    (local.set $s (local.get $m2))
    (local.set $t (local.get $m5))
    (local.set $pos (i32.const 0))
    (local.set $wm1 (i32.sub (local.get $width) (i32.const 1)))
    (local.set $hm1 (i32.sub (local.get $height) (i32.const 1)))

    (local.set $i (i32.const 0))
    (block $i_done
      (loop $i_loop
        (br_if $i_done (i32.ge_s (local.get $i) (local.get $warp_size)))
        (local.set $r (f64.add (local.get $r) (local.get $m7)))
        (local.set $s (f64.add (local.get $s) (local.get $m1)))
        (local.set $t (f64.add (local.get $t) (local.get $m4)))

        (local.set $u (local.get $r))
        (local.set $v (local.get $s))
        (local.set $w (local.get $t))

        (local.set $j (i32.const 0))
        (block $j_done
          (loop $j_loop
            (br_if $j_done (i32.ge_s (local.get $j) (local.get $warp_size)))
            (local.set $u (f64.add (local.get $u) (local.get $m6)))
            (local.set $v (f64.add (local.get $v) (local.get $m0)))
            (local.set $w (f64.add (local.get $w) (local.get $m3)))

            (local.set $x (f64.div (local.get $v) (local.get $u)))
            (local.set $y (f64.div (local.get $w) (local.get $u)))

            (local.set $sx1 (i32.trunc_sat_f64_u (local.get $x)))
            (local.set $sx2 (select (local.get $sx1) (i32.add (local.get $sx1) (i32.const 1)) (i32.eq (local.get $sx1) (local.get $wm1))))
            (local.set $dx1 (f64.sub (local.get $x) (f64.convert_i32_u (local.get $sx1))))
            (local.set $dx2 (f64.sub (f64.const 1.0) (local.get $dx1)))

            (local.set $sy1 (i32.trunc_sat_f64_u (local.get $y)))
            (local.set $sy2 (select (local.get $sy1) (i32.add (local.get $sy1) (i32.const 1)) (i32.eq (local.get $sy1) (local.get $hm1))))
            (local.set $dy1 (f64.sub (local.get $y) (f64.convert_i32_u (local.get $sy1))))
            (local.set $dy2 (f64.sub (f64.const 1.0) (local.get $dy1)))

            (local.set $p1 (i32.mul (local.get $sy1) (local.get $width)))
            (local.set $p3 (i32.mul (local.get $sy2) (local.get $width)))

            (i32.store8
              (i32.add (local.get $dst) (local.get $pos))
              (i32.and
                (i32.trunc_sat_f64_s
                  (f64.add
                    (f64.mul
                      (local.get $dy2)
                      (f64.add
                        (f64.mul (local.get $dx2) (f64.convert_i32_u (i32.load8_u (i32.add (local.get $src) (i32.add (local.get $p1) (local.get $sx1))))))
                        (f64.mul (local.get $dx1) (f64.convert_i32_u (i32.load8_u (i32.add (local.get $src) (i32.add (local.get $p1) (local.get $sx2))))))
                      )
                    )
                    (f64.mul
                      (local.get $dy1)
                      (f64.add
                        (f64.mul (local.get $dx2) (f64.convert_i32_u (i32.load8_u (i32.add (local.get $src) (i32.add (local.get $p3) (local.get $sx1))))))
                        (f64.mul (local.get $dx1) (f64.convert_i32_u (i32.load8_u (i32.add (local.get $src) (i32.add (local.get $p3) (local.get $sx2))))))
                      )
                    )
                  )
                )
                (i32.const 255)
              )
            )
            (local.set $pos (i32.add (local.get $pos) (i32.const 1)))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $j_loop)
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $i_loop)
      )
    )
  )

  ;; CV.isContourConvex ($poly_pts: array of [i32 x, i32 y])
  (func $cv_is_contour_convex (export "cv_is_contour_convex") (param $poly_pts i32) (param $len i32) (result i32)
    (local $orientation i32) (local $convex i32) (local $i i32) (local $j i32)
    (local $cur_x i32) (local $cur_y i32) (local $prev_x i32) (local $prev_y i32)
    (local $dx0 f64) (local $dy0 f64) (local $dx f64) (local $dy f64) (local $dxdy0 f64) (local $dydx0 f64) (local $pt_ptr i32)

    (local.set $orientation (i32.const 0))
    (local.set $convex (i32.const 1))

    (local.set $pt_ptr (i32.add (local.get $poly_pts) (i32.shl (i32.sub (local.get $len) (i32.const 1)) (i32.const 3))))
    (local.set $prev_x (i32.load (local.get $pt_ptr)))
    (local.set $prev_y (i32.load offset=4 (local.get $pt_ptr)))

    (local.set $cur_x (i32.load (local.get $poly_pts)))
    (local.set $cur_y (i32.load offset=4 (local.get $poly_pts)))

    (local.set $dx0 (f64.convert_i32_s (i32.sub (local.get $cur_x) (local.get $prev_x))))
    (local.set $dy0 (f64.convert_i32_s (i32.sub (local.get $cur_y) (local.get $prev_y))))

    (local.set $i (i32.const 0))
    (local.set $j (i32.const 0))
    (block $done
      (loop $loop
        (br_if $done (i32.ge_s (local.get $i) (local.get $len)))
        (local.set $j (i32.add (local.get $j) (i32.const 1)))
        (if (i32.eq (local.get $j) (local.get $len)) (then (local.set $j (i32.const 0))))

        (local.set $prev_x (local.get $cur_x))
        (local.set $prev_y (local.get $cur_y))

        (local.set $pt_ptr (i32.add (local.get $poly_pts) (i32.shl (local.get $j) (i32.const 3))))
        (local.set $cur_x (i32.load (local.get $pt_ptr)))
        (local.set $cur_y (i32.load offset=4 (local.get $pt_ptr)))

        (local.set $dx (f64.convert_i32_s (i32.sub (local.get $cur_x) (local.get $prev_x))))
        (local.set $dy (f64.convert_i32_s (i32.sub (local.get $cur_y) (local.get $prev_y))))
        (local.set $dxdy0 (f64.mul (local.get $dx) (local.get $dy0)))
        (local.set $dydx0 (f64.mul (local.get $dy) (local.get $dx0)))

        (local.set $orientation
          (i32.or
            (local.get $orientation)
            (if (result i32) (f64.gt (local.get $dydx0) (local.get $dxdy0))
              (then (i32.const 1))
              (else
                (select (i32.const 2) (i32.const 3) (f64.lt (local.get $dydx0) (local.get $dxdy0)))
              )
            )
          )
        )

        (if (i32.eq (local.get $orientation) (i32.const 3))
          (then
            (local.set $convex (i32.const 0))
            (br $done)
          )
        )

        (local.set $dx0 (local.get $dx))
        (local.set $dy0 (local.get $dy))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    (local.get $convex)
  )

  ;; CV.perimeter ($poly_pts: array of [i32 x, i32 y])
  (func $cv_perimeter (export "cv_perimeter") (param $poly_pts i32) (param $len i32) (result f64)
    (local $i i32) (local $j i32) (local $p f64) (local $dx f64) (local $dy f64) (local $pi i32) (local $pj i32)
    (local.set $i (i32.const 0))
    (local.set $j (i32.sub (local.get $len) (i32.const 1)))
    (local.set $p (f64.const 0.0))

    (block $done
      (loop $loop
        (br_if $done (i32.ge_s (local.get $i) (local.get $len)))
        (local.set $pi (i32.add (local.get $poly_pts) (i32.shl (local.get $i) (i32.const 3))))
        (local.set $pj (i32.add (local.get $poly_pts) (i32.shl (local.get $j) (i32.const 3))))
        (local.set $dx (f64.convert_i32_s (i32.sub (i32.load (local.get $pi)) (i32.load (local.get $pj)))))
        (local.set $dy (f64.convert_i32_s (i32.sub (i32.load offset=4 (local.get $pi)) (i32.load offset=4 (local.get $pj)))))
        (local.set $p (f64.add (local.get $p) (f64.sqrt (f64.add (f64.mul (local.get $dx) (local.get $dx)) (f64.mul (local.get $dy) (local.get $dy))))))
        (local.set $j (local.get $i))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    (local.get $p)
  )

  ;; CV.minEdgeLength ($poly_pts: array of [i32 x, i32 y])
  (func $cv_min_edge_length (export "cv_min_edge_length") (param $poly_pts i32) (param $len i32) (result f64)
    (local $i i32) (local $j i32) (local $min f64) (local $d f64) (local $dx f64) (local $dy f64) (local $pi i32) (local $pj i32)
    (local.set $i (i32.const 0))
    (local.set $j (i32.sub (local.get $len) (i32.const 1)))
    (local.set $min (f64.const inf))

    (block $done
      (loop $loop
        (br_if $done (i32.ge_s (local.get $i) (local.get $len)))
        (local.set $pi (i32.add (local.get $poly_pts) (i32.shl (local.get $i) (i32.const 3))))
        (local.set $pj (i32.add (local.get $poly_pts) (i32.shl (local.get $j) (i32.const 3))))
        (local.set $dx (f64.convert_i32_s (i32.sub (i32.load (local.get $pi)) (i32.load (local.get $pj)))))
        (local.set $dy (f64.convert_i32_s (i32.sub (i32.load offset=4 (local.get $pi)) (i32.load offset=4 (local.get $pj)))))
        (local.set $d (f64.add (f64.mul (local.get $dx) (local.get $dx)) (f64.mul (local.get $dy) (local.get $dy))))
        (if (f64.lt (local.get $d) (local.get $min))
          (then (local.set $min (local.get $d)))
        )
        (local.set $j (local.get $i))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    (f64.sqrt (local.get $min))
  )

  ;; CV.countNonZero: SIMD accelerated non-zero pixel count in ROI square
  (func $cv_count_non_zero (export "cv_count_non_zero")
    (param $src i32) (param $img_width i32) (param $sq_x i32) (param $sq_y i32) (param $sq_w i32) (param $sq_h i32)
    (result i32)
    (local $pos i32) (local $span i32) (local $nz i32) (local $i i32) (local $j i32) (local $simd_w i32)
    (local $v_zero v128) (local $v_one v128) (local $v_sum16 v128) (local $v_chunk v128) (local $v_sum32 v128)

    (local.set $pos (i32.add (local.get $sq_x) (i32.mul (local.get $sq_y) (local.get $img_width))))
    (local.set $span (i32.sub (local.get $img_width) (local.get $sq_w)))
    (local.set $nz (i32.const 0))
    (local.set $simd_w (i32.and (local.get $sq_w) (i32.const -16)))
    (local.set $v_zero (i8x16.splat (i32.const 0)))
    (local.set $v_one (i8x16.splat (i32.const 1)))
    (local.set $v_sum32 (i32x4.splat (i32.const 0)))

    (local.set $i (i32.const 0))
    (block $i_done
      (loop $i_loop
        (br_if $i_done (i32.ge_s (local.get $i) (local.get $sq_h)))
        (local.set $j (i32.const 0))

        ;; SIMD 16-byte loop when square width >= 16
        (block $simd_done
          (loop $simd_loop
            (br_if $simd_done (i32.ge_s (local.get $j) (local.get $simd_w)))
            (local.set $v_chunk
              (v128.andnot
                (local.get $v_one)
                (i8x16.eq (v128.load (i32.add (local.get $src) (local.get $pos))) (local.get $v_zero))
              )
            )
            (local.set $v_sum16 (i16x8.extadd_pairwise_i8x16_u (local.get $v_chunk)))
            (local.set $v_sum32 (i32x4.add (local.get $v_sum32) (i32x4.extadd_pairwise_i16x8_u (local.get $v_sum16))))
            (local.set $pos (i32.add (local.get $pos) (i32.const 16)))
            (local.set $j (i32.add (local.get $j) (i32.const 16)))
            (br $simd_loop)
          )
        )

        ;; Scalar tail
        (block $j_done
          (loop $j_loop
            (br_if $j_done (i32.ge_s (local.get $j) (local.get $sq_w)))
            (if (i32.ne (i32.load8_u (i32.add (local.get $src) (local.get $pos))) (i32.const 0))
              (then (local.set $nz (i32.add (local.get $nz) (i32.const 1))))
            )
            (local.set $pos (i32.add (local.get $pos) (i32.const 1)))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $j_loop)
          )
        )
        (local.set $pos (i32.add (local.get $pos) (local.get $span)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $i_loop)
      )
    )

    (i32.add
      (local.get $nz)
      (i32.add
        (i32.add (i32x4.extract_lane 0 (local.get $v_sum32)) (i32x4.extract_lane 1 (local.get $v_sum32)))
        (i32.add (i32x4.extract_lane 2 (local.get $v_sum32)) (i32x4.extract_lane 3 (local.get $v_sum32)))
      )
    )
  )

  ;; ============================================================================
  ;; Section 2: Dictionary & Bitwise Hamming Distance (64-bit POPCNT)
  ;; ============================================================================
  ;; Dictionary Layout at $dict_ptr:
  ;;   0..3   : nBits (i32)
  ;;   4..7   : markSize (i32)
  ;;   8..11  : tau (i32)
  ;;   12..15 : count (i32)
  ;;   16..   : codes array (count * 8 bytes, each code stored as i64 bitmask)

  (func $dict_calculate_tau (export "dict_calculate_tau") (param $dict_ptr i32) (result i32)
    (local $count i32) (local $tau i32) (local $i i32) (local $j i32) (local $dist i32)
    (local $c1 i64) (local $c2 i64) (local $codes_ptr i32)

    (local.set $count (i32.load offset=12 (local.get $dict_ptr)))
    (local.set $codes_ptr (i32.add (local.get $dict_ptr) (i32.const 16)))
    (local.set $tau (i32.const 2147483647))

    (local.set $i (i32.const 0))
    (block $i_done
      (loop $i_loop
        (br_if $i_done (i32.ge_s (local.get $i) (local.get $count)))
        (local.set $c1 (i64.load (i32.add (local.get $codes_ptr) (i32.shl (local.get $i) (i32.const 3)))))
        (local.set $j (i32.add (local.get $i) (i32.const 1)))
        (block $j_done
          (loop $j_loop
            (br_if $j_done (i32.ge_s (local.get $j) (local.get $count)))
            (local.set $c2 (i64.load (i32.add (local.get $codes_ptr) (i32.shl (local.get $j) (i32.const 3)))))
            (local.set $dist (i32.wrap_i64 (i64.popcnt (i64.xor (local.get $c1) (local.get $c2)))))
            (if (i32.lt_s (local.get $dist) (local.get $tau))
              (then (local.set $tau (local.get $dist)))
            )
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $j_loop)
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $i_loop)
      )
    )
    (i32.store offset=8 (local.get $dict_ptr) (local.get $tau))
    (local.get $tau)
  )

  ;; Dictionary.find: Returns packed i64 ((distance << 32) | id), or -1 if no match < tau
  (func $dict_find_bits (export "dict_find_bits") (param $dict_ptr i32) (param $bits_mask i64) (result i64)
    (local $count i32) (local $tau i32) (local $i i32) (local $dist i32)
    (local $best_id i32) (local $best_dist i32) (local $codes_ptr i32)

    (local.set $tau (i32.load offset=8 (local.get $dict_ptr)))
    (local.set $count (i32.load offset=12 (local.get $dict_ptr)))
    (local.set $codes_ptr (i32.add (local.get $dict_ptr) (i32.const 16)))
    (local.set $best_id (i32.const -1))
    (local.set $best_dist (i32.const 2147483647))

    (local.set $i (i32.const 0))
    (block $done
      (loop $loop
        (br_if $done (i32.ge_s (local.get $i) (local.get $count)))
        (local.set $dist
          (i32.wrap_i64
            (i64.popcnt
              (i64.xor
                (local.get $bits_mask)
                (i64.load (i32.add (local.get $codes_ptr) (i32.shl (local.get $i) (i32.const 3))))
              )
            )
          )
        )
        (if (i32.and (i32.lt_s (local.get $dist) (local.get $tau)) (i32.lt_s (local.get $dist) (local.get $best_dist)))
          (then
            (local.set $best_dist (local.get $dist))
            (local.set $best_id (local.get $i))
            (br_if $done (i32.eq (local.get $best_dist) (i32.const 0)))
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )

    (if (result i64) (i32.lt_s (local.get $best_id) (i32.const 0))
      (then (i64.const -1))
      (else
        (i64.or
          (i64.shl (i64.extend_i32_u (local.get $best_dist)) (i64.const 32))
          (i64.extend_i32_u (local.get $best_id))
        )
      )
    )
  )

  ;; Packs a square 2D array of i32 bits ($size x $size) into an i64 bitmask
  (func $pack_bits_matrix (param $bits_ptr i32) (param $size i32) (result i64)
    (local $len i32) (local $i i32) (local $mask i64)
    (local.set $len (i32.mul (local.get $size) (local.get $size)))
    (local.set $mask (i64.const 0))
    (local.set $i (i32.const 0))
    (block $done
      (loop $loop
        (br_if $done (i32.ge_s (local.get $i) (local.get $len)))
        (local.set $mask
          (i64.or
            (i64.shl (local.get $mask) (i64.const 1))
            (i64.extend_i32_u (i32.and (i32.load (i32.add (local.get $bits_ptr) (i32.shl (local.get $i) (i32.const 2)))) (i32.const 1)))
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    (local.get $mask)
  )

  ;; ============================================================================
  ;; Section 3: Detector Pipeline (Candidates, Corners, Markers)
  ;; ============================================================================

  ;; Detector.rotate: Rotates $size x $size matrix of i32 90 degrees clockwise
  (func $detector_rotate (export "detector_rotate") (param $src i32) (param $dst i32) (param $size i32)
    (local $i i32) (local $j i32) (local $src_idx i32) (local $dst_idx i32)
    (local.set $i (i32.const 0))
    (block $i_done
      (loop $i_loop
        (br_if $i_done (i32.ge_s (local.get $i) (local.get $size)))
        (local.set $j (i32.const 0))
        (block $j_done
          (loop $j_loop
            (br_if $j_done (i32.ge_s (local.get $j) (local.get $size)))
            ;; dst[i][j] = src[size - j - 1][i]
            (local.set $src_idx (i32.add (i32.mul (i32.sub (i32.sub (local.get $size) (local.get $j)) (i32.const 1)) (local.get $size)) (local.get $i)))
            (local.set $dst_idx (i32.add (i32.mul (local.get $i) (local.get $size)) (local.get $j)))
            (i32.store
              (i32.add (local.get $dst) (i32.shl (local.get $dst_idx) (i32.const 2)))
              (i32.load (i32.add (local.get $src) (i32.shl (local.get $src_idx) (i32.const 2))))
            )
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $j_loop)
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $i_loop)
      )
    )
  )

  ;; Detector.rotate2: Rotates 4-corner array ([i32 x, i32 y] x 4 = 32 bytes)
  (func $detector_rotate2 (export "detector_rotate2") (param $src i32) (param $rotation i32) (param $dst i32)
    (local $i i32) (local $idx i32)
    (local.set $i (i32.const 0))
    (block $done
      (loop $loop
        (br_if $done (i32.ge_s (local.get $i) (i32.const 4)))
        (local.set $idx (i32.and (i32.add (local.get $rotation) (local.get $i)) (i32.const 3)))
        (i64.store
          (i32.add (local.get $dst) (i32.shl (local.get $i) (i32.const 3)))
          (i64.load (i32.add (local.get $src) (i32.shl (local.get $idx) (i32.const 3))))
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
  )

  ;; Detector.findCandidates:
  ;; Reads contours from $contours_headers, approximates polygons, filters 4-vertex convex quads >= minLength
  ;; Writes 4-corner candidates (32 bytes each: 4 x [i32 x, i32 y]) to $candidates_out
  (func $detector_find_candidates (export "detector_find_candidates")
    (param $contours_headers i32) (param $num_contours i32) (param $min_size f64) (param $epsilon f64) (param $min_length f64)
    (param $candidates_out i32) (param $poly_scratch i32) (param $stack_scratch i32)
    (result i32)
    (local $i i32) (local $num_cand i32) (local $hdr i32) (local $pts_ptr i32) (local $c_len i32) (local $poly_len i32)

    (local.set $num_cand (i32.const 0))
    (local.set $i (i32.const 0))
    (block $done
      (loop $loop
        (br_if $done (i32.ge_s (local.get $i) (local.get $num_contours)))
        (local.set $hdr (i32.add (local.get $contours_headers) (i32.mul (local.get $i) (i32.const 12))))
        (local.set $pts_ptr (i32.load (local.get $hdr)))
        (local.set $c_len (i32.load offset=4 (local.get $hdr)))

        (if (f64.ge (f64.convert_i32_s (local.get $c_len)) (local.get $min_size))
          (then
            (local.set $poly_len
              (call $cv_approx_poly_dp
                (local.get $pts_ptr)
                (local.get $c_len)
                (f64.mul (f64.convert_i32_s (local.get $c_len)) (local.get $epsilon))
                (local.get $poly_scratch)
                (local.get $stack_scratch)
              )
            )
            (if (i32.eq (local.get $poly_len) (i32.const 4))
              (then
                (if (call $cv_is_contour_convex (local.get $poly_scratch) (i32.const 4))
                  (then
                    (if (f64.ge (call $cv_min_edge_length (local.get $poly_scratch) (i32.const 4)) (local.get $min_length))
                      (then
                        (memory.copy
                          (i32.add (local.get $candidates_out) (i32.shl (local.get $num_cand) (i32.const 5)))
                          (local.get $poly_scratch)
                          (i32.const 32)
                        )
                        (local.set $num_cand (i32.add (local.get $num_cand) (i32.const 1)))
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    (local.get $num_cand)
  )

  ;; Detector.clockwiseCorners: In-place corner winding order normalization
  (func $detector_clockwise_corners (export "detector_clockwise_corners") (param $candidates i32) (param $len i32)
    (local $i i32) (local $c_ptr i32) (local $dx1 f64) (local $dy1 f64) (local $dx2 f64) (local $dy2 f64) (local $swap i64)
    (local.set $i (i32.const 0))
    (block $done
      (loop $loop
        (br_if $done (i32.ge_s (local.get $i) (local.get $len)))
        (local.set $c_ptr (i32.add (local.get $candidates) (i32.shl (local.get $i) (i32.const 5))))
        (local.set $dx1 (f64.convert_i32_s (i32.sub (i32.load offset=8 (local.get $c_ptr)) (i32.load (local.get $c_ptr)))))
        (local.set $dy1 (f64.convert_i32_s (i32.sub (i32.load offset=12 (local.get $c_ptr)) (i32.load offset=4 (local.get $c_ptr)))))
        (local.set $dx2 (f64.convert_i32_s (i32.sub (i32.load offset=16 (local.get $c_ptr)) (i32.load (local.get $c_ptr)))))
        (local.set $dy2 (f64.convert_i32_s (i32.sub (i32.load offset=20 (local.get $c_ptr)) (i32.load offset=4 (local.get $c_ptr)))))

        (if (f64.lt (f64.sub (f64.mul (local.get $dx1) (local.get $dy2)) (f64.mul (local.get $dy1) (local.get $dx2))) (f64.const 0.0))
          (then
            (local.set $swap (i64.load offset=8 (local.get $c_ptr)))
            (i64.store offset=8 (local.get $c_ptr) (i64.load offset=24 (local.get $c_ptr)))
            (i64.store offset=24 (local.get $c_ptr) (local.get $swap))
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
  )

  ;; Detector.notTooNear: Filters overlapping candidates based on perimeter
  (func $detector_not_too_near (export "detector_not_too_near")
    (param $candidates i32) (param $len i32) (param $min_dist f64) (param $out_candidates i32) (param $flags_scratch i32)
    (result i32)
    (local $i i32) (local $j i32) (local $k i32) (local $out_len i32)
    (local $ci i32) (local $cj i32) (local $dx f64) (local $dy f64) (local $dist f64) (local $min_sq f64)

    (memory.fill (local.get $flags_scratch) (i32.const 0) (local.get $len))
    (local.set $min_sq (f64.mul (local.get $min_dist) (local.get $min_dist)))

    (local.set $i (i32.const 0))
    (block $i_done
      (loop $i_loop
        (br_if $i_done (i32.ge_s (local.get $i) (local.get $len)))
        (local.set $ci (i32.add (local.get $candidates) (i32.shl (local.get $i) (i32.const 5))))
        (local.set $j (i32.add (local.get $i) (i32.const 1)))
        (block $j_done
          (loop $j_loop
            (br_if $j_done (i32.ge_s (local.get $j) (local.get $len)))
            (local.set $cj (i32.add (local.get $candidates) (i32.shl (local.get $j) (i32.const 5))))
            (local.set $dist (f64.const 0.0))

            (local.set $k (i32.const 0))
            (block $k_done
              (loop $k_loop
                (br_if $k_done (i32.ge_s (local.get $k) (i32.const 4)))
                (local.set $dx
                  (f64.convert_i32_s
                    (i32.sub
                      (i32.load (i32.add (local.get $ci) (i32.shl (local.get $k) (i32.const 3))))
                      (i32.load (i32.add (local.get $cj) (i32.shl (local.get $k) (i32.const 3))))
                    )
                  )
                )
                (local.set $dy
                  (f64.convert_i32_s
                    (i32.sub
                      (i32.load offset=4 (i32.add (local.get $ci) (i32.shl (local.get $k) (i32.const 3))))
                      (i32.load offset=4 (i32.add (local.get $cj) (i32.shl (local.get $k) (i32.const 3))))
                    )
                  )
                )
                (local.set $dist (f64.add (local.get $dist) (f64.add (f64.mul (local.get $dx) (local.get $dx)) (f64.mul (local.get $dy) (local.get $dy)))))
                (local.set $k (i32.add (local.get $k) (i32.const 1)))
                (br $k_loop)
              )
            )

            (if (f64.lt (f64.mul (local.get $dist) (f64.const 0.25)) (local.get $min_sq))
              (then
                (if (f64.lt (call $cv_perimeter (local.get $ci) (i32.const 4)) (call $cv_perimeter (local.get $cj) (i32.const 4)))
                  (then (i32.store8 (i32.add (local.get $flags_scratch) (local.get $i)) (i32.const 1)))
                  (else (i32.store8 (i32.add (local.get $flags_scratch) (local.get $j)) (i32.const 1)))
                )
              )
            )
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $j_loop)
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $i_loop)
      )
    )

    (local.set $out_len (i32.const 0))
    (local.set $i (i32.const 0))
    (block $f_done
      (loop $f_loop
        (br_if $f_done (i32.ge_s (local.get $i) (local.get $len)))
        (if (i32.eq (i32.load8_u (i32.add (local.get $flags_scratch) (local.get $i))) (i32.const 0))
          (then
            (memory.copy
              (i32.add (local.get $out_candidates) (i32.shl (local.get $out_len) (i32.const 5)))
              (i32.add (local.get $candidates) (i32.shl (local.get $i) (i32.const 5)))
              (i32.const 32)
            )
            (local.set $out_len (i32.add (local.get $out_len) (i32.const 1)))
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $f_loop)
      )
    )
    (local.get $out_len)
  )

  ;; Detector.getMarker:
  ;; Validates border cells, extracts inner grid bits, checks 4 rotations against Dictionary
  ;; If matched, writes Marker struct (40 bytes: [id: i32, dist: i32, corners: 4 x [i32 x, i32 y]]) to $marker_out and returns 1
  (func $detector_get_marker (export "detector_get_marker")
    (param $homography i32) (param $warp_size i32) (param $candidate_ptr i32) (param $dict_ptr i32) (param $marker_out i32)
    (result i32)
    (local $mark_size i32) (local $inner_size i32) (local $cell_w i32) (local $min_zero i32)
    (local $inc i32) (local $i i32) (local $j i32)
    (local $rot0 i32) (local $rot1 i32) (local $src_rot i32) (local $dst_rot i32)
    (local $found i64) (local $f_id i32) (local $f_dist i32)
    (local $best_id i32) (local $best_dist i32) (local $best_rot i32)

    (local.set $mark_size (i32.load offset=4 (local.get $dict_ptr)))
    (local.set $inner_size (i32.sub (local.get $mark_size) (i32.const 2)))
    (local.set $cell_w (i32.div_u (local.get $warp_size) (local.get $mark_size)))
    (local.set $min_zero (i32.shr_s (i32.mul (local.get $cell_w) (local.get $cell_w)) (i32.const 1)))

    ;; 1. Verify black border cells
    (local.set $i (i32.const 0))
    (block $b_done
      (loop $b_loop
        (br_if $b_done (i32.ge_s (local.get $i) (local.get $mark_size)))
        (local.set $inc
          (select
            (i32.const 1)
            (i32.sub (local.get $mark_size) (i32.const 1))
            (i32.or (i32.eq (local.get $i) (i32.const 0)) (i32.eq (local.get $i) (i32.sub (local.get $mark_size) (i32.const 1))))
          )
        )
        (local.set $j (i32.const 0))
        (block $bj_done
          (loop $bj_loop
            (br_if $bj_done (i32.ge_s (local.get $j) (local.get $mark_size)))
            (if
              (i32.gt_s
                (call $cv_count_non_zero
                  (local.get $homography)
                  (local.get $warp_size)
                  (i32.mul (local.get $j) (local.get $cell_w))
                  (i32.mul (local.get $i) (local.get $cell_w))
                  (local.get $cell_w)
                  (local.get $cell_w)
                )
                (local.get $min_zero)
              )
              (then (return (i32.const 0)))
            )
            (local.set $j (i32.add (local.get $j) (local.get $inc)))
            (br $bj_loop)
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $b_loop)
      )
    )

    ;; 2. Extract interior bit matrix into scratch $rot0 (0x1500 = 5376) and $rot1 (0x1600 = 5632)
    (local.set $rot0 (i32.const 5376))
    (local.set $rot1 (i32.const 5632))

    (local.set $i (i32.const 0))
    (block $g_done
      (loop $g_loop
        (br_if $g_done (i32.ge_s (local.get $i) (local.get $inner_size)))
        (local.set $j (i32.const 0))
        (block $gj_done
          (loop $gj_loop
            (br_if $gj_done (i32.ge_s (local.get $j) (local.get $inner_size)))
            (i32.store
              (i32.add (local.get $rot0) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $inner_size)) (local.get $j)) (i32.const 2)))
              (select
                (i32.const 1)
                (i32.const 0)
                (i32.gt_s
                  (call $cv_count_non_zero
                    (local.get $homography)
                    (local.get $warp_size)
                    (i32.mul (i32.add (local.get $j) (i32.const 1)) (local.get $cell_w))
                    (i32.mul (i32.add (local.get $i) (i32.const 1)) (local.get $cell_w))
                    (local.get $cell_w)
                    (local.get $cell_w)
                  )
                  (local.get $min_zero)
                )
              )
            )
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $gj_loop)
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $g_loop)
      )
    )

    ;; 3. Check all 4 rotations
    (local.set $best_id (i32.const -1))
    (local.set $best_dist (i32.const 2147483647))
    (local.set $best_rot (i32.const 0))
    (local.set $src_rot (local.get $rot0))
    (local.set $dst_rot (local.get $rot1))

    (local.set $i (i32.const 0))
    (block $r_done
      (loop $r_loop
        (br_if $r_done (i32.ge_s (local.get $i) (i32.const 4)))
        (local.set $found (call $dict_find_bits (local.get $dict_ptr) (call $pack_bits_matrix (local.get $src_rot) (local.get $inner_size))))
        (if (i64.ge_s (local.get $found) (i64.const 0))
          (then
            (local.set $f_id (i32.wrap_i64 (local.get $found)))
            (local.set $f_dist (i32.wrap_i64 (i64.shr_u (local.get $found) (i64.const 32))))
            (if (i32.lt_s (local.get $f_dist) (local.get $best_dist))
              (then
                (local.set $best_id (local.get $f_id))
                (local.set $best_dist (local.get $f_dist))
                (local.set $best_rot (local.get $i))
                (br_if $r_done (i32.eq (local.get $best_dist) (i32.const 0)))
              )
            )
          )
        )
        (call $detector_rotate (local.get $src_rot) (local.get $dst_rot) (local.get $inner_size))
        ;; Swap rotation buffers
        (local.set $j (local.get $src_rot))
        (local.set $src_rot (local.get $dst_rot))
        (local.set $dst_rot (local.get $j))

        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $r_loop)
      )
    )

    (if (result i32) (i32.ge_s (local.get $best_id) (i32.const 0))
      (then
        (i32.store (local.get $marker_out) (local.get $best_id))
        (i32.store offset=4 (local.get $marker_out) (local.get $best_dist))
        (call $detector_rotate2 (local.get $candidate_ptr) (i32.sub (i32.const 4) (local.get $best_rot)) (i32.add (local.get $marker_out) (i32.const 8)))
        (i32.const 1)
      )
      (else (i32.const 0))
    )
  )

  ;; Detector.findMarkers:
  ;; Warps each candidate to $warp_size x $warp_size, Otsu thresholds, and decodes Marker
  (func $detector_find_markers (export "detector_find_markers")
    (param $grey_src i32) (param $width i32) (param $height i32) (param $candidates i32) (param $num_candidates i32)
    (param $warp_size i32) (param $homography_buf i32) (param $dict_ptr i32) (param $markers_out i32)
    (result i32)
    (local $i i32) (local $k i32) (local $num_markers i32) (local $cand_ptr i32) (local $f64_quad i32) (local $warp_len i32)

    (local.set $num_markers (i32.const 0))
    (local.set $f64_quad (i32.const 5888)) ;; Scratch 0x1700 for 4 x [f64 x, f64 y]
    (local.set $warp_len (i32.mul (local.get $warp_size) (local.get $warp_size)))

    (local.set $i (i32.const 0))
    (block $done
      (loop $loop
        (br_if $done (i32.ge_s (local.get $i) (local.get $num_candidates)))
        (local.set $cand_ptr (i32.add (local.get $candidates) (i32.shl (local.get $i) (i32.const 5))))

        ;; Convert i32 candidate corners to f64 for CV.warp
        (local.set $k (i32.const 0))
        (block $k_done
          (loop $k_loop
            (br_if $k_done (i32.ge_s (local.get $k) (i32.const 4)))
            (f64.store
              (i32.add (local.get $f64_quad) (i32.shl (local.get $k) (i32.const 4)))
              (f64.convert_i32_s (i32.load (i32.add (local.get $cand_ptr) (i32.shl (local.get $k) (i32.const 3)))))
            )
            (f64.store
              offset=8
              (i32.add (local.get $f64_quad) (i32.shl (local.get $k) (i32.const 4)))
              (f64.convert_i32_s (i32.load offset=4 (i32.add (local.get $cand_ptr) (i32.shl (local.get $k) (i32.const 3)))))
            )
            (local.set $k (i32.add (local.get $k) (i32.const 1)))
            (br $k_loop)
          )
        )

        (call $cv_warp (local.get $grey_src) (local.get $homography_buf) (local.get $width) (local.get $height) (local.get $f64_quad) (local.get $warp_size))
        (call $cv_threshold
          (local.get $homography_buf)
          (local.get $homography_buf)
          (local.get $warp_len)
          (call $cv_otsu (local.get $homography_buf) (local.get $warp_len))
        )

        (if
          (call $detector_get_marker
            (local.get $homography_buf)
            (local.get $warp_size)
            (local.get $cand_ptr)
            (local.get $dict_ptr)
            (i32.add (local.get $markers_out) (i32.mul (local.get $num_markers) (i32.const 40)))
          )
          (then (local.set $num_markers (i32.add (local.get $num_markers) (i32.const 1))))
        )

        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    (local.get $num_markers)
  )

  ;; Detector.detect: Full end-to-end ArUco detection pipeline in Wasm
  ;; Returns number of markers written to $markers_out
  (func $detector_detect (export "detector_detect")
    (param $rgba_src i32) (param $width i32) (param $height i32) (param $dict_ptr i32)
    (param $grey_buf i32) (param $thres_buf i32) (param $binary_buf i32)
    (param $contours_headers i32) (param $contours_pts i32)
    (param $candidates_buf1 i32) (param $candidates_buf2 i32)
    (param $poly_scratch i32) (param $stack_scratch i32) (param $homography_buf i32)
    (param $markers_out i32)
    (result i32)
    (local $num_contours i32) (local $num_cand i32) (local $min_dist f64)

    ;; 1. RGBA -> Grayscale
    (call $cv_grayscale (local.get $rgba_src) (local.get $grey_buf) (local.get $width) (local.get $height))

    ;; 2. Adaptive Threshold (radius=2, threshold=7)
    (call $cv_adaptive_threshold (local.get $grey_buf) (local.get $thres_buf) (local.get $width) (local.get $height) (i32.const 2) (i32.const 7))

    ;; 3. Find Contours
    (local.set $num_contours
      (call $cv_find_contours
        (local.get $thres_buf) (local.get $width) (local.get $height)
        (local.get $binary_buf) (local.get $contours_headers) (local.get $contours_pts)
      )
    )

    ;; 4. Find Quad Candidates (minSize = width * 0.01, epsilon = 0.05, minLength = 10)
    (local.set $num_cand
      (call $detector_find_candidates
        (local.get $contours_headers)
        (local.get $num_contours)
        (f64.mul (f64.convert_i32_s (local.get $width)) (f64.const 0.01))
        (f64.const 0.05)
        (f64.const 10.0)
        (local.get $candidates_buf1)
        (local.get $poly_scratch)
        (local.get $stack_scratch)
      )
    )

    ;; 5. Clockwise Corners
    (call $detector_clockwise_corners (local.get $candidates_buf1) (local.get $num_cand))

    ;; 6. Filter NotTooNear (minDist = max(30, width * 0.05))
    (local.set $min_dist (f64.max (f64.const 30.0) (f64.mul (f64.convert_i32_s (local.get $width)) (f64.const 0.05))))
    (local.set $num_cand
      (call $detector_not_too_near
        (local.get $candidates_buf1)
        (local.get $num_cand)
        (local.get $min_dist)
        (local.get $candidates_buf2)
        (local.get $poly_scratch)
      )
    )

    ;; 7. Warp & Identify Markers (warpSize = 49)
    (call $detector_find_markers
      (local.get $grey_buf) (local.get $width) (local.get $height)
      (local.get $candidates_buf2) (local.get $num_cand)
      (i32.const 49) (local.get $homography_buf)
      (local.get $dict_ptr) (local.get $markers_out)
    )
  )
)