(module
  ;; ============================================================================
  ;; Host Math Imports (Required for Posit & Gaussian Kernel)
  ;; ============================================================================
  (import "env" "sin" (func $sin (param f64) (result f64)))
  (import "env" "cos" (func $cos (param f64) (result f64)))
  (import "env" "acos" (func $acos (param f64) (result f64)))
  (import "env" "atan" (func $atan (param f64) (result f64)))
  (import "env" "exp" (func $exp (param f64) (result f64)))

  ;; ============================================================================
  ;; Memory & Static Data Layout
  ;; ============================================================================
  ;; 512 pages = 32 MB linear memory
  ;; 0x000000 - 0x00FFFF : Static Lookup Tables, Constants, & Scratch Registers
  ;; 0x010000 - 0x01FFFF : Dictionary Bitmasks & Posit State
  ;; 0x020000 - 0x1FFFFF : Dynamic Heap &Detector Buffers (Managed by TS/Bump Allocator)
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

  ;; Default 4x4_1000 Dictionary (50 x u16 codes at 0x0200 = 512)
  (data (i32.const 512)
    "\32\b5\9a\0f\2d\33\46\99\9e\54\cd\79\2e\9e\f2\c4"
    "\da\fe\56\cf\91\f9\a7\11\b7\0e\0f\2a\b1\24\3e\26"
    "\65\46\00\66\5e\6c\af\76\8b\86\2b\b0\d5\cc\82\dd"
    "\47\fe\71\94\e4\ac\54\a5\23\21\6f\34\15\44\b2\57"
    "\cf\9e\cb\f0\ae\08\29\09\75\18\ff\04\f6\0d\5a\1c"
    "\18\17\28\2a\8c\32\b2\38\e8\24\eb\2e\3f\2d\64\4b"
    "\2e\50\13\50"
  )

  ;; Legacy ArUco 5x5_1024 row IDs (4 x 5 x i32 at 0x0280 = 640)
  (data (i32.const 640)
    "\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
    "\01\00\00\00\00\00\00\00\01\00\00\00\01\00\00\00\01\00\00\00"
    "\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00"
    "\00\00\00\00\01\00\00\00\01\00\00\00\01\00\00\00\00\00\00\00"
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
    (local.set $res (global.get $heap_ptr))
    ;; 16-byte align for SIMD safety
    (global.set $heap_ptr
      (i32.and
        (i32.add (i32.add (local.get $res) (local.get $size)) (i32.const 15))
        (i32.const -16)
      )
    )
    (local.get $res)
  )

  ;; ============================================================================
  ;; Section 1: SIMD Vec3 & Mat3 Math Operations (Float32)
  ;; ============================================================================

  (func $vec3_add (export "vec3_add") (param $a i32) (param $b i32) (param $out i32)
    (local $v v128)
    (local.set $v (f32x4.add (v128.load (local.get $a)) (v128.load (local.get $b))))
    (f32.store (local.get $out) (f32x4.extract_lane 0 (local.get $v)))
    (f32.store offset=4 (local.get $out) (f32x4.extract_lane 1 (local.get $v)))
    (f32.store offset=8 (local.get $out) (f32x4.extract_lane 2 (local.get $v)))
  )

  (func $vec3_sub (export "vec3_sub") (param $a i32) (param $b i32) (param $out i32)
    (local $v v128)
    (local.set $v (f32x4.sub (v128.load (local.get $a)) (v128.load (local.get $b))))
    (f32.store (local.get $out) (f32x4.extract_lane 0 (local.get $v)))
    (f32.store offset=4 (local.get $out) (f32x4.extract_lane 1 (local.get $v)))
    (f32.store offset=8 (local.get $out) (f32x4.extract_lane 2 (local.get $v)))
  )

  (func $vec3_mult (export "vec3_mult") (param $a i32) (param $b i32) (param $out i32)
    (local $v v128)
    (local.set $v (f32x4.mul (v128.load (local.get $a)) (v128.load (local.get $b))))
    (f32.store (local.get $out) (f32x4.extract_lane 0 (local.get $v)))
    (f32.store offset=4 (local.get $out) (f32x4.extract_lane 1 (local.get $v)))
    (f32.store offset=8 (local.get $out) (f32x4.extract_lane 2 (local.get $v)))
  )

  (func $vec3_add_scalar (export "vec3_add_scalar") (param $a i32) (param $s f32) (param $out i32)
    (local $v v128)
    (local.set $v (f32x4.add (v128.load (local.get $a)) (f32x4.splat (local.get $s))))
    (f32.store (local.get $out) (f32x4.extract_lane 0 (local.get $v)))
    (f32.store offset=4 (local.get $out) (f32x4.extract_lane 1 (local.get $v)))
    (f32.store offset=8 (local.get $out) (f32x4.extract_lane 2 (local.get $v)))
  )

  (func $vec3_mult_scalar (export "vec3_mult_scalar") (param $a i32) (param $s f32) (param $out i32)
    (local $v v128)
    (local.set $v (f32x4.mul (v128.load (local.get $a)) (f32x4.splat (local.get $s))))
    (f32.store (local.get $out) (f32x4.extract_lane 0 (local.get $v)))
    (f32.store offset=4 (local.get $out) (f32x4.extract_lane 1 (local.get $v)))
    (f32.store offset=8 (local.get $out) (f32x4.extract_lane 2 (local.get $v)))
  )

  (func $vec3_dot (export "vec3_dot") (param $a i32) (param $b i32) (result f32)
    (local $v v128)
    (local.set $v (f32x4.mul (v128.load (local.get $a)) (v128.load (local.get $b))))
    (f32.add
      (f32.add (f32x4.extract_lane 0 (local.get $v)) (f32x4.extract_lane 1 (local.get $v)))
      (f32x4.extract_lane 2 (local.get $v))
    )
  )

  (func $vec3_cross (export "vec3_cross") (param $a i32) (param $b i32) (param $out i32)
    (local $ax f32) (local $ay f32) (local $az f32)
    (local $bx f32) (local $by f32) (local $bz f32)
    (local.set $ax (f32.load (local.get $a)))
    (local.set $ay (f32.load offset=4 (local.get $a)))
    (local.set $az (f32.load offset=8 (local.get $a)))
    (local.set $bx (f32.load (local.get $b)))
    (local.set $by (f32.load offset=4 (local.get $b)))
    (local.set $bz (f32.load offset=8 (local.get $b)))
    (f32.store (local.get $out) (f32.sub (f32.mul (local.get $ay) (local.get $bz)) (f32.mul (local.get $az) (local.get $by))))
    (f32.store offset=4 (local.get $out) (f32.sub (f32.mul (local.get $az) (local.get $bx)) (f32.mul (local.get $ax) (local.get $bz))))
    (f32.store offset=8 (local.get $out) (f32.sub (f32.mul (local.get $ax) (local.get $by)) (f32.mul (local.get $ay) (local.get $bx))))
  )

  (func $vec3_square (export "vec3_square") (param $a i32) (result f32)
    (call $vec3_dot (local.get $a) (local.get $a))
  )

  (func $vec3_normalize (export "vec3_normalize") (param $a i32) (result f32)
    (local $len f32)
    (local $inv f32)
    (local.set $len (f32.sqrt (call $vec3_square (local.get $a))))
    (if (f32.gt (local.get $len) (f32.const 0.0))
      (then
        (local.set $inv (f32.div (f32.const 1.0) (local.get $len)))
        (call $vec3_mult_scalar (local.get $a) (local.get $inv) (local.get $a))
      )
    )
    (local.get $len)
  )

  (func $vec3_inverse (export "vec3_inverse") (param $a i32) (param $out i32)
    (local $x f32) (local $y f32) (local $z f32)
    (local.set $x (f32.load (local.get $a)))
    (local.set $y (f32.load offset=4 (local.get $a)))
    (local.set $z (f32.load offset=8 (local.get $a)))
    (f32.store (local.get $out) (select (f32.div (f32.const 1.0) (local.get $x)) (f32.const 0.0) (f32.ne (local.get $x) (f32.const 0.0))))
    (f32.store offset=4 (local.get $out) (select (f32.div (f32.const 1.0) (local.get $y)) (f32.const 0.0) (f32.ne (local.get $y) (f32.const 0.0))))
    (f32.store offset=8 (local.get $out) (select (f32.div (f32.const 1.0) (local.get $z)) (f32.const 0.0) (f32.ne (local.get $z) (f32.const 0.0))))
  )

  (func $vec3_min_index (export "vec3_min_index") (param $a i32) (result i32)
    (local $x f32) (local $y f32) (local $z f32)
    (local.set $x (f32.load (local.get $a)))
    (local.set $y (f32.load offset=4 (local.get $a)))
    (local.set $z (f32.load offset=8 (local.get $a)))
    (if (result i32) (f32.lt (local.get $x) (local.get $y))
      (then (select (i32.const 0) (i32.const 2) (f32.lt (local.get $x) (local.get $z))))
      (else (select (i32.const 1) (i32.const 2) (f32.lt (local.get $y) (local.get $z))))
    )
  )

  (func $mat3_transpose (export "mat3_transpose") (param $a i32) (param $out i32)
    (local $m00 f32) (local $m01 f32) (local $m02 f32)
    (local $m10 f32) (local $m11 f32) (local $m12 f32)
    (local $m20 f32) (local $m21 f32) (local $m22 f32)
    (local.set $m00 (f32.load (local.get $a)))
    (local.set $m01 (f32.load offset=4 (local.get $a)))
    (local.set $m02 (f32.load offset=8 (local.get $a)))
    (local.set $m10 (f32.load offset=12 (local.get $a)))
    (local.set $m11 (f32.load offset=16 (local.get $a)))
    (local.set $m12 (f32.load offset=20 (local.get $a)))
    (local.set $m20 (f32.load offset=24 (local.get $a)))
    (local.set $m21 (f32.load offset=28 (local.get $a)))
    (local.set $m22 (f32.load offset=32 (local.get $a)))

    (f32.store (local.get $out) (local.get $m00))
    (f32.store offset=4 (local.get $out) (local.get $m10))
    (f32.store offset=8 (local.get $out) (local.get $m20))
    (f32.store offset=12 (local.get $out) (local.get $m01))
    (f32.store offset=16 (local.get $out) (local.get $m11))
    (f32.store offset=20 (local.get $out) (local.get $m21))
    (f32.store offset=24 (local.get $out) (local.get $m02))
    (f32.store offset=28 (local.get $out) (local.get $m12))
    (f32.store offset=32 (local.get $out) (local.get $m22))
  )

  (func $mat3_mult (export "mat3_mult") (param $a i32) (param $b i32) (param $out i32)
    (local $a00 f32) (local $a01 f32) (local $a02 f32)
    (local $a10 f32) (local $a11 f32) (local $a12 f32)
    (local $a20 f32) (local $a21 f32) (local $a22 f32)
    (local $b00 f32) (local $b01 f32) (local $b02 f32)
    (local $b10 f32) (local $b11 f32) (local $b12 f32)
    (local $b20 f32) (local $b21 f32) (local $b22 f32)

    (local.set $a00 (f32.load (local.get $a)))
    (local.set $a01 (f32.load offset=4 (local.get $a)))
    (local.set $a02 (f32.load offset=8 (local.get $a)))
    (local.set $a10 (f32.load offset=12 (local.get $a)))
    (local.set $a11 (f32.load offset=16 (local.get $a)))
    (local.set $a12 (f32.load offset=20 (local.get $a)))
    (local.set $a20 (f32.load offset=24 (local.get $a)))
    (local.set $a21 (f32.load offset=28 (local.get $a)))
    (local.set $a22 (f32.load offset=32 (local.get $a)))

    (local.set $b00 (f32.load (local.get $b)))
    (local.set $b01 (f32.load offset=4 (local.get $b)))
    (local.set $b02 (f32.load offset=8 (local.get $b)))
    (local.set $b10 (f32.load offset=12 (local.get $b)))
    (local.set $b11 (f32.load offset=16 (local.get $b)))
    (local.set $b12 (f32.load offset=20 (local.get $b)))
    (local.set $b20 (f32.load offset=24 (local.get $b)))
    (local.set $b21 (f32.load offset=28 (local.get $b)))
    (local.set $b22 (f32.load offset=32 (local.get $b)))

    (f32.store (local.get $out) (f32.add (f32.add (f32.mul (local.get $a00) (local.get $b00)) (f32.mul (local.get $a01) (local.get $b10))) (f32.mul (local.get $a02) (local.get $b20))))
    (f32.store offset=4 (local.get $out) (f32.add (f32.add (f32.mul (local.get $a00) (local.get $b01)) (f32.mul (local.get $a01) (local.get $b11))) (f32.mul (local.get $a02) (local.get $b21))))
    (f32.store offset=8 (local.get $out) (f32.add (f32.add (f32.mul (local.get $a00) (local.get $b02)) (f32.mul (local.get $a01) (local.get $b12))) (f32.mul (local.get $a02) (local.get $b22))))

    (f32.store offset=12 (local.get $out) (f32.add (f32.add (f32.mul (local.get $a10) (local.get $b00)) (f32.mul (local.get $a11) (local.get $b10))) (f32.mul (local.get $a12) (local.get $b20))))
    (f32.store offset=16 (local.get $out) (f32.add (f32.add (f32.mul (local.get $a10) (local.get $b01)) (f32.mul (local.get $a11) (local.get $b11))) (f32.mul (local.get $a12) (local.get $b21))))
    (f32.store offset=20 (local.get $out) (f32.add (f32.add (f32.mul (local.get $a10) (local.get $b02)) (f32.mul (local.get $a11) (local.get $b12))) (f32.mul (local.get $a12) (local.get $b22))))

    (f32.store offset=24 (local.get $out) (f32.add (f32.add (f32.mul (local.get $a20) (local.get $b00)) (f32.mul (local.get $a21) (local.get $b10))) (f32.mul (local.get $a22) (local.get $b20))))
    (f32.store offset=28 (local.get $out) (f32.add (f32.add (f32.mul (local.get $a20) (local.get $b01)) (f32.mul (local.get $a21) (local.get $b11))) (f32.mul (local.get $a22) (local.get $b21))))
    (f32.store offset=32 (local.get $out) (f32.add (f32.add (f32.mul (local.get $a20) (local.get $b02)) (f32.mul (local.get $a21) (local.get $b12))) (f32.mul (local.get $a22) (local.get $b22))))
  )

  (func $mat3_mult_vector (export "mat3_mult_vector") (param $m i32) (param $v i32) (param $out i32)
    (local $vx f32) (local $vy f32) (local $vz f32)
    (local $x f32) (local $y f32) (local $z f32)
    (local.set $vx (f32.load (local.get $v)))
    (local.set $vy (f32.load offset=4 (local.get $v)))
    (local.set $vz (f32.load offset=8 (local.get $v)))

    (local.set $x (f32.add (f32.add (f32.mul (f32.load (local.get $m)) (local.get $vx)) (f32.mul (f32.load offset=4 (local.get $m)) (local.get $vy))) (f32.mul (f32.load offset=8 (local.get $m)) (local.get $vz))))
    (local.set $y (f32.add (f32.add (f32.mul (f32.load offset=12 (local.get $m)) (local.get $vx)) (f32.mul (f32.load offset=16 (local.get $m)) (local.get $vy))) (f32.mul (f32.load offset=20 (local.get $m)) (local.get $vz))))
    (local.set $z (f32.add (f32.add (f32.mul (f32.load offset=24 (local.get $m)) (local.get $vx)) (f32.mul (f32.load offset=28 (local.get $m)) (local.get $vy))) (f32.mul (f32.load offset=32 (local.get $m)) (local.get $vz))))

    (f32.store (local.get $out) (local.get $x))
    (f32.store offset=4 (local.get $out) (local.get $y))
    (f32.store offset=8 (local.get $out) (local.get $z))
  )

  (func $mat3_from_rows (export "mat3_from_rows") (param $a i32) (param $b i32) (param $c i32) (param $out i32)
    (f32.store (local.get $out) (f32.load (local.get $a)))
    (f32.store offset=4 (local.get $out) (f32.load offset=4 (local.get $a)))
    (f32.store offset=8 (local.get $out) (f32.load offset=8 (local.get $a)))
    (f32.store offset=12 (local.get $out) (f32.load (local.get $b)))
    (f32.store offset=16 (local.get $out) (f32.load offset=4 (local.get $b)))
    (f32.store offset=20 (local.get $out) (f32.load offset=8 (local.get $b)))
    (f32.store offset=24 (local.get $out) (f32.load (local.get $c)))
    (f32.store offset=28 (local.get $out) (f32.load offset=4 (local.get $c)))
    (f32.store offset=32 (local.get $out) (f32.load offset=8 (local.get $c)))
  )

  (func $mat3_from_diagonal (export "mat3_from_diagonal") (param $a i32) (param $out i32)
    (memory.fill (local.get $out) (i32.const 0) (i32.const 36))
    (f32.store (local.get $out) (f32.load (local.get $a)))
    (f32.store offset=16 (local.get $out) (f32.load offset=4 (local.get $a)))
    (f32.store offset=32 (local.get $out) (f32.load offset=8 (local.get $a)))
  )

  ;; ============================================================================
  ;; Section 2: Computer Vision (CV) with SIMD Acceleration
  ;; ============================================================================

  ;; CV.grayscale: RGBA (4 bytes/pixel) -> Grayscale (1 byte/pixel)
  (func $cv_grayscale (export "cv_grayscale") (param $src i32) (param $dst i32) (param $width i32) (param $height i32)
    (local $len i32) (local $i i32) (local $j i32)
    (local.set $len (i32.shl (i32.mul (local.get $width) (local.get $height)) (i32.const 2)))
    (local.set $i (i32.const 0))
    (local.set $j (i32.const 0))
    (block $break
      (loop $loop
        (br_if $break (i32.ge_u (local.get $i) (local.get $len)))
        (i32.store8
          (i32.add (local.get $dst) (local.get $j))
          (i32.and
            (i32.trunc_sat_f64_s
              (f64.add
                (f64.add
                  (f64.mul (f64.convert_i32_u (i32.load8_u (i32.add (local.get $src) (local.get $i)))) (f64.const 0.299))
                  (f64.mul (f64.convert_i32_u (i32.load8_u offset=1 (i32.add (local.get $src) (local.get $i)))) (f64.const 0.587))
                )
                (f64.add
                  (f64.mul (f64.convert_i32_u (i32.load8_u offset=2 (i32.add (local.get $src) (local.get $i)))) (f64.const 0.114))
                  (f64.const 0.5)
                )
              )
            )
            (i32.const 255)
          )
        )
        (local.set $j (i32.add (local.get $j) (i32.const 1)))
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $loop)
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

  ;; CV.gaussianKernel: Generates f64 weights at $kernel_ptr
  (func $cv_gaussian_kernel (export "cv_gaussian_kernel") (param $kernel_size i32) (param $kernel_ptr i32)
    (local $center f64) (local $sigma f64) (local $scale2X f64) (local $sum f64) (local $x f64) (local $val f64) (local $i i32)
    (if (i32.and (i32.le_s (local.get $kernel_size) (i32.const 7)) (i32.eq (i32.and (local.get $kernel_size) (i32.const 1)) (i32.const 1)))
      (then
        (if (i32.eq (local.get $kernel_size) (i32.const 1))
          (then (f64.store (local.get $kernel_ptr) (f64.const 1.0)))
        )
        (if (i32.eq (local.get $kernel_size) (i32.const 3))
          (then
            (f64.store (local.get $kernel_ptr) (f64.const 0.25))
            (f64.store offset=8 (local.get $kernel_ptr) (f64.const 0.5))
            (f64.store offset=16 (local.get $kernel_ptr) (f64.const 0.25))
          )
        )
        (if (i32.eq (local.get $kernel_size) (i32.const 5))
          (then
            (f64.store (local.get $kernel_ptr) (f64.const 0.0625))
            (f64.store offset=8 (local.get $kernel_ptr) (f64.const 0.25))
            (f64.store offset=16 (local.get $kernel_ptr) (f64.const 0.375))
            (f64.store offset=24 (local.get $kernel_ptr) (f64.const 0.25))
            (f64.store offset=32 (local.get $kernel_ptr) (f64.const 0.0625))
          )
        )
        (if (i32.eq (local.get $kernel_size) (i32.const 7))
          (then
            (f64.store (local.get $kernel_ptr) (f64.const 0.03125))
            (f64.store offset=8 (local.get $kernel_ptr) (f64.const 0.109375))
            (f64.store offset=16 (local.get $kernel_ptr) (f64.const 0.21875))
            (f64.store offset=24 (local.get $kernel_ptr) (f64.const 0.28125))
            (f64.store offset=32 (local.get $kernel_ptr) (f64.const 0.21875))
            (f64.store offset=40 (local.get $kernel_ptr) (f64.const 0.109375))
            (f64.store offset=48 (local.get $kernel_ptr) (f64.const 0.03125))
          )
        )
      )
      (else
        (local.set $center (f64.mul (f64.convert_i32_s (i32.sub (local.get $kernel_size) (i32.const 1))) (f64.const 0.5)))
        (local.set $sigma (f64.add (f64.const 0.8) (f64.mul (f64.const 0.3) (f64.sub (local.get $center) (f64.const 1.0)))))
        (local.set $scale2X (f64.div (f64.const -0.5) (f64.mul (local.get $sigma) (local.get $sigma))))
        (local.set $sum (f64.const 0.0))
        (local.set $i (i32.const 0))
        (block $k1_done
          (loop $k1_loop
            (br_if $k1_done (i32.ge_s (local.get $i) (local.get $kernel_size)))
            (local.set $x (f64.sub (f64.convert_i32_s (local.get $i)) (local.get $center)))
            (local.set $val (call $exp (f64.mul (local.get $scale2X) (f64.mul (local.get $x) (local.get $x)))))
            (f64.store (i32.add (local.get $kernel_ptr) (i32.shl (local.get $i) (i32.const 3))) (local.get $val))
            (local.set $sum (f64.add (local.get $sum) (local.get $val)))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $k1_loop)
          )
        )
        (local.set $sum (f64.div (f64.const 1.0) (local.get $sum)))
        (local.set $i (i32.const 0))
        (block $k2_done
          (loop $k2_loop
            (br_if $k2_done (i32.ge_s (local.get $i) (local.get $kernel_size)))
            (f64.store
              (i32.add (local.get $kernel_ptr) (i32.shl (local.get $i) (i32.const 3)))
              (f64.mul (f64.load (i32.add (local.get $kernel_ptr) (i32.shl (local.get $i) (i32.const 3)))) (local.get $sum))
            )
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $k2_loop)
          )
        )
      )
    )
  )

  ;; CV.gaussianBlurFilter
  (func $cv_gaussian_blur_filter (export "cv_gaussian_blur_filter")
    (param $src i32) (param $dst i32) (param $width i32) (param $height i32) (param $kernel_ptr i32) (param $kernel_len i32) (param $horizontal i32)
    (local $pos i32) (local $limit i32) (local $i i32) (local $j i32) (local $k i32) (local $cur i32) (local $value f64)
    (local.set $pos (i32.const 0))
    (local.set $limit (i32.shr_s (local.get $kernel_len) (i32.const 1)))
    (local.set $i (i32.const 0))
    (block $i_done
      (loop $i_loop
        (br_if $i_done (i32.ge_s (local.get $i) (local.get $height)))
        (local.set $j (i32.const 0))
        (block $j_done
          (loop $j_loop
            (br_if $j_done (i32.ge_s (local.get $j) (local.get $width)))
            (local.set $value (f64.const 0.0))
            (local.set $k (i32.sub (i32.const 0) (local.get $limit)))
            (block $k_done
              (loop $k_loop
                (br_if $k_done (i32.gt_s (local.get $k) (local.get $limit)))
                (if (local.get $horizontal)
                  (then
                    (local.set $cur (i32.add (local.get $pos) (local.get $k)))
                    (if (i32.or (i32.lt_s (i32.add (local.get $j) (local.get $k)) (i32.const 0)) (i32.ge_s (i32.add (local.get $j) (local.get $k)) (local.get $width)))
                      (then (local.set $cur (local.get $pos)))
                    )
                  )
                  (else
                    (local.set $cur (i32.add (local.get $pos) (i32.mul (local.get $k) (local.get $width))))
                    (if (i32.or (i32.lt_s (i32.add (local.get $i) (local.get $k)) (i32.const 0)) (i32.ge_s (i32.add (local.get $i) (local.get $k)) (local.get $height)))
                      (then (local.set $cur (local.get $pos)))
                    )
                  )
                )
                (local.set $value
                  (f64.add
                    (local.get $value)
                    (f64.mul
                      (f64.load (i32.add (local.get $kernel_ptr) (i32.shl (i32.add (local.get $limit) (local.get $k)) (i32.const 3))))
                      (f64.convert_i32_u (i32.load8_u (i32.add (local.get $src) (local.get $cur))))
                    )
                  )
                )
                (local.set $k (i32.add (local.get $k) (i32.const 1)))
                (br $k_loop)
              )
            )
            (if (local.get $horizontal)
              (then
                ;; Uint8ClampedArray round-to-even semantics for horizontal float intermediate
                (i32.store8 (i32.add (local.get $dst) (local.get $pos)) (i32.trunc_sat_f64_s (f64.nearest (local.get $value))))
              )
              (else
                (i32.store8 (i32.add (local.get $dst) (local.get $pos)) (i32.trunc_sat_f64_s (f64.add (local.get $value) (f64.const 0.5))))
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

  (func $cv_gaussian_blur (export "cv_gaussian_blur")
    (param $src i32) (param $dst i32) (param $mean i32) (param $width i32) (param $height i32) (param $kernel_size i32)
    (call $cv_gaussian_kernel (local.get $kernel_size) (i32.const 3072))
    (call $cv_gaussian_blur_filter (local.get $src) (local.get $mean) (local.get $width) (local.get $height) (i32.const 3072) (local.get $kernel_size) (i32.const 1))
    (call $cv_gaussian_blur_filter (local.get $mean) (local.get $dst) (local.get $width) (local.get $height) (i32.const 3072) (local.get $kernel_size) (i32.const 0))
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
  ;; Section 3: Singular Value Decomposition (SVD)
  ;; ============================================================================

  (func $svd_pythag (export "svd_pythag") (param $a f64) (param $b f64) (result f64)
    (local $at f64) (local $bt f64) (local $ct f64)
    (local.set $at (f64.abs (local.get $a)))
    (local.set $bt (f64.abs (local.get $b)))
    (if (result f64) (f64.gt (local.get $at) (local.get $bt))
      (then
        (local.set $ct (f64.div (local.get $bt) (local.get $at)))
        (f64.mul (local.get $at) (f64.sqrt (f64.add (f64.const 1.0) (f64.mul (local.get $ct) (local.get $ct)))))
      )
      (else
        (if (result f64) (f64.eq (local.get $bt) (f64.const 0.0))
          (then (f64.const 0.0))
          (else
            (local.set $ct (f64.div (local.get $at) (local.get $bt)))
            (f64.mul (local.get $bt) (f64.sqrt (f64.add (f64.const 1.0) (f64.mul (local.get $ct) (local.get $ct)))))
          )
        )
      )
    )
  )

  (func $svd_sign (export "svd_sign") (param $a f64) (param $b f64) (result f64)
    (select
      (f64.abs (local.get $a))
      (f64.neg (f64.abs (local.get $a)))
      (f64.ge (local.get $b) (f64.const 0.0))
    )
  )

  ;; SVD.svdcmp: Operates on flat f64 arrays:
  ;; $a: m * n * 8 bytes, $w: n * 8 bytes, $v: n * n * 8 bytes
  (func $svd_svdcmp (export "svd_svdcmp")
    (param $a i32) (param $m i32) (param $n i32) (param $w i32) (param $v i32)
    (result i32)
    (local $flag i32) (local $i i32) (local $its i32) (local $j i32) (local $jj i32)
    (local $k i32) (local $l i32) (local $nm i32) (local $min_mn i32)
    (local $anorm f64) (local $c f64) (local $f f64) (local $g f64) (local $h f64)
    (local $s f64) (local $scale f64) (local $x f64) (local $y f64) (local $z f64)
    (local $rv1 i32) (local $ptr1 i32) (local $ptr2 i32)

    ;; rv1 scratch at 0x1000 (4096)
    (local.set $rv1 (i32.const 4096))
    (local.set $g (f64.const 0.0))
    (local.set $scale (f64.const 0.0))
    (local.set $anorm (f64.const 0.0))

    ;; 1. Householder reduction to bidiagonal form
    (local.set $i (i32.const 0))
    (block $h_done
      (loop $h_loop
        (br_if $h_done (i32.ge_s (local.get $i) (local.get $n)))
        (local.set $l (i32.add (local.get $i) (i32.const 1)))
        (f64.store (i32.add (local.get $rv1) (i32.shl (local.get $i) (i32.const 3))) (f64.mul (local.get $scale) (local.get $g)))
        (local.set $g (f64.const 0.0))
        (local.set $s (f64.const 0.0))
        (local.set $scale (f64.const 0.0))

        (if (i32.lt_s (local.get $i) (local.get $m))
          (then
            (local.set $k (local.get $i))
            (block $k1_done
              (loop $k1_loop
                (br_if $k1_done (i32.ge_s (local.get $k) (local.get $m)))
                (local.set $scale
                  (f64.add
                    (local.get $scale)
                    (f64.abs (f64.load (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $k) (local.get $n)) (local.get $i)) (i32.const 3)))))
                  )
                )
                (local.set $k (i32.add (local.get $k) (i32.const 1)))
                (br $k1_loop)
              )
            )
            (if (f64.ne (local.get $scale) (f64.const 0.0))
              (then
                (local.set $k (local.get $i))
                (block $k2_done
                  (loop $k2_loop
                    (br_if $k2_done (i32.ge_s (local.get $k) (local.get $m)))
                    (local.set $ptr1 (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $k) (local.get $n)) (local.get $i)) (i32.const 3))))
                    (local.set $x (f64.div (f64.load (local.get $ptr1)) (local.get $scale)))
                    (f64.store (local.get $ptr1) (local.get $x))
                    (local.set $s (f64.add (local.get $s) (f64.mul (local.get $x) (local.get $x))))
                    (local.set $k (i32.add (local.get $k) (i32.const 1)))
                    (br $k2_loop)
                  )
                )
                (local.set $ptr1 (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $n)) (local.get $i)) (i32.const 3))))
                (local.set $f (f64.load (local.get $ptr1)))
                (local.set $g (f64.neg (call $svd_sign (f64.sqrt (local.get $s)) (local.get $f))))
                (local.set $h (f64.sub (f64.mul (local.get $f) (local.get $g)) (local.get $s)))
                (f64.store (local.get $ptr1) (f64.sub (local.get $f) (local.get $g)))

                (local.set $j (local.get $l))
                (block $j1_done
                  (loop $j1_loop
                    (br_if $j1_done (i32.ge_s (local.get $j) (local.get $n)))
                    (local.set $s (f64.const 0.0))
                    (local.set $k (local.get $i))
                    (block $k3_done
                      (loop $k3_loop
                        (br_if $k3_done (i32.ge_s (local.get $k) (local.get $m)))
                        (local.set $s
                          (f64.add
                            (local.get $s)
                            (f64.mul
                              (f64.load (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $k) (local.get $n)) (local.get $i)) (i32.const 3))))
                              (f64.load (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $k) (local.get $n)) (local.get $j)) (i32.const 3))))
                            )
                          )
                        )
                        (local.set $k (i32.add (local.get $k) (i32.const 1)))
                        (br $k3_loop)
                      )
                    )
                    (local.set $f (f64.div (local.get $s) (local.get $h)))
                    (local.set $k (local.get $i))
                    (block $k4_done
                      (loop $k4_loop
                        (br_if $k4_done (i32.ge_s (local.get $k) (local.get $m)))
                        (local.set $ptr2 (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $k) (local.get $n)) (local.get $j)) (i32.const 3))))
                        (f64.store
                          (local.get $ptr2)
                          (f64.add
                            (f64.load (local.get $ptr2))
                            (f64.mul (local.get $f) (f64.load (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $k) (local.get $n)) (local.get $i)) (i32.const 3)))))
                          )
                        )
                        (local.set $k (i32.add (local.get $k) (i32.const 1)))
                        (br $k4_loop)
                      )
                    )
                    (local.set $j (i32.add (local.get $j) (i32.const 1)))
                    (br $j1_loop)
                  )
                )
                (local.set $k (local.get $i))
                (block $k5_done
                  (loop $k5_loop
                    (br_if $k5_done (i32.ge_s (local.get $k) (local.get $m)))
                    (local.set $ptr1 (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $k) (local.get $n)) (local.get $i)) (i32.const 3))))
                    (f64.store (local.get $ptr1) (f64.mul (f64.load (local.get $ptr1)) (local.get $scale)))
                    (local.set $k (i32.add (local.get $k) (i32.const 1)))
                    (br $k5_loop)
                  )
                )
              )
            )
          )
        )

        (f64.store (i32.add (local.get $w) (i32.shl (local.get $i) (i32.const 3))) (f64.mul (local.get $scale) (local.get $g)))
        (local.set $g (f64.const 0.0))
        (local.set $s (f64.const 0.0))
        (local.set $scale (f64.const 0.0))

        (if (i32.and (i32.lt_s (local.get $i) (local.get $m)) (i32.ne (local.get $i) (i32.sub (local.get $n) (i32.const 1))))
          (then
            (local.set $k (local.get $l))
            (block $rk1_done
              (loop $rk1_loop
                (br_if $rk1_done (i32.ge_s (local.get $k) (local.get $n)))
                (local.set $scale
                  (f64.add
                    (local.get $scale)
                    (f64.abs (f64.load (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $n)) (local.get $k)) (i32.const 3)))))
                  )
                )
                (local.set $k (i32.add (local.get $k) (i32.const 1)))
                (br $rk1_loop)
              )
            )
            (if (f64.ne (local.get $scale) (f64.const 0.0))
              (then
                (local.set $k (local.get $l))
                (block $rk2_done
                  (loop $rk2_loop
                    (br_if $rk2_done (i32.ge_s (local.get $k) (local.get $n)))
                    (local.set $ptr1 (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $n)) (local.get $k)) (i32.const 3))))
                    (local.set $x (f64.div (f64.load (local.get $ptr1)) (local.get $scale)))
                    (f64.store (local.get $ptr1) (local.get $x))
                    (local.set $s (f64.add (local.get $s) (f64.mul (local.get $x) (local.get $x))))
                    (local.set $k (i32.add (local.get $k) (i32.const 1)))
                    (br $rk2_loop)
                  )
                )
                (local.set $ptr1 (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $n)) (local.get $l)) (i32.const 3))))
                (local.set $f (f64.load (local.get $ptr1)))
                (local.set $g (f64.neg (call $svd_sign (f64.sqrt (local.get $s)) (local.get $f))))
                (local.set $h (f64.sub (f64.mul (local.get $f) (local.get $g)) (local.get $s)))
                (f64.store (local.get $ptr1) (f64.sub (local.get $f) (local.get $g)))

                (local.set $k (local.get $l))
                (block $rk3_done
                  (loop $rk3_loop
                    (br_if $rk3_done (i32.ge_s (local.get $k) (local.get $n)))
                    (f64.store
                      (i32.add (local.get $rv1) (i32.shl (local.get $k) (i32.const 3)))
                      (f64.div (f64.load (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $n)) (local.get $k)) (i32.const 3)))) (local.get $h))
                    )
                    (local.set $k (i32.add (local.get $k) (i32.const 1)))
                    (br $rk3_loop)
                  )
                )

                (local.set $j (local.get $l))
                (block $rj1_done
                  (loop $rj1_loop
                    (br_if $rj1_done (i32.ge_s (local.get $j) (local.get $m)))
                    (local.set $s (f64.const 0.0))
                    (local.set $k (local.get $l))
                    (block $rk4_done
                      (loop $rk4_loop
                        (br_if $rk4_done (i32.ge_s (local.get $k) (local.get $n)))
                        (local.set $s
                          (f64.add
                            (local.get $s)
                            (f64.mul
                              (f64.load (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $j) (local.get $n)) (local.get $k)) (i32.const 3))))
                              (f64.load (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $n)) (local.get $k)) (i32.const 3))))
                            )
                          )
                        )
                        (local.set $k (i32.add (local.get $k) (i32.const 1)))
                        (br $rk4_loop)
                      )
                    )
                    (local.set $k (local.get $l))
                    (block $rk5_done
                      (loop $rk5_loop
                        (br_if $rk5_done (i32.ge_s (local.get $k) (local.get $n)))
                        (local.set $ptr2 (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $j) (local.get $n)) (local.get $k)) (i32.const 3))))
                        (f64.store
                          (local.get $ptr2)
                          (f64.add (f64.load (local.get $ptr2)) (f64.mul (local.get $s) (f64.load (i32.add (local.get $rv1) (i32.shl (local.get $k) (i32.const 3))))))
                        )
                        (local.set $k (i32.add (local.get $k) (i32.const 1)))
                        (br $rk5_loop)
                      )
                    )
                    (local.set $j (i32.add (local.get $j) (i32.const 1)))
                    (br $rj1_loop)
                  )
                )

                (local.set $k (local.get $l))
                (block $rk6_done
                  (loop $rk6_loop
                    (br_if $rk6_done (i32.ge_s (local.get $k) (local.get $n)))
                    (local.set $ptr1 (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $n)) (local.get $k)) (i32.const 3))))
                    (f64.store (local.get $ptr1) (f64.mul (f64.load (local.get $ptr1)) (local.get $scale)))
                    (local.set $k (i32.add (local.get $k) (i32.const 1)))
                    (br $rk6_loop)
                  )
                )
              )
            )
          )
        )

        (local.set $anorm
          (f64.max
            (local.get $anorm)
            (f64.add
              (f64.abs (f64.load (i32.add (local.get $w) (i32.shl (local.get $i) (i32.const 3)))))
              (f64.abs (f64.load (i32.add (local.get $rv1) (i32.shl (local.get $i) (i32.const 3)))))
            )
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $h_loop)
      )
    )

    ;; 2. Accumulation of right-hand transformation
    (local.set $i (i32.sub (local.get $n) (i32.const 1)))
    (block $r_done
      (loop $r_loop
        (br_if $r_done (i32.lt_s (local.get $i) (i32.const 0)))
        (if (i32.lt_s (local.get $i) (i32.sub (local.get $n) (i32.const 1)))
          (then
            (if (f64.ne (local.get $g) (f64.const 0.0))
              (then
                (local.set $j (local.get $l))
                (block $rj2_done
                  (loop $rj2_loop
                    (br_if $rj2_done (i32.ge_s (local.get $j) (local.get $n)))
                    (f64.store
                      (i32.add (local.get $v) (i32.shl (i32.add (i32.mul (local.get $j) (local.get $n)) (local.get $i)) (i32.const 3)))
                      (f64.div
                        (f64.div
                          (f64.load (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $n)) (local.get $j)) (i32.const 3))))
                          (f64.load (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $n)) (local.get $l)) (i32.const 3))))
                        )
                        (local.get $g)
                      )
                    )
                    (local.set $j (i32.add (local.get $j) (i32.const 1)))
                    (br $rj2_loop)
                  )
                )
                (local.set $j (local.get $l))
                (block $rj3_done
                  (loop $rj3_loop
                    (br_if $rj3_done (i32.ge_s (local.get $j) (local.get $n)))
                    (local.set $s (f64.const 0.0))
                    (local.set $k (local.get $l))
                    (block $rk7_done
                      (loop $rk7_loop
                        (br_if $rk7_done (i32.ge_s (local.get $k) (local.get $n)))
                        (local.set $s
                          (f64.add
                            (local.get $s)
                            (f64.mul
                              (f64.load (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $n)) (local.get $k)) (i32.const 3))))
                              (f64.load (i32.add (local.get $v) (i32.shl (i32.add (i32.mul (local.get $k) (local.get $n)) (local.get $j)) (i32.const 3))))
                            )
                          )
                        )
                        (local.set $k (i32.add (local.get $k) (i32.const 1)))
                        (br $rk7_loop)
                      )
                    )
                    (local.set $k (local.get $l))
                    (block $rk8_done
                      (loop $rk8_loop
                        (br_if $rk8_done (i32.ge_s (local.get $k) (local.get $n)))
                        (local.set $ptr1 (i32.add (local.get $v) (i32.shl (i32.add (i32.mul (local.get $k) (local.get $n)) (local.get $j)) (i32.const 3))))
                        (f64.store
                          (local.get $ptr1)
                          (f64.add
                            (f64.load (local.get $ptr1))
                            (f64.mul (local.get $s) (f64.load (i32.add (local.get $v) (i32.shl (i32.add (i32.mul (local.get $k) (local.get $n)) (local.get $i)) (i32.const 3)))))
                          )
                        )
                        (local.set $k (i32.add (local.get $k) (i32.const 1)))
                        (br $rk8_loop)
                      )
                    )
                    (local.set $j (i32.add (local.get $j) (i32.const 1)))
                    (br $rj3_loop)
                  )
                )
              )
            )
            (local.set $j (local.get $l))
            (block $rj4_done
              (loop $rj4_loop
                (br_if $rj4_done (i32.ge_s (local.get $j) (local.get $n)))
                (f64.store (i32.add (local.get $v) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $n)) (local.get $j)) (i32.const 3))) (f64.const 0.0))
                (f64.store (i32.add (local.get $v) (i32.shl (i32.add (i32.mul (local.get $j) (local.get $n)) (local.get $i)) (i32.const 3))) (f64.const 0.0))
                (local.set $j (i32.add (local.get $j) (i32.const 1)))
                (br $rj4_loop)
              )
            )
          )
        )
        (f64.store (i32.add (local.get $v) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $n)) (local.get $i)) (i32.const 3))) (f64.const 1.0))
        (local.set $g (f64.load (i32.add (local.get $rv1) (i32.shl (local.get $i) (i32.const 3)))))
        (local.set $l (local.get $i))
        (local.set $i (i32.sub (local.get $i) (i32.const 1)))
        (br $r_loop)
      )
    )

    ;; 3. Accumulation of left-hand transformation
    (local.set $min_mn (select (local.get $n) (local.get $m) (i32.lt_s (local.get $n) (local.get $m))))
    (local.set $i (i32.sub (local.get $min_mn) (i32.const 1)))
    (block $l_done
      (loop $l_loop
        (br_if $l_done (i32.lt_s (local.get $i) (i32.const 0)))
        (local.set $l (i32.add (local.get $i) (i32.const 1)))
        (local.set $g (f64.load (i32.add (local.get $w) (i32.shl (local.get $i) (i32.const 3)))))

        (local.set $j (local.get $l))
        (block $lj1_done
          (loop $lj1_loop
            (br_if $lj1_done (i32.ge_s (local.get $j) (local.get $n)))
            (f64.store (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $n)) (local.get $j)) (i32.const 3))) (f64.const 0.0))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $lj1_loop)
          )
        )

        (if (f64.ne (local.get $g) (f64.const 0.0))
          (then
            (local.set $g (f64.div (f64.const 1.0) (local.get $g)))
            (local.set $j (local.get $l))
            (block $lj2_done
              (loop $lj2_loop
                (br_if $lj2_done (i32.ge_s (local.get $j) (local.get $n)))
                (local.set $s (f64.const 0.0))
                (local.set $k (local.get $l))
                (block $lk1_done
                  (loop $lk1_loop
                    (br_if $lk1_done (i32.ge_s (local.get $k) (local.get $m)))
                    (local.set $s
                      (f64.add
                        (local.get $s)
                        (f64.mul
                          (f64.load (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $k) (local.get $n)) (local.get $i)) (i32.const 3))))
                          (f64.load (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $k) (local.get $n)) (local.get $j)) (i32.const 3))))
                        )
                      )
                    )
                    (local.set $k (i32.add (local.get $k) (i32.const 1)))
                    (br $lk1_loop)
                  )
                )
                (local.set $f
                  (f64.mul
                    (f64.div (local.get $s) (f64.load (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $n)) (local.get $i)) (i32.const 3)))))
                    (local.get $g)
                  )
                )
                (local.set $k (local.get $i))
                (block $lk2_done
                  (loop $lk2_loop
                    (br_if $lk2_done (i32.ge_s (local.get $k) (local.get $m)))
                    (local.set $ptr1 (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $k) (local.get $n)) (local.get $j)) (i32.const 3))))
                    (f64.store
                      (local.get $ptr1)
                      (f64.add
                        (f64.load (local.get $ptr1))
                        (f64.mul (local.get $f) (f64.load (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $k) (local.get $n)) (local.get $i)) (i32.const 3)))))
                      )
                    )
                    (local.set $k (i32.add (local.get $k) (i32.const 1)))
                    (br $lk2_loop)
                  )
                )
                (local.set $j (i32.add (local.get $j) (i32.const 1)))
                (br $lj2_loop)
              )
            )
            (local.set $j (local.get $i))
            (block $lj3_done
              (loop $lj3_loop
                (br_if $lj3_done (i32.ge_s (local.get $j) (local.get $m)))
                (local.set $ptr1 (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $j) (local.get $n)) (local.get $i)) (i32.const 3))))
                (f64.store (local.get $ptr1) (f64.mul (f64.load (local.get $ptr1)) (local.get $g)))
                (local.set $j (i32.add (local.get $j) (i32.const 1)))
                (br $lj3_loop)
              )
            )
          )
          (else
            (local.set $j (local.get $i))
            (block $lj4_done
              (loop $lj4_loop
                (br_if $lj4_done (i32.ge_s (local.get $j) (local.get $m)))
                (f64.store (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $j) (local.get $n)) (local.get $i)) (i32.const 3))) (f64.const 0.0))
                (local.set $j (i32.add (local.get $j) (i32.const 1)))
                (br $lj4_loop)
              )
            )
          )
        )
        (local.set $ptr1 (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $n)) (local.get $i)) (i32.const 3))))
        (f64.store (local.get $ptr1) (f64.add (f64.load (local.get $ptr1)) (f64.const 1.0)))
        (local.set $i (i32.sub (local.get $i) (i32.const 1)))
        (br $l_loop)
      )
    )

    ;; 4. Diagonalization of the bidiagonal form
    (local.set $k (i32.sub (local.get $n) (i32.const 1)))
    (block $d_done
      (loop $d_loop
        (br_if $d_done (i32.lt_s (local.get $k) (i32.const 0)))
        (local.set $its (i32.const 1))
        (block $its_done
          (loop $its_loop
            (br_if $its_done (i32.gt_s (local.get $its) (i32.const 30)))
            (local.set $flag (i32.const 1))
            (local.set $l (local.get $k))
            (block $split_done
              (loop $split_loop
                (br_if $split_done (i32.lt_s (local.get $l) (i32.const 0)))
                (local.set $nm (i32.sub (local.get $l) (i32.const 1)))
                (if (f64.eq (f64.add (f64.abs (f64.load (i32.add (local.get $rv1) (i32.shl (local.get $l) (i32.const 3))))) (local.get $anorm)) (local.get $anorm))
                  (then
                    (local.set $flag (i32.const 0))
                    (br $split_done)
                  )
                )
                (if (f64.eq (f64.add (f64.abs (f64.load (i32.add (local.get $w) (i32.shl (local.get $nm) (i32.const 3))))) (local.get $anorm)) (local.get $anorm))
                  (then (br $split_done))
                )
                (local.set $l (i32.sub (local.get $l) (i32.const 1)))
                (br $split_loop)
              )
            )

            (if (local.get $flag)
              (then
                (local.set $c (f64.const 0.0))
                (local.set $s (f64.const 1.0))
                (local.set $i (local.get $l))
                (block $f_done
                  (loop $f_loop
                    (br_if $f_done (i32.gt_s (local.get $i) (local.get $k)))
                    (local.set $f (f64.mul (local.get $s) (f64.load (i32.add (local.get $rv1) (i32.shl (local.get $i) (i32.const 3))))))
                    (br_if $f_done (f64.eq (f64.add (f64.abs (local.get $f)) (local.get $anorm)) (local.get $anorm)))
                    (local.set $g (f64.load (i32.add (local.get $w) (i32.shl (local.get $i) (i32.const 3)))))
                    (local.set $h (call $svd_pythag (local.get $f) (local.get $g)))
                    (f64.store (i32.add (local.get $w) (i32.shl (local.get $i) (i32.const 3))) (local.get $h))
                    (local.set $h (f64.div (f64.const 1.0) (local.get $h)))
                    (local.set $c (f64.mul (local.get $g) (local.get $h)))
                    (local.set $s (f64.mul (f64.neg (local.get $f)) (local.get $h)))

                    (local.set $j (i32.const 0))
                    (block $fj_done
                      (loop $fj_loop
                        (br_if $fj_done (i32.ge_s (local.get $j) (local.get $m)))
                        (local.set $ptr1 (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $j) (local.get $n)) (local.get $nm)) (i32.const 3))))
                        (local.set $ptr2 (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $j) (local.get $n)) (local.get $i)) (i32.const 3))))
                        (local.set $y (f64.load (local.get $ptr1)))
                        (local.set $z (f64.load (local.get $ptr2)))
                        (f64.store (local.get $ptr1) (f64.add (f64.mul (local.get $y) (local.get $c)) (f64.mul (local.get $z) (local.get $s))))
                        (f64.store (local.get $ptr2) (f64.sub (f64.mul (local.get $z) (local.get $c)) (f64.mul (local.get $y) (local.get $s))))
                        (local.set $j (i32.add (local.get $j) (i32.const 1)))
                        (br $fj_loop)
                      )
                    )
                    (local.set $i (i32.add (local.get $i) (i32.const 1)))
                    (br $f_loop)
                  )
                )
              )
            )

            (local.set $z (f64.load (i32.add (local.get $w) (i32.shl (local.get $k) (i32.const 3)))))
            (if (i32.eq (local.get $l) (local.get $k))
              (then
                (if (f64.lt (local.get $z) (f64.const 0.0))
                  (then
                    (f64.store (i32.add (local.get $w) (i32.shl (local.get $k) (i32.const 3))) (f64.neg (local.get $z)))
                    (local.set $j (i32.const 0))
                    (block $vj_done
                      (loop $vj_loop
                        (br_if $vj_done (i32.ge_s (local.get $j) (local.get $n)))
                        (local.set $ptr1 (i32.add (local.get $v) (i32.shl (i32.add (i32.mul (local.get $j) (local.get $n)) (local.get $k)) (i32.const 3))))
                        (f64.store (local.get $ptr1) (f64.neg (f64.load (local.get $ptr1))))
                        (local.set $j (i32.add (local.get $j) (i32.const 1)))
                        (br $vj_loop)
                      )
                    )
                  )
                )
                (br $its_done)
              )
            )

            (if (i32.eq (local.get $its) (i32.const 30))
              (then (return (i32.const 0)))
            )

            ;; Shift from bottom 2-by-2 minor
            (local.set $x (f64.load (i32.add (local.get $w) (i32.shl (local.get $l) (i32.const 3)))))
            (local.set $nm (i32.sub (local.get $k) (i32.const 1)))
            (local.set $y (f64.load (i32.add (local.get $w) (i32.shl (local.get $nm) (i32.const 3)))))
            (local.set $g (f64.load (i32.add (local.get $rv1) (i32.shl (local.get $nm) (i32.const 3)))))
            (local.set $h (f64.load (i32.add (local.get $rv1) (i32.shl (local.get $k) (i32.const 3)))))
            (local.set $f
              (f64.div
                (f64.add
                  (f64.mul (f64.sub (local.get $y) (local.get $z)) (f64.add (local.get $y) (local.get $z)))
                  (f64.mul (f64.sub (local.get $g) (local.get $h)) (f64.add (local.get $g) (local.get $h)))
                )
                (f64.mul (f64.const 2.0) (f64.mul (local.get $h) (local.get $y)))
              )
            )
            (local.set $g (call $svd_pythag (local.get $f) (f64.const 1.0)))
            (local.set $f
              (f64.div
                (f64.add
                  (f64.mul (f64.sub (local.get $x) (local.get $z)) (f64.add (local.get $x) (local.get $z)))
                  (f64.mul (local.get $h) (f64.sub (f64.div (local.get $y) (f64.add (local.get $f) (call $svd_sign (local.get $g) (local.get $f)))) (local.get $h)))
                )
                (local.get $x)
              )
            )

            ;; Next QR transformation
            (local.set $c (f64.const 1.0))
            (local.set $s (f64.const 1.0))
            (local.set $j (local.get $l))
            (block $qr_done
              (loop $qr_loop
                (br_if $qr_done (i32.gt_s (local.get $j) (local.get $nm)))
                (local.set $i (i32.add (local.get $j) (i32.const 1)))
                (local.set $g (f64.load (i32.add (local.get $rv1) (i32.shl (local.get $i) (i32.const 3)))))
                (local.set $y (f64.load (i32.add (local.get $w) (i32.shl (local.get $i) (i32.const 3)))))
                (local.set $h (f64.mul (local.get $s) (local.get $g)))
                (local.set $g (f64.mul (local.get $c) (local.get $g)))
                (local.set $z (call $svd_pythag (local.get $f) (local.get $h)))
                (f64.store (i32.add (local.get $rv1) (i32.shl (local.get $j) (i32.const 3))) (local.get $z))
                (local.set $c (f64.div (local.get $f) (local.get $z)))
                (local.set $s (f64.div (local.get $h) (local.get $z)))
                (local.set $f (f64.add (f64.mul (local.get $x) (local.get $c)) (f64.mul (local.get $g) (local.get $s))))
                (local.set $g (f64.sub (f64.mul (local.get $g) (local.get $c)) (f64.mul (local.get $x) (local.get $s))))
                (local.set $h (f64.mul (local.get $y) (local.get $s)))
                (local.set $y (f64.mul (local.get $y) (local.get $c)))

                (local.set $jj (i32.const 0))
                (block $qr1_done
                  (loop $qr1_loop
                    (br_if $qr1_done (i32.ge_s (local.get $jj) (local.get $n)))
                    (local.set $ptr1 (i32.add (local.get $v) (i32.shl (i32.add (i32.mul (local.get $jj) (local.get $n)) (local.get $j)) (i32.const 3))))
                    (local.set $ptr2 (i32.add (local.get $v) (i32.shl (i32.add (i32.mul (local.get $jj) (local.get $n)) (local.get $i)) (i32.const 3))))
                    (local.set $x (f64.load (local.get $ptr1)))
                    (local.set $z (f64.load (local.get $ptr2)))
                    (f64.store (local.get $ptr1) (f64.add (f64.mul (local.get $x) (local.get $c)) (f64.mul (local.get $z) (local.get $s))))
                    (f64.store (local.get $ptr2) (f64.sub (f64.mul (local.get $z) (local.get $c)) (f64.mul (local.get $x) (local.get $s))))
                    (local.set $jj (i32.add (local.get $jj) (i32.const 1)))
                    (br $qr1_loop)
                  )
                )

                (local.set $z (call $svd_pythag (local.get $f) (local.get $h)))
                (f64.store (i32.add (local.get $w) (i32.shl (local.get $j) (i32.const 3))) (local.get $z))
                (if (f64.ne (local.get $z) (f64.const 0.0))
                  (then
                    (local.set $z (f64.div (f64.const 1.0) (local.get $z)))
                    (local.set $c (f64.mul (local.get $f) (local.get $z)))
                    (local.set $s (f64.mul (local.get $h) (local.get $z)))
                  )
                )
                (local.set $f (f64.add (f64.mul (local.get $c) (local.get $g)) (f64.mul (local.get $s) (local.get $y))))
                (local.set $x (f64.sub (f64.mul (local.get $c) (local.get $y)) (f64.mul (local.get $s) (local.get $g))))

                (local.set $jj (i32.const 0))
                (block $qr2_done
                  (loop $qr2_loop
                    (br_if $qr2_done (i32.ge_s (local.get $jj) (local.get $m)))
                    (local.set $ptr1 (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $jj) (local.get $n)) (local.get $j)) (i32.const 3))))
                    (local.set $ptr2 (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $jj) (local.get $n)) (local.get $i)) (i32.const 3))))
                    (local.set $y (f64.load (local.get $ptr1)))
                    (local.set $z (f64.load (local.get $ptr2)))
                    (f64.store (local.get $ptr1) (f64.add (f64.mul (local.get $y) (local.get $c)) (f64.mul (local.get $z) (local.get $s))))
                    (f64.store (local.get $ptr2) (f64.sub (f64.mul (local.get $z) (local.get $c)) (f64.mul (local.get $y) (local.get $s))))
                    (local.set $jj (i32.add (local.get $jj) (i32.const 1)))
                    (br $qr2_loop)
                  )
                )
                (local.set $j (i32.add (local.get $j) (i32.const 1)))
                (br $qr_loop)
              )
            )
            (f64.store (i32.add (local.get $rv1) (i32.shl (local.get $l) (i32.const 3))) (f64.const 0.0))
            (f64.store (i32.add (local.get $rv1) (i32.shl (local.get $k) (i32.const 3))) (local.get $f))
            (f64.store (i32.add (local.get $w) (i32.shl (local.get $k) (i32.const 3))) (local.get $x))

            (local.set $its (i32.add (local.get $its) (i32.const 1)))
            (br $its_loop)
          )
        )
        (local.set $k (i32.sub (local.get $k) (i32.const 1)))
        (br $d_loop)
      )
    )
    (i32.const 1)
  )

  ;; ============================================================================
  ;; Section 4: POSIT 3D Pose Estimation (Float64 Precision)
  ;; ============================================================================
  ;; Posit State Layout at $posit_ptr (272 bytes):
  ;;   0..95    : model (4 x 3 x f64)
  ;;   96..103  : focalLength (f64)
  ;;   104..175 : modelVectors (9 x f64)
  ;;   176..199 : modelNormal (3 x f64)
  ;;   200..271 : modelPseudoInverse (9 x f64)

  (func $f64_mat3_mult_vec (param $m i32) (param $vx f64) (param $vy f64) (param $vz f64) (param $out i32)
    (f64.store (local.get $out)
      (f64.add (f64.add (f64.mul (f64.load (local.get $m)) (local.get $vx)) (f64.mul (f64.load offset=8 (local.get $m)) (local.get $vy))) (f64.mul (f64.load offset=16 (local.get $m)) (local.get $vz))))
    (f64.store offset=8 (local.get $out)
      (f64.add (f64.add (f64.mul (f64.load offset=24 (local.get $m)) (local.get $vx)) (f64.mul (f64.load offset=32 (local.get $m)) (local.get $vy))) (f64.mul (f64.load offset=40 (local.get $m)) (local.get $vz))))
    (f64.store offset=16 (local.get $out)
      (f64.add (f64.add (f64.mul (f64.load offset=48 (local.get $m)) (local.get $vx)) (f64.mul (f64.load offset=56 (local.get $m)) (local.get $vy))) (f64.mul (f64.load offset=64 (local.get $m)) (local.get $vz))))
  )

  (func $posit_angle (export "posit_angle")
    (param $ax f64) (param $ay f64) (param $bx f64) (param $by f64) (param $cx f64) (param $cy f64)
    (result f64)
    (local $x1 f64) (local $y1 f64) (local $x2 f64) (local $y2 f64)
    (local.set $x1 (f64.sub (local.get $bx) (local.get $ax)))
    (local.set $y1 (f64.sub (local.get $by) (local.get $ay)))
    (local.set $x2 (f64.sub (local.get $cx) (local.get $ax)))
    (local.set $y2 (f64.sub (local.get $cy) (local.get $ay)))
    (f64.div
      (f64.mul
        (call $acos
          (f64.div
            (f64.add (f64.mul (local.get $x1) (local.get $x2)) (f64.mul (local.get $y1) (local.get $y2)))
            (f64.mul
              (f64.sqrt (f64.add (f64.mul (local.get $x1) (local.get $x1)) (f64.mul (local.get $y1) (local.get $y1))))
              (f64.sqrt (f64.add (f64.mul (local.get $x2) (local.get $x2)) (f64.mul (local.get $y2) (local.get $y2))))
            )
          )
        )
        (f64.const 180.0)
      )
      (f64.const 3.141592653589793)
    )
  )

  (func $posit_init (export "posit_init") (param $posit_ptr i32) (param $model_size f64) (param $focal_length f64)
    (local $half f64) (local $u i32) (local $d i32) (local $v i32)
    (local $d0 f64) (local $d1 f64) (local $d2 f64)
    (local $inv0 f64) (local $inv1 f64) (local $inv2 f64)
    (local $min_idx i32) (local $i i32) (local $j i32)

    (local.set $half (f64.mul (local.get $model_size) (f64.const 0.5)))

    ;; model[0]: [-half, half, 0]
    (f64.store (local.get $posit_ptr) (f64.neg (local.get $half)))
    (f64.store offset=8 (local.get $posit_ptr) (local.get $half))
    (f64.store offset=16 (local.get $posit_ptr) (f64.const 0.0))
    ;; model[1]: [half, half, 0]
    (f64.store offset=24 (local.get $posit_ptr) (local.get $half))
    (f64.store offset=32 (local.get $posit_ptr) (local.get $half))
    (f64.store offset=40 (local.get $posit_ptr) (f64.const 0.0))
    ;; model[2]: [half, -half, 0]
    (f64.store offset=48 (local.get $posit_ptr) (local.get $half))
    (f64.store offset=56 (local.get $posit_ptr) (f64.neg (local.get $half)))
    (f64.store offset=64 (local.get $posit_ptr) (f64.const 0.0))
    ;; model[3]: [-half, -half, 0]
    (f64.store offset=72 (local.get $posit_ptr) (f64.neg (local.get $half)))
    (f64.store offset=80 (local.get $posit_ptr) (f64.neg (local.get $half)))
    (f64.store offset=88 (local.get $posit_ptr) (f64.const 0.0))

    (f64.store offset=96 (local.get $posit_ptr) (local.get $focal_length))

    ;; modelVectors (rows: model[1]-model[0], model[2]-model[0], model[3]-model[0])
    (f64.store offset=104 (local.get $posit_ptr) (f64.sub (f64.load offset=24 (local.get $posit_ptr)) (f64.load (local.get $posit_ptr))))
    (f64.store offset=112 (local.get $posit_ptr) (f64.sub (f64.load offset=32 (local.get $posit_ptr)) (f64.load offset=8 (local.get $posit_ptr))))
    (f64.store offset=120 (local.get $posit_ptr) (f64.const 0.0))

    (f64.store offset=128 (local.get $posit_ptr) (f64.sub (f64.load offset=48 (local.get $posit_ptr)) (f64.load (local.get $posit_ptr))))
    (f64.store offset=136 (local.get $posit_ptr) (f64.sub (f64.load offset=56 (local.get $posit_ptr)) (f64.load offset=8 (local.get $posit_ptr))))
    (f64.store offset=144 (local.get $posit_ptr) (f64.const 0.0))

    (f64.store offset=152 (local.get $posit_ptr) (f64.sub (f64.load offset=72 (local.get $posit_ptr)) (f64.load (local.get $posit_ptr))))
    (f64.store offset=160 (local.get $posit_ptr) (f64.sub (f64.load offset=80 (local.get $posit_ptr)) (f64.load offset=8 (local.get $posit_ptr))))
    (f64.store offset=168 (local.get $posit_ptr) (f64.const 0.0))

    ;; Scratch for SVD: u at 4352, d at 4432, v at 4464
    (local.set $u (i32.const 4352))
    (local.set $d (i32.const 4432))
    (local.set $v (i32.const 4464))
    (memory.copy (local.get $u) (i32.add (local.get $posit_ptr) (i32.const 104)) (i32.const 72))

    (drop (call $svd_svdcmp (local.get $u) (i32.const 3) (i32.const 3) (local.get $d) (local.get $v)))

    (local.set $d0 (f64.load (local.get $d)))
    (local.set $d1 (f64.load offset=8 (local.get $d)))
    (local.set $d2 (f64.load offset=16 (local.get $d)))

    (local.set $inv0 (select (f64.div (f64.const 1.0) (local.get $d0)) (f64.const 0.0) (f64.ne (local.get $d0) (f64.const 0.0))))
    (local.set $inv1 (select (f64.div (f64.const 1.0) (local.get $d1)) (f64.const 0.0) (f64.ne (local.get $d1) (f64.const 0.0))))
    (local.set $inv2 (select (f64.div (f64.const 1.0) (local.get $d2)) (f64.const 0.0) (f64.ne (local.get $d2) (f64.const 0.0))))

    ;; modelPseudoInverse = v * diag(inv) * u^T
    ;; Entry (i, j) = v[i][0]*inv0*u[j][0] + v[i][1]*inv1*u[j][1] + v[i][2]*inv2*u[j][2]
    (local.set $i (i32.const 0))
    (block $pi_done
      (loop $pi_loop
        (br_if $pi_done (i32.ge_s (local.get $i) (i32.const 3)))
        (local.set $j (i32.const 0))
        (block $pj_done
          (loop $pj_loop
            (br_if $pj_done (i32.ge_s (local.get $j) (i32.const 3)))
            (f64.store
              (i32.add (i32.add (local.get $posit_ptr) (i32.const 200)) (i32.shl (i32.add (i32.mul (local.get $i) (i32.const 3)) (local.get $j)) (i32.const 3)))
              (f64.add
                (f64.add
                  (f64.mul
                    (f64.mul (f64.load (i32.add (local.get $v) (i32.shl (i32.mul (local.get $i) (i32.const 3)) (i32.const 3)))) (local.get $inv0))
                    (f64.load (i32.add (local.get $u) (i32.shl (i32.mul (local.get $j) (i32.const 3)) (i32.const 3))))
                  )
                  (f64.mul
                    (f64.mul (f64.load offset=8 (i32.add (local.get $v) (i32.shl (i32.mul (local.get $i) (i32.const 3)) (i32.const 3)))) (local.get $inv1))
                    (f64.load offset=8 (i32.add (local.get $u) (i32.shl (i32.mul (local.get $j) (i32.const 3)) (i32.const 3))))
                  )
                )
                (f64.mul
                  (f64.mul (f64.load offset=16 (i32.add (local.get $v) (i32.shl (i32.mul (local.get $i) (i32.const 3)) (i32.const 3)))) (local.get $inv2))
                  (f64.load offset=16 (i32.add (local.get $u) (i32.shl (i32.mul (local.get $j) (i32.const 3)) (i32.const 3))))
                )
              )
            )
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $pj_loop)
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $pi_loop)
      )
    )

    ;; modelNormal = v.column(d.minIndex())
    (local.set $min_idx
      (if (result i32) (f64.lt (local.get $d0) (local.get $d1))
        (then (select (i32.const 0) (i32.const 2) (f64.lt (local.get $d0) (local.get $d2))))
        (else (select (i32.const 1) (i32.const 2) (f64.lt (local.get $d1) (local.get $d2))))
      )
    )
    (f64.store offset=176 (local.get $posit_ptr) (f64.load (i32.add (local.get $v) (i32.shl (local.get $min_idx) (i32.const 3)))))
    (f64.store offset=184 (local.get $posit_ptr) (f64.load (i32.add (local.get $v) (i32.shl (i32.add (local.get $min_idx) (i32.const 3)) (i32.const 3)))))
    (f64.store offset=192 (local.get $posit_ptr) (f64.load (i32.add (local.get $v) (i32.shl (i32.add (local.get $min_idx) (i32.const 6)) (i32.const 3)))))
  )

  ;; Posit.pos
  (func $posit_pos (export "posit_pos")
    (param $posit_ptr i32) (param $pts_ptr i32) (param $eps_ptr i32)
    (param $r1_ptr i32) (param $r2_ptr i32) (param $t1_ptr i32) (param $t2_ptr i32)
    (local $p0x f64) (local $p0y f64) (local $xs0 f64) (local $xs1 f64) (local $xs2 f64)
    (local $ys0 f64) (local $ys1 f64) (local $ys2 f64)
    (local $i0_ptr i32) (local $j0_ptr i32)
    (local $i0x f64) (local $i0y f64) (local $i0z f64)
    (local $j0x f64) (local $j0y f64) (local $j0z f64)
    (local $mnx f64) (local $mny f64) (local $mnz f64)
    (local $s f64) (local $ij f64) (local $r f64) (local $theta f64) (local $lambda f64) (local $mu f64)
    (local $ix f64) (local $iy f64) (local $iz f64)
    (local $jx f64) (local $jy f64) (local $jz f64)
    (local $kx f64) (local $ky f64) (local $kz f64)
    (local $inorm f64) (local $jnorm f64) (local $scale f64) (local $fl f64)
    (local $tmp_ptr i32)

    (local.set $p0x (f64.load (local.get $pts_ptr)))
    (local.set $p0y (f64.load offset=8 (local.get $pts_ptr)))

    (local.set $xs0 (f64.sub (f64.mul (f64.load offset=16 (local.get $pts_ptr)) (f64.load (local.get $eps_ptr))) (local.get $p0x)))
    (local.set $xs1 (f64.sub (f64.mul (f64.load offset=32 (local.get $pts_ptr)) (f64.load offset=8 (local.get $eps_ptr))) (local.get $p0x)))
    (local.set $xs2 (f64.sub (f64.mul (f64.load offset=48 (local.get $pts_ptr)) (f64.load offset=16 (local.get $eps_ptr))) (local.get $p0x)))

    (local.set $ys0 (f64.sub (f64.mul (f64.load offset=24 (local.get $pts_ptr)) (f64.load (local.get $eps_ptr))) (local.get $p0y)))
    (local.set $ys1 (f64.sub (f64.mul (f64.load offset=40 (local.get $pts_ptr)) (f64.load offset=8 (local.get $eps_ptr))) (local.get $p0y)))
    (local.set $ys2 (f64.sub (f64.mul (f64.load offset=56 (local.get $pts_ptr)) (f64.load offset=16 (local.get $eps_ptr))) (local.get $p0y)))

    (local.set $i0_ptr (i32.const 4608))
    (local.set $j0_ptr (i32.const 4632))
    (local.set $tmp_ptr (i32.const 4656))

    (call $f64_mat3_mult_vec (i32.add (local.get $posit_ptr) (i32.const 200)) (local.get $xs0) (local.get $xs1) (local.get $xs2) (local.get $i0_ptr))
    (call $f64_mat3_mult_vec (i32.add (local.get $posit_ptr) (i32.const 200)) (local.get $ys0) (local.get $ys1) (local.get $ys2) (local.get $j0_ptr))

    (local.set $i0x (f64.load (local.get $i0_ptr)))
    (local.set $i0y (f64.load offset=8 (local.get $i0_ptr)))
    (local.set $i0z (f64.load offset=16 (local.get $i0_ptr)))
    (local.set $j0x (f64.load (local.get $j0_ptr)))
    (local.set $j0y (f64.load offset=8 (local.get $j0_ptr)))
    (local.set $j0z (f64.load offset=16 (local.get $j0_ptr)))

    (local.set $s
      (f64.sub
        (f64.add (f64.add (f64.mul (local.get $j0x) (local.get $j0x)) (f64.mul (local.get $j0y) (local.get $j0y))) (f64.mul (local.get $j0z) (local.get $j0z)))
        (f64.add (f64.add (f64.mul (local.get $i0x) (local.get $i0x)) (f64.mul (local.get $i0y) (local.get $i0y))) (f64.mul (local.get $i0z) (local.get $i0z)))
      )
    )
    (local.set $ij (f64.add (f64.add (f64.mul (local.get $i0x) (local.get $j0x)) (f64.mul (local.get $i0y) (local.get $j0y))) (f64.mul (local.get $i0z) (local.get $j0z))))

    (if (f64.eq (local.get $s) (f64.const 0.0))
      (then
        (local.set $r (f64.sqrt (f64.abs (f64.mul (f64.const 2.0) (local.get $ij)))))
        (local.set $theta
          (f64.mul
            (f64.const -1.5707963267948966)
            (if (result f64) (f64.lt (local.get $ij) (f64.const 0.0))
              (then (f64.const -1.0))
              (else (select (f64.const 1.0) (f64.const 0.0) (f64.gt (local.get $ij) (f64.const 0.0))))
            )
          )
        )
      )
      (else
        (local.set $r (f64.sqrt (f64.sqrt (f64.add (f64.mul (local.get $s) (local.get $s)) (f64.mul (f64.const 4.0) (f64.mul (local.get $ij) (local.get $ij)))))))
        (local.set $theta (call $atan (f64.div (f64.mul (f64.const -2.0) (local.get $ij)) (local.get $s))))
        (if (f64.lt (local.get $s) (f64.const 0.0))
          (then (local.set $theta (f64.add (local.get $theta) (f64.const 3.141592653589793))))
        )
        (local.set $theta (f64.mul (local.get $theta) (f64.const 0.5)))
      )
    )

    (local.set $lambda (f64.mul (local.get $r) (call $cos (local.get $theta))))
    (local.set $mu (f64.mul (local.get $r) (call $sin (local.get $theta))))

    (local.set $mnx (f64.load offset=176 (local.get $posit_ptr)))
    (local.set $mny (f64.load offset=184 (local.get $posit_ptr)))
    (local.set $mnz (f64.load offset=192 (local.get $posit_ptr)))
    (local.set $fl (f64.load offset=96 (local.get $posit_ptr)))

    ;; First possible rotation/translation
    (local.set $ix (f64.add (local.get $i0x) (f64.mul (local.get $mnx) (local.get $lambda))))
    (local.set $iy (f64.add (local.get $i0y) (f64.mul (local.get $mny) (local.get $lambda))))
    (local.set $iz (f64.add (local.get $i0z) (f64.mul (local.get $mnz) (local.get $lambda))))

    (local.set $jx (f64.add (local.get $j0x) (f64.mul (local.get $mnx) (local.get $mu))))
    (local.set $jy (f64.add (local.get $j0y) (f64.mul (local.get $mny) (local.get $mu))))
    (local.set $jz (f64.add (local.get $j0z) (f64.mul (local.get $mnz) (local.get $mu))))

    (local.set $inorm (f64.sqrt (f64.add (f64.add (f64.mul (local.get $ix) (local.get $ix)) (f64.mul (local.get $iy) (local.get $iy))) (f64.mul (local.get $iz) (local.get $iz)))))
    (if (f64.gt (local.get $inorm) (f64.const 0.0))
      (then
        (local.set $ix (f64.div (local.get $ix) (local.get $inorm)))
        (local.set $iy (f64.div (local.get $iy) (local.get $inorm)))
        (local.set $iz (f64.div (local.get $iz) (local.get $inorm)))
      )
    )
    (local.set $jnorm (f64.sqrt (f64.add (f64.add (f64.mul (local.get $jx) (local.get $jx)) (f64.mul (local.get $jy) (local.get $jy))) (f64.mul (local.get $jz) (local.get $jz)))))
    (if (f64.gt (local.get $jnorm) (f64.const 0.0))
      (then
        (local.set $jx (f64.div (local.get $jx) (local.get $jnorm)))
        (local.set $jy (f64.div (local.get $jy) (local.get $jnorm)))
        (local.set $jz (f64.div (local.get $jz) (local.get $jnorm)))
      )
    )

    (local.set $kx (f64.sub (f64.mul (local.get $iy) (local.get $jz)) (f64.mul (local.get $iz) (local.get $jy))))
    (local.set $ky (f64.sub (f64.mul (local.get $iz) (local.get $jx)) (f64.mul (local.get $ix) (local.get $jz))))
    (local.set $kz (f64.sub (f64.mul (local.get $ix) (local.get $jy)) (f64.mul (local.get $iy) (local.get $jx))))

    (f64.store (local.get $r1_ptr) (local.get $ix))
    (f64.store offset=8 (local.get $r1_ptr) (local.get $iy))
    (f64.store offset=16 (local.get $r1_ptr) (local.get $iz))
    (f64.store offset=24 (local.get $r1_ptr) (local.get $jx))
    (f64.store offset=32 (local.get $r1_ptr) (local.get $jy))
    (f64.store offset=40 (local.get $r1_ptr) (local.get $jz))
    (f64.store offset=48 (local.get $r1_ptr) (local.get $kx))
    (f64.store offset=56 (local.get $r1_ptr) (local.get $ky))
    (f64.store offset=64 (local.get $r1_ptr) (local.get $kz))

    (local.set $scale (f64.mul (f64.add (local.get $inorm) (local.get $jnorm)) (f64.const 0.5)))
    (call $f64_mat3_mult_vec (local.get $r1_ptr) (f64.load (local.get $posit_ptr)) (f64.load offset=8 (local.get $posit_ptr)) (f64.load offset=16 (local.get $posit_ptr)) (local.get $tmp_ptr))
    (f64.store (local.get $t1_ptr) (f64.sub (f64.div (local.get $p0x) (local.get $scale)) (f64.load (local.get $tmp_ptr))))
    (f64.store offset=8 (local.get $t1_ptr) (f64.sub (f64.div (local.get $p0y) (local.get $scale)) (f64.load offset=8 (local.get $tmp_ptr))))
    (f64.store offset=16 (local.get $t1_ptr) (f64.div (local.get $fl) (local.get $scale)))

    ;; Second possible rotation/translation
    (local.set $ix (f64.sub (local.get $i0x) (f64.mul (local.get $mnx) (local.get $lambda))))
    (local.set $iy (f64.sub (local.get $i0y) (f64.mul (local.get $mny) (local.get $lambda))))
    (local.set $iz (f64.sub (local.get $i0z) (f64.mul (local.get $mnz) (local.get $lambda))))

    (local.set $jx (f64.sub (local.get $j0x) (f64.mul (local.get $mnx) (local.get $mu))))
    (local.set $jy (f64.sub (local.get $j0y) (f64.mul (local.get $mny) (local.get $mu))))
    (local.set $jz (f64.sub (local.get $j0z) (f64.mul (local.get $mnz) (local.get $mu))))

    (local.set $inorm (f64.sqrt (f64.add (f64.add (f64.mul (local.get $ix) (local.get $ix)) (f64.mul (local.get $iy) (local.get $iy))) (f64.mul (local.get $iz) (local.get $iz)))))
    (if (f64.gt (local.get $inorm) (f64.const 0.0))
      (then
        (local.set $ix (f64.div (local.get $ix) (local.get $inorm)))
        (local.set $iy (f64.div (local.get $iy) (local.get $inorm)))
        (local.set $iz (f64.div (local.get $iz) (local.get $inorm)))
      )
    )
    (local.set $jnorm (f64.sqrt (f64.add (f64.add (f64.mul (local.get $jx) (local.get $jx)) (f64.mul (local.get $jy) (local.get $jy))) (f64.mul (local.get $jz) (local.get $jz)))))
    (if (f64.gt (local.get $jnorm) (f64.const 0.0))
      (then
        (local.set $jx (f64.div (local.get $jx) (local.get $jnorm)))
        (local.set $jy (f64.div (local.get $jy) (local.get $jnorm)))
        (local.set $jz (f64.div (local.get $jz) (local.get $jnorm)))
      )
    )

    (local.set $kx (f64.sub (f64.mul (local.get $iy) (local.get $jz)) (f64.mul (local.get $iz) (local.get $jy))))
    (local.set $ky (f64.sub (f64.mul (local.get $iz) (local.get $jx)) (f64.mul (local.get $ix) (local.get $jz))))
    (local.set $kz (f64.sub (f64.mul (local.get $ix) (local.get $jy)) (f64.mul (local.get $iy) (local.get $jx))))

    (f64.store (local.get $r2_ptr) (local.get $ix))
    (f64.store offset=8 (local.get $r2_ptr) (local.get $iy))
    (f64.store offset=16 (local.get $r2_ptr) (local.get $iz))
    (f64.store offset=24 (local.get $r2_ptr) (local.get $jx))
    (f64.store offset=32 (local.get $r2_ptr) (local.get $jy))
    (f64.store offset=40 (local.get $r2_ptr) (local.get $jz))
    (f64.store offset=48 (local.get $r2_ptr) (local.get $kx))
    (f64.store offset=56 (local.get $r2_ptr) (local.get $ky))
    (f64.store offset=64 (local.get $r2_ptr) (local.get $kz))

    (local.set $scale (f64.mul (f64.add (local.get $inorm) (local.get $jnorm)) (f64.const 0.5)))
    (call $f64_mat3_mult_vec (local.get $r2_ptr) (f64.load (local.get $posit_ptr)) (f64.load offset=8 (local.get $posit_ptr)) (f64.load offset=16 (local.get $posit_ptr)) (local.get $tmp_ptr))
    (f64.store (local.get $t2_ptr) (f64.sub (f64.div (local.get $p0x) (local.get $scale)) (f64.load (local.get $tmp_ptr))))
    (f64.store offset=8 (local.get $t2_ptr) (f64.sub (f64.div (local.get $p0y) (local.get $scale)) (f64.load offset=8 (local.get $tmp_ptr))))
    (f64.store offset=16 (local.get $t2_ptr) (f64.div (local.get $fl) (local.get $scale)))
  )

  ;; Posit.getError
  (func $posit_get_error (export "posit_get_error")
    (param $posit_ptr i32) (param $pts_ptr i32) (param $r_ptr i32) (param $t_ptr i32)
    (result f64)
    (local $fl f64) (local $tx f64) (local $ty f64) (local $tz f64)
    (local $vec_ptr i32) (local $mod_ptr i32) (local $i i32) (local $m_off i32)
    (local $vx f64) (local $vy f64) (local $vz f64)
    (local $ia1 f64) (local $ia2 f64) (local $ia3 f64) (local $ia4 f64)
    (local $ma1 f64) (local $ma2 f64) (local $ma3 f64) (local $ma4 f64)

    (local.set $fl (f64.load offset=96 (local.get $posit_ptr)))
    (local.set $tx (f64.load (local.get $t_ptr)))
    (local.set $ty (f64.load offset=8 (local.get $t_ptr)))
    (local.set $tz (f64.load offset=16 (local.get $t_ptr)))
    (local.set $vec_ptr (i32.const 4680))
    (local.set $mod_ptr (i32.const 4704))

    (local.set $i (i32.const 0))
    (block $done
      (loop $loop
        (br_if $done (i32.ge_s (local.get $i) (i32.const 4)))
        (local.set $m_off (i32.add (local.get $posit_ptr) (i32.mul (local.get $i) (i32.const 24))))
        (call $f64_mat3_mult_vec (local.get $r_ptr) (f64.load (local.get $m_off)) (f64.load offset=8 (local.get $m_off)) (f64.load offset=16 (local.get $m_off)) (local.get $vec_ptr))
        (local.set $vx (f64.add (f64.load (local.get $vec_ptr)) (local.get $tx)))
        (local.set $vy (f64.add (f64.load offset=8 (local.get $vec_ptr)) (local.get $ty)))
        (local.set $vz (f64.add (f64.load offset=16 (local.get $vec_ptr)) (local.get $tz)))

        (f64.store (i32.add (local.get $mod_ptr) (i32.shl (local.get $i) (i32.const 4))) (f64.div (f64.mul (local.get $vx) (local.get $fl)) (local.get $vz)))
        (f64.store offset=8 (i32.add (local.get $mod_ptr) (i32.shl (local.get $i) (i32.const 4))) (f64.div (f64.mul (local.get $vy) (local.get $fl)) (local.get $vz)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )

    (local.set $ia1 (call $posit_angle (f64.load (local.get $pts_ptr)) (f64.load offset=8 (local.get $pts_ptr)) (f64.load offset=16 (local.get $pts_ptr)) (f64.load offset=24 (local.get $pts_ptr)) (f64.load offset=48 (local.get $pts_ptr)) (f64.load offset=56 (local.get $pts_ptr))))
    (local.set $ia2 (call $posit_angle (f64.load offset=16 (local.get $pts_ptr)) (f64.load offset=24 (local.get $pts_ptr)) (f64.load offset=32 (local.get $pts_ptr)) (f64.load offset=40 (local.get $pts_ptr)) (f64.load (local.get $pts_ptr)) (f64.load offset=8 (local.get $pts_ptr))))
    (local.set $ia3 (call $posit_angle (f64.load offset=32 (local.get $pts_ptr)) (f64.load offset=40 (local.get $pts_ptr)) (f64.load offset=48 (local.get $pts_ptr)) (f64.load offset=56 (local.get $pts_ptr)) (f64.load offset=16 (local.get $pts_ptr)) (f64.load offset=24 (local.get $pts_ptr))))
    (local.set $ia4 (call $posit_angle (f64.load offset=48 (local.get $pts_ptr)) (f64.load offset=56 (local.get $pts_ptr)) (f64.load (local.get $pts_ptr)) (f64.load offset=8 (local.get $pts_ptr)) (f64.load offset=32 (local.get $pts_ptr)) (f64.load offset=40 (local.get $pts_ptr))))

    (local.set $ma1 (call $posit_angle (f64.load (local.get $mod_ptr)) (f64.load offset=8 (local.get $mod_ptr)) (f64.load offset=16 (local.get $mod_ptr)) (f64.load offset=24 (local.get $mod_ptr)) (f64.load offset=48 (local.get $mod_ptr)) (f64.load offset=56 (local.get $mod_ptr))))
    (local.set $ma2 (call $posit_angle (f64.load offset=16 (local.get $mod_ptr)) (f64.load offset=24 (local.get $mod_ptr)) (f64.load offset=32 (local.get $mod_ptr)) (f64.load offset=40 (local.get $mod_ptr)) (f64.load (local.get $mod_ptr)) (f64.load offset=8 (local.get $mod_ptr))))
    (local.set $ma3 (call $posit_angle (f64.load offset=32 (local.get $mod_ptr)) (f64.load offset=40 (local.get $mod_ptr)) (f64.load offset=48 (local.get $mod_ptr)) (f64.load offset=56 (local.get $mod_ptr)) (f64.load offset=16 (local.get $mod_ptr)) (f64.load offset=24 (local.get $mod_ptr))))
    (local.set $ma4 (call $posit_angle (f64.load offset=48 (local.get $mod_ptr)) (f64.load offset=56 (local.get $mod_ptr)) (f64.load (local.get $mod_ptr)) (f64.load offset=8 (local.get $mod_ptr)) (f64.load offset=32 (local.get $mod_ptr)) (f64.load offset=40 (local.get $mod_ptr))))

    (f64.mul
      (f64.add
        (f64.add (f64.abs (f64.sub (local.get $ia1) (local.get $ma1))) (f64.abs (f64.sub (local.get $ia2) (local.get $ma2))))
        (f64.add (f64.abs (f64.sub (local.get $ia3) (local.get $ma3))) (f64.abs (f64.sub (local.get $ia4) (local.get $ma4))))
      )
      (f64.const 0.25)
    )
  )

  ;; Posit.iterate
  (func $posit_iterate (export "posit_iterate")
    (param $posit_ptr i32) (param $pts_ptr i32) (param $r_ptr i32) (param $t_ptr i32)
    (result f64)
    (local $prev_error f64) (local $error f64) (local $error1 f64) (local $error2 f64)
    (local $i i32) (local $inv_tz f64)
    (local $eps_ptr i32) (local $r1_ptr i32) (local $r2_ptr i32) (local $t1_ptr i32) (local $t2_ptr i32)

    (local.set $prev_error (f64.const inf))
    (local.set $error (f64.const 0.0))
    (local.set $eps_ptr (i32.const 4800))
    (local.set $r1_ptr (i32.const 4832))
    (local.set $r2_ptr (i32.const 4904))
    (local.set $t1_ptr (i32.const 4976))
    (local.set $t2_ptr (i32.const 5000))

    (local.set $i (i32.const 0))
    (block $done
      (loop $loop
        (br_if $done (i32.ge_s (local.get $i) (i32.const 100)))
        ;; eps = modelVectors * rotation.row(2) / translation[2] + 1.0
        (call $f64_mat3_mult_vec
          (i32.add (local.get $posit_ptr) (i32.const 104))
          (f64.load offset=48 (local.get $r_ptr))
          (f64.load offset=56 (local.get $r_ptr))
          (f64.load offset=64 (local.get $r_ptr))
          (local.get $eps_ptr)
        )
        (local.set $inv_tz (f64.div (f64.const 1.0) (f64.load offset=16 (local.get $t_ptr))))
        (f64.store (local.get $eps_ptr) (f64.add (f64.mul (f64.load (local.get $eps_ptr)) (local.get $inv_tz)) (f64.const 1.0)))
        (f64.store offset=8 (local.get $eps_ptr) (f64.add (f64.mul (f64.load offset=8 (local.get $eps_ptr)) (local.get $inv_tz)) (f64.const 1.0)))
        (f64.store offset=16 (local.get $eps_ptr) (f64.add (f64.mul (f64.load offset=16 (local.get $eps_ptr)) (local.get $inv_tz)) (f64.const 1.0)))

        (call $posit_pos (local.get $posit_ptr) (local.get $pts_ptr) (local.get $eps_ptr) (local.get $r1_ptr) (local.get $r2_ptr) (local.get $t1_ptr) (local.get $t2_ptr))

        (local.set $error1 (call $posit_get_error (local.get $posit_ptr) (local.get $pts_ptr) (local.get $r1_ptr) (local.get $t1_ptr)))
        (local.set $error2 (call $posit_get_error (local.get $posit_ptr) (local.get $pts_ptr) (local.get $r2_ptr) (local.get $t2_ptr)))

        (if (f64.lt (local.get $error1) (local.get $error2))
          (then
            (memory.copy (local.get $r_ptr) (local.get $r1_ptr) (i32.const 72))
            (memory.copy (local.get $t_ptr) (local.get $t1_ptr) (i32.const 24))
            (local.set $error (local.get $error1))
          )
          (else
            (memory.copy (local.get $r_ptr) (local.get $r2_ptr) (i32.const 72))
            (memory.copy (local.get $t_ptr) (local.get $t2_ptr) (i32.const 24))
            (local.set $error (local.get $error2))
          )
        )

        (br_if $done (i32.or (f64.le (local.get $error) (f64.const 2.0)) (f64.gt (local.get $error) (local.get $prev_error))))
        (local.set $prev_error (local.get $error))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    (local.get $error)
  )

  ;; Posit.pose: Writes Pose output to $pose_out (208 bytes):
  ;;   0..7     : bestError (f64)
  ;;   8..79    : bestRotation (9 x f64)
  ;;   80..103  : bestTranslation (3 x f64)
  ;;   104..111 : alternativeError (f64)
  ;;   112..183 : alternativeRotation (9 x f64)
  ;;   184..207 : alternativeTranslation (3 x f64)
  (func $posit_pose (export "posit_pose")
    (param $posit_ptr i32) (param $pts_ptr i32) (param $pose_out i32)
    (local $eps_ptr i32) (local $r1_ptr i32) (local $r2_ptr i32) (local $t1_ptr i32) (local $t2_ptr i32)
    (local $error1 f64) (local $error2 f64)

    (local.set $eps_ptr (i32.const 5120))
    (local.set $r1_ptr (i32.const 5152))
    (local.set $r2_ptr (i32.const 5224))
    (local.set $t1_ptr (i32.const 5296))
    (local.set $t2_ptr (i32.const 5320))

    (f64.store (local.get $eps_ptr) (f64.const 1.0))
    (f64.store offset=8 (local.get $eps_ptr) (f64.const 1.0))
    (f64.store offset=16 (local.get $eps_ptr) (f64.const 1.0))

    (call $posit_pos (local.get $posit_ptr) (local.get $pts_ptr) (local.get $eps_ptr) (local.get $r1_ptr) (local.get $r2_ptr) (local.get $t1_ptr) (local.get $t2_ptr))

    (local.set $error1 (call $posit_iterate (local.get $posit_ptr) (local.get $pts_ptr) (local.get $r1_ptr) (local.get $t1_ptr)))
    (local.set $error2 (call $posit_iterate (local.get $posit_ptr) (local.get $pts_ptr) (local.get $r2_ptr) (local.get $t2_ptr)))

    (if (f64.lt (local.get $error1) (local.get $error2))
      (then
        (f64.store (local.get $pose_out) (local.get $error1))
        (memory.copy (i32.add (local.get $pose_out) (i32.const 8)) (local.get $r1_ptr) (i32.const 72))
        (memory.copy (i32.add (local.get $pose_out) (i32.const 80)) (local.get $t1_ptr) (i32.const 24))
        (f64.store offset=104 (local.get $pose_out) (local.get $error2))
        (memory.copy (i32.add (local.get $pose_out) (i32.const 112)) (local.get $r2_ptr) (i32.const 72))
        (memory.copy (i32.add (local.get $pose_out) (i32.const 184)) (local.get $t2_ptr) (i32.const 24))
      )
      (else
        (f64.store (local.get $pose_out) (local.get $error2))
        (memory.copy (i32.add (local.get $pose_out) (i32.const 8)) (local.get $r2_ptr) (i32.const 72))
        (memory.copy (i32.add (local.get $pose_out) (i32.const 80)) (local.get $t2_ptr) (i32.const 24))
        (f64.store offset=104 (local.get $pose_out) (local.get $error1))
        (memory.copy (i32.add (local.get $pose_out) (i32.const 112)) (local.get $r1_ptr) (i32.const 72))
        (memory.copy (i32.add (local.get $pose_out) (i32.const 184)) (local.get $t1_ptr) (i32.const 24))
      )
    )
  )

  ;; ============================================================================
  ;; Section 5: Dictionary & Bitwise Hamming Distance (64-bit POPCNT)
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
  ;; Section 6: Detector Pipeline (Candidates, Corners, Markers)
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

  ;; Detector.hammingDistance (5x5 legacy ArUco matrix at $bits_ptr)
  (func $detector_hamming_distance (export "detector_hamming_distance") (param $bits_ptr i32) (result i32)
    (local $dist i32) (local $sum i32) (local $min_sum i32) (local $i i32) (local $j i32) (local $k i32)
    (local.set $dist (i32.const 0))
    (local.set $i (i32.const 0))
    (block $i_done
      (loop $i_loop
        (br_if $i_done (i32.ge_s (local.get $i) (i32.const 5)))
        (local.set $min_sum (i32.const 2147483647))
        (local.set $j (i32.const 0))
        (block $j_done
          (loop $j_loop
            (br_if $j_done (i32.ge_s (local.get $j) (i32.const 4)))
            (local.set $sum (i32.const 0))
            (local.set $k (i32.const 0))
            (block $k_done
              (loop $k_loop
                (br_if $k_done (i32.ge_s (local.get $k) (i32.const 5)))
                (local.set $sum
                  (i32.add
                    (local.get $sum)
                    (select
                      (i32.const 0)
                      (i32.const 1)
                      (i32.eq
                        (i32.load (i32.add (local.get $bits_ptr) (i32.shl (i32.add (i32.mul (local.get $i) (i32.const 5)) (local.get $k)) (i32.const 2))))
                        (i32.load (i32.add (i32.const 640) (i32.shl (i32.add (i32.mul (local.get $j) (i32.const 5)) (local.get $k)) (i32.const 2))))
                      )
                    )
                  )
                )
                (local.set $k (i32.add (local.get $k) (i32.const 1)))
                (br $k_loop)
              )
            )
            (if (i32.lt_s (local.get $sum) (local.get $min_sum))
              (then (local.set $min_sum (local.get $sum)))
            )
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $j_loop)
          )
        )
        (local.set $dist (i32.add (local.get $dist) (local.get $min_sum)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $i_loop)
      )
    )
    (local.get $dist)
  )

  ;; Detector.mat2id (5x5 legacy ArUco matrix at $bits_ptr)
  (func $detector_mat2id (export "detector_mat2id") (param $bits_ptr i32) (result i32)
    (local $id i32) (local $i i32)
    (local.set $id (i32.const 0))
    (local.set $i (i32.const 0))
    (block $done
      (loop $loop
        (br_if $done (i32.ge_s (local.get $i) (i32.const 5)))
        (local.set $id
          (i32.or
            (i32.shl (local.get $id) (i32.const 1))
            (i32.load (i32.add (local.get $bits_ptr) (i32.shl (i32.add (i32.mul (local.get $i) (i32.const 5)) (i32.const 1)) (i32.const 2))))
          )
        )
        (local.set $id
          (i32.or
            (i32.shl (local.get $id) (i32.const 1))
            (i32.load (i32.add (local.get $bits_ptr) (i32.shl (i32.add (i32.mul (local.get $i) (i32.const 5)) (i32.const 3)) (i32.const 2))))
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    (local.get $id)
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