// Tokens vs. values in the `async` dialect.
//
//   !async.token       -- a future "done" signal. Carries no data.
//                         Await it for its *effect* (a side-effecting task
//                         finished). `await` returns nothing.
//
//   !async.value<T>    -- a future of type T. Carries a value once ready.
//                         `await` on it produces a T.
//
// Verify with:
//   mlir-opt concurrrent/case_2.mlir

module {

  // ---------- Style 1: token + side-effecting memref --------------------
  // The async task has no SSA result; the only way to get information out
  // is to write into memory the caller already owns.
  func.func @produce_into_buf(%out: memref<1xf32>) -> !async.token {
    %tok = async.execute {
      %c0 = arith.constant 0 : index
      %v  = arith.constant 4.200000e+01 : f32
      memref.store %v, %out[%c0] : memref<1xf32>
      async.yield
    }
    return %tok : !async.token
  }

  // ---------- Style 2: async.value -------------------------------------
  // The async task yields a value. The caller awaits it and receives the
  // f32 directly -- no shared buffer, no aliasing concerns.
  func.func @produce_value() -> !async.value<f32> {
    // `async.execute -> !async.value<f32>` returns BOTH a token (for the
    // task's completion) and a value-future for the yielded f32.
    %tok, %res = async.execute -> !async.value<f32> {
      %v = arith.constant 4.200000e+01 : f32
      async.yield %v : f32
    }
    return %res : !async.value<f32>
  }

  // ---------- Comparing them at the call site --------------------------
  func.func @main() -> f32 {
    // Style 1: caller must allocate, pass in, await the token, then load.
    %buf = memref.alloc() : memref<1xf32>
    %tok = func.call @produce_into_buf(%buf) : (memref<1xf32>) -> !async.token
    async.await %tok : !async.token            // await returns nothing
    %c0 = arith.constant 0 : index
    %a  = memref.load %buf[%c0] : memref<1xf32>
    memref.dealloc %buf : memref<1xf32>

    // Style 2: caller awaits the value-future and gets the f32 back.
    %fut = func.call @produce_value() : () -> !async.value<f32>
    %b   = async.await %fut : !async.value<f32>  // <-- await *produces* f32

    %sum = arith.addf %a, %b : f32
    return %sum : f32
  }

  // ---------- Style 3: async.func --------------------------------------
  // A genuinely-async function. Note:
  //   - declared with `async.func`, not `func.func`
  //   - body uses ordinary scalar ops + a plain `return` of an f32
  //   - the *function's* return type is wrapped: !async.value<f32>
  //   - callers invoke it with `async.call`, which returns the future
  // No `async.execute` block needed inside.
  async.func @produce_value_native() -> !async.value<f32> {
    %v = arith.constant 4.200000e+01 : f32
    return %v : f32
  }

  func.func @use_native() -> f32 {
    %fut = async.call @produce_value_native() : () -> !async.value<f32>
    %x   = async.await %fut : !async.value<f32>
    return %x : f32
  }

  // ---------- Bonus: chaining on a value -------------------------------
  // A value-future can be passed into another `async.execute` as an operand.
  // The body sees it unwrapped to the underlying type, and the new task
  // does not start until the input value is ready.
  func.func @double_async(%x: !async.value<f32>) -> !async.value<f32> {
    %tok, %res = async.execute (%x as %xv: !async.value<f32>)
                    -> !async.value<f32> {
      %two = arith.constant 2.000000e+00 : f32
      %y   = arith.mulf %xv, %two : f32
      async.yield %y : f32
    }
    return %res : !async.value<f32>
  }
}
