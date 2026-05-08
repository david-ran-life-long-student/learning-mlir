// A minimal illustration of the `async` dialect.
//
// Verify with:
//   mlir-opt concurrrent/case_1.mlir
//
// Lower toward LLVM with (for example):
//   mlir-opt concurrrent/case_1.mlir \
//     -async-to-async-runtime -async-runtime-ref-counting \
//     -convert-async-to-llvm

module {

  // A fake "compute" kernel: write the index as f32 into each slot of %arg.
  // Synchronous on its own -- concurrency comes from how main() *calls* it.
  func.func @operation(%arg: memref<1024xf32>) {
    affine.for %i = 0 to 1024 {
      %f = arith.index_cast %i : index to i32
      %v = arith.sitofp %f : i32 to f32
      affine.store %v, %arg[%i] : memref<1024xf32>
    }
    return
  }

  func.func @main() {
    // Two independent buffers so the two async tasks have no data dependency.
    %buf0 = memref.alloc() : memref<1024xf32>
    %buf1 = memref.alloc() : memref<1024xf32>

    // Launch @operation asynchronously on %buf0.
    // `async.execute` returns an !async.token that completes when the
    // body's `async.yield` runs.
    %tok0 = async.execute {
      func.call @operation(%buf0) : (memref<1024xf32>) -> ()
      async.yield
    }

    // Launch a second async task on %buf1 -- runs concurrently with %tok0.
    %tok1 = async.execute {
      func.call @operation(%buf1) : (memref<1024xf32>) -> ()
      async.yield
    }


    %tok2 = async.execute {  // ERROR! two tasks on the same data without await
      func.call @operation(%buf1) : (memref<1024xf32>) -> ()
      async.yield
    }

    // Block until the first task is done.
    async.await %tok0 : !async.token
    async.await %tok1 : !async.token
    async.await %tok2 : !async.token

    memref.dealloc %buf0 : memref<1024xf32>
    memref.dealloc %buf1 : memref<1024xf32>
    return
  }
}
