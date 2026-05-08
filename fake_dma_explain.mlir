module {

  // -------------------------------------------------------------------
  // 1. THE STUB DECLARATION
  // -------------------------------------------------------------------
  // This is the placeholder. 'private' tells the compiler that this
  // function has external linkage and will be provided at link-time
  // (e.g., by a C runtime mock or a hardware simulator). No body is needed.
  func.func private @hardware_dma_stub(%alloc: memref<4x1x128x64xi8, 1 : i32>,
                                       %arg12: memref<512x512xi8>,
                                       %dyn_offset_row: index,
                                       %dyn_offset_col: index)


  // -------------------------------------------------------------------
  // 2. THE COMPUTE KERNEL
  // -------------------------------------------------------------------
  // This represents the host function where your DMA operation originally lived.
  func.func @compute_kernel(%arg12: memref<512x512xi8>,
                            %dyn_offset_row: index,
                            %dyn_offset_col: index) {

    // Allocate the local accelerator memory (Memory Space 1)
    // This provides the %alloc SSA value required by the DMA transfer.
    %alloc = memref.alloc() : memref<4x1x128x64xi8, 1 : i32>

    // ...
    // [Pretend your accelerator compute ops happen here, filling %alloc]
    // ...

    // THE original: Instead of the opaque custom_hw.dma_transfer,
    // we use a standard func.call to hit the stub.
    // The type signature after the colon strictly defines the inputs and the empty () return.
    // func.call @hardware_dma_stub(%alloc, %arg12, %dyn_offset_row, %dyn_offset_col)
    //     : (memref<4x1x128x64xi8, 1 : i32>, memref<512x512xi8>, index, index) -> ()

    // 1. The Fused Routing Logic
    // d0=i, d1=j, d2=k, d3=l (Loop Iterators)
    // s0=dyn_offset_row, s1=dyn_offset_col (Symbols/Runtime Offsets)
    #writeback_map = affine_map<(d0, d1, d2, d3)[s0, s1] -> (d0 * 128 + d2 + s0, d1 * 64 + d3 + s1)>

    // 2. The Iteration Space (Bound to the 32,768 elements of the local tile)
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 1 {
        affine.for %k = 0 to 128 {
          affine.for %l = 0 to 64 {

            // 3. Dense Sequential Read
            // Since the loop variables perfectly match the dimensions of the local memory,
            // affine.load implicitly uses the standard identity map (i, j, k, l).
            %val = affine.load %alloc[%i, %j, %k, %l] : memref<4x1x128x64xi8, 1 : i32>

            // 4. Strided Scatter Writeback
            // We feed the 4 dimensions and 2 symbols directly into the fused map.
            // The polyhedral analyzer sees exactly where every single byte lands.
            affine.store %val, %arg12[%i, %j, %k, %l, %dyn_offset_row, %dyn_offset_col] {
                map = #writeback_map
            } : memref<512x512xi8>

          }
        }
      }
    }

    // Clean up the local memory allocation
    memref.dealloc %alloc : memref<4x1x128x64xi8, 1 : i32>

    // Standard return for a void function
    func.return
  }

}