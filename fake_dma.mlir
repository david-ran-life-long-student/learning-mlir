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
    func.call @hardware_dma_stub(%alloc, %arg12, %dyn_offset_row, %dyn_offset_col)
        : (memref<4x1x128x64xi8, 1 : i32>, memref<512x512xi8>, index, index) -> ()

    // Clean up the local memory allocation
    memref.dealloc %alloc : memref<4x1x128x64xi8, 1 : i32>

    // Standard return for a void function
    func.return
  }

}