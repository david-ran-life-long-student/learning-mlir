module {
    // input is unkown shape and unkown rank, but should be compatible with 1024
    func.func @reshape_static(%a: memref<*xf32>) -> memref<1x1x1024xf32> {
        // always reshape down to 1D with subview maybe with cast?
        %0 = memref.subview %a : memref<*xf32> to memref<?xf32>
        // then go back up with view
        %1 = memref.view %0 : memref<?xf32> to memref<1x1x1024xf32>

        return %1 : memref<1x1x1024xf32>
    }
}