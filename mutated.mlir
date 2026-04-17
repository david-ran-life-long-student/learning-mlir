module {
  func.func @simple_matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: index, %arg4: f32, %arg5: f32) {
    affine.for %arg6 = 0 to %arg3 {
      %true = arith.constant true
      scf.if %true {
        affine.for %arg7 = 0 to %arg3 {
          %true_0 = arith.constant true
          scf.if %true_0 {
            %0 = affine.load %arg2[%arg6, %arg7] : memref<?x?xf32>
            %1 = arith.mulf %0, %arg5 : f32
            affine.store %1, %arg2[%arg6, %arg7] : memref<?x?xf32>
          }
        }
      }
    }
    affine.for %arg6 = 0 to %arg3 {
      %true = arith.constant true
      scf.if %true {
        affine.for %arg7 = 0 to %arg3 {
          %true_0 = arith.constant true
          scf.if %true_0 {
            affine.for %arg8 = 0 to %arg3 {
              %true_1 = arith.constant true
              scf.if %true_1 {
                %0 = affine.load %arg0[%arg7, %arg6] : memref<?x?xf32>
                %1 = affine.load %arg1[%arg6, %arg8] : memref<?x?xf32>
                %2 = affine.load %arg2[%arg7, %arg8] : memref<?x?xf32>
                %3 = arith.mulf %0, %1 : f32
                %4 = arith.mulf %arg4, %3 : f32
                %5 = arith.addf %2, %4 : f32
                affine.store %5, %arg2[%arg7, %arg8] : memref<?x?xf32>
              }
            }
          }
        }
      }
    }
    return
  }
}

