module {
  func.func @simple_matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>,
                           %N: index, %alpha: f32, %beta: f32) {

    // 1. Scale C by beta
    affine.for %i = 0 to %N {
      affine.for %j = 0 to %N {
        %c_val = affine.load %C[%i, %j] : memref<?x?xf32>
        %scaled = arith.mulf %c_val, %beta : f32
        affine.store %scaled, %C[%i, %j] : memref<?x?xf32>
      }
    }

    // 2. Generalized Matrix Multiplication (k-i-j order)
    affine.for %k = 0 to %N {
      affine.for %i = 0 to %N {
        affine.for %j = 0 to %N {

          // Load the values
          %a_val = affine.load %A[%i, %k] : memref<?x?xf32>
          %b_val = affine.load %B[%k, %j] : memref<?x?xf32>
          %c_val = affine.load %C[%i, %j] : memref<?x?xf32>

          // Calculate: alpha * A[i][k] * B[k][j]
          %mul_ab = arith.mulf %a_val, %b_val : f32
          %mul_alpha = arith.mulf %alpha, %mul_ab : f32

          // Accumulate: C[i][j] += ...
          %add = arith.addf %c_val, %mul_alpha : f32

          // Store the result
          affine.store %add, %C[%i, %j] : memref<?x?xf32>
        }
      }
    }
    return
  }
}