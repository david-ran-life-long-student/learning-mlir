//
// Created by dran on 3/9/26.
//

void simple_matmul (const restrict float* A, const restrict float* B, restrict float* C,
    const int N, const float alpha, const float beta) {

    // scale first
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
            C[i][j] *= beta;

    // generalized mat mul
    for (int k=0; k<N; k++)
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                C[i][j] += alpha * A[i][k] * B[k][j];
}
