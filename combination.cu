/**
 * Thi simpler example just to see how to use simple combination
 * in a parralel way on GPU. The elaborated combination don't follow 
 * the lexograpical order
 */

#include <stdio.h>

__device__ int Choose(int n, int k)
{
    if (n < k)
        return 0;  // special case
    if (n == k)
        return 1;

    int delta, iMax;

    if (k < n-k) // ex: Choose(100,3)
    {
        delta = n-k;
        iMax = k;
    }
    else         // ex: Choose(100,97)
    {
        delta = k;
        iMax = n-k;
    }

    int ans = delta + 1;

    for (int i = 2; i <= iMax; ++i)
    {
        ans = (ans * (delta + i)) / i;
    }

    return ans;
} // Choose()

// diaplay combination with given index
__global__ void combination(int n, int k, int tot_comb, int *final)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i;
    if (idx < tot_comb) {
        // printf("%d\n", idx);
        int a = n;
        int b = k;
        int x = idx; // x is the "dual" of m

        for (i = 0; i < k; ++i)
        {
            --a;
            while (Choose(a,b) > x)
                --a;
            x = x - Choose(a,b);
            final[idx*k+i] = a;
            b = b-1;
        }
    }
} // combination()
 
int main(void)
{
    int n = 5;
    int k = 3;

    // where all final combination will bi stored
    int *final, *dev_final;

    // calculate number of combinations
    int i;
    int n_f = 1; // nominatore fattoriale
    for (i = n; i > k; i--) n_f *= i;
    int d_f = 1; // denominatore fattoriale
    for (i = 1; i <= n - k ; i++) d_f *= i;

    int tot_comb = n_f/d_f;
    cudaMalloc(&dev_final, k*tot_comb*sizeof(int));
    final = (int *)malloc(k*tot_comb*sizeof(int));

    printf("number of total combination is: %d\n\n", tot_comb);

    combination<<<1, tot_comb>>>(n, k, tot_comb, dev_final);

    // for(i = 0; i < k*tot_comb; i++) final[i] = -1;

    cudaMemcpy(final, dev_final, k*tot_comb*sizeof(int), cudaMemcpyDeviceToHost);

    for(i = 0; i < k*tot_comb; i++) {
        if (i % k == 0) printf("\n%d) ", i/k+1);
        printf("%d ", final[i]);
    }
    printf("\n");

    cudaFree(dev_final);

    cudaDeviceReset();
    return 0;
}
 