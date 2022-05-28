/**
 * Max Shared memory obatained is  49152
 * Max thread obtained are 1024
 * Max blocks are not defined
 * 
 */

#include<stdio.h>

__global__ void test_shared_memory(int quantity)
{
    extern  __shared__ unsigned char s[];

    if(threadIdx.x == 0)
        printf("Allocate %d memory\n", quantity);
}

int main(int argc, char const *argv[])
{
    for(int i = 44000; i < 50000; i++) {
        test_shared_memory<<<9,1024,i>>>(i);
        cudaDeviceSynchronize();
    }
    return 0;
}
