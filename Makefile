CC=gcc
NVCC=nvcc

combination_v3: dfg_v3.c
	$(CC) -o dfg_v3_CPU dfg_v3.c

combination_v4: dfg_v4.c
	$(CC) -o dfg_v4_CPU dfg_v4.c

dfg_v8: dfg_v8.cu
	$(NVCC) -o dfg_v8 dfg_v8.cu

dfg_v9: dfg_v9.cu
	$(NVCC) -o dfg_v9 dfg_v9.cu

dfg_v9_1: dfg_v9_1.cu
	$(NVCC) -o dfg_v9_1 dfg_v9_1.cu

dfg_v9_2: dfg_v9_2.cu
	$(NVCC) -o dfg_v9_2 dfg_v9_2.cu

dfg_v9_3: dfg_v9_3.cu
	$(NVCC) -o dfg_v9_3 dfg_v9_3.cu

dfg_v9_4: dfg_v9_4.cu
	$(NVCC) -o dfg_v9_4 dfg_v9_4.cu

dfg_v10: dfg_v10.cu
	$(NVCC) -o dfg_v10 dfg_v10.cu

dfg_v11: dfg_v11.cu
	$(NVCC) -o dfg_v11 dfg_v11.cu

dfg_v11_1: dfg_v11_1.cu
	$(NVCC) -o dfg_v11_1 dfg_v11_1.cu