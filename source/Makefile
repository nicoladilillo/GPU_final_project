aCC=gcc
NVCC=nvcc
	
clean: 
	rm *.out

clean_log: 
	rm *.log

dfg_v1_CPU: dfg_v1_CPU.c
	$(CC) -o dfg_v1_CPU.out dfg_v1_CPU.c

dfg_v2_CPU: dfg_v2_CPU.c
	$(CC) -o dfg_v2_CPU.out dfg_v2_CPU.c

dfg_v1_0: dfg_v1_0.cu
	$(NVCC) -o dfg_v1_0.out dfg_v1_0.cu

dfg_v2_0: dfg_v2_0.cu
	$(NVCC) -o dfg_v2_0.out dfg_v2_0.cu

dfg_v2_1: dfg_v2_1.cu
	$(NVCC) -o dfg_v2_1.out dfg_v2_1.cu

dfg_v2_2: dfg_v2_2.cu
	$(NVCC) -o dfg_v2_2.out dfg_v2_2.cu

dfg_v2_3: dfg_v2_3.cu
	$(NVCC) -o dfg_v2_3.out dfg_v2_3.cu

dfg_v3_0: dfg_v3_0.cu
	$(NVCC) -o dfg_v3_0.out dfg_v3_0.cu

dfg_v4_0: dfg_v4_0.cu
	$(NVCC) -o dfg_v4_0.out dfg_v4_0.cu

dfg_v4_1: dfg_v4_1.cu
	$(NVCC) -o dfg_v4_1.out dfg_v4_1.cu

all: dfg_v1_CPU dfg_v2_CPU dfg_v1_0 dfg_v2_0 dfg_v2_1 dfg_v2_2 dfg_v2_3 dfg_v3_0 dfg_v4_0 dfg_v4_1
	mv *.out executable
	echo "done"
