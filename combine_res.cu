/**
 * Thi simpler example just to see how to use simple combination
 * in a parralel way on GPU. The elaborated combination don't follow 
 * the lexograpical order
 */

 #include <stdio.h>
 #include "dfg.h"

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
 __global__ void combination(int n, int k, int tot_comb, int *final, int *final_time, operation_t *Operation_init, int operation_number, node_t *node_init, int node_number)
 {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i, j, z;
    // can speed up the overall process
    operation_t Operation[10];
    node_t node[15];

    for(i = 0; i < operation_number; i++)
    {
        Operation[i] = Operation_init[i];
        for(j = 0; j < Operation[i].index_next_node_occurency; j++)
            Operation[i].index_next_node[j] = Operation_init[i].index_next_node[j];
        for(j = 0; j < Operation[i].res_occurency; j++)
            Operation[i].res[j] = Operation_init[i].res[j];
    }

    for(i = 0; i < node_number; i++)
    {
        node[i] = node_init[i];
        for(j = 0; j < node[i].index_next_node_occurency; j++)
            node[i].index_next_node[j] = node_init[i].index_next_node[j];
    }

    if (idx < tot_comb) {
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

        // assign resources and check if resources used cover all operations
        for(z = 0; z < k; ++z)
        {
            for(i = 0; i < operation_number; i++)
            {
                for(j = 0; j < Operation[i].res_occurency; j++)
                {
                    if (Operation[i].res[j].id == final[idx*k+z]) {
                        Operation[i].res[j].occurency = 1;
                        Operation[i].covered = 1;
                    }
                }
            }
        }

        j = 1;
        for(i = 0; i < operation_number; i++)
        {
            if (Operation[i].covered != 1) {
                j = 0;
            }
        }

        // printf("%d has covered %d\n", idx+1, j);
        int time = -1;
        if(j == 1) 
        {
            // print all avaiable resources
            // if (idx == 8)
            // {
            //     for(i = 0; i < operation_number; i++)
            //     {
            //         for(j = 0; j < Operation[i].res_occurency; j++)
            //         {
            //             if (Operation[i].res[j].occurency > 0)
            //             {
            //                 printf("OPERATION: %s - SPEED: %d - AREA: %d - OCCURENCY: %d\n", Operation[i].name, 
            //                 Operation[i].res[j].speed, Operation[i].res[j].area, Operation[i].res[j].occurency);
            //             }
            //         }
            //     }
            // }

            // calculate time for the following resource combination
            int flag = 1;
            uint8_t node_index;
            while (flag)
            {
                flag = 0;
                // check between all operation and find node that can be scheduled or that are in execution, 
                // in case you find nothing this means that all nodes hande been scheduled
                for(i = 0; i < operation_number; i++) 
                {
                    // Put some node from idle to executed state
                    for(j = 0; j < Operation[i].index_next_node_occurency; j++)
                    {
                        node_index = Operation[i].index_next_node[j];
                        // Check if exist a node that has parents scheduled and is in Idle state
                        if(node[node_index].dependecies_level_satisfy == 0 && node[node_index].state == Idle)
                        {
                            flag = 1;
                            // Check if there is some free resorce
                            for(j = 0; j < Operation[i].res_occurency; j++)
                            {
                                if (Operation[i].res[j].occurency > 0)
                                {
                                    // Associate the resources to the node and decrease the occurency
                                    node[node_index].remain_time = Operation[i].res[j].speed;
                                    node[node_index].id_resource = j;
                                    node[node_index].state = Execution;
                                    Operation[i].res[j].occurency--;
                                }
                            }
                        }
                    }
                }
                for(i = 0; i < operation_number; i++) 
                {
                    // See how much time is needed for operation to terminated and they can free the resource
                    for(j = 0; j < Operation[i].index_next_node_occurency; j++)
                    {
                        node_index = Operation[i].index_next_node[j];
                        // Check if exist a node that has parents scheduled and is in Idle o Execution state
                        if(node[node_index].dependecies_level_satisfy == 0 && node[node_index].state == Execution)
                        {
                            flag = 1;
                            if (node[node_index].remain_time == 1) 
                            {
                                // Node terminates to use the resource and all his dependencies have to be free
                                node[node_index].state = Finish;
                                Operation[i].res[node[node_index].id_resource].occurency++;
                                for(k = 0; k < node[node_index].index_next_node_occurency; k++)
                                    node[node[node_index].index_next_node[k]].dependecies_level_satisfy--;
                            } else {
                                node[node_index].remain_time--;
                            }
                        }
                    }
                }
                time++;
            }
        }
        final_time[idx] = time;
    }

 } // combination()
  
 int main(int argc, char const *argv[])
 {
    int app;            // for read int
    uint8_t i, j, k;    // use like iterator

    if (argc != 3 && argc != 6)
    {
        printf("Error in argument, expected 1 but was %d!\n", argc);
        return -1;
    }

    /** Read resources */

    FILE *fp = fopen(argv[2], "r");
    if (fp == NULL) 
    {
        printf("Error file name: %s doesn't exist!\n", argv[2]);
        return -2;
    }
    
    // initilize resources
    uint8_t operation_number;
    fscanf(fp, "%d", &app);
    operation_number = app;


    operation_t *Operation;
    Operation = (operation_t *)malloc(sizeof(operation_t)*operation_number);

    uint8_t count = 0;
    uint8_t len;
    for(i = 0; i < operation_number; i++)
    {   
        fscanf(fp, "%s", Operation[i].name);
        fscanf(fp, "%d\n", &app);
        len = app;
        Operation[i].res_occurency = len;
        // assign id to operation in a increase order
        Operation[i].operation_id  = i;
        Operation[i].covered = 0;
        Operation[i].index_next_node_occurency = 0; 
        // Operation[i].res = (resource_t *)malloc(sizeof(resource_t)*len);
        // Read how many resources are avaiable for executed this operation and
        // read all its property (speed and area)
        for(j = 0; j < len; j++)
        {
            // use app to avoid problem whit int scanf that use 32 bits
            fscanf(fp, "%d", &app);
            Operation[i].res[j].area = app;
            fscanf(fp, "%d", &app);
            Operation[i].res[j].speed = app;
            // assign id to resources in a increase order
            Operation[i].res[j].id = count++;
        }
    }

    /** Read node_t */

    fp = fopen(argv[1], "r");
    if (fp == NULL) 
    {
        printf("Error file name: %s doesn't exist!\n", argv[1]);
        return -2;
    }

    // initilize the node
    uint8_t len_node;
    fscanf(fp, "%d", &app);
    len_node = app;

    node_t *node;
    node = (node_t *)malloc(sizeof(node_t)*len_node);

    char temp1[8];
    char temp2[8];
    for(i = 0; i < len_node; i++) 
    {
        fscanf(fp, "%s", temp1);
        fscanf(fp, "%s", temp2);
        strcpy(node[i].name, temp1);
        node[i].id_node = i;
        node[i].state = Idle;
        node[i].dep1_index = EMPTY_INDEX;
        node[i].dep2_index = EMPTY_INDEX;
        node[i].index_next_node_occurency = 0;
        node[i].dependecies_level         = 0;
        node[i].dependecies_level_satisfy = 0;
        for(j = 0; j < operation_number; j++)
        {
            if (strcmp(temp2, Operation[j].name) == 0)
            {
                node[i].index_operation = j;
                // Add index to list of operation
                Operation[j].index_next_node[Operation[j].index_next_node_occurency++] = i;
                break;
            }
        }
    }
    
    // inizialize edge
    uint8_t len_edge;
    fscanf(fp, "%d", &app);
    len_edge = app;

    uint8_t v, u;
    for(i = 0; i < len_edge; i++) 
    {
        // Read source node
        fscanf(fp, "%s", temp1);
        // Read destination node
        fscanf(fp, "%s", temp2);
        // Check the index of two nodes
        for (j = 0; j < len_node; j++)
        {
            if (strcmp(node[j].name, temp1) == 0)
                u = j;
            else if (strcmp(node[j].name, temp2) == 0)
                v = j;
        }

        // Put as one of next node for the one read first
        node[u].index_next_node[node[u].index_next_node_occurency++] = v;
        
        // Put like next node for the one read in second place
        if (node[v].dep1_index == EMPTY_INDEX) 
            node[v].dep1_index = u;
        else
            node[v].dep2_index = u;
        node[v].dependecies_level++;
        node[v].dependecies_level_satisfy++;
        
        printf("Node %s(%s) va in nodo %s(%s)\n",  
            node[u].name, Operation[node[u].index_operation].name, 
            node[v].name, Operation[node[v].index_operation].name);
    }

    /** Print all read data to check the correct assimilation*/

    printf("\nNODE\n\n");
    for(i = 0; i < len_node; i++)
    {
        printf("%d) Node: %s - Operation: %s" , node[i].id_node, node[i].name, Operation[node[i].index_operation].name);
        if (node[i].dependecies_level != 0) {
            printf(" - Dependecies: ");
            if (node[i].dep1_index != EMPTY_INDEX)
                printf("%s ", node[node[i].dep1_index].name);
            if (node[i].dep2_index != EMPTY_INDEX)
                printf("%s ", node[node[i].dep2_index].name);
        }
        if (node[i].index_next_node_occurency > 0) 
        {
            printf(" - Next node:   ");
            for(j = 0; j < node[i].index_next_node_occurency; j++)
                printf("%s ", node[node[i].index_next_node[j]].name);
        }
        printf("\n");
    }

    printf("\nRESOURCES\n\n");
    for(i = 0; i < operation_number; i++)
    {
        printf("For %s the node are: ", Operation[i].name);
        for(j = 0; j < Operation[i].index_next_node_occurency; j++)
            printf("%s ", node[Operation[i].index_next_node[j]].name);
        printf("\n");
        printf("\tID Area Speed\n");
        for(j = 0; j < Operation[i].res_occurency; j++)
        {
            printf("%d)\t%2d %4d %4d\n", j, Operation[i].res[j].id, Operation[i].res[j].area, Operation[i].res[j].speed);
        }
    }
    printf("\n");

    // data to pass to GPU
    int *final, *dev_final;
    int *final_time, *dev_final_time;
    operation_t *dev_Operation;
    node_t *dev_node;

    printf("Number of possible resource is %d\n", count);

    // calculate number of combinations
    k = 3; // how big are the cutset
    int n_f = 1; // nominatore fattoriale
    for (i = count; i > k; i--) n_f *= i;
    int d_f = 1; // denominatore fattoriale
    for (i = 1; i <= count - k ; i++) d_f *= i;
    int tot_comb = n_f/d_f;

    // Allocatr GPU memory
    cudaMalloc(&dev_Operation, operation_number*sizeof(operation_t));
    cudaMemcpy(dev_Operation, Operation, operation_number*sizeof(operation_t), cudaMemcpyHostToDevice);
    // for(i = 0; i < operation_number; i++)
    // {
    //     cudaMalloc((void**) &(dev_Operation[i].res), Operation[i].res_occurency*sizeof(resource_t));
    //     cudaMemcpy(dev_Operation[i].res, Operation[i].res, Operation[i].res_occurency*sizeof(resource_t), cudaMemcpyHostToDevice);
    // }

    cudaMalloc(&dev_node, len_node*sizeof(node_t));
    cudaMemcpy(dev_node, node, len_node*sizeof(node_t), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_final_time, tot_comb*sizeof(int));
    final_time = (int *)malloc(tot_comb*sizeof(int));

    cudaMalloc(&dev_final, k*tot_comb*sizeof(int));
    final = (int *)malloc(k*tot_comb*sizeof(int));

    printf("Number of total combination is: %d\n\n", tot_comb);

    // call kernel
    combination<<<1, tot_comb>>>(count, k, tot_comb, dev_final, dev_final_time, dev_Operation, operation_number, dev_node, len_node);

    cudaMemcpy(final, dev_final, k*tot_comb*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(final_time, dev_final_time, tot_comb*sizeof(int), cudaMemcpyDeviceToHost);

    printf("   1) ");
    for(i = 0; i < k*tot_comb; i++) {
        if (i % k == 0 && i > 0) 
            printf(" - Time: %2d\n%4d) ", final_time[i/k-1], i/k+1);

        printf("%d ", final[i]);
    }
    printf(" - Time: %d\n", final_time[i/k-1]);
    printf("\n");

    cudaFree(dev_final);
    cudaFree(dev_final_time);
    cudaFree(dev_node);
    cudaFree(dev_Operation);

    cudaDeviceReset();
    return 0;
 }
  