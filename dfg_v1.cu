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
 __global__ void combination(int n, int k_comb, int tot_comb, 
    operation_t *Operation_init, const int operation_number, node_t *node_init, const int node_number, 
    const int area_limit, int *final_best_combination, int *final_best_repetition, int *final_best_time, int *final_area_calculated)
 {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i, j, z, k;
    int best_time = 0x7fffffff;
    int area_calculated = 0x7fffffff;
    int area = 0;
    int final[10];
    int all_aree[10];

    // can speed up the overall process coping in the local register
    // int operation_covered[10];
    operation_t Operation[10];
    node_t node[10];

    // Copy operations information
    for(i = 0; i < operation_number; i++)
    {
        Operation[i] = Operation_init[i];
        for(j = 0; j < Operation[i].index_next_node_occurency; j++)
            Operation[i].index_next_node[j] = Operation_init[i].index_next_node[j];
        for(j = 0; j < Operation[i].res_occurency; j++)
            Operation[i].res[j] = Operation_init[i].res[j];
    }

    // Copy nodes information
    for(i = 0; i < node_number; i++)
    {
        node[i] = node_init[i];
        for(j = 0; j < node[i].index_next_node_occurency; j++)
            node[i].index_next_node[j] = node_init[i].index_next_node[j];
    }

    if (idx < tot_comb) {
        int a = n;
        int b = k_comb;
        int x = idx; // x is the "dual" of m

        // calculate the combination
        for (i = 0; i < k_comb; ++i)
        {
            --a;
            while (Choose(a,b) > x)
                --a;
            x = x - Choose(a,b);
            final[i] = a;
            // final_best_combination[idx*k_comb+i] = a;
            b = b-1;
        }

        //synchronize the local threads writing to the local memory cache
        // __syncthreads();

        // // check the best time
        // if(idx <= 0)
        // {
        //     best_time = 0x7fffffff;
        //     printf("I am HERE 1\n");
        //     for(i = 0; i < tot_comb; i++)
        //     {   
        //         printf("%d) ", i);
        //         for(j = 0; j < k_comb; j++) 
        //             printf("%d - ", final_best_combination[i*k_comb+j]);
        //         printf("\n");
        //     }
        // }

        // __syncthreads();

        // assign resources and check if resources used cover all operations
        k = 0;
        for(z = 0; z < k_comb; z++)
        {
            for(i = 0; i < operation_number; i++)
            {
                for(j = 0; j < Operation[i].res_occurency; j++)
                {
                    if (Operation[i].res[j].id == final[z])
                   {
                        all_aree[k++] = Operation[i].res[j].area;
                        Operation[i].res[j].occurency = 1;
                        Operation[i].covered = 1;
                    }
                }
            }
        }

        // work with repetition, with a maximum of area_limit
        int repeat[10];
        int index = 0;
        int end_index = 0;

        area = 0;
        for (j=0; j < k_comb; j++)
        {
            repeat[j] = 1;
            area += all_aree[j];
        }

        for(i = 0; i < operation_number; i++)
        {
            if (Operation[i].covered != 1)
                end_index = k_comb;
        }

       
        int flag;
        int time;
        uint8_t node_index;
        // start repeat combination
        while(end_index != k_comb)
        {
            // set occurency for each resources
            for(z = 0; z < k_comb; z++)
            {
                for(i = 0; i < operation_number; i++)
                {
                    for(j = 0; j < Operation[i].res_occurency; j++)
                    {
                        if (Operation[i].res[j].id == final[z])
                            Operation[i].res[j].occurency = repeat[z];
                    }
                }
            }

            /** Scheduling operation */
            flag = 0;
            if (area <= area_limit)
                flag = 1;
            time = -1;
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
                                    node[node_index].id_resource = Operation[i].res[j].id;
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
                                for(z = 0; z < node[node_index].index_next_node_occurency; z++)
                                    node[node[node_index].index_next_node[z]].dependecies_level_satisfy--;
                            } else {
                                node[node_index].remain_time--;
                            }
                        }
                    }
                }
                time++;
            } // End scheduling

            // see if a better result has been achived
            if(time > -1 && ((time < best_time) || (time == best_time && area < area_calculated)))
            {
                for(i = 0; i < k_comb; i++) 
                {
                    // TO_DO1: save them in variable and then copy nack in local memory
                    // TO_DO2: save them in variable and then copy nack in shared memory
                    final_best_combination[idx*k_comb+i] = final[i];
                    final_best_repetition[idx*k_comb+i] = repeat[i];
                }
                final_best_time[idx] = time;
                area_calculated = area;
                best_time = time;
            }


            // Calculate the new repetition and the new area value 
            index = 0;
            while(repeat[index] == 5 && index < end_index)   
                repeat[index++] = 1;

            if(index == end_index) 
            {
                if (repeat[end_index] == 5) 
                {
                    repeat[end_index++] = 1;
                }

                if (end_index != k_comb) 
                {
                    repeat[end_index]++;
                }
            } else {
                repeat[index]++;
            }

            area = 0;
            for(i = 0; i < k_comb; i++)
                area += (all_aree[i]*repeat[i]);

            // Restart node property
            for(i = 0; i < node_number; i++)
            {
                node[i].dependecies_level_satisfy = node[i].dependecies_level;
                node[i].state = Idle;
                node[i].remain_time = 0;
            }
            
        }// End repeat combination

        // set operation as the beginning
        for(z = 0; z < k_comb; ++z)
        {
            for(i = 0; i < operation_number; i++)
            {
                for(j = 0; j < Operation[i].res_occurency; j++)
                {
                    if (Operation[i].res[j].id == final[z]) {
                        Operation[i].res[j].occurency = 0;
                        Operation[i].covered = 0;
                    }
                }
            }
        }

        // TO_DO1: save result using temporaly register
        final_best_time[idx] = best_time;
        final_area_calculated[idx] = area_calculated;
    } // End check if rigth thread

    //synchronize the local threads writing to the local memory cache
    __syncthreads();

    // check the best time
    if(idx <= 0)
    {
        for(i = 1; i < tot_comb; i++)
        {   
            // printf("%d) ", i);
            // for(j = 0; j < k_comb; j++) 
            //     printf("%d %d - ", final_best_combination[i*k_comb+j], final_best_repetition[i*k_comb+j], final_best_time[i]);
            // printf("\n");

            if (best_time > final_best_time[i])
            {
                final_best_time[0] = final_best_time[i];
                final_area_calculated[0] = final_area_calculated[i];
                for(j = 0; j < k; j++) 
                {
                    final_best_combination[j] = final_best_combination[i*k_comb+j];
                    final_best_repetition[j] = final_best_repetition[i*k_comb+j];
                }
            }
        }
    }

 } // End combination()
  
 int main(int argc, char const *argv[])
 {
    int app;            // for read int
    uint8_t i, j, k;    // use like iterator

    if (argc != 4)
    {
        printf("Error in argument, expected 3 but was %d!\n", argc-1);
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
        Operation[i].max_index_next_node_occurency = 4; 
        Operation[i].index_next_node = (uint8_t *)malloc(sizeof(uint8_t)*4);
        Operation[i].index_next_node_occurency = 0;
        Operation[i].res = (resource_t *)malloc(sizeof(resource_t)*len);
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
        node[i].max_index_next_node_occurency = 4;
        node[i].index_next_node = (uint8_t * )malloc(sizeof(uint8_t)*4);
        node[i].index_next_node_occurency = 0;
        node[i].dependecies_level         = 0;
        node[i].dependecies_level_satisfy = 0;
        for(j = 0; j < operation_number; j++)
        {
            if (strcmp(temp2, Operation[j].name) == 0)
            {
                node[i].index_operation = j;
                // Add index to list of node in the propr operation
                if(Operation[j].max_index_next_node_occurency = Operation[j].index_next_node_occurency) 
                {
                    Operation[i].max_index_next_node_occurency *= 2;
                    Operation[i].index_next_node = (uint8_t *)realloc((uint8_t *)Operation[i].index_next_node, sizeof(uint8_t)*Operation[i].max_index_next_node_occurency);
                }
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
        if(node[u].max_index_next_node_occurency = Operation[u].index_next_node_occurency) 
        {
            node[u].max_index_next_node_occurency *= 2;
            node[u].index_next_node = (uint8_t *)realloc((uint8_t *)node[u].index_next_node, sizeof(uint8_t)*node[u].max_index_next_node_occurency);
        }
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

    // variables used for GPU
    int final_best_time, *dev_final_best_time;
    int final_area_calculated, *dev_final_area_calculated;
    int *final_best_combination, *dev_final_best_combination;
    int *final_best_repetition, *dev_final_best_repetition;
    operation_t *dev_Operation;
    node_t *dev_node;

    uint8_t *dev_app;

    printf("Number of possible resource is %d\n\n\n", count);

    // Allocatr GPU memory
    cudaMalloc(&dev_Operation, operation_number*sizeof(operation_t));
    cudaMemcpy(dev_Operation, Operation, operation_number*sizeof(operation_t), cudaMemcpyHostToDevice);
    // Allocate the right quantity for store the proper dimension of array in each structure
    for(i = 0; i < operation_number; i++)
    {
        // Copy resources
        cudaMalloc(&dev_app, Operation[i].res_occurency*sizeof(resource_t));
        cudaMemcpy(dev_app, Operation[i].res, Operation[i].res_occurency*sizeof(resource_t), cudaMemcpyHostToDevice);
        cudaMemcpy(&(dev_Operation[i].res), &dev_app, sizeof(uint8_t *), cudaMemcpyHostToDevice);
        // Copy nodes
        cudaMalloc(&dev_app, Operation[i].index_next_node_occurency*sizeof(uint8_t));
        cudaMemcpy(dev_app, Operation[i].index_next_node, Operation[i].index_next_node_occurency*sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(&(dev_Operation[i].index_next_node), &dev_app, sizeof(uint8_t *), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&dev_node, len_node*sizeof(node_t));
    cudaMemcpy(dev_node, node, len_node*sizeof(node_t), cudaMemcpyHostToDevice);

    for(i = 0; i < len_node; i++)
    {
        // Copy nodes
        cudaMalloc(&dev_app, node[i].index_next_node_occurency*sizeof(uint8_t));
        cudaMemcpy(dev_app, node[i].index_next_node, node[i].index_next_node_occurency*sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(&(dev_node[i].index_next_node), &dev_app, sizeof(uint8_t *), cudaMemcpyHostToDevice);
    }

    // store the value for comparison
    int *best_final = (int *)malloc(sizeof(int)*(count+1));   
    int *best_final_repetition = (int *)malloc(sizeof(int)*count);
    int best_time = 0x7fffffff;
    int area_calculated = 0x7fffffff;
    int area_limit = atoi(argv[3]);;

    // to store the execution time of code
    double time_spent = 0.0;
 
    clock_t begin = clock();
    // how big are the cutset, modify it iteratively
    for(k = 3; k <= count; k++) {
        // calculate number of combinations
        int n_f = 1; // nominatore fattoriale
        for (i = count; i > k; i--) n_f *= i;
        int d_f = 1; // denominatore fattoriale
        for (i = 1; i <= count - k ; i++) d_f *= i;
        int tot_comb = n_f/d_f;

        cudaMalloc(&dev_final_best_time, tot_comb*sizeof(int));
        
        cudaMalloc(&dev_final_area_calculated, tot_comb*sizeof(int));

        cudaMalloc(&dev_final_best_combination, k*tot_comb*sizeof(int));
        final_best_combination = (int *)malloc(k*sizeof(int));

        cudaMalloc(&dev_final_best_repetition, k*tot_comb*sizeof(int));
        final_best_repetition = (int *)malloc(k*sizeof(int));

        #ifdef TESTING
        printf("Number of total combination witk k equal to %d are: %d\n\n", k, tot_comb);
        #endif

        // call kernel
        combination<<<1, tot_comb>>>(count, k, tot_comb, dev_Operation, operation_number, dev_node, len_node, area_limit, 
            dev_final_best_combination, dev_final_best_repetition, dev_final_best_time, dev_final_area_calculated);

        cudaMemcpy(&final_best_time, dev_final_best_time, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&final_area_calculated, dev_final_area_calculated, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(final_best_combination, dev_final_best_combination, k*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(final_best_repetition, dev_final_best_repetition, k*sizeof(int), cudaMemcpyDeviceToHost);

        #ifdef TESTING
        printf("Best Combination: ");
        for(i = 0; i < k; i++)
            printf(" %2d %2d,", final_best_combination[i], final_best_repetition[i]);
        printf(" - Time: %d\n", final_best_time);
        printf("\n");
        #endif

        if(final_best_time > -1 && ((final_best_time < best_time) 
            || (final_best_time == best_time && final_area_calculated < area_calculated)))
        {
            for(i = 0; i < k; i++)
            {
                best_final[i] = final_best_combination[i];
                best_final_repetition[i] = final_best_repetition[i];
            }
            best_final[i] = -1;
            best_time = final_best_time;
            area_calculated = final_area_calculated;
        }

        cudaFree(dev_final_best_time);
        cudaFree(dev_final_area_calculated);
        cudaFree(dev_final_best_combination);
        cudaFree(dev_final_best_repetition);
    }

    /** Print the best solution obtained */

    fprintf(stdout, "\n\nArea Limit is %d\n", area_limit);
    fprintf(stdout, "\n\nBest solution has time %d:\n", best_time);
    for(i = 0; i < count && best_final[i] != -1; i++) 
    {
        for(j = 0; j < operation_number; j++) 
        {
            for(k = 0; k < Operation[j].res_occurency; k++) 
            {
                if (best_final[i] == Operation[j].res[k].id)
                {
                    fprintf(stdout, "\tOPERATION: %4s - ID RESOURCE: %2d - SPEED: %2d - AREA: %2d - OCCURENCY: %2d\n", 
                    Operation[j].name, Operation[j].res[k].id, Operation[j].res[k].speed, Operation[j].res[k].area, best_final_repetition[i]);
                }
            }
        }
    }

    fprintf(stdout, "Final area is %d\n", area_calculated);

    clock_t end = clock();
 
    // calculate elapsed time by finding difference (end - begin) and
    // dividing the difference by CLOCKS_PER_SEC to convert to seconds
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
 
    printf("\n\nThe elapsed time is %f seconds\n", time_spent);

    cudaFree(dev_node);
    cudaFree(dev_Operation);

    cudaDeviceReset();
    return 0;
 }
  