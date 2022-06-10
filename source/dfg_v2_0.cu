/**
 * Thi simpler example just to see how to use simple combination
 * in a parralel way on GPU. The elaborated combination don't follow 
 * the lexograpical order.
 * 
 * single thread care about a single repetition.
 */

 #include <stdio.h>
 #include "dfg.h"

#define MAX_BLOCKS  40
#define MAX_THREADS 1024
#define MAX_SHARED_MEMORY 49152

// #define TESTING_OP_AND_NODE
// #define TESTING
// #define TESTING_MEMORY
// #define TESTING_SCHEDULING

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
 __global__ void combination(const int n, int r, const int tot_comb, const int start_comb, const int end_comb,
    int const shared_memory_size, int const shared_memory_size_offset, int const max_rep, int const factor,
    const operation_GPU_t *Operation_init, const int operation_number, const node_GPU_t *node_init,
    const int node_number, const int area_limit_app, const uint8_t resources_number, uint8_t *final_best_combination,
    uint8_t *final_best_repetition, int *final_best_time, int *final_area_calculated)
 {
    int idx = threadIdx.x + start_comb;
    
    if (idx >= start_comb && idx < end_comb) {
        // printf("\tInside %d\n", idx);

        extern  __shared__ unsigned char s[];

        int i, j, z;

        const int k_comb = r;
        const int area_limit = area_limit_app;
        int area = 0;
        int time = -1;
        
        const uint8_t max_repetition = (uint8_t) max_rep;

        // This variable can be shared between threads in the same block
        node_GPU_t *node;
        operation_GPU_t *Operation;

        // offset between group of array thread
        unsigned long memory_trace = 0;

        node = (node_GPU_t *) &(s[memory_trace]);
        memory_trace += (((unsigned long) node_number)*sizeof(node_GPU_t));            
        Operation = (operation_GPU_t *) &(s[(int) memory_trace]);
        memory_trace += (((unsigned long) operation_number)*sizeof(operation_GPU_t));             
        
        int *final_time = (int *) &(s[(int) memory_trace]);
        memory_trace += (sizeof(int));            
        int *final_area = (int *) &(s[(int) memory_trace]);
        memory_trace += (sizeof(int));            

        // use only one instanze for all nodes and operation information
        if (idx == start_comb)
        {
            // Copy operations information
            for(i = 0; i < operation_number; i++) 
                Operation[i] = Operation_init[i];

            // Copy nodes information
            for(i = 0; i < node_number; i++)
                node[i] = node_init[i];

            *final_time = 0x7fffffff;
            *final_area = 0x7fffffff;

            // printf("Calculated offset %d vs real one %d\n", shared_memory_size_offset, (int) memory_trace);
            
        } 
        __syncthreads();

        // offset between group of array thread
        memory_trace = (unsigned long) (shared_memory_size_offset + threadIdx.x*shared_memory_size);
        
        // lenght k_comb
        resource_t resources[30];
        // resource_t *resources = (resource_t *) &(s[memory_trace]);
        // memory_trace += (((unsigned long) resources_number)*sizeof(resource_t));
        uint8_t *final = (uint8_t *) &(s[(int) memory_trace]);
        memory_trace += (((unsigned long) k_comb)*sizeof(uint8_t));
        uint8_t *repeat = (uint8_t *) &(s[(int) memory_trace]);
        memory_trace += (((unsigned long) k_comb)*sizeof(uint8_t));

        // lenght operation_number
        // variable used for operation covered
        uint8_t *operation_covered = (uint8_t *) &(s[(int) memory_trace]);
        memory_trace += (((unsigned long) operation_number)*sizeof(uint8_t));
        
        // variable used from scheduling node
        uint8_t *state;
        uint8_t *remain_time;
        uint8_t *id_resource;
        uint8_t *dependecies_level_satisfy; 

        for(i = 0; i < operation_number; i++) 
            operation_covered[i] = 0;

        int a = n;
        int b = k_comb;
        int x = idx/factor; // x is the "dual" of m
        int rep_id = idx%factor;
        

        // calculate the combination
        for(i = 0; i < k_comb; i++)
        {
            --a;
            while (Choose(a,b) > x)
                --a;
            x = x - Choose(a,b);
            final[i] = (uint8_t) a;
            b = b-1;
            // Calculate the new repetition
            repeat[i] = (uint8_t) rep_id%max_repetition+1;
            rep_id    = rep_id/max_repetition;       
        }


        #ifdef TESTING_OP_AND_NODE
        // synchronize the local threads writing to the local memory cache
        __syncthreads();

        // check the best time
        if(idx == 0)
        {
            printf("\nNODE inside kernel\n\n");
            for(i = 0; i < node_number; i++)
            {
                printf("%d) Node: %d - Operation: %d - Dependency_level: %d" , node[i].id_node, node[i].id_node, Operation[node[i].index_operation].operation_id, node[i].dependecies_level);
                if (node[i].dependecies_level != 0) {
                    printf(" - Dependecies: ");
                    if (node[i].dep1_index != EMPTY_INDEX)
                        printf("%d ", node[node[i].dep1_index].id_node);
                    if (node[i].dep2_index != EMPTY_INDEX)
                        printf("%d ", node[node[i].dep2_index].id_node);
                }
                if (node[i].index_next_node_occurency > 0) 
                {
                    printf(" - Next node:   ");
                    for(j = 0; j < node[i].index_next_node_occurency; j++)
                        printf("%d ", node[node[i].index_next_node[j]].id_node);
                }
                printf("\n");
            }

            printf("\nRESOURCES inside kernel\n\n");
            for(i = 0; i < operation_number; i++)
            {
                printf("For %d the node are: ", Operation[i].operation_id);
                for(j = 0; j < Operation[i].index_next_node_occurency; j++)
                    printf("%d ", Operation[i].index_next_node[j]);
                printf("\n");
                printf("\tID Area Speed Occ\n");
                for(j = 0; j < Operation[i].res_occurency; j++)
                {
                    printf("%d)\t%2d %4d %4d %4d\n", j, Operation[i].res[j].id, Operation[i].res[j].area, Operation[i].res[j].speed, Operation[i].res[j].occurency);
                }
            }
            printf("\n");
        }
        #endif

        // assign resources and check if resources used cover all operations
        #ifdef TESTING
        for(i = 0; i < operation_number; i++)
            for(j = 0; j < Operation[i].res_occurency; j++) {
                resources[(int) (i*Operation[i].res_occurency+j)].id              = (uint8_t) Operation[i].res[j].id;
                resources[(int) (i*Operation[i].res_occurency+j)].area            = (int)     Operation[i].res[j].area;
                resources[(int) (i*Operation[i].res_occurency+j)].speed           = (uint8_t) Operation[i].res[j].speed;
                resources[(int) (i*Operation[i].res_occurency+j)].index_operation = (uint8_t) Operation[i].res[j].index_operation;
                resources[(int) (i*Operation[i].res_occurency+j)].occurency       = (uint8_t) 0;
            }
        #endif
        
        for(z = 0; z < k_comb; z++)
        {
            for(i = 0; i < operation_number; i++)
            {
                for(j = 0; j < Operation[i].res_occurency; j++)
                {
                    if (Operation[i].res[j].id == final[z])
                    {
                        // resources[final[z]].id              = (uint8_t) Operation[i].res[j].id;
                        // resources[final[z]].speed           = (uint8_t) Operation[i].res[j].speed;
                        // resources[final[z]].index_operation = (uint8_t) Operation[i].res[j].index_operation;
                        // resources[final[z]].area            = (int)     Operation[i].res[j].area;
                        resources[final[z]]                 = Operation[i].res[j];
                        resources[final[z]].occurency       = (uint8_t) repeat[z];
                        operation_covered[i]                = (uint8_t) 1;
                        area                               += (Operation[i].res[j].area * repeat[z]);
                    }
                }
            }
        }

        #ifdef TESTING
        for(i = start_comb; i < end_comb; i++)
        {   
            __syncthreads();
            if(idx == i)
            {
                printf("\t%d) ", idx);
                for(j = 0; j < k_comb; j++) 
                    printf("%2d %d A: %3d S: %2d   ", resources[final[j]].speed, repeat[j], resources[final[j]].area, resources[final[j]].speed);                
                printf(" -- Memory is from %d to %d \n",
                    (int)( shared_memory_size_offset + threadIdx.x*shared_memory_size),
                    (int)( shared_memory_size_offset + (threadIdx.x+1)*shared_memory_size));
            }
        }
        __syncthreads();
        #endif

        // all others repeated combination will be bigger 
        uint8_t flag = 0;
        if (area <= area_limit)
        {
            flag = 1;
            // work with repetition, with a maximum of area_limit
            for(i = 0; i < operation_number; i++)
            {
                if (operation_covered[i] != 1)
                    flag = 0;
            } 
            // initialize if all operation are covered
            if(flag == 1) 
            {
                // variable used from scheduling node
                state         = (uint8_t *) &(s[(int) memory_trace]);
                memory_trace += (((unsigned long) node_number)*sizeof(uint8_t));
                remain_time   = (uint8_t *) &(s[(int) memory_trace]);
                memory_trace += (((unsigned long) node_number)*sizeof(uint8_t));
                id_resource   = (uint8_t *) &(s[(int) memory_trace]);
                memory_trace += (((unsigned long) node_number)*sizeof(uint8_t));
                dependecies_level_satisfy = (uint8_t *) &(s[(int) memory_trace]);
                memory_trace += (((unsigned long) node_number)*sizeof(uint8_t));

                #ifdef TESTING_MEMORY
                printf("%d is covered with memory from %d to %d -- node number %d -- op %d\n", idx, 
                    (int) (shared_memory_size_offset + threadIdx.x*shared_memory_size), (int) memory_trace, node_number, operation_number);
                __syncthreads();
                #endif
                
                // Set intial node property
                for(i = 0; i < node_number; i++)
                {
                    dependecies_level_satisfy[i] = (uint8_t) node[i].dependecies_level;
                    state[i]                     = (uint8_t) Idle;
                }

                #ifdef TESTING_SCHEDULING
                if(idx == 1035 && k_comb == 3) {
                    printf("START SCHEDULING WITH: \n");
                    for(i = 0; i < k_comb; i++)
                        printf("\t%d %d\n", final[i], repeat[i]);
                    printf("\n");

                    printf("RESOURCES: \n");
                    for(i = 0; i < resources_number; i++)
                        printf("\t%d %d %d %d\n", resources[i].id,  resources[i].area,  resources[i].speed, resources[i].occurency);
                    printf("\n");
                }
                #endif

                uint8_t index_node;
                while (flag)
                {
                    #ifdef TESTING_SCHEDULING
                    if(idx == 1035 && k_comb == 3) {
                        printf("START time %d\n", time+1);
                        printf("See IDLE node\n");
                    }
                    #endif
                    flag = 0;
                    // check between all operation and find node that can be scheduled or that are in execution, 
                    // in case you find nothing this means that all nodes hande been scheduled
                    for(i = 0; i < k_comb; i++) 
                    {
                        #ifdef TESTING_SCHEDULING
                        if(idx == 1035 && k_comb == 3) {
                            printf("res %d - op %d - occ %d\n", final[i], resources[final[i]].index_operation, resources[final[i]].occurency);
                        }
                        #endif
                        // Put some node from idle to executed state
                        if(resources[final[i]].occurency > 0)
                        {
                            // TO DO 3: improvo exit cycle
                            for(j = 0; j < Operation[resources[final[i]].index_operation].index_next_node_occurency; j++)
                            {
                                index_node = Operation[resources[final[i]].index_operation].index_next_node[j];
                                // Check if exist a node that has parents scheduled and is in Idle state
                                if(dependecies_level_satisfy[index_node] == 0 && state[index_node] == Idle)
                                {
                                    flag = 1;
                                    // Associate the resources to the node and decrease the occurency
                                    remain_time[index_node] =  (uint8_t) resources[final[i]].speed;
                                    id_resource[index_node] =  (uint8_t) final[i];
                                    state[index_node]       =  (uint8_t) Execution;                               
                                    resources[final[i]].occurency--;
                                    #ifdef TESTING_SCHEDULING
                                    if(idx == 1035 && k_comb == 3) {
                                        printf("Scheduling node %d at time %d with resources %d (remainign %d) - will finish at %d\n", index_node, time+1, 
                                            id_resource[index_node], resources[final[i]].occurency, time + remain_time[index_node]);
                                    }
                                    #endif
                                    if (resources[final[i]].occurency == 0)
                                        break;
                                }
                            }
                        }
                    }
                    
                    #ifdef TESTING_SCHEDULING
                    if(idx == 1035 && k_comb == 3) {
                        printf("See EXECUTE node\n");
                        for(j = 0; j < node_number; j++)
                            printf("Node %d state %d dep %d\n", node[j].id_node, state[j], dependecies_level_satisfy[j]);
                    }
                    #endif

                    // Put some node from idle to executed state
                    for(j = 0; j < node_number; j++)
                    {
                        // Check if exist a node that has parents scheduled and is in Idle state
                        if(state[j] == Execution)
                        {
                            flag = 1;
                            if (remain_time[j] == 1) 
                            {
                                #ifdef TESTING_SCHEDULING
                                if(idx == 1035 && k_comb == 3) {
                                    printf("END node %d (op %d -- state %d) at time %d with resources %d\n", j, node[j].index_operation, state[j], time+1, id_resource[j]);
                                }
                                #endif
                                // Node terminates to use the resource and all his dependencies have to be free
                                state[j] = Finish;
                                resources[id_resource[j]].occurency++;
                                for(z = 0; z < node[j].index_next_node_occurency; z++)
                                    dependecies_level_satisfy[node[j].index_next_node[z]]--; 
                            } else {
                                remain_time[j]--;
                                #ifdef TESTING_SCHEDULING
                                if(idx == 1035 && k_comb == 3) {
                                    printf("Node %d (op %d -- state %d) at time %d with resources %d\n", j, node[j].index_operation, state[j], time+1, id_resource[j]);
                                }
                                #endif
                            }
                        }
                    }
                    
                    #ifdef TESTING_SCHEDULING
                    if(idx == 1035 && k_comb == 3) {
                        printf("End time %d\n\n", time+1);
                    }
                    #endif

                    time++;
                } // End scheduling
            }
        }
       
        
        #ifdef TESTING
        for(j = start_comb; j < end_comb; j++)
        {   
            __syncthreads();
            if(j == idx)
            {
                if (time == -1)
                {
                    printf("idx: %d --> No combination for ", idx);
                    for(i = 0; i < k_comb; i++)
                        printf("%d  ", final[i]);
                } else {
                    printf("idx: %d - Best time: %d - area: %d\n", idx, time, area);
                    for(i = 0; i < k_comb; i++)
                        printf("%d  ", final[i]);
                        printf("\n");
                    for(i = 0; i < k_comb; i++)
                    {                        
                        printf("\tid: %d - occurency: %d - area: %d - speed: %d\n ",
                            final[i], repeat[i], 
                            resources[final[i]].area, resources[final[i]].speed);
                    }
                }
                printf("\n");   
            }
        }
        #endif

        // check the best time
        for(i = start_comb; i < end_comb; i++)
        {
            __syncthreads();
            if(i == idx) 
            {
                if (time > -1 && (*final_time > time
                    || (time == *final_time && *final_area > area)))
                {
                    *final_time = time;
                    *final_area = area;
                    for(j = 0; j < k_comb; j++) 
                    {
                        final_best_combination[j] = final[j];
                        final_best_repetition[j] =  repeat[j];
                    }
                }
            }
        }

        if(idx == start_comb) 
        {
            // copy here all data
            *final_best_time = *final_time;
            *final_area_calculated = *final_area;
        }
    } // End check if rigth thread
    return;
 } // End combination()
  
 int main(int argc, char const *argv[])
 {
    int app;            // for read int
    int i, j, k, z;    // use like iterator

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

    printf("START reading operations\n");

    operation_t *Operation;
    Operation = (operation_t *)malloc(sizeof(operation_t)*operation_number);

    uint8_t resource_number = 0;
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
        Operation[i].used    = 0; 
        Operation[i].max_index_next_node_occurency = 4; 
        Operation[i].index_next_node = (uint8_t *)malloc(sizeof(uint8_t)*4);
        Operation[i].index_next_node_occurency = 0;
        Operation[i].res = (resource_t *)malloc(sizeof(resource_t)*len);
        // Read how many resources are avaiable for executed this operation and
        // read all its property (speed and area)
        for(j = 0; j < len; j++)
        {
            // use app to avoid problem whit int scanf that use 32 bits
            fscanf(fp, "%d", &Operation[i].res[j].area);
            fscanf(fp, "%d", &app);
            Operation[i].res[j].speed = app;
            Operation[i].res[j].id    = resource_number++;
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
    uint8_t node_number;
    fscanf(fp, "%d", &app);
    node_number = app;

    printf("START reading nodes\n");

    node_t *node;
    node = (node_t *)malloc(sizeof(node_t)*node_number);

    uint8_t operation_used = 0;
    resource_number = 0;

    char temp1[8];
    char temp2[8];
    for(i = 0; i < node_number; i++) 
    {
        fscanf(fp, "%s", temp1);
        fscanf(fp, "%s", temp2);
        printf("%d %s %s\n", i, temp1, temp2);
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
                if(Operation[j].used == 0)
                {
                    Operation[j].used = 1;
                    operation_used++;
                }
                node[i].index_operation = j;
                // Add index to list of node in the propr operation
                if(Operation[j].max_index_next_node_occurency == Operation[j].index_next_node_occurency) 
                {
                    #ifdef TESTING
                    printf("\tREALLOC from %d ... ", Operation[j].max_index_next_node_occurency);
                    #endif
                    Operation[j].max_index_next_node_occurency *= 2;
                    #ifdef TESTING
                    printf("to %d ... ", Operation[j].max_index_next_node_occurency);
                    #endif
                    Operation[j].index_next_node = (uint8_t *)realloc((uint8_t *)Operation[j].index_next_node, sizeof(uint8_t)*Operation[j].max_index_next_node_occurency);
                    #ifdef TESTING
                    printf("done\n");
                    #endif
                }
                Operation[j].index_next_node[Operation[j].index_next_node_occurency++] = i;
                break;
            }
        }
        if (j == operation_number)
        {
            printf("Node with operation that doesn't exist!\n");
            return -2;
        }
    }
    
    // inizialize edge
    uint8_t len_edge;
    fscanf(fp, "%d", &app);
    len_edge = app;

    printf("START reading edge\n");
    uint8_t v, u;
    for(i = 0; i < len_edge; i++) 
    {
        // Read source node
        fscanf(fp, "%s", temp1);
        // Read destination node
        fscanf(fp, "%s", temp2);
        // Check the index of two nodes
        for (j = 0; j < node_number; j++)
        {
            if (strcmp(node[j].name, temp1) == 0)
                u = j;
            else if (strcmp(node[j].name, temp2) == 0)
                v = j;
        }
        
        // Put as one of next node for the one read first
        if(node[u].max_index_next_node_occurency == Operation[u].index_next_node_occurency) 
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
    for(i = 0; i < node_number; i++)
    {
        printf("%d) Node: %s(%d) - Operation: %s" , node[i].id_node, node[i].name, node[i].id_node, Operation[node[i].index_operation].name);
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
        printf("For %s (USED %d) the node are: ", Operation[i].name, Operation[i].used);
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

    // Copy variable to use for GPU purpose
    node_GPU_t *node_GPU = (node_GPU_t *)malloc(node_number*sizeof(node_GPU_t));
    for (i = 0; i < node_number; i++)
    {
        node_GPU[i].id_node           = node[i].id_node;
        node_GPU[i].dep1_index        = node[i].dep1_index;
        node_GPU[i].dep2_index        = node[i].dep2_index;
        node_GPU[i].dependecies_level = node[i].dependecies_level;
        node_GPU[i].index_operation   = node[i].index_operation;
        node_GPU[i].index_next_node_occurency = node[i].index_next_node_occurency;
        node_GPU[i].index_next_node = (uint8_t *)malloc(sizeof(uint8_t)*node[i].index_next_node_occurency);
        for (j = 0; j < node[i].index_next_node_occurency; j++)
            node_GPU[i].index_next_node[j] = node[i].index_next_node[j];
    }

    operation_t *New_Operation     = (operation_t *)malloc(operation_used*sizeof(operation_t));
    operation_GPU_t *Operation_GPU = (operation_GPU_t *)malloc(operation_used*sizeof(operation_GPU_t));
    for(i = 0, resource_number = 0, k = 0; i < operation_number && k < operation_used; i++)
    {   
        if(Operation[i].used == 1)
        {
            New_Operation[k] = Operation[i];
            New_Operation[k].operation_id = k;
            Operation_GPU[k].operation_id = k;
            // copy next node occurency
            Operation_GPU[k].index_next_node_occurency = Operation[i].index_next_node_occurency;
            Operation_GPU[k].index_next_node           = Operation[i].index_next_node;
            for(j = 0; j < Operation[i].index_next_node_occurency; j++){
                node[Operation[i].index_next_node[j]].index_operation     = k;
                node_GPU[Operation[i].index_next_node[j]].index_operation = k;
            }
            // copy resources occurency
            Operation_GPU[k].res_occurency = Operation[i].res_occurency;
            Operation_GPU[k].res           = Operation[i].res;
            for (j = 0; j < Operation[i].res_occurency; j++)
            {
                Operation_GPU[k].res[j].id = resource_number++;
                Operation_GPU[k].res[j].index_operation = k;
            }
            k++;
        }
    }
    operation_number = operation_used;
    Operation = New_Operation;

    printf("\nNODE to GPU\n\n");
    for(i = 0; i < node_number; i++)
    {
        printf("%d) Node: %s(%d) - Operation: %s(%d)" , node_GPU[i].id_node, node[node_GPU[i].id_node].name, node_GPU[i].id_node, Operation[node_GPU[i].index_operation].name, node_GPU[i].index_operation);
        if (node[i].dependecies_level != 0) {
            printf(" - Dependecies: ");
            if (node[i].dep1_index != EMPTY_INDEX)
                printf("%s ", node[node_GPU[i].dep1_index].name);
            if (node[i].dep2_index != EMPTY_INDEX)
                printf("%s ", node[node_GPU[i].dep2_index].name);
        }
        if (node[i].index_next_node_occurency > 0) 
        {
            printf(" - Next node:   ");
            for(j = 0; j < node_GPU[i].index_next_node_occurency; j++)
                printf("%s ", node[node_GPU[i].index_next_node[j]].name);
        }
        printf("\n");
    }

    printf("\nRESOURCES to GPU\n\n");
    for(i = 0; i < operation_number; i++)
    {
        printf("For %s(%d) the node are: ", Operation[Operation_GPU[i].operation_id].name, Operation_GPU[i].operation_id);
        for(j = 0; j < Operation[i].index_next_node_occurency; j++)
            printf("%s ", node[Operation_GPU[i].index_next_node[j]].name);
        printf("\n");
        printf("\tID Area Speed\n");
        for(j = 0; j < Operation_GPU[i].res_occurency; j++)
        {
            printf("%d)\t%2d %4d %4d\n", j, Operation[i].res[j].id, Operation[i].res[j].area, Operation[i].res[j].speed);
        }
    }
    printf("\n");

    // variables used for GPU
    int stream_number = 0;
    const int max_stream_number = MAX_BLOCKS;
    cudaStream_t streams[max_stream_number];
    for(i = 0; i < max_stream_number; i++)
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);

    int final_best_time[max_stream_number], *dev_final_best_time; 
    int final_area_calculated[max_stream_number], *dev_final_area_calculated; 
    uint8_t *final_best_combination[max_stream_number], *dev_final_best_combination[max_stream_number]; 
    uint8_t *final_best_repetition[max_stream_number], *dev_final_best_repetition[max_stream_number]; 
    operation_GPU_t *dev_Operation;
    node_GPU_t *dev_node;

    uint8_t *dev_app;

    // Allocatr GPU memory
    cudaMalloc(&dev_Operation, operation_number*sizeof(operation_GPU_t));
    cudaMemcpy(dev_Operation, Operation_GPU, operation_number*sizeof(operation_GPU_t), cudaMemcpyHostToDevice);
    // Allocate the right quantity for store the proper dimension of array in each structure
    for(i = 0; i < operation_number; i++)
    {
        // Copy resources
        cudaMalloc(&dev_app, Operation_GPU[i].res_occurency*sizeof(resource_t));
        cudaMemcpy(dev_app, Operation_GPU[i].res, Operation_GPU[i].res_occurency*sizeof(resource_t), cudaMemcpyHostToDevice);
        cudaMemcpy(&(dev_Operation[i].res), &dev_app, sizeof(uint8_t *), cudaMemcpyHostToDevice);
        // Copy index nodes
        cudaMalloc(&dev_app, Operation_GPU[i].index_next_node_occurency*sizeof(uint8_t));
        cudaMemcpy(dev_app, Operation_GPU[i].index_next_node, Operation_GPU[i].index_next_node_occurency*sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(&(dev_Operation[i].index_next_node), &dev_app, sizeof(uint8_t *), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&dev_node, node_number*sizeof(node_GPU_t));
    cudaMemcpy(dev_node, node_GPU, node_number*sizeof(node_GPU_t), cudaMemcpyHostToDevice);

    for(i = 0; i < node_number; i++)
    {
        // Copy next index nodes
        cudaMalloc(&dev_app, node_GPU[i].index_next_node_occurency*sizeof(uint8_t));
        cudaMemcpy(dev_app, node_GPU[i].index_next_node, node_GPU[i].index_next_node_occurency*sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(&(dev_node[i].index_next_node), &dev_app, sizeof(uint8_t *), cudaMemcpyHostToDevice);
    }

    // store the value for comparison
    uint8_t *best_final            = (uint8_t *)malloc(sizeof(uint8_t)*(resource_number+1));   
    uint8_t *best_final_repetition = (uint8_t *)malloc(sizeof(uint8_t)*resource_number);
    int best_time = 0x7fffffff;
    int area_calculated = 0x7fffffff;
    int area_limit = atoi(argv[3]);

    int shared_memory_size;
    int tot_shared_memory;
    int offset_shared_memory_size = int (operation_number*sizeof(operation_GPU_t) +
                                        node_number*sizeof(node_GPU_t)) +
                                        2*sizeof(int);

    printf("Number of possible resource is %d\n", resource_number);
    printf("k min is %d and k max is %d\n", operation_used, resource_number);
    printf("Offset shared memory is %d\n", offset_shared_memory_size);
    printf("\n");

    cudaMalloc(&dev_final_best_time, sizeof(int)*max_stream_number);
    cudaMalloc(&dev_final_area_calculated, sizeof(int)*max_stream_number);

    for(i = 0; i < max_stream_number; i++)
    {
        // allocate with max number possible
        cudaMalloc(&dev_final_best_combination[i], resource_number*sizeof(uint8_t));
        final_best_combination[i] = (uint8_t *)malloc(resource_number*sizeof(uint8_t));

        cudaMalloc(&dev_final_best_repetition[i], resource_number*sizeof(uint8_t));
        final_best_repetition[i] = (uint8_t *)malloc(resource_number*sizeof(uint8_t));
    }

    // variable used for calculate internaly the idx for combination and the proper repetition
    int max_repetition = 3;
    int factor;

    // Invoke kernel
    int threadsPerBlock_d;
    int end_comb = 0;
    int start_comb = 0;
    int saved_k[max_stream_number];
    
    // to store the execution time of code
    double time_spent = 0.0;
    cudaError_t cuda_error;
 
    clock_t begin = clock();
    // how big are the cutset, modify it iteratively
    //for(k = 5; k <= 5; k++) {
    for(k = operation_used; k <= resource_number; k++) {
        // calculate number of combinations
        int n_f = 1; // nominatore fattoriale
        for (i = resource_number; i > k; i--) n_f *= i;
        int d_f = 1; // denominatore fattoriale
        for (i = 1; i <= resource_number - k ; i++) d_f *= i;
        int tot_comb = n_f/d_f;

        // sum of all vector inside kernel
        shared_memory_size = (int) (k*((int) sizeof(uint8_t))*2 +
                                operation_number*((int) sizeof(uint8_t)) +
                                node_number*((int) sizeof(uint8_t))*4);
        
        printf("Number of total combination witk k equal to %d are: %d -- ", k, tot_comb);
        
        factor = 1;
        for(i = 0; i < max_repetition; i++)
            factor *= k;
        tot_comb *= factor;
        printf("thread are %d\n", tot_comb);
        #ifdef TESTING_MEMORY
            printf("Piece of shared memory is %d\n", shared_memory_size);
        #endif     

        end_comb = 0;
        // Go among group of MAX_BLOCKS
        while(end_comb != tot_comb)
        {
            start_comb        = end_comb;
            threadsPerBlock_d = (int) (MAX_SHARED_MEMORY - offset_shared_memory_size)/shared_memory_size;
            if(threadsPerBlock_d > MAX_THREADS)
                threadsPerBlock_d = MAX_THREADS;
            end_comb   = threadsPerBlock_d + start_comb;
            if (end_comb > tot_comb)
            {
                end_comb = tot_comb;
                threadsPerBlock_d = tot_comb - start_comb;
            }

            tot_shared_memory = offset_shared_memory_size + (shared_memory_size*threadsPerBlock_d);

            #ifdef TESTING_MEMORY
            printf("\tStart comb is %d -- end comb is %d -- thread are %d -- sahred memory is %d\n",
                start_comb, end_comb, threadsPerBlock_d, tot_shared_memory);
            #endif

            // call kernel
            combination<<<1, threadsPerBlock_d, tot_shared_memory, streams[stream_number]>>>(
                resource_number, k, tot_comb, start_comb, end_comb, 
                shared_memory_size, offset_shared_memory_size, max_repetition, factor,
                 dev_Operation, operation_number, dev_node, node_number, area_limit, resource_number,
                dev_final_best_combination[stream_number], dev_final_best_repetition[stream_number],
                &(dev_final_best_time[stream_number]), &(dev_final_area_calculated[stream_number]));
            
            cuda_error = cudaGetLastError();
            if(cuda_error != cudaSuccess)
            {
                printf("ERROR : %s\n", cudaGetErrorString(cuda_error));
                return -1;
            }
            
            saved_k[stream_number++] = k;

            if (stream_number == max_stream_number || (k == resource_number && end_comb == tot_comb))
            {
                printf("Arrived with waiting %d streams\n", stream_number);

                for(i = 0; i < stream_number; i++) 
                {   
                    #ifdef TESTING_MEMORY
                    printf("\tWaiting for stream %d ... ", i);
                    #endif
                    cudaStreamSynchronize(streams[i]);
                   
                    cudaMemcpyAsync(&(final_best_time[i]),       &(dev_final_best_time[i]),       sizeof(int),       cudaMemcpyDeviceToHost, streams[i]);
                    cudaMemcpyAsync(&(final_area_calculated[i]), &(dev_final_area_calculated[i]), sizeof(int),       cudaMemcpyDeviceToHost, streams[i]);

                    if(final_best_time[i] > -1 && ((final_best_time[i]  < best_time) 
                        || (final_best_time[i]  == best_time && final_area_calculated[i]  < area_calculated)))
                    {
                        cudaMemcpyAsync(final_best_combination[i], dev_final_best_combination[i], saved_k[i]*sizeof(uint8_t), cudaMemcpyDeviceToHost, streams[i]);
                        cudaMemcpyAsync(final_best_repetition[i],  dev_final_best_repetition[i],  saved_k[i]*sizeof(uint8_t), cudaMemcpyDeviceToHost, streams[i]);

                        for(z = 0; z < saved_k[i]; z++)
                        {
                            best_final[z]            = final_best_combination[i][z];
                            best_final_repetition[z] = final_best_repetition[i][z];
                        }

                        best_final[z]   = EMPTY_INDEX;
                        best_time       = final_best_time[i];
                        area_calculated = final_area_calculated[i];
                    }

                    #ifdef TESTING_MEMORY
                    printf(" \t\t ... END\n");
                    #endif
                } 
                stream_number = 0;  
            }
            
        }
                            
    } // END For k subset

    clock_t end = clock();

    // calculate elapsed time by finding difference (end - begin) and
    // dividing the difference by CLOCKS_PER_SEC to convert to seconds
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;

    cudaFree(dev_final_best_time);
    cudaFree(dev_final_area_calculated);
    for(i = 0; i < max_stream_number; i++)
    {        
        cudaFree(dev_final_best_repetition[i]);
        cudaFree(dev_final_best_combination[i]);
        free(final_best_combination[i]);
        free(final_best_repetition[i]);
    }
    
    /** Print the best solution obtained */
    fp = fopen("log_v2_0.log", "a");
    
    time_t rawtime;
    struct tm * timeinfo;

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );

    fprintf (fp, "--------------------------------------------------\n");
    fprintf (fp, "Current local time and date: %s\n", asctime(timeinfo) );
    fprintf (fp, "DFG is %s\n", argv[1]);
    fprintf (fp, "Reasources are %s\n", argv[2]);
    fprintf(fp, "Area Limit is %d\n", area_limit);
    fprintf (fp, "--------------------------------------------------\n\n");


    fprintf(fp, "\nArea Limit is %d\n", area_limit);
    fprintf(stdout, "\nArea Limit is %d\n", area_limit);
    fprintf(fp, "Best solution has time %d:\n", best_time);
    fprintf(stdout, "Best solution has time %d:\n", best_time);
    for(i = 0; i < resource_number && best_final[i] != EMPTY_INDEX; i++) 
    {
        for(j = 0; j < operation_number; j++) 
        {
            for(k = 0; k < Operation[j].res_occurency; k++) 
            {
                if (best_final[i] == Operation[j].res[k].id)
                {
                    fprintf(stdout, "\tOPERATION: %4s - ID RESOURCE: %2d - SPEED: %2d - AREA: %2d - OCCURENCY: %2d\n", 
                        Operation[j].name, Operation[j].res[k].id, Operation[j].res[k].speed, Operation[j].res[k].area, best_final_repetition[i]);
                    fprintf(fp, "\tOPERATION: %4s - ID RESOURCE: %2d - SPEED: %2d - AREA: %2d - OCCURENCY: %2d\n", 
                        Operation[j].name, Operation[j].res[k].id, Operation[j].res[k].speed, Operation[j].res[k].area, best_final_repetition[i]);

                }
            }
        }
    }

    fprintf(stdout, "Final area is %d\n", area_calculated);
    fprintf(fp, "Final area is %d\n", area_calculated);
  
    printf("\nThe elapsed time is %f seconds\n", time_spent);
    fprintf(fp,"\nThe elapsed time is %f seconds\n\n", time_spent);

    cudaFree(dev_node);
    cudaFree(dev_Operation);

    cudaDeviceReset();
    return 0;
 }
  