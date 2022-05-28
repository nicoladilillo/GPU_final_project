/**
 * Thi simpler example just to see how to use simple combination
 * in a parralel way on GPU. The elaborated combination don't follow 
 * the lexograpical order.
 * 
 * Use shared memory
 */

 #include <stdio.h>
 #include "dfg.h"

// #define TESTING_OP_AND_NODE
// #define TESTING
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
 __global__ void combination(const uint8_t n, int r, const int tot_comb, int const shared_memory_size,
    int const shared_memory_size_offset, const operation_GPU_t *Operation_init, const node_GPU_t *node_init,
    const int operation_number, const int node_number, const uint8_t resources_number, const int area_limit,
    uint8_t *final_best_combination, uint8_t *final_best_repetition,
    int *final_best_time, int *final_area_calculated)
 {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < tot_comb) {
        extern  __shared__ unsigned char s[];

        uint8_t i, j, z, k;

        int k_comb = r;
        int best_time = 0x7fffffff;
        int area_calculated = 0x7fffffff;
        int area = 0;
        
        uint8_t index = 0;
        uint8_t flag;
        int time;

        // This variable can be shared between threads in the same block
        node_GPU_t *node;
        operation_GPU_t *Operation;

        node = (node_GPU_t *) &(s[0]);            
        Operation = (operation_GPU_t *) &(s[(int) (node_number*sizeof(node_GPU_t))]);

        // use only one instanze for all nodes and operation information
        if (threadIdx.x == 0)
        {
            // TO.DO: include also the index node and resources

            // Copy operations information
            for(i = 0; i < operation_number; i++) 
                Operation[i] = Operation_init[i];

            // Copy nodes information
            for(i = 0; i < node_number; i++)
                node[i] = node_init[i];
        } 
        __syncthreads();

        // offset between group of array thread
        int memory_trace = (int) (shared_memory_size_offset + idx*shared_memory_size);
        
        // lenght k_comb
        uint8_t *final = (uint8_t *) &(s[memory_trace]);
        memory_trace += (int) (k_comb*sizeof(uint8_t));
        // lenght k_comb
        int *all_aree = (int *) &(s[memory_trace]);
        memory_trace += (int) (k_comb*sizeof(int));

        // can speed up the overall process coping in the local register
        // lenght operation_number
        resource_t *resources = (resource_t *) &(s[memory_trace]);
        memory_trace += (int) (resources_number*sizeof(resource_t));
        // variable used for operation covered
        uint8_t *operation_covered = (uint8_t *) &(s[memory_trace]);
        memory_trace += (int) (operation_number*sizeof(uint8_t));
        
        // variable used from scheduling node
        uint8_t *state = (uint8_t *) &(s[memory_trace]);
        memory_trace += (int) (node_number*sizeof(uint8_t));
        uint8_t *remain_time = (uint8_t *) &(s[memory_trace]);
        memory_trace += (int) (node_number*sizeof(uint8_t));
        uint8_t *id_resource = (uint8_t *) &(s[memory_trace]);
        memory_trace += (int) (node_number*sizeof(uint8_t));
        uint8_t *dependecies_level_satisfy = (uint8_t *) &(s[memory_trace]);
        memory_trace += (int) (node_number*sizeof(uint8_t));

        // lenght K_comb
        uint8_t *repeat = (uint8_t *) &(s[memory_trace]);
        memory_trace += (int) (k_comb*sizeof(uint8_t));
        uint8_t *best_combination = (uint8_t *) &(s[memory_trace]);
        memory_trace += (int) (k_comb*sizeof(uint8_t));
        uint8_t *best_repetition = (uint8_t *) &(s[memory_trace]);
        memory_trace += (int) (k_comb*sizeof(uint8_t));        

        for(i = 0; i < operation_number; i++) 
            operation_covered[i] = 0;

        uint8_t a = n;
        int b = k_comb;
        int x = idx; // x is the "dual" of m

        // calculate the combination
        for (i = 0; i < k_comb; i++)
        {
            --a;
            while (Choose((int) a,b) > x)
                --a;
            x = x - Choose((int) a,b);
            final[i] = a;
            b = b-1;
        }
        
        #ifdef TESTING_OP_AND_NODE
        // synchronize the local threads writing to the local memory cache
        __syncthreads();

        // check the best time
        if(idx == tot_comb-1)
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

        #ifdef TESTING
        for(i = 0; i < tot_comb ; i++)
        {   
            __syncthreads();
            if(idx == i)
            {
                printf("\t%d) ", i);
                for(j = 0; j < k_comb; j++) 
                    printf("%d  ", final[j]);
                printf(" -- Memory is %d, from %d to %d \n", memory_trace,
                    (int)( shared_memory_size_offset + idx*shared_memory_size),
                    (int)( shared_memory_size_offset + (idx+1)*shared_memory_size));
            }
        }
        __syncthreads();
        #endif

        // assign resources and check if resources used cover all operations
        k = 0;
        area = 0;
        for(z = 0; z < k_comb; z++)
        {
            for(i = 0; i < operation_number; i++)
            {
                for(j = 0; j < Operation[i].res_occurency; j++)
                {
                    if (Operation[i].res[j].id == final[z])
                    {
                        operation_covered[i] = 1;
                        resources[final[z]] = Operation[i].res[j];
                        resources[final[z]].index_operation = i;
                        resources[final[z]].occurency = 1;
                        repeat[k] = 1;
                        all_aree[k++] = resources[final[z]].area;
                        area += resources[final[z]].area;
                    }
                }
            }
        }

        // all others repeated combination will be bigger 
        if (area > area_limit)
            index = k_comb;
        else {
            // work with repetition, with a maximum of area_limit
            for(i = 0; i < operation_number; i++)
            {
                if (operation_covered[i] != 1)
                    index = k_comb;
            }
        }
   
        // start repeat combination
        while(index != k_comb)
        {            
            // set occurency for each resources
            for(i = 0; i < k_comb; i++)
                resources[final[i]].occurency = repeat[i];

            // Set intial node property
            for(i = 0; i < node_number; i++)
            {
                dependecies_level_satisfy[i] = node[i].dependecies_level;
                state[i] = Idle;
                remain_time[i] = 0;
            }

            #ifdef TESING_SCHEDULING
            if(idx == 7 && k_comb == 3) {
                printf("START SCHEDULING WITH: \n");
                for(i = 0; i < k_comb; i++)
                    printf("\t%d %d\n", final[i], repeat[i]);
                printf("\n");
            }
            #endif

            flag = 0;
            if (area <= area_limit)
                flag = 1;
            time = -1;
            uint8_t index_node;
            while (flag)
            {
                #ifdef TESING_SCHEDULING
                if(idx == 7 && k_comb == 3) {
                    printf("START time %d\n", time+1);
                    printf("See IDLE node\n");
                }
                #endif
                flag = 0;
                // check between all operation and find node that can be scheduled or that are in execution, 
                // in case you find nothing this means that all nodes hande been scheduled
                for(i = 0; i < k_comb; i++) 
                {
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
                                remain_time[index_node] = resources[final[i]].speed;
                                id_resource[index_node] = final[i];
                                state[index_node] = Execution;                               
                                resources[final[i]].occurency--;
                                #ifdef TESING_SCHEDULING
                                if(idx == 7 && k_comb == 3) {
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
                
                #ifdef TESING_SCHEDULING
                if(idx == 7 && k_comb == 3) {
                    printf("See EXECUTE node\n");
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
                            #ifdef TESING_SCHEDULING
                            if(idx == 7 && k_comb == 3) {
                                printf("END node %d (op %d -- state %d) at time %d with resources %d\n", j, node[j].index_operation, state[index], time+1, id_resource[j]);
                            }
                            #endif
                            // Node terminates to use the resource and all his dependencies have to be free
                            state[j] = Finish;
                            resources[id_resource[j]].occurency++;
                            for(z = 0; z < node[j].index_next_node_occurency; z++)
                                dependecies_level_satisfy[node[j].index_next_node[z]]--; 
                        } else {
                            remain_time[j]--;
                            #ifdef TESING_SCHEDULING
                            if(idx == 7 && k_comb == 3) {
                                printf("Node %d (op %d -- state %d) at time %d with resources %d\n", j, node[j].index_operation, state[index], time+1, id_resource[j]);
                            }
                            #endif
                        }
                    }
                }
                
                #ifdef TESING_SCHEDULING
                if(idx == 7 && k_comb == 3) {
                    printf("End time %d\n\n", time+1);
                }
                #endif

                time++;
            } // End scheduling

            // see if a better result has been achived
            if(time > -1 && ((time < best_time) || (time == best_time && area < area_calculated)))
            //if(time > -1 && time < best_time)
            {
                for(i = 0; i < k_comb; i++) 
                {
                    // TO_DO2: save them in variable and then copy nack in shared memory
                    best_combination[i] = final[i];
                    best_repetition[i]  = repeat[i];
                }
                area_calculated = area;
                best_time = time;
            }


            // Calculate the new repetition and the new area value 
            // go haed only if are is lesser than area_limit
            index = 0;
            int max_repetition = 4;
            do {
                while(index < k_comb && ((area + all_aree[index] > area_limit) || repeat[index] == max_repetition))
                {
                    for(i = 1; i < repeat[index]; i++)
                        area -= all_aree[index];
                    repeat[index] = 1;
                    index++;
                }
                
                if (index < k_comb)
                {
                    repeat[index]++;
                    area += all_aree[index];
                }

            } while (index != k_comb && area > area_limit);
            
        }// End repeat combination
        
        // TO_DO1: save result using temporaly register
        final_best_time[idx] = best_time;
        final_area_calculated[idx] = area_calculated;
        for(i = 0; i < k_comb; i++)
        {
            final_best_combination[idx*k_comb+i] = best_combination[i];
            final_best_repetition[idx*k_comb+i] = best_repetition[i];
        }

        #ifdef TESTING
        for(j = 0; j < tot_comb; j++)
        {   
            int area_app, speed_app;
            __syncthreads();
            if(idx == j)
            {
                if (best_time == 0x7fffffff)
                {
                    printf("idx: %d --> No combination for ", j);
                    for(i = 0; i < k_comb; i++)
                        printf("%d  ", final[i]);
                } else {
                    printf("idx: %d - Best time: %d - area: %d\n", j, best_time, area_calculated);
                    for(i = 0; i < k_comb; i++)
                    {
                        for(z = 0; z < operation_number; z++)
                        {
                            for (k = 0; k < Operation[z].res_occurency; k++){
                                if(Operation[z].res[k].id== final_best_combination[idx*k_comb+i])
                                {
                                    area_app = Operation[z].res[k].area;
                                    speed_app = Operation[z].res[k].speed;
                                }
                            }
                        }
                            
                        printf("\tid: %d - occurency: %d - area: %d - speed: %d\n ", final_best_combination[idx*k_comb+i], final_best_repetition[idx*k_comb+i], area_app, speed_app);
                    }
                }
                printf("\n");   
            }
        }
        #endif
        
        // check the best time
        if(idx == 0)
        {
            __syncthreads();
            for(i = 1; i < tot_comb; i++)
            {   
                // TO-DO 2: use shared memory
                if (best_time > -1 && (best_time > final_best_time[i]
                    || (best_time == final_best_time[i] && area_calculated > final_area_calculated[i])))
                {
                    final_best_time[0] = final_best_time[i];
                    best_time = final_best_time[i];
                    final_area_calculated[0] = final_area_calculated[i];
                    area_calculated = final_area_calculated[i];
                    for(j = 0; j < k_comb; j++) 
                    {
                        final_best_combination[j] = final_best_combination[i*k_comb+j];
                        final_best_repetition[j] = final_best_repetition[i*k_comb+j];
                    }
                }
            }
        }
    } // End check if rigth thread
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
            //Operation_GPU[k].res = (resource_t *)malloc(sizeof(resource_t)*Operation[i].res_occurency);
            for (j = 0; j < Operation[i].res_occurency; j++)
            {
                // Operation_GPU[k].res[j] = Operation[i].res[j];
                // Change id to resources and index operation
                Operation_GPU[k].res[j].id = resource_number++;
                Operation_GPU[k].res[j].index_operation = k;
            }
            // Operation_GPU[k].index_next_node_occurency = Operation[i].index_next_node_occurency;
            // Operation_GPU[k].index_next_node = (uint8_t *)malloc(sizeof(uint8_t)*Operation[i].index_next_node_occurency);
            // for (j = 0; j < Operation[i].index_next_node_occurency; j++)
            // {
            //     // if(i == 0)
            //     //     printf("%s %d %d %s\n", Operation[i].name, Operation[i].index_next_node_occurency, j, node[Operation[i].index_next_node[j]].name);
            //     Operation_GPU[k].index_next_node[j] = Operation[i].index_next_node[j];
            // }
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
    int final_best_time, *dev_final_best_time;
    int final_area_calculated, *dev_final_area_calculated;
    uint8_t *final_best_combination, *dev_final_best_combination;
    uint8_t *final_best_repetition, *dev_final_best_repetition;
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
    uint8_t *best_final = (uint8_t *)malloc(sizeof(uint8_t)*(resource_number+1));   
    uint8_t *best_final_repetition = (uint8_t *)malloc(sizeof(uint8_t)*resource_number);
    int best_time = 0x7fffffff;
    int area_calculated = 0x7fffffff;
    int area_limit = atoi(argv[3]);

    printf("Number of possible resource is %d\n", resource_number);
    printf("k min is %d and k max is %d\n\n", operation_used, resource_number);

    int shared_memory_size;
    int tot_shared_memory;
    int offset_shared_memory_size = int (operation_number*sizeof(operation_GPU_t) + 
                                        node_number*sizeof(node_GPU_t));

    // to store the execution time of code
    double time_spent = 0.0;
 
    clock_t begin = clock();
    // how big are the cutset, modify it iteratively
    for(k = operation_used; k <= resource_number; k++) {
        // calculate number of combinations
        int n_f = 1; // nominatore fattoriale
        for (i = resource_number; i > k; i--) n_f *= i;
        int d_f = 1; // denominatore fattoriale
        for (i = 1; i <= resource_number - k ; i++) d_f *= i;
        int tot_comb = n_f/d_f;

        // sum of all vector inside kernel
        shared_memory_size = (int) (k*sizeof(uint8_t)*4 +
                                k*sizeof(int) +
                                operation_number*sizeof(uint8_t) +
                                resource_number*sizeof(resource_t) +
                                node_number*sizeof(uint8_t)*4);
        
        tot_shared_memory = offset_shared_memory_size + (shared_memory_size*tot_comb);
        printf("Number of total combination witk k equal to %d are: %d\n\tShared memory Total is %d -- Piece of shared memory is %d -- Offset is %d\n",
            k, tot_comb, tot_shared_memory, shared_memory_size, offset_shared_memory_size);

        cudaMalloc(&dev_final_best_time, tot_comb*sizeof(int));
        
        cudaMalloc(&dev_final_area_calculated, tot_comb*sizeof(int));

        cudaMalloc(&dev_final_best_combination, k*tot_comb*sizeof(uint8_t));
        final_best_combination = (uint8_t *)malloc(k*sizeof(uint8_t));

        cudaMalloc(&dev_final_best_repetition, k*tot_comb*sizeof(uint8_t));
        final_best_repetition = (uint8_t *)malloc(k*sizeof(uint8_t));

        // call kernel
        combination<<<1, tot_comb, tot_shared_memory>>>(resource_number, k, tot_comb, shared_memory_size,
            offset_shared_memory_size, dev_Operation, dev_node, operation_number, node_number, resource_number, area_limit,
            dev_final_best_combination, dev_final_best_repetition, dev_final_best_time, dev_final_area_calculated);
        
        cudaDeviceSynchronize();

        cudaMemcpy(&final_best_time, dev_final_best_time, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&final_area_calculated, dev_final_area_calculated, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(final_best_combination, dev_final_best_combination, k*sizeof(uint8_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(final_best_repetition, dev_final_best_repetition, k*sizeof(uint8_t), cudaMemcpyDeviceToHost);

        #ifdef TESTING
        printf("Best Combination: ");
        for(i = 0; i < k; i++)
            printf(" %2d %2d,", final_best_combination[i], final_best_repetition[i]);
        printf(" - Time: %d - Area: %d\n", final_best_time, final_area_calculated);
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
            best_final[i] = EMPTY_INDEX;
            best_time = final_best_time;
            area_calculated = final_area_calculated;
        }

        cudaFree(dev_final_best_time);
        cudaFree(dev_final_area_calculated);
        cudaFree(dev_final_best_combination);
        cudaFree(dev_final_best_repetition);
    } // END For k subset
    clock_t end = clock();

    // calculate elapsed time by finding difference (end - begin) and
    // dividing the difference by CLOCKS_PER_SEC to convert to seconds
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    
    /** Print the best solution obtained */
    fp = fopen("log_v6.log", "a");
    
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
  