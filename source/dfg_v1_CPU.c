/**
 * @file dfg_combination_v2.c
 * In this version not all repeated combination has calculated but
 * only the one that have the area minos than area_limit. In order to have 
 * an accettable time of elaboration has been added a further parameter that
 * give a max number of repetition for each resources
 * 
 */
#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <time.h>       // for clock_t, clock(), CLOCKS_PER_SEC

// #define TESTING
// #define SCHEDULING

#define MAX_LIMITATOR 0x7fffffff

#include "dfg.h"

void scheduling_dfg(const int n, const int r, const int final[], FILE *fp, operation_t *Operation, int operation_number, node_t *node, 
const int node_number, int *best_final, int *best_final_repetition, const int area_limit, int *best_time, int *area_calculated)
{
    int i, j, z, k; 
    int area, all_aree[r];    
    int limitator;  
    
    /** Perform a scheduling operation according avaiable resources */
    // assign resources and check if resources used cover all operations
    k = 0;
    for(z = 0; z < r; z++)
    {
        for(i = 0; i < operation_number; i++)
        {
            for(j = 0; j < Operation[i].res_occurency; j++)
            {
                if (Operation[i].res[j].id == final[z]) {
                    Operation[i].covered = 1;
                    all_aree[k++] = Operation[i].res[j].area;
                    #ifdef TESTING
                    if (k > 1)
                        fprintf(fp, "--  ");
                    fprintf(fp, "id: %d - area: %d  ", final[z], Operation[i].res[j].area);
                    #endif
                }
            }
        }
    }
    #ifdef TESTING
    fprintf(fp, "\n");
    #endif
    
    // work with repetition, with a maximum of area 10
    int repeat[r];
    int index = 0;

    area = 0;
    for (j=0; j < r; j++)
    {
        repeat[j] = 1;
        area += all_aree[j];
    }
    // all others repeated combination will be bigger 
    if (area > area_limit)
        index = r;
    else {
        for(i = 0; i < operation_number; i++)
        {
            // check if operation is used and if one of the the resources covers it
            if (Operation[i].covered != 1)
                index = r;
        }
    }

    int flag;
    int time;
    uint8_t node_index;
    limitator = 0;
    while(index != r && limitator < MAX_LIMITATOR) {
        #ifdef TESTING
        fprintf(fp, "\t");
        for (j=0; j<r; j++)
            fprintf(fp, "%d ", repeat[j]);
        fprintf(fp, "\n");
        #endif

        // sett occurency for each resources
        for(z = 0; z < r; z++)
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

        #ifdef SCHEDULING
        if(r == 3 && final[0] == 0 && final[1] == 3 && final[2] == 4) {
            printf("START SCHEDULING WITH: \n");
            for(i = 0; i < r; i++)
                printf("\t%d %d\n", final[i], repeat[i]);
            printf("\n");
        }
        #endif

        /** Scheduling operation */
        flag = 0;
        if (area <= area_limit)
            flag = 1;
        time = -1;
        while (flag)
        {
            #ifdef SCHEDULING
            if(r == 3 && final[0] == 0 && final[1] == 3 && final[2] == 4) {
                printf("START time %d\n", time+1);
                printf("See IDLE node\n");
            }
            #endif
            flag = 0;
            // check between all operation and find node that can be scheduled or that are in execution, 
            // in case you find nothing this means that all nodes hande been scheduled  
            for(i = operation_number-1; i >= 0; i--) 
            {
                for(z = Operation[i].res_occurency-1; z >= 0; z--)
                {
                    // Check if there is some free resorce
                    if (Operation[i].res[z].occurency > 0)
                    {
                        // Try to put some node from idle to executed state
                        for(j = 0; j < Operation[i].index_next_node_occurency; j++)
                        {
                            node_index = Operation[i].index_next_node[j];
                            // Check if exist a node that has parents scheduled and is in Idle state
                            if(node[node_index].dependecies_level_satisfy == 0 && node[node_index].state == Idle)
                            {
                                flag = 1;
                                // Associate the resources to the node and decrease the occurency
                                node[node_index].remain_time = Operation[i].res[z].speed;
                                node[node_index].id_resource = z;
                                node[node_index].state = Execution;
                                Operation[i].res[z].occurency--;
                                #ifdef SCHEDULING
                                if(r == 3 && final[0] == 0 && final[1] == 3 && final[2] == 4) {
                                    printf("Scheduling node %d at time %d with resources %d (remainign %d) - will finish at %d\n", node_index, time+1, 
                                    Operation[i].res[z].id, Operation[i].res[z].occurency, time + node[node_index].remain_time);
                                }
                                #endif
                                // maybe in the gpu you have mispredicted this break
                                if(Operation[i].res[z].occurency == 0)
                                    break;
                            }                                
                        }
                    }
                }
            }

            #ifdef SCHEDULING
            if(r == 3 && final[0] == 0 && final[1] == 3 && final[2] == 4) {
                printf("See EXECUTE node\n");
            }
            #endif
            
            // See how much time is needed for operation to terminated and they can free the resource
            for(j = 0; j < node_number; j++)
            {
                // Check if exist a node that has parents scheduled and is in Idle o Execution state
                if(node[j].state == Execution)
                {
                    flag = 1;
                    if (node[j].remain_time == 1) 
                    {
                        #ifdef SCHEDULING
                        if(r == 3 && final[0] == 0 && final[1] == 3 && final[2] == 4) {
                            printf("END node %d (op %d -- state %d) at time %d with resources %d\n", j, node[j].index_operation, node[j].state,
                                time+1, Operation[node[j].index_operation].res[node[j].id_resource].id);
                        }
                        #endif
                        // printf("\t\t%d %s %d\n",time+1, node[j].name, node[j].index_next_node_occurency);
                        // Node terminates to use the resource and all his dependencies have to be free
                        node[j].state = Finish;
                        Operation[node[j].index_operation].res[node[j].id_resource].occurency++;
                        for(z = 0; z < node[j].index_next_node_occurency; z++)
                            node[node[j].index_next_node[z]].dependecies_level_satisfy--;
                    } else {
                        node[j].remain_time--;
                        #ifdef SCHEDULING
                        if(r == 3 && final[0] == 0 && final[1] == 3 && final[2] == 4) {
                            printf("Node %d (op %d -- state %d) at time %d with resources %d\n", j, node[j].index_operation, node[j].state, time+1, 
                                Operation[node[j].index_operation].res[node[j].id_resource].id);
                        }
                        #endif
                    }
                }
            }

            #ifdef SCHEDULING
            if(r == 3 && final[0] == 0 && final[1] == 3 && final[2] == 4) {
                printf("End time %d\n\n", time+1);
            }
            #endif

            time++;
        }

        #ifdef TESTING
        fprintf(fp, " -- time is %d vs %d-- area is %d\n", time, *best_time, area);
        #endif

        // see if a better result has been achived
        if(time > -1 && ((time < *best_time) || (time == *best_time && area < *area_calculated))) {
            for(i = 0; i < r; i++) {
                best_final[i] = final[i];
                best_final_repetition[i] = repeat[i];
                *best_time = time;
            }
            best_final[i] = -1;
            *area_calculated = area;
            limitator = 0;
        } else 
            limitator++;


        /** Calculate the new repetition and the new area value */
        // go haed only if are is lesser than area_limit
        index = 0;
        int max_repetition = 3;
        do {
            while(index < r && repeat[index] == max_repetition)
            {
                for(i = 1; i < repeat[index]; i++)
                    area -= all_aree[index];
                repeat[index] = 1;
                index++;
            }
            
            if (index < r)
            {
                repeat[index]++;
                area += all_aree[index];
            }

        } while (index != r && area > area_limit);

        for(i = 0; i < node_number; i++)
        {
            node[i].dependecies_level_satisfy = node[i].dependecies_level;
            node[i].state = Idle;
            node[i].remain_time = 0;
        }
    }
    

    for(z = 0; z < r; ++z)
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

    #ifdef TESTING
    fprintf(fp, "\n");
    #endif
}

void combinationUtil(int n,int r,int index,int data[],int i, FILE *fp, operation_t *Operation, int number_operation, node_t *node, 
const int node_number, int *best_final, int *best_final_repetition, const int area_limit, int *best_time, int *area_calculated);
 
// The main function that prints all combinations of size r
// in arr[] of size n. This function mainly uses combinationUtil()
void printCombination(int n, int r, FILE *fp, operation_t *Operation, int number_operation, node_t *node, int node_number, 
int *best_final, int *best_final_repetition, const int area_limit, int *best_time, int *area_calculated)
{
    // A temporary array to store all combination one by one
    int data[r];
 
    // Print all combination using temporary array 'data[]'
    combinationUtil(n, r, 0, data, 0, fp, Operation, number_operation, node, node_number, best_final, best_final_repetition, area_limit, best_time, area_calculated);
}
 
/*
   n      ---> Size of input array
   r      ---> Size of a combination to be printed
   index  ---> Current index in data[]
   data[] ---> Temporary array to store current combination
   i      ---> index of current element in arr[]     */
void combinationUtil(int n, int r, int index, int data[], int i, FILE *fp, operation_t *Operation, int number_operation, node_t *node, 
int node_number, int *best_final, int *best_final_repetition, const int area_limit, int *best_time, int *area_calculated)
{
    // Current combination is ready, print it, 
    if (index == r)
    {
        scheduling_dfg(n, r, data, fp, Operation, number_operation, node, node_number, best_final, best_final_repetition, area_limit, best_time, area_calculated);
        return;
    }
 
    // When no more elements are there to put in data[]
    if (i >= n)
        return;
 
    // current is included, put next at next location
    data[index] = i;
    combinationUtil(n, r, index+1, data, i+1, fp, Operation, number_operation, node, node_number, best_final, best_final_repetition, area_limit, best_time, area_calculated);
 
    // current is excluded, replace it with next (Note that
    // i+1 is passed, but index is not changed)
    combinationUtil(n, r, index, data, i+1, fp, Operation, number_operation, node, node_number, best_final, best_final_repetition, area_limit, best_time, area_calculated);
}

int main(int argc, char const *argv[])
{
    int app;               // for read int
    int i, j, k;       // use like iterator

    if (argc != 4)
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

    printf("START reading operations\n");

    operation_t *Operation;
    Operation = (operation_t *)malloc(sizeof(operation_t)*operation_number);

    int area;
    uint8_t len, speed;
    for(i = 0; i < operation_number; i++)
    {   
        fscanf(fp, "%s", Operation[i].name);
        fscanf(fp, "%d\n", &app);
        len = app;
        Operation[i].res_occurency = len;
        // assign id to operation in a increase order
        Operation[i].operation_id  = i;
        Operation[i].index_next_node = malloc(sizeof(uint8_t)*100);
        Operation[i].index_next_node_occurency = 0; 
        Operation[i].covered = 0;
        Operation[i].used    = 0;
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
            Operation[i].res[j].occurency = 0;
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

    int operation_used = 0;
    int resource_number = 0;

    printf("START reading nodes\n");
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
        node[i].index_next_node = malloc(sizeof(uint8_t)*100);
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
                // Add index to list of operation
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

    // work only with used operation, fix data organization
    operation_t *New_Operation = malloc(operation_used*sizeof(operation_t));
    
    k = 0;
    for(i = 0; i < operation_number; i++)
    {
        if(Operation[i].used == 1)
        {
            New_Operation[k] = Operation[i];
            for(j = 0; j < Operation[i].res_occurency; j++)
                New_Operation[k].res[j].id = resource_number++;
            for(j = 0; j < len_node; j++)
            {
                if(node[j].index_operation == Operation[i].operation_id)
                    node[j].index_operation = k;
            }
            New_Operation[k].operation_id = k;
            k++;
        }
    }

    operation_number = k;
    Operation = New_Operation;
    int n = resource_number;
    
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
        printf("For %s(%d) the node are: ", Operation[i].name, Operation[i].operation_id);
        for(j = 0; j < Operation[i].index_next_node_occurency; j++)
            printf("%s ", node[Operation[i].index_next_node[j]].name);
        printf("\n");
        printf("\tID Area Speed\n");
        for(j = 0; j < Operation[i].res_occurency; j++)
        {
            printf("%d)\t%2d %4d %4d\n", j, Operation[i].res[j].id, Operation[i].res[j].area, Operation[i].res[j].speed);
        }
    }

    long double tot_combination;
    int *final;

    // time information
    long int start_time;
    long int time_difference;
    struct timespec gettime_now;
    time_t rawtime;
    struct tm * timeinfo;

    clock_gettime(CLOCK_REALTIME, &gettime_now);
    start_time = gettime_now.tv_sec;		//Get nS value

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );

    // variable definition
    int *best_final = malloc(sizeof(int)*n);
    int *best_final_repetition = malloc(sizeof(int)*n);
    int area_limit = atoi(argv[3]);
    // assign a big value
    int best_time = 0x7fffffff;
    int area_calculated = 0x7fffffff;

    fp = stdout;
    #ifdef TESTING
    fp = fopen("output_v3.txt", "w");
    #endif
    fprintf(stdout, "The total os resources is %d\n", n);
    fprintf(stdout, "k min is %d and k max is %d\n\n", operation_used, resource_number);
    for(k = operation_used; k <= resource_number; k++) {
    // for(k = 12; k <= 12; k++) {
        tot_combination = 1;
        // calculate number of combination
        for(i = 2; i <= n; i++) tot_combination = tot_combination*((long double)i);
        for (i = 2; i <= k; i++) tot_combination = tot_combination/((long double)i);
        for (i = 2; i <= n-k; i++) tot_combination = tot_combination/((long double)i);
        
        fprintf(stdout, "Combinations for k = %d with %d combination\n", k, (int)tot_combination);

        // display combination in a non recursive way
        printCombination(n, k, fp, Operation, operation_number, node, len_node, best_final, best_final_repetition, area_limit, &best_time, &area_calculated);        
    }

    // calculate elapsed time by finding difference (end - begin) and
    // dividing the difference by CLOCKS_PER_SEC to convert to seconds
    clock_gettime(CLOCK_REALTIME, &gettime_now);
    time_difference = gettime_now.tv_sec - start_time;
    
    fp = fopen("log_v1_CPU.log", "a");

    fprintf (fp, "--------------------------------------------------\n");
    fprintf (fp, "Start local time and date: %s\n", asctime(timeinfo) );
    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    fprintf (fp, "End local time and date: %s\n", asctime(timeinfo) );
    fprintf (fp, "DFG is %s\n", argv[1]);
    fprintf (fp, "Reasources are %s\n", argv[2]);
    fprintf (fp, "Area Limit is %d\n", area_limit);
    fprintf (fp, "--------------------------------------------------\n\n");

    fprintf(fp, "Best solution has time %d:\n", best_time);
    fprintf(stdout, "Best solution has time %d:\n", best_time);
    for(i = 0; i < n && best_final[i] != -1; i++) {
        for(j = 0; j < operation_number; j++) {
            for(k = 0; k < Operation[j].res_occurency; k++) {
                if (best_final[i] == Operation[j].res[k].id)
                {
                    fprintf(fp, "\tOPERATION: %4s - ID RESOURCE: %2d - SPEED: %2d - AREA: %2d - OCCURENCY: %2d\n", 
                        Operation[j].name, Operation[j].res[k].id, Operation[j].res[k].speed, Operation[j].res[k].area, best_final_repetition[i]);
                    fprintf(stdout, "\tOPERATION: %4s - ID RESOURCE: %2d - SPEED: %2d - AREA: %2d - OCCURENCY: %2d\n", 
                        Operation[j].name, Operation[j].res[k].id, Operation[j].res[k].speed, Operation[j].res[k].area, best_final_repetition[i]);
                }
            }
        }
    }

    fprintf(fp, "Final area is %d\n", area_calculated);
    fprintf(stdout, "Final area is %d\n", area_calculated);
 
    fprintf(stdout, "\nThe elapsed time is %ld seconds\n", time_difference);
    fprintf(fp,"\nThe elapsed time is %ld seconds\n\n", time_difference);

    return 0;
}
