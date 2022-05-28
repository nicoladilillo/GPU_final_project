#include <stdio.h>
#include <malloc.h>
#include <string.h>

typedef signed char uint8_t;

#define EMPTY_INDEX 0xFF

enum node_state {
    Idle, Execution, Finish
};

typedef struct node {
    char name[8];
    char operation[8];
    uint8_t id_node;
    uint8_t index_operation;
    uint8_t dep1_index;
    uint8_t dep2_index;
    struct list_node *list_next_node;
    uint8_t state;
    uint8_t remain_time;
    uint8_t id_resource;
} node_t;

typedef struct list_node {
    struct list_node *next;
    node_t *node;
} list_node_t;

typedef struct resource {
    uint8_t area;
    uint8_t speed;
    uint8_t occurency;
    uint8_t id;
} resource_t;

typedef struct operation {
    char name[8];
    uint8_t operation_id;
    uint8_t res_occurency;
    // all the node that perform this operation
    struct list_node *list_node;
    resource_t *res;
} operation_t;

int main(int argc, char const *argv[])
{
    int app;            // for read int
    list_node_t *app_h; // use for modify list
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

    uint8_t area, speed, count = 0;
    uint8_t len;
    for(i = 0; i < operation_number; i++)
    {   
        fscanf(fp, "%s", Operation[i].name);
        fscanf(fp, "%d\n", &app);
        len = app;
        Operation[i].res_occurency = len;
        // assign id to operation in a increase order
        Operation[i].operation_id  = i;
        Operation[i].list_node  = NULL;
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
        strcpy(node[i].operation, temp2);
        node[i].id_node = i;
        node[i].state = Idle;
        node[i].dep1_index = EMPTY_INDEX;
        node[i].dep2_index = EMPTY_INDEX;
        node[i].list_next_node = NULL;
        for(j = 0; j < operation_number; j++)
        {
            if (strcmp(temp2, Operation[j].name) == 0)
            {
                node[i].index_operation = j;
                // Add node to list of operation
                app_h = (list_node_t *)malloc(sizeof(list_node_t));
                app_h->next = Operation[j].list_node;
                app_h->node = &node[i];
                Operation[j].list_node = app_h;
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
        app_h = (list_node_t *)malloc(sizeof(list_node_t));
        app_h->next = node[u].list_next_node;
        app_h->node = &node[v];
        node[u].list_next_node = app_h;

        // Put like next node for the one read in second place
        if (node[v].dep1_index == EMPTY_INDEX) 
            node[v].dep1_index = u;
        else
            node[v].dep2_index = u;
        
        // printf("Node %s(%s) va in nodo %s(%s)\n",  
        //     node[u].name, Operation[node[u].index_operation].name, 
        //     node[v].name, Operation[node[v].index_operation].name);
    }

    /** Print all read data to check the correct assimilation*/

    printf("\nNODE\n\n");
    for(i = 0; i < len_node; i++)
    {
        printf("%d) Node: %s - Operation: %s" , node[i].id_node, node[i].name, Operation[node[i].index_operation].name);
        if (node[i].dep1_index != EMPTY_INDEX &&  node[i].dep2_index != EMPTY_INDEX) {
            printf(" - Dependecies: ");
            if (node[i].dep1_index != EMPTY_INDEX)
                printf("%s ", node[node[i].dep1_index].name);
            if (node[i].dep2_index != EMPTY_INDEX)
                printf("%s ", node[node[i].dep2_index].name);
        }
        if (node[i].list_next_node != NULL) 
        {
            printf(" - Next node:   ");
            for(app_h = node[i].list_next_node; app_h != NULL; app_h = app_h->next)
                printf("%s ", app_h->node->name);
            printf("\n");
        } else {
            printf("\n");
        }
    }

    printf("\nRESOURCES\n\n");
    for(i = 0; i < operation_number; i++)
    {
        printf("For %s the node are: ", Operation[i].name);
        for(app_h = Operation[i].list_node; app_h != NULL; app_h = app_h->next)
            printf("%s ", app_h->node->name);
        printf("\n");
        printf("\tID Area Speed\n");
        for(j = 0; j < Operation[i].res_occurency; j++)
        {
            printf("%d)\t%2d %4d %4d\n", j, Operation[i].res[j].id, Operation[i].res[j].area, Operation[i].res[j].speed);
        }
    }

    /** Perform a scheduling operation according avaiable resources */
    // Moclop value
    Operation[0].res[0].occurency  = 2;
    Operation[1].res[0].occurency  = 1;
    Operation[2].res[0].occurency  = 1;

    // OLD ALGORITHM
    // i = 1;
    // j = 0;
    // int time = -1;
    // while(i == 1)
    // {
    //     i = 0;
    //     for(j = 0; j < len_node; j++)
    //     {
    //         // only in this condition it's possible to schedule the node, when all parents has been scheduled
    //         if(node[j].dep1_index == EMPTY_INDEX && node[j].dep2_index == EMPTY_INDEX && node[j].state != Finish)
    //         {
    //             if(node[j].state == Idle) {
    //                 i = 1;
    //                 // check if there is some free resource
    //                 for(k = 0; k < Operation[node[j].index_operation].res_occurency; k++ )
    //                 {
    //                     if ( Operation[node[j].index_operation].res[k].occurency > 0) {
    //                         // Associate the resources to the node
    //                         node[j].remain_time = Operation[node[j].index_operation].res[k].speed;
    //                         node[j].id_resource = k;
    //                         node[j].state = Execution;
    //                         Operation[node[j].index_operation].res[k].occurency--;
    //                     }
    //                 }
    //             } else if(node[j].state == Execution) {
    //                 i = 1;
    //                 if ( node[j].remain_time == 1) 
    //                 {
    //                     // Remove associate the resources to the node
    //                     node[j].state = Finish;
    //                     Operation[node[j].index_operation].res[node[j].id_resource].occurency++;
    //                     if (node[node[j].next_index].dep1_index == j)
    //                         node[node[j].next_index].dep1_index = EMPTY_INDEX;
    //                     else 
    //                         node[node[j].next_index].dep2_index = EMPTY_INDEX;
    //                 } else {
    //                     node[j].remain_time--;
    //                 }
    //             }
    //         }
    //     }
    //     time++;
    // }

    // NEW ALGORITHM
    int flag = 1;
    int time = -1;
    while (flag)
    {
        flag = 0;
        // check between all operation and find node that can be scheduled or that are in execution, 
        // in case you find nothing this means that all nodes hande been scheduled
        for(i = 0; i < operation_number; i++) 
        {
            for(app_h = Operation[i].list_node; app_h != NULL; app_h = app_h->next)
            {
                // Check if exist a node that has parents scheduled and is in Idle o Execution state
                if(app_h->node->dep1_index == EMPTY_INDEX && 
                    app_h->node->dep2_index == EMPTY_INDEX && app_h->node->state != Finish)
                {
                    flag = 1;
                    if (app_h->node->state == Idle) {
                        // Check if there is some free resorce
                        for(j = 0; j < Operation[i].res_occurency; j++)
                        {
                            if (Operation[i].res[j].occurency > 0)
                            {
                                // Associate the resources to the node and decrease the occurency
                                app_h->node->remain_time = Operation[i].res[j].speed;
                                app_h->node->id_resource = k;
                                app_h->node->state = Execution;
                                Operation[i].res[j].occurency--;
                            }
                        }
                    } else if (app_h->node->state == Execution) {
                        if ( app_h->node->remain_time == 1) 
                        {
                            // Node terminates to use the resource and all his dependencies have to be free
                            app_h->node->state = Finish;
                            Operation[i].res[app_h->node->id_resource].occurency++;
                            while (app_h->node->list_next_node != NULL)
                            {
                                if (app_h->node->list_next_node->node->dep1_index == app_h->node->id_node)
                                    app_h->node->list_next_node->node->dep1_index = EMPTY_INDEX;
                                else 
                                    app_h->node->list_next_node->node->dep2_index = EMPTY_INDEX;
                                app_h->node->list_next_node = app_h->node->list_next_node->next;
                            }
                        } else {
                            app_h->node->remain_time = app_h->node->remain_time - 1;
                        }
                    }
                }
            }
        }
        time++;
    }
    
    printf("\n\nScheduling time is %d\n", time);

    return 0;
}