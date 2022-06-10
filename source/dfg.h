#include <malloc.h>
#include <string.h>

typedef unsigned char uint8_t;
// typedef int uint8_t;

#define EMPTY_INDEX 0xFF

enum node_state {
    Idle, Execution, Finish
};

typedef struct node {
    char name[8];
    uint8_t id_node;
    uint8_t index_operation;
    uint8_t dep1_index;
    uint8_t dep2_index;
    // one is keep constant and the other change over different scheduling
    uint8_t dependecies_level;
    uint8_t dependecies_level_satisfy;
    // All child nodes
    uint8_t *index_next_node;
    uint8_t index_next_node_occurency;
    uint8_t max_index_next_node_occurency;
    uint8_t state;
    uint8_t remain_time;
    uint8_t id_resource;
} node_t;

typedef struct resource {
    int area;
    uint8_t speed;
    uint8_t occurency;
    uint8_t index_operation;
    uint8_t id;
} resource_t;

typedef struct operation {
    char name[8];
    uint8_t operation_id;
    resource_t *res;
    uint8_t res_occurency;
    // all the node that perform this operation
    uint8_t *index_next_node;
    uint8_t max_index_next_node_occurency;    
    uint8_t index_next_node_occurency;    
    uint8_t covered;
    uint8_t used;
} operation_t;

// Used in version 2 to improve memory capability

typedef struct node_GPU {
    uint8_t id_node;
    uint8_t index_operation;
    uint8_t dep1_index;
    uint8_t dep2_index;
    // one is keep constant and the other change over different scheduling
    uint8_t dependecies_level;
    // All child nodes
    uint8_t *index_next_node;
    uint8_t index_next_node_occurency;
} node_GPU_t;

typedef struct operation_GPU {
    uint8_t operation_id;
    resource_t *res;
    uint8_t res_occurency;
    // all the node that perform this operation
    uint8_t *index_next_node;
    uint8_t index_next_node_occurency;    
} operation_GPU_t;
