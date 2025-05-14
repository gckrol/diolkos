#include <netinet/tcp.h>
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>
#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "net.h"

//////////
// Types

typedef struct Node {
    uint16_t id;
    const char *address;
    uint16_t port;

    pthread_t send_thread;
    pthread_t receive_thread;

    int socket;
    pthread_mutex_t socket_write_mutex;
} Node;

typedef struct NodeMsg {
    uint16_t id;
    char address[256];
    uint16_t port;
} NodeMsg;

// Sent by the root node to all nodes to initialize them.
typedef struct InitMsg {
    uint16_t my_id;
    uint16_t your_id;
    uint16_t num_nodes;
} InitMsg;

typedef struct RingPingMsg {
    uint16_t sender_id;
    int8_t direction;
} RingPingMsg;

typedef enum NodeMsgType {
    // Initialization
    MSG_INIT,
    MSG_NODES,
    MSG_NODES_ACK,    
    MSG_CONNECT_FULL_MESH,
    MSG_CONNECT_FULL_MESH_ACK,

    // Normal operation
    MSG_PING,
    MSG_PING_ACK,
    MSG_RING_PING,
} NodeMsgType;

////////////
// Globals

static uint16_t num_nodes = 0;
static Node *nodes = NULL;

static uint16_t my_node_id = 0;
static uint16_t listen_port = 0;
// Socket we listen on for incoming connections.
// Note that the root node doesn't have this (it only has outgoing connections).
static int listen_socket = -1;
static pthread_t listen_thread = 0;

//////////////
// Functions

// Process incoming messages from other nodes, starting compute tasks if needed.
static void *node_receive_fn(void *arg) {
    Node *node = arg;

    printf("Started receive thread for node %d\n", node->id);

    for (;;) {
        int16_t msg_type = -1;
        read_full(node->socket, &msg_type, sizeof(msg_type));
        printf("Node %d: Received message type: %d\n", node->id, msg_type);

        switch (msg_type) {
            case MSG_PING: {
                // Send back a ping response.
                msg_type = MSG_PING_ACK;
                write_full(node->socket, &msg_type, sizeof(msg_type));
                break;
            }
            case MSG_PING_ACK: {
                break;
            }
            case MSG_RING_PING: {
                RingPingMsg ping_msg;
                read_full(node->socket, &ping_msg, sizeof(ping_msg));
                printf("Node %d: Received ring ping originating from node %d\n", node->id, ping_msg.sender_id);
                if (ping_msg.sender_id != my_node_id) {
                    // Forward the ping to the next node.
                    int next_node_id = (my_node_id + ping_msg.direction) % num_nodes;
                    if (next_node_id < 0) {
                        next_node_id += num_nodes;
                    }
                    printf("Node %d: Forwarding ring ping to node %d\n", node->id, next_node_id);
                    Node *next_node =
                        &nodes[next_node_id];
                    pthread_mutex_lock(&next_node->socket_write_mutex);
                    msg_type = MSG_RING_PING;
                    write_full(next_node->socket, &msg_type, sizeof(msg_type));
                    write_full(next_node->socket, &ping_msg, sizeof(ping_msg));
                    pthread_mutex_unlock(&next_node->socket_write_mutex);
                    printf("Node %d: Sent ring ping to node %d\n", node->id, next_node->id);
                } else {
                    // We received our own ping, so we can stop the ring.
                    printf("Node %d: Stopping ring ping\n", node->id);
                }
                break;
            }
            default: {
                fprintf(stderr, "Node %d: Unknown message type: %d\n", node->id, msg_type);
                exit(1);
                break;
            }
        }
    }
}

// Listen to incoming connections from other nodes.
// Only used during intialization.
static void* client_listen_fn(void *arg) {
    (void)arg;

    int one = 1;
    for (;;) {
        // Listen for incoming connections from other nodes.
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);        
        int incoming_socket = accept(listen_socket, (struct sockaddr*)&client_addr, &addr_len);
        if (incoming_socket < 0) {
            perror("accept failed");
            exit(1);
        }
        printf("Accepted connection from %s:%d\n", inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));
        // Tune for speed.
        setsockopt(incoming_socket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        setsockopt(incoming_socket, SOL_SOCKET, SO_BUSY_POLL, &one, sizeof(one));

        uint16_t peer_id;
        read_full(incoming_socket, &peer_id, sizeof(peer_id));

        Node *node = &nodes[peer_id];
        node->socket = incoming_socket;
        pthread_create(&node->receive_thread, NULL, node_receive_fn, node);

        // Allow sending on this socket.
        pthread_mutex_unlock(&node->socket_write_mutex);

        // Check how many connections are ready for writing.
        int num_ready = 0;
        for (int i = 0; i < num_nodes; i++) {
            if (pthread_mutex_trylock(&nodes[i].socket_write_mutex) == 0) {
                pthread_mutex_unlock(&nodes[i].socket_write_mutex);
                num_ready++;
            }
        }
        printf("Number of nodes ready for writing: %d/%d\n", num_ready, num_nodes);
    }
}

static void client_init() {
    int one = 1;
    
    struct sockaddr_in server_addr = {.sin_family = AF_INET, .sin_port = htons(listen_port), .sin_addr.s_addr = INADDR_ANY};
    listen_socket = socket(AF_INET, SOCK_STREAM, 0);

    setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (bind(listen_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        exit(1);
    }

    if (listen(listen_socket, 1) < 0) {
        perror("listen failed");
        exit(1);
    }

    printf("Client listening on port %d...\n", listen_port);
    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);

    int incoming_socket = accept(listen_socket, (struct sockaddr*)&client_addr, &addr_len);
    if (incoming_socket < 0) {
        perror("accept failed");
        exit(1);
    }
    // Tune for speed.
    setsockopt(incoming_socket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    setsockopt(incoming_socket, SOL_SOCKET, SO_BUSY_POLL, &one, sizeof(one));

    // The first one to connect is the root node.
    int16_t msg_type = -1;
    read_full(incoming_socket, &msg_type, sizeof(msg_type));
    assert(msg_type == MSG_INIT);
    InitMsg init_msg;
    read_full(incoming_socket, &init_msg, sizeof(init_msg));
    printf("Received init message: your id=%d, my id=%d, num_nodes=%d\n",
           init_msg.my_id, init_msg.your_id, init_msg.num_nodes);
    assert(init_msg.my_id == 0);
    my_node_id = init_msg.your_id;
    num_nodes = init_msg.num_nodes;

    printf("Connected to root node. My id: %d\n", init_msg.my_id);

    nodes = calloc(num_nodes, sizeof(Node));
    read_full(incoming_socket, &msg_type, sizeof(msg_type));
    assert(msg_type == MSG_NODES);
    NodeMsg nodes_msg[num_nodes];
    read_full(incoming_socket, nodes_msg, sizeof(nodes_msg));

    for (int i = 0; i < num_nodes; i++) {
        nodes[i].id = nodes_msg[i].id;
        nodes[i].address = strdup(nodes_msg[i].address);
        nodes[i].port = nodes_msg[i].port;

        pthread_mutex_init(&nodes[i].socket_write_mutex, NULL);
        pthread_mutex_lock(&nodes[i].socket_write_mutex);
        if (nodes[i].id == init_msg.my_id) {
            nodes[i].socket = incoming_socket;
        }
    }

    printf("Received %d nodes:\n", num_nodes);

    // Start our listen thread. This will accept incoming connections from other nodes.
    pthread_create(&listen_thread, NULL, client_listen_fn, NULL);

    // Send confirmation that we're ready to accept connections.
    printf("Sending MSG_NODES_ACK...\n");
    msg_type = MSG_NODES_ACK;
    write_full(incoming_socket, &msg_type, sizeof(msg_type));

    // Wait until we should connect.
    printf("Waiting for MSG_CONNECT_FULL_MESH message...\n");
    read_full(incoming_socket, &msg_type, sizeof(msg_type));
    assert(msg_type == MSG_CONNECT_FULL_MESH);

    printf("Connecting to all nodes...\n");

    // Connect to all nodes with an id higher than us.
    for (int i = my_node_id + 1; i < num_nodes; i++) {
        nodes[i].socket = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in peer_addr;
        peer_addr.sin_family = AF_INET;
        peer_addr.sin_port = htons(nodes[i].port);
        inet_pton(AF_INET, nodes[i].address, &peer_addr.sin_addr);

        printf("Connecting to %s:%d...\n", nodes[i].address, nodes[i].port);
        connect(nodes[i].socket, (struct sockaddr *)&peer_addr,
                sizeof(peer_addr));
        // Tune for speed.
        setsockopt(nodes[i].socket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        setsockopt(nodes[i].socket, SOL_SOCKET, SO_BUSY_POLL, &one, sizeof(one));

        // Send our id to the other node.
        write_full(nodes[i].socket, &my_node_id, sizeof(my_node_id));

        // Start receiving messages from this node.
        pthread_create(&nodes[i].receive_thread, NULL, node_receive_fn, &nodes[i]);

        // Allow sending on this socket.
        pthread_mutex_unlock(&nodes[i].socket_write_mutex);
        printf("Connected to %s:%d\n", nodes[i].address, nodes[i].port);
    }

    // Send confirmation to the root node.
    msg_type = MSG_CONNECT_FULL_MESH_ACK;
    write_full(incoming_socket, &msg_type, sizeof(msg_type));

    pthread_mutex_unlock(&nodes[0].socket_write_mutex);
    pthread_create(&nodes[0].receive_thread, NULL, node_receive_fn, &nodes[0]);
}

int main(int argc, char *_argv[]) {
    (void)_argv;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <node_port> [root]\n", _argv[0]);
        return 1;
    }

    listen_port = atoi(_argv[1]);

    if (argc > 2) {
        // Root node, start the mesh network.
        printf("Starting root node...\n");
        num_nodes = 3;
        nodes = calloc(num_nodes, sizeof(Node));
        nodes[0] = (Node){.id = 0, .address = "127.0.0.1", .port = 5000};
        nodes[1] = (Node){.id = 1, .address = "127.0.0.1", .port = 5001};
        nodes[2] = (Node){.id = 2, .address = "127.0.0.1", .port = 5002};

        for (int i = 0; i < num_nodes; i++) {
            pthread_mutex_init(&nodes[i].socket_write_mutex, NULL);
            pthread_mutex_lock(&nodes[i].socket_write_mutex);
        }

        // First, connect to every node, and send them the node list.
        uint16_t msg_type;

        NodeMsg nodes_msg[num_nodes];
        for (int j = 0; j < num_nodes; j++) {
            nodes_msg[j].id = nodes[j].id;
            assert(strlen(nodes[j].address) < sizeof(nodes_msg[j].address));
            snprintf(nodes_msg[j].address, sizeof(nodes_msg[j].address), "%s", nodes[j].address);
            nodes_msg[j].port = nodes[j].port;
        }

        for (int i = 1; i < num_nodes; i++) {
            Node *node = &nodes[i];
            struct sockaddr_in server_addr;
            server_addr.sin_family = AF_INET;
            server_addr.sin_port = htons(node->port);
            inet_pton(AF_INET, node->address, &server_addr.sin_addr);
            node->socket = socket(AF_INET, SOCK_STREAM, 0);
            if (connect(node->socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
                perror("connect");
                exit(EXIT_FAILURE);
            }
            printf("Connected to %s:%d\n", node->address, node->port);
            int one = 1;
            setsockopt(node->socket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            setsockopt(node->socket, SOL_SOCKET, SO_BUSY_POLL, &one, sizeof(one));

            InitMsg hello_msg = {.my_id = my_node_id,
                                 .your_id = node->id,
                                 .num_nodes = num_nodes};
            msg_type = MSG_INIT;
            write_full(node->socket, &msg_type, sizeof(msg_type));
            write_full(node->socket, &hello_msg, sizeof(hello_msg));
            
            msg_type = MSG_NODES;
            write_full(node->socket, &msg_type, sizeof(msg_type));
            write_full(node->socket, nodes_msg, sizeof(nodes_msg));
        }
        // Wait for all nodes to respond that they are ready.
        printf("Waiting for MSG_NODES_ACK from all nodes...\n");
        for (int i = 1; i < num_nodes; i++) {
            read_full(nodes[i].socket, &msg_type, sizeof(msg_type));
            assert(msg_type == MSG_NODES_ACK);
        }
        // Now, we ask all nodes to connect to each other.
        printf("Asking all nodes to connect to each other...\n");
        for (int i = 1; i < num_nodes; i++) {
            Node *node = &nodes[i];
            msg_type = MSG_CONNECT_FULL_MESH;
            write_full(node->socket, &msg_type, sizeof(msg_type));
        }
        printf("Waiting for MSG_CONNECT_FULL_MESH_ACK from all nodes...\n");
        for (int i = 1; i < num_nodes; i++) {
            read_full(nodes[i].socket, &msg_type, sizeof(msg_type));
            assert(msg_type == MSG_CONNECT_FULL_MESH_ACK);
        }

        // Start receive threads for all nodes.
        printf("Starting receive threads for all nodes...\n");
        for (int i = 1; i < num_nodes; i++) {
            pthread_create(&nodes[i].receive_thread, NULL, node_receive_fn, &nodes[i]);
        }
        // We're ready, we can unlock the write mutexes.
        printf("Unlocking write mutexes for all nodes...\n");
        for (int i = 1; i < num_nodes; i++) {
            pthread_mutex_unlock(&nodes[i].socket_write_mutex);
        }

        // TODO: inject the prompt, then wait until inference is complete.
        // TODO: disconnect from all nodes. This will cause them to reset their state.

        // Forward ring ping.
        Node *first_node = &nodes[1];
        RingPingMsg ping_msg = {.sender_id = 0, .direction = 1};
        msg_type = MSG_RING_PING;
        pthread_mutex_lock(&first_node->socket_write_mutex);
        write_full(first_node->socket, &msg_type, sizeof(msg_type));
        write_full(first_node->socket, &ping_msg, sizeof(ping_msg));
        printf("Sent ring ping to node %d\n", first_node->id);
        pthread_mutex_unlock(&first_node->socket_write_mutex);

        // Backward ring ping.
        Node *last_node = &nodes[num_nodes - 1];
        ping_msg.direction = -1;
        msg_type = MSG_RING_PING;
        pthread_mutex_lock(&last_node->socket_write_mutex);
        write_full(last_node->socket, &msg_type, sizeof(msg_type));
        write_full(last_node->socket, &ping_msg, sizeof(ping_msg));
        printf("Sent ring ping to node %d\n", last_node->id);
        pthread_mutex_unlock(&last_node->socket_write_mutex);

    } else {
        printf("Starting client node...\n");
        // Start listening for the connection from the root node.
        client_init();
    }

    printf("Waiting forever...\n");
    while (1) {
        pause();
    }
    return 0;
}
