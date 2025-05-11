#ifndef REMOTE_WORKERS_H
#define REMOTE_WORKERS_H

typedef struct RemoteWorker {
    int fd;
    const char *address;
    int port;
    float start;
    float end;

    float send_time;
    float receive_time;
} RemoteWorker;

extern RemoteWorker *workers;
extern int num_workers;

#endif