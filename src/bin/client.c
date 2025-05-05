/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>

#include "transformer.h"
#include "tokenizer.h"
#include "sampler.h"
#include "transformer_info.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include "safetensors.h"
#include "worker_commands.h"
#include <assert.h>
#include "net.h"

typedef struct RemoteWorker {
    int fd;
    const char *address;
    int port;
    float start;
    float end;
} RemoteWorker;

RemoteWorker *workers = NULL;

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms(void) {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
// VS Code will complain about CLOCK_REALTIME being undefined otherwise.
#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME 0
#endif
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

float generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, const char *prompt, int steps) {
    const char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        Tensor* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, data_f32(logits));
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    float tokens_per_second = 0.0f;
    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        tokens_per_second = (float)(pos - 1) * 1000.0f / (end - start);
        fprintf(stderr, "achieved tok/s: %.2f\n", tokens_per_second);
    }

    free(prompt_tokens);
    return tokens_per_second;
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

void send_slice(RemoteWorker *worker, int slice_id, Tensor *matrix) {
    uint32_t start_offset = (uint32_t)(matrix->vdim * worker->start); // Inclusive.
    uint32_t end_offset = (uint32_t)(matrix->vdim * worker->end); // Exclusive.

    // send the slice id and input vector to the worker
    uint16_t command = CMD_LOAD_MATRIX;
    uint32_t type = matrix->type;
    uint32_t dim_in = matrix->hdim;
    uint32_t dim_out = end_offset - start_offset;
    uint32_t end = 0xCAFEF00D;

    write_full(worker->fd, &command, sizeof(command));
    write_full(worker->fd, &slice_id, sizeof(slice_id));
    write_full(worker->fd, &type, sizeof(type));
    write_full(worker->fd, &dim_in, sizeof(dim_in));
    write_full(worker->fd, &dim_out, sizeof(dim_out));

    size_t start_index = start_offset * matrix->hdim * quant_size(matrix->type);
    size_t end_index = end_offset * matrix->hdim * quant_size(matrix->type);

    size_t bytes_sent = 0;

    u_int8_t *data_start = (uint8_t*)matrix->data + start_index * quant_size(matrix->type);
    size_t data_size = (end_index - start_index) * quant_size(matrix->type);
    assert(data_start + data_size <= (uint8_t*)matrix->data + Tensor_storage_size(matrix));
    write_full(worker->fd, data_start, data_size);
    bytes_sent += data_size;

    if (matrix->type == Q8_0) {
        float *scale_start = matrix->scale + start_offset * matrix->hdim / group_size(matrix->type);
        size_t scale_size = (end_offset - start_offset) * matrix->hdim / group_size(matrix->type) * sizeof(float);
        assert((uint8_t*)scale_start + scale_size <= (float*)matrix->scale + matrix->dim / group_size(matrix->type));

        write_full(worker->fd, scale_start, scale_size);
        bytes_sent += scale_size;
    }
    printf("Sent %zu bytes for slice %d\n", bytes_sent, slice_id);

    write_full(worker->fd, &end, sizeof(end));
}

void error_usage(void) {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> tokenizer path (default: python/tokenizer.bin)\n");
    fprintf(stderr, "  -m <string> mode: generate|chat (default: generate)\n");
    fprintf(stderr, "  -y <string> system prompt for chat mode (default: NULL)\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    const char *tokenizer_path = "python/tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    const char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode
    
    // Start timing for initialization
    long start_time = time_in_ms();

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer_from_safetensors(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // Connect to the workers and upload their matrices.

    // Define the workers. TODO: load from config file, or have them register.
    const int num_workers = 1;
    workers = calloc(num_workers, sizeof(RemoteWorker));
    workers[0].address = "127.0.0.1";
    workers[0].port = 1234;
    workers[0].start = 0.0f;
    workers[0].end = 0.5f;
    if (num_workers > 1) {
        workers[1].address = "127.0.0.1";
        workers[1].port = 1235;
        workers[1].start = 0.5f;
        workers[1].end = 1.0f;
    }

    // Connect to each of them.
    for (int i = 0; i < num_workers; i++) {
        workers[i].fd = socket(AF_INET, SOCK_STREAM, 0);
        if (workers[i].fd < 0) {
            perror("socket");
            exit(EXIT_FAILURE);
        }

        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(workers[i].port);
        inet_pton(AF_INET, workers[i].address, &server_addr.sin_addr);

        printf("Connecting to worker %d at %s:%d\n", i, workers[i].address, workers[i].port);
        if (connect(workers[i].fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
            perror("connect");
            exit(EXIT_FAILURE);
        }

        // Loop over all matrices, and send the slice to the worker.
        Model *m = transformer.safetensors;
        int matrix_id = 0;
        send_slice(&workers[i], matrix_id++, m->token_embedding_table);
        send_slice(&workers[i], matrix_id++, m->rms_final_weight);
        send_slice(&workers[i], matrix_id++, m->wcls);
        for (int j = 0; j < m->config->n_layers; j++) {
            Layer *layer = &m->layers[j];
            send_slice(&workers[i], matrix_id++, layer->rms_att_weight);
            send_slice(&workers[i], matrix_id++, layer->rms_ffn_weight);
            send_slice(&workers[i], matrix_id++, layer->wq);
            send_slice(&workers[i], matrix_id++, layer->wk);
            send_slice(&workers[i], matrix_id++, layer->wv);
            send_slice(&workers[i], matrix_id++, layer->wo);
            send_slice(&workers[i], matrix_id++, layer->w1);
            send_slice(&workers[i], matrix_id++, layer->w2);
            send_slice(&workers[i], matrix_id++, layer->w3);
        }
    }

    // Print elapsed initialization time
    long end_time = time_in_ms();
    fprintf(stderr, "Initialization took %ld ms\n", (end_time - start_time));

    size_t total_params = print_transformer_info(&transformer);

    float tokens_per_second = generate(&transformer, &tokenizer, &sampler, prompt, steps);
    fprintf(stderr, "GMAC: %.1f\n", (double)tokens_per_second*total_params / 1000000000.0);

    return 0;
}

