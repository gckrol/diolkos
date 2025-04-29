#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>

typedef struct {
    const char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
    int bos_id; // beginning of sentence token
    int eos_id; // end of sentence token
} Tokenizer;

void build_tokenizer(Tokenizer* t, const char* tokenizer_path);
void free_tokenizer(Tokenizer* t);
char* decode(Tokenizer* t, int prev_token, int token);
void safe_printf(const char *piece);
void encode(Tokenizer* t, const char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
int str_lookup(const char *str, TokenIndex *sorted_vocab, int vocab_size);

#endif // TOKENIZER_H
