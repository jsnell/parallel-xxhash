// -*- mode: c++; c-basic-offset: 4 indent-tabs-mode: nil -*- */
//
// Copyright 2017 Juho Snellman, released under a MIT license

#include "parallel-murmur3.h"

static const uint32_t KEY_LENGTH = 3;

static uint32_t rows[][KEY_LENGTH] = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9},
    {10, 11, 12},
    {13, 14, 15},
    {16, 17, 18},
    {19, 20, 21},
    {22, 23, 24},
};

static uint32_t cols[][8] = {
    {1, 4, 7, 10, 13, 16, 19, 22},
    {2, 5, 8, 11, 14, 17, 20, 23},
    {3, 6, 9, 12, 15, 18, 21, 24},
};

static uint32_t seeds[] = { 0x3afc8e77, 0x924f408d };
static const uint32_t seed_count = sizeof(seeds) / sizeof(uint32_t);

void example_scalar() {
    for (int s = 0; s < seed_count; ++s) {
        for (int i = 0; i < 8; ++i) {
            uint32_t* row = rows[i];
            uint32_t res = murmur3<3>::scalar(row, seeds[s]);
            printf("seed=%08x, key={%u,%u,%u}, hash=%u\n",
                   seeds[s], row[0], row[1], row[2], res);
        }
    }
}

void example_parallel() {
    for (int s = 0; s < seed_count; ++s) {
        __m256i hash = murmur3<3>::parallel(cols[0], seeds[s]);
        uint32_t res[8];
        _mm256_storeu_si256((__m256i*) res, hash);
        for (int i = 0; i < 8; ++i) {
            printf("seed=%08x, key={%u,%u,%u}, hash=%u\n",
                   seeds[s], cols[0][i], cols[1][i], cols[2][i],
                   res[i]);
        }
    }
}

void example_parallel_multiseed() {
    __m256i hash[seed_count];
    murmur3<3>::parallel_multiseed<seed_count>(cols[0], seeds, hash);
    for (int s = 0; s < seed_count; ++s) {
        uint32_t res[8];
        _mm256_storeu_si256((__m256i*) res, hash[s]);
        for (int i = 0; i < 8; ++i) {
            printf("seed=%08x, key={%u,%u,%u}, hash=%u\n",
                   seeds[s], cols[0][i], cols[1][i], cols[2][i],
                   res[i]);
        }
    }
}

int main (void) {
    example_scalar();
    example_parallel();
    example_parallel_multiseed();
}
