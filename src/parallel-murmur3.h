// -*- mode: c++; c-basic-offset: 4 indent-tabs-mode: nil -*- */
//
// Copyright 2017 Juho Snellman, released under a MIT license:
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// README
// ------
//
// Three AVX2 implementations of the Murmur3 hash functions.
//
// - parallel: Computing the hash values of 8 keys in parallel.
// - parallel_multiseed: Computing N hash values for each of 8 keys
//   in parallel. Each of the N hash values for a key will be computed
//   using a different key.
// - scalar: Computing the hash value of a single key.
//
// All of the above implementations have the same constraints on
// keys:
//
// - A constant size (so not e.g. on strings)
// - A size that's an multiple of 4 bytes.
//
// These are very special purpose implementations, and will not be
// of any interest in most programs. Use these functions if:
//
// - You have an application that can receive and process inputs
//   in batches such that you usually have 8 keys to process at once.
//   (Doesn't need to be full batches of 8, but the break-even point
//   vs. a fast scalar hash is probably around batches of 3-4 keys).
// - The keys are fairly small. Many scalar hash functions benefit
//   from having large key sizes. These implementations don't. The
//   break-even point should be at around 64 bytes.
// - You can arrange for your hash keys to be in a column-major
//   order without too much pain.
//
// If the above points aren't true for you application, you're
// probably better off using e.g. CityHash.
//
// The "scalar" implementation is only included as a fallback, for
// programs that generally use the parallel versions, but have some
// exceptional cases that need identical hash codes for individual
// keys.
//
// DATA LAYOUT
// -----------
//
// For the parallel implementations the keys should be laid out
// adjacent to each other, in column-major order. That is, the first
// word in "keys" should be the first word of the first key. The
// second word of "keys should be the first word of the second
// key. And so on:
//
//   key1[0] key2[0] ... key7[0]
//   key2[1] key2[1] ... key7[1]
//   ...
//   key1[SizeWords-1] key2[SizeWords-1] ... key7[SizeWords-1]
//
// EXAMPLES
// --------
//
// Assume the following definitions:
//
//   static const uint32_t KEY_LENGTH = 3;
//
//    static uint32_t rows[][KEY_LENGTH] = {
//       {1, 2, 3},
//       {4, 5, 6},
//       {7, 8, 9},
//       {10, 11, 12},
//       {13, 14, 15},
//       {16, 17, 18},
//       {19, 20, 21},
//       {22, 23, 24},
//   };
//
//   static uint32_t cols[][8] = {
//       {1, 4, 7, 10, 13, 16, 19, 22},
//       {2, 5, 8, 11, 14, 17, 20, 23},
//       {3, 6, 9, 12, 15, 18, 21, 24},
//   };
//
//   static uint32_t seeds[] = { 0x3afc8e77, 0x924f408d };
//   static const uint32_t seed_count = sizeof(seeds) / sizeof(uint32_t);
//
// The three implementations could be used as follows to compute
// hash values for each of the key / seed combinations:
//
//   void example_scalar() {
//       for (int s = 0; s < seed_count; ++s) {
//           for (int i = 0; i < 8; ++i) {
//               uint32_t* row = rows[i];
//               uint32_t res = murmur3<3>::scalar(row, seeds[s]);
//               printf("seed=%08x, key={%u,%u,%u}, hash=%u\n",
//                      seeds[s], row[0], row[1], row[2], res);
//           }
//       }
//   }
//
//   void example_parallel() {
//       for (int s = 0; s < seed_count; ++s) {
//           __m256i hash = murmur3<3>::parallel(cols[0], seeds[s]);
//           uint32_t res[8];
//           _mm256_storeu_si256((__m256i*) res, hash);
//           for (int i = 0; i < 8; ++i) {
//               printf("seed=%08x, key={%u,%u,%u}, hash=%u\n",
//                      seeds[s], cols[0][i], cols[1][i], cols[2][i],
//                      res[i]);
//           }
//       }
//   }
//
//   void example_parallel_multiseed() {
//       __m256i hash[seed_count];
//       murmur3<3>::parallel_multiseed<seed_count>(cols[0], seeds, hash);
//       for (int s = 0; s < seed_count; ++s) {
//           uint32_t res[8];
//           _mm256_storeu_si256((__m256i*) res, hash[s]);
//           for (int i = 0; i < 8; ++i) {
//               printf("seed=%08x, key={%u,%u,%u}, hash=%u\n",
//                      seeds[s], cols[0][i], cols[1][i], cols[2][i],
//                      res[i]);
//           }
//       }
//   }


#ifndef PARALLEL_MURMUR3_H
#define PARALLEL_MURMUR3_H

#include <cstdint>
#include <cstdio>
#include <immintrin.h>

template<int SizeWords>
struct murmur3 {

    // Compute a hash value for 8 keys of SizeWords*4 bytes each.
    static __m256i parallel(const uint32_t* keys, uint32_t seed) {
        const __m256i c1 = _mm256_set1_epi32(0xcc9e2d51);
        const __m256i c2 = _mm256_set1_epi32(0x1b873593);
        __m256i h = _mm256_set1_epi32(seed);

        for (int i = 0; i < SizeWords; ++i) {
            __m256i k = _mm256_load_si256((__m256i*) (keys + i * 8));
            k = _mm256_mullo_epi32(k, c1);
            k = mm256_rol32<15>(k);
            k = _mm256_mullo_epi32(k, c2);

            h = _mm256_xor_si256(h, k);
            h = mm256_rol32<13>(h);
            h = _mm256_add_epi32(_mm256_mullo_epi32(h,
                                                    _mm256_set1_epi32(5)),
                                 _mm256_set1_epi32(0xe6546b64));
        }

        // Mixing in the length here is pretty silly, since it's always
        // constant. But there's probably some value in producing bitwise
        // identical results to the original murmur3 code.
        h = _mm256_xor_si256(h, _mm256_set1_epi32(SizeWords * 4));

        return mm256_fmix32(h);
    }

    // For each of 8 keys, compute N hash values each with a different
    // starting seed value. The hash values will be written to "res".
    template<int N>
    static void parallel_multiseed(const uint32_t* keys, uint32_t seeds[N],
                                   __m256i res[N]) {
        const __m256i c1 = _mm256_set1_epi32(0xcc9e2d51);
        const __m256i c2 = _mm256_set1_epi32(0x1b873593);
        __m256i h[N];
        for (int j = 0; j < N; ++j) {
            h[j] = _mm256_set1_epi32(seeds[j]);
        }

        for (int i = 0; i < SizeWords; ++i) {
            __m256i k = _mm256_load_si256((__m256i*) (keys + i * 8));
            k = _mm256_mullo_epi32(k, c1);
            k = mm256_rol32<15>(k);
            k = _mm256_mullo_epi32(k, c2);

            for (int j = 0; j < N; ++j) {
                h[j] = _mm256_xor_si256(h[j], k);
                h[j] = mm256_rol32<13>(h[j]);
                h[j] = _mm256_add_epi32(_mm256_mullo_epi32(h[j],
                                                           _mm256_set1_epi32(5)),
                                        _mm256_set1_epi32(0xe6546b64));
            }
        }

        for (int j = 0; j < N; ++j) {
            h[j] = _mm256_xor_si256(h[j], _mm256_set1_epi32(SizeWords * 4));
            res[j] = mm256_fmix32(h[j]);
        }
    }

    // Compute a hash value for the key.
    static uint32_t scalar(uint32_t* key, uint32_t seed) {
        const uint32_t c1s = 0xcc9e2d51;
        const uint32_t c2s = 0x1b873593;
        uint32_t h = seed;

        for (int i = 0; i < SizeWords; ++i) {
            uint32_t k = key[i];
            k *= c1s;
            k = rol32<15>(k);
            k *= c2s;

            h ^= k;
            h = rol32<13>(h);
            h = h*5 + 0xe6546b64;
        }

        h ^= SizeWords * 4;

        return fmix32(h);
    }

private:
    static __m256i mm256_fmix32(__m256i h) {
        h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16));
        h = _mm256_mullo_epi32(h, _mm256_set1_epi32(0x85ebca6b));
        h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 13));
        h = _mm256_mullo_epi32(h, _mm256_set1_epi32(0xc2b2ae35));
        h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16));

        return h;
    }

    template<int r>
    static __m256i mm256_rol32(__m256i x) {
        return _mm256_or_si256(_mm256_slli_epi32(x, r),
                               _mm256_srli_epi32(x, 32 - r));
    }

    static uint32_t fmix32(uint32_t h) {
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;

        return h;
    }

    template<int r>
    static uint32_t rol32(uint32_t x) {
        return (x << r) | (x >> (32 - r));
    }
};

#endif // PARALLEL_MURMUR3_H
