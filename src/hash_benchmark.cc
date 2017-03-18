#include "parallel-murmur3.h"
#include "parallel-xxhash.h"

#include <chrono>
#include <cstdio>
extern "C" {
#include "third-party/MurmurHash3.h"
}
#include "third-party/cityhash/city.h"
#include "third-party/xxhash.h"

#if !defined(KEY_LENGTH)
#error "Remember to pass in a -DKEY_LENGTH"
#endif

struct test_parallel {
    __attribute__((noinline))
    void run(uint32_t* keys, uint32_t* seed, uint32_t* res) {
        auto h = hash(keys, seed[0]);
        _mm256_storeu_si256((__m256i*) res, h);
    }

    __attribute__((noinline))
    __m256i hash(uint32_t* keys, uint32 seed) {
        return murmur3<KEY_LENGTH>::parallel(keys, seed);
    }
};

template<int SeedCount>
struct test_parallel_multiseed {
    __attribute__((noinline))
    void run(uint32_t* keys, uint32_t* seed, uint32_t* res) {
        // No point in playing further noinline games here.
        murmur3<KEY_LENGTH>::parallel_multiseed<SeedCount>(
            keys, seed, (__m256i*) res);
    }
};

struct test_scalar {
    __attribute__((noinline))
    void run(uint32_t* keys, uint32_t* seed, uint32_t* res) {
        for (int i = 0; i < 8; ++i) {
            res[i] = hash(&keys[KEY_LENGTH * i], seed[0]);
        }
    }

    __attribute__((noinline))
    uint32_t hash(uint32_t* key, uint32 seed) {
        return murmur3<KEY_LENGTH>::scalar(key, seed);
    }
};

struct test_parallel_xxhash32 {
    __attribute__((noinline))
    void run(uint32_t* keys, uint32_t* seed, uint32_t* res) {
        auto h = hash(keys, seed[0]);
        _mm256_storeu_si256((__m256i*) res, h);
    }

    __attribute__((noinline))
    __m256i hash(uint32_t* keys, uint32 seed) {
        return xxhash32<KEY_LENGTH>::parallel(keys, seed);
    }
};

struct test_scalar_xxhash32 {
    __attribute__((noinline))
    void run(uint32_t* keys, uint32_t* seed, uint32_t* res) {
        for (int i = 0; i < 8; ++i) {
            res[i] = hash(&keys[KEY_LENGTH * i], seed[0]);
        }
    }

    __attribute__((noinline))
    uint32_t hash(uint32_t* key, uint32 seed) {
        return xxhash32<KEY_LENGTH>::scalar(key, seed);
    }
};


struct test_original {
    __attribute__((noinline))
    void run(uint32_t* keys, uint32_t* seed, uint32_t* res) {
        for (int i = 0; i < 8; ++i) {
            MurmurHash3_x86_32(&keys[KEY_LENGTH * i],
                               4 * KEY_LENGTH,
                               seed[0],
                               &res[i]);
        }
    }
};

struct test_cityhash {
    __attribute__((noinline))
    void run(uint32_t* keys, uint32_t* seed, uint32_t* res) {
        for (int i = 0; i < 8; ++i) {
            res[i] = CityHash64WithSeed((const char*)
                                        &keys[KEY_LENGTH * i],
                                        4 * KEY_LENGTH,
                                        seed[0]);
        }
    }
};

struct test_cityhash32 {
    __attribute__((noinline))
    void run(uint32_t* keys, uint32_t* seed, uint32_t* res) {
        for (int i = 0; i < 8; ++i) {
            res[i] = CityHash32((const char*)
                                &keys[KEY_LENGTH * i],
                                4 * KEY_LENGTH);
        }
    }
};

struct test_xxhash32 {
    __attribute__((noinline))
    void run(uint32_t* keys, uint32_t* seed, uint32_t* res) {
        for (int i = 0; i < 8; ++i) {
            res[i] = XXH32((const char*)
                           &keys[KEY_LENGTH * i],
                           4 * KEY_LENGTH,
                           i);
        }
    }
};

struct test_xxhash64 {
    __attribute__((noinline))
    void run(uint32_t* keys, uint32_t* seed, uint32_t* res) {
        for (int i = 0; i < 8; ++i) {
            res[i] = XXH64((const char*)
                           &keys[KEY_LENGTH * i],
                           4 * KEY_LENGTH,
                           i);
        }
    }
};

template<typename Q, int WorkFactor=1>
bool bench(const char* label, uint64_t n, uint32_t* keys) {
    Q tester;
    uint32_t seed[] = {
        0x3afc8e77, 0x924f408d,
        0x8c2a315e, 0x78884cdb,
        0xd2ef9767, 0xee5e590c,
        0x06201e43, 0xb2e4d8df,
    };
    uint32_t res[8 * 8];

    auto start = std::chrono::system_clock::now();

    for (int i = 0; i < n; ++i) {
        tester.run(keys, seed, res);
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> t = end - start;

    uint64_t total_bytes = n * KEY_LENGTH * 4 * 8 * WorkFactor;
    printf("%s,%d,%lf,%ld,%lf,%lf\n",
           label,
           4 * KEY_LENGTH,
           t.count(),
           total_bytes,
           t.count() * 1e9 / total_bytes,
           t.count() * 1e9 / (n * 8 * WorkFactor));

    return true;
}

void init_keys(uint32_t* rows, uint32_t* cols) {
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < KEY_LENGTH; ++j) {
            uint32_t value = (i + 1) * ((j << 8) + 1);
            rows[i * KEY_LENGTH + j] = value;
            cols[j * 8 + i] = value;
        }
    }
}

int main(void) {
    uint32_t rows[KEY_LENGTH * 8 * sizeof(uint32_t)];
    uint32_t cols[KEY_LENGTH * 8 * sizeof(uint32_t)];

    int n = (1 << 27) / KEY_LENGTH;

    if (KEY_LENGTH == 1) {
        printf("impl,keysize,time,bytes,ns_per_byte,ns_per_key\n");
    }

    init_keys(rows, cols);
    bench<test_parallel>("parallel murmur3", n, cols);
    bench<test_parallel_xxhash32>("parallel xxhash32", n, cols);
    // bench<test_parallel_multiseed<1>, 1>("parallel_multiseed<1>", n,
    //                                      cols);
    // bench<test_parallel_multiseed<2>, 2>("parallel_multiseed<2>", n,
    //                                      cols);
    // bench<test_parallel_multiseed<4>, 4>("parallel_multiseed<4>", n,
    //                                      cols);
    // bench<test_scalar>("scalar murmur3", n, rows);
    // bench<test_original>("original murmur3", n, rows);
    bench<test_scalar_xxhash32>("scalar xxhash32", n, cols);
    bench<test_cityhash>("cityhash", n, rows);
    bench<test_cityhash32>("cityhash32", n, rows);
    bench<test_xxhash32>("xxhash32", n, rows);
    bench<test_xxhash64>("xxhash64", n, rows);
}
