#define IS_POWER_OF_TWO(x) ((x) > 0 && ((x) & ((x) - 1)) == 0)
#define ENABLE_VISUAL
#define ENABLE_VERBOSE
// #define ENABLE_TIMING 2  // undefined for no timing, 1 for high-level timing (negligible performance impact), 2 for low-level timing (high performance impact)
// #define ENABLE_STATS 2 // undefined for no stats, 1 for high-level stats (negligible performance impact), 2 for low-level stats (high performance impact), 3 for even higher performance impact stats

#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <tuple>
#include <vector>

#include "omp.h"

#ifdef ENABLE_VISUAL
#include <unistd.h>
#endif

using namespace std;
using namespace chrono;

// patterns
enum pattern
{
    nothing,  // test
    r_pentomino,
    glider,  // test
    lightweight_spaceship,
    twenty_cell_quadratic_growth,  // test
    methuselah_126932979M,
    lidka,  // test
};

// pattern to simulate
constexpr auto pattern = lidka;

// viewport parameters
constexpr int viewport_half_height = 100;
constexpr int viewport_half_width = 400;
constexpr int n_timesteps = 20000;

// parallelism parameters
constexpr int global_log_n_threads = 2;
constexpr int global_log_n_shards = 10;

// derived parallelism parameters
constexpr int global_n_threads = 1 << global_log_n_threads;
constexpr int global_n_shards = 1 << global_log_n_shards;

// timers
#ifdef ENABLE_TIMING
high_resolution_clock::duration duration_viewports = high_resolution_clock::duration::zero();
high_resolution_clock::duration duration_rehashing = high_resolution_clock::duration::zero();
high_resolution_clock::duration durations_show_viewports[global_n_threads + 1];  // an extra space to dump time that we don't want to count 
high_resolution_clock::duration durations_show_viewports_planning[global_n_threads + 1];
high_resolution_clock::duration durations_show_viewports_solution[global_n_threads + 1];
high_resolution_clock::duration durations_show_viewports_output[global_n_threads + 1];
high_resolution_clock::duration durations_hashmap[global_n_threads + 1];
high_resolution_clock::duration durations_hashmap_set_lock[global_n_threads + 1];
high_resolution_clock::duration durations_hashmap_unset_lock[global_n_threads + 1];
high_resolution_clock::duration durations_hashmap_hash_operations[global_n_threads + 1];
#endif

// stats
#ifdef ENABLE_STATS
long long n_constructs[global_n_threads + 1];
long long n_gets[global_n_threads + 1];
long long thread_buckets_probed[global_n_threads + 1];
long long thread_locks_set[global_n_threads + 1];
long long thread_locks_contended[global_n_threads + 1];
#endif

int round_two(int number, int exponent, bool round_up) {
    // round number up or down to the nearest 2^exponent
    int power_of_two = 1 << exponent;
    if (number < 0) {
        round_up = !round_up;
    }
    int abs_number = abs(number);
    if (round_up) {
        abs_number = (abs_number + power_of_two - 1) & ~(power_of_two - 1);
    } else {
        abs_number &= ~(power_of_two - 1);
    }
    return (number < 0) ? -abs_number : abs_number;
}

struct quad {
    int log_size; // square macrocell with side lengths 2^size
    quad* ne;
    quad* nw;
    quad* sw;
    quad* se;
    quad* result;

    quad() 
        : log_size(0), ne(nullptr), nw(nullptr), sw(nullptr), se(nullptr), result(nullptr) {}

    quad(quad* ne, quad* nw, quad* sw, quad* se) 
        : log_size(ne->log_size + 1), ne(ne), nw(nw), sw(sw), se(se), result(nullptr) {}

    bool operator==(const quad& other) const {
        return ne == other.ne && nw == other.nw && sw == other.sw && se == other.se;
    }
};

size_t custom_hash(quad* ptr) {
    size_t h = reinterpret_cast<size_t>(ptr);
    h ^= (h >> 17);
    h *= 0x85ebca6b;
    h ^= (h >> 13);
    h *= 0xc2b2ae35;
    h ^= (h >> 16);
    return h;
}

size_t quad_hash(quad* ne, quad* nw, quad* sw, quad* se) {
    size_t h1 = custom_hash(ne);
    size_t h2 = custom_hash(nw);
    size_t h3 = custom_hash(sw);
    size_t h4 = custom_hash(se);
    return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
}

class concurrent_hashmap {
    // a rather "dumb" concurrent hashmap datastructure
    // it only supports a single get_or_insert operation
    // it uses linear probing for collision resolution
    // it tells you when it needs to be rehashed (>=0.7 load factor in any shard)
    // get_or_construct() is meant to be used by many concurrent threads
    // rehash() is meant to be called outside a parallel block, and handles concurrency itself
    // to avoid deadlocks, make sure there are many more shards compared to threads
public:

    // constants
    int log_capacity;
    int capacity;
    int log_n_shards;
    int n_shards;
    int log_shard_capacity;
    int rehash_threshold;

    bool rehash_needed;
    tuple<quad*, quad*, quad*, quad*, quad*, size_t>** buckets;
    int* shard_sizes;
    omp_lock_t* shard_locks;

    concurrent_hashmap(int log_capacity, int log_n_shards) {

        if (log_capacity <= log_n_shards) {
            cout << "log_capacity (" << log_capacity << ") must be greater than log_n_shards (" << log_n_shards << ")." << endl;
            throw;
        }

        this->log_capacity = log_capacity;
        capacity = 1 << log_capacity;
        this->log_n_shards = log_n_shards;
        n_shards = 1 << log_n_shards;
        log_shard_capacity = log_capacity - log_n_shards;
        rehash_threshold = 1 << (log_shard_capacity - 1);

        rehash_needed = false;
        buckets = new tuple<quad*, quad*, quad*, quad*, quad*, size_t>*[capacity]();
        shard_sizes = new int[n_shards]();
        shard_locks = new omp_lock_t[n_shards]();
        for(int i = 0; i < n_shards; i++) {
            omp_init_lock(&shard_locks[i]);
        }
    }

    ~concurrent_hashmap() {
        // tuples are reused in the rehashed hashmap, so don't delete them
        // this probably violates some programming principles regarding memory management
        // for (int i = 0; i < capacity; ++i) {
        //     if (buckets[i] != nullptr) {
        //         delete buckets[i];
        //     }
        // }
        delete[] buckets;
        for (int i = 0; i < n_shards; i++) {
            omp_destroy_lock(&shard_locks[i]);
        }
        delete[] shard_locks;
    }

    concurrent_hashmap(concurrent_hashmap&&) = delete;
    concurrent_hashmap& operator=(concurrent_hashmap&&) = delete;

    int bucket_to_shard_index(size_t bucket_index) {
        return bucket_index >> log_shard_capacity;
    }

    quad* get_or_construct(quad* ne, quad* nw, quad* sw, quad* se) {
        // thread_index is just for doing timing profiling and isn't necessary for the actual algorithm
        return get_or_construct(ne, nw, sw, se, global_n_threads);
    }

    quad* get_or_construct(quad* ne, quad* nw, quad* sw, quad* se, int thread_index) {
        // thread_index is just for doing timing profiling and isn't necessary for the actual algorithm
#if ENABLE_STATS >= 2
        thread_buckets_probed[thread_index] += 1;
        thread_locks_set[thread_index] += 1;
#endif
#if ENABLE_TIMING >= 2
        auto hash_operations_start = high_resolution_clock::now();
        auto hash_start = hash_operations_start;
#endif
        size_t hash_value = quad_hash(ne, nw, sw, se);
        int bucket_index = hash_value % capacity;  // hash_value & (capacity - 1);
        int first_bucket_index = bucket_index;
        int shard_index = bucket_to_shard_index(bucket_index);
        quad* ptr_to_return;
#if ENABLE_TIMING >= 2
        auto hash_operations_end = high_resolution_clock::now();
        durations_hashmap_hash_operations[thread_index] += hash_operations_end - hash_operations_start;
        auto set_lock_start = high_resolution_clock::now();
#endif
#if ENABLE_STATS >= 3
        if (!omp_test_lock(&shard_locks[shard_index])) {
            thread_locks_contended[thread_index] += 1;
            omp_set_lock(&shard_locks[shard_index]);
        }
#else
        omp_set_lock(&shard_locks[shard_index]);
#endif
#if ENABLE_TIMING >= 2
        auto set_lock_end = high_resolution_clock::now();
        durations_hashmap_set_lock[thread_index] += set_lock_end - set_lock_start;
        hash_operations_start = high_resolution_clock::now();
#endif
        for(;;) {
            if (buckets[bucket_index] == nullptr) {
                // key not found; construct and insert a new quad
                ptr_to_return = new quad(ne, nw, sw, se);
                buckets[bucket_index] = new tuple<quad*, quad*, quad*, quad*, quad*, size_t>(ne, nw, sw, se, ptr_to_return, hash_value);
                shard_sizes[shard_index]++;
                if (shard_sizes[shard_index] >= rehash_threshold) {
                    #pragma omp atomic write
                    rehash_needed = true;
                }
#if ENABLE_STATS >= 2
                n_constructs[thread_index] += 1;
#endif
                break;
            } else if (get<0>(*buckets[bucket_index]) == ne && 
                       get<1>(*buckets[bucket_index]) == nw && 
                       get<2>(*buckets[bucket_index]) == sw && 
                       get<3>(*buckets[bucket_index]) == se) {
                // found our key; return the corresponding quad
                ptr_to_return = get<4>(*buckets[bucket_index]);
#if ENABLE_STATS >= 2
                n_gets[thread_index] += 1;
#endif
                break;
            } else {
                // hash collision; continue linearly probing, switching locks if necessary
#if ENABLE_STATS >= 2
                thread_buckets_probed[thread_index] += 1;
#endif
                bucket_index = (bucket_index + 1) % capacity;  // (bucket_index + 1) & (capacity - 1);
                if (bucket_index == first_bucket_index) {
                    cout << "Attempted to construct and insert a new item, but all hashmap shards were full. This is fatal."  << endl;
                    throw;
                }
                int new_shard_index = bucket_to_shard_index(bucket_index);
                if (shard_index != new_shard_index) {
#if ENABLE_STATS >= 2
                    thread_locks_set[thread_index] += 1;
#endif
#if ENABLE_TIMING >= 2
                    hash_operations_end = high_resolution_clock::now();
                    durations_hashmap_hash_operations[thread_index] += hash_operations_end - hash_operations_start;
                    auto unset_lock_start = high_resolution_clock::now();
#endif
                    omp_unset_lock(&shard_locks[shard_index]);
#if ENABLE_TIMING >= 2
                    auto unset_lock_end = high_resolution_clock::now();
                    durations_hashmap_unset_lock[thread_index] += unset_lock_end - unset_lock_start;
                    set_lock_start = high_resolution_clock::now();
#endif
#if ENABLE_STATS >= 3
                    if (!omp_test_lock(&shard_locks[new_shard_index])) {
                        thread_locks_contended[thread_index] += 1;
                        omp_set_lock(&shard_locks[new_shard_index]);
                    }
#else
                    omp_set_lock(&shard_locks[new_shard_index]);
#endif
#if ENABLE_TIMING >= 2
                    set_lock_end = high_resolution_clock::now();
                    durations_hashmap_set_lock[thread_index] += set_lock_end - set_lock_start;
                    hash_operations_start = high_resolution_clock::now();
#endif
                    shard_index = new_shard_index;
                }
            }
        }
#if ENABLE_TIMING >= 2
        hash_operations_end = high_resolution_clock::now();
        durations_hashmap_hash_operations[thread_index] += hash_operations_end - hash_operations_start;
        auto unset_lock_start = high_resolution_clock::now();
#endif
        omp_unset_lock(&shard_locks[shard_index]);
#if ENABLE_TIMING >= 2
        auto unset_lock_end = high_resolution_clock::now();
        auto hash_end = unset_lock_end;
        durations_hashmap_unset_lock[thread_index] += unset_lock_end - unset_lock_start;
        durations_hashmap[thread_index] += hash_end - hash_start;
#endif
        return ptr_to_return;
    }

    static concurrent_hashmap* rehash(concurrent_hashmap* old_hashmap, int n_threads) {
        // constructs a rehashed version the old hashmap using multiple threads
        // caller is responsible for managing deletion of both the old and new hashmap
        // caller is responsible for ensuring that this is called when no writes to the old hashmap are active
        concurrent_hashmap* new_hashmap = new concurrent_hashmap(old_hashmap->log_capacity + 1, old_hashmap->log_n_shards);
#if ENABLE_STATS
        int minimum;
        int maximum;
        int total;
        minimum = old_hashmap->shard_sizes[0];
        maximum = old_hashmap->shard_sizes[0];
        total = 0;
        for (int i = 0; i < old_hashmap->n_shards; i++) {
            if (minimum > old_hashmap->shard_sizes[i]) { minimum = old_hashmap->shard_sizes[i]; }
            if (maximum < old_hashmap->shard_sizes[i]) { maximum = old_hashmap->shard_sizes[i]; }
            total += old_hashmap->shard_sizes[i];
        }
        cout << "Old hashmap shard size statistics: min = " << minimum << ", max = " << maximum << ", avg = " << total / old_hashmap->n_shards << "." << endl;
#endif
        #pragma omp parallel for num_threads(n_threads)
        for (int i = 0; i < old_hashmap->capacity; ++i) {
            if (old_hashmap->buckets[i] != nullptr) {
                new_hashmap->rehash_insert(old_hashmap->buckets[i]);
            }
        }
#if ENABLE_STATS
        minimum = new_hashmap->shard_sizes[0];
        maximum = new_hashmap->shard_sizes[0];
        total = 0;
        for (int i = 0; i < new_hashmap->n_shards; i++) {
            if (minimum > new_hashmap->shard_sizes[i]) { minimum = new_hashmap->shard_sizes[i]; }
            if (maximum < new_hashmap->shard_sizes[i]) { maximum = new_hashmap->shard_sizes[i]; }
            total += new_hashmap->shard_sizes[i];
        }
        cout << "New hashmap shard size statistics: min = " << minimum << ", max = " << maximum << ", avg = " << total / new_hashmap->n_shards << "." << endl;
#endif
        return new_hashmap;
    }

    // int test() {
    //     int blah = 0;
    //     for (int i = 0; i < capacity; i++) {
    //         if (buckets[i] != nullptr) {
    //             int test1 = get<0>(*buckets[i])->log_size;
    //             int test2 = get<1>(*buckets[i])->log_size;
    //             int test3 = get<2>(*buckets[i])->log_size;
    //             int test4 = get<3>(*buckets[i])->log_size;
    //             int test5 = get<4>(*buckets[i])->log_size;
    //             blah ^= test1 ^ test2 ^ test3 ^ test4 ^ test5;
    //         }
    //     }
    //     return blah;
    // }

private:
    void rehash_insert(tuple<quad*, quad*, quad*, quad*, quad*, size_t>* item) {
        size_t hash_value = get<5>(*item);
        int bucket_index = hash_value % capacity;  // hash_value & (capacity - 1);
        int first_bucket_index = bucket_index;
        int shard_index = bucket_to_shard_index(bucket_index);
        int new_shard_index;
        omp_set_lock(&shard_locks[shard_index]);
        for(;;) {
            if (buckets[bucket_index] == nullptr) {
                // key not found; insert the provided quad
                buckets[bucket_index] = item;
                shard_sizes[shard_index]++;
                if (shard_sizes[shard_index] >= rehash_threshold) {
                    #pragma omp atomic write
                    rehash_needed = true;
                }
                break;
            } else if (get<0>(*buckets[bucket_index]) == get<0>(*item) && 
                       get<1>(*buckets[bucket_index]) == get<1>(*item) && 
                       get<2>(*buckets[bucket_index]) == get<2>(*item) && 
                       get<3>(*buckets[bucket_index]) == get<3>(*item)) {
                // found our key; this is an error
                cout << "Tried to insert a duplicate key using concurrent_hashmap::insert()." << endl;
                throw;
            } else {
                // hash collision; continue linearly probing, switching locks if necessary
                bucket_index = (bucket_index + 1) % capacity; // (bucket_index + 1) % capacity;
                if (bucket_index == first_bucket_index) {
                    cout << "During rehashing, attempted to insert an item, but all hashmap shards were full. This is fatal."  << endl;
                    throw;
                }
                new_shard_index = bucket_to_shard_index(bucket_index);
                if (shard_index != new_shard_index) {
                    omp_unset_lock(&shard_locks[shard_index]);
                    shard_index = new_shard_index;
                    omp_set_lock(&shard_locks[shard_index]);
                }
            }
        }
        omp_unset_lock(&shard_locks[shard_index]);
    }
};

class hashlife {
public:
    bool initialized = false;
    quad* dead_cell = new quad();
    quad* live_cell = new quad();
    concurrent_hashmap* hashmap;
    quad* top_quad;
    vector<quad*> dead_quads = {dead_cell};

    void initialize_hashmap(int log_n_shards) {

        // this function should be called only once
        if (initialized) {
            cout << "Hashmap cannot be initialized more than once." << endl;
            throw;
        }
        initialized = true;

        // create the hashmap
        hashmap = new concurrent_hashmap(18, log_n_shards);

        // enumerate both (1x1) macrocells; these aren't memoized
        quad* quads_1x1[] = {dead_cell, live_cell};

        // generate and memoize all 16 (2x2) macrocells; they don't have results
        vector<quad*> quads_2x2;
        tuple<quad*, quad*, quad*, quad*> next_key;
        quad* next_quad;
        for (int ne_idx = 0; ne_idx < 2; ne_idx++) {
            for (int nw_idx = 0; nw_idx < 2; nw_idx++) {
                for (int sw_idx = 0; sw_idx < 2; sw_idx++) {
                    for (int se_idx = 0; se_idx < 2; se_idx++) {
                        next_quad = hashmap->get_or_construct(quads_1x1[ne_idx], quads_1x1[nw_idx], quads_1x1[sw_idx], quads_1x1[se_idx]);
                        quads_2x2.push_back(next_quad);
                    }
                }
            }
        }

        // generate and memoize all 65536 (4x4) macrocells with their results
        int results_ne_neighbors;
        int results_nw_neighbors;
        int results_sw_neighbors;
        int results_se_neighbors;
        quad* results_ne;
        quad* results_nw;
        quad* results_sw;
        quad* results_se;
        quad* next_results;
        for (int ne_idx = 0; ne_idx < quads_2x2.size(); ne_idx++) {
            for (int nw_idx = 0; nw_idx < quads_2x2.size(); nw_idx++) {
                for (int sw_idx = 0; sw_idx < quads_2x2.size(); sw_idx++) {
                    for (int se_idx = 0; se_idx < quads_2x2.size(); se_idx++) {
                        
                        // manually calculate the result for each 4x4 macrocell using standard conway rules
                        results_ne_neighbors = 0;
                        results_nw_neighbors = 0;
                        results_sw_neighbors = 0;
                        results_se_neighbors = 0;
                        if (quads_2x2[ne_idx]->ne == quads_1x1[1]) { results_ne_neighbors++; }
                        if (quads_2x2[ne_idx]->nw == quads_1x1[1]) { results_ne_neighbors++; results_nw_neighbors++; }
                        if (quads_2x2[ne_idx]->sw == quads_1x1[1]) { results_nw_neighbors++; results_sw_neighbors++; results_se_neighbors++; }
                        if (quads_2x2[ne_idx]->se == quads_1x1[1]) { results_ne_neighbors++; results_se_neighbors++; }
                        if (quads_2x2[nw_idx]->ne == quads_1x1[1]) { results_nw_neighbors++; results_ne_neighbors++; }
                        if (quads_2x2[nw_idx]->nw == quads_1x1[1]) { results_nw_neighbors++; }
                        if (quads_2x2[nw_idx]->sw == quads_1x1[1]) { results_nw_neighbors++; results_sw_neighbors++; }
                        if (quads_2x2[nw_idx]->se == quads_1x1[1]) { results_ne_neighbors++; results_sw_neighbors++; results_se_neighbors++; }
                        if (quads_2x2[sw_idx]->ne == quads_1x1[1]) { results_ne_neighbors++; results_nw_neighbors++; results_se_neighbors++; }
                        if (quads_2x2[sw_idx]->nw == quads_1x1[1]) { results_sw_neighbors++; results_nw_neighbors++; }
                        if (quads_2x2[sw_idx]->sw == quads_1x1[1]) { results_sw_neighbors++; }
                        if (quads_2x2[sw_idx]->se == quads_1x1[1]) { results_sw_neighbors++; results_se_neighbors++; }
                        if (quads_2x2[se_idx]->ne == quads_1x1[1]) { results_se_neighbors++; results_ne_neighbors++; }
                        if (quads_2x2[se_idx]->nw == quads_1x1[1]) { results_ne_neighbors++; results_nw_neighbors++; results_sw_neighbors++; }
                        if (quads_2x2[se_idx]->sw == quads_1x1[1]) { results_se_neighbors++; results_sw_neighbors++; }
                        if (quads_2x2[se_idx]->se == quads_1x1[1]) { results_se_neighbors++; }
                        results_ne = (results_ne_neighbors == 3 || (results_ne_neighbors == 2 && quads_2x2[ne_idx]->sw == quads_1x1[1])) ? quads_1x1[1] : quads_1x1[0];
                        results_nw = (results_nw_neighbors == 3 || (results_nw_neighbors == 2 && quads_2x2[nw_idx]->se == quads_1x1[1])) ? quads_1x1[1] : quads_1x1[0];
                        results_sw = (results_sw_neighbors == 3 || (results_sw_neighbors == 2 && quads_2x2[sw_idx]->ne == quads_1x1[1])) ? quads_1x1[1] : quads_1x1[0];
                        results_se = (results_se_neighbors == 3 || (results_se_neighbors == 2 && quads_2x2[se_idx]->nw == quads_1x1[1])) ? quads_1x1[1] : quads_1x1[0];
                        next_results = hashmap->get_or_construct(results_ne, results_nw, results_sw, results_se);
                        
                        // store in hashmap
                        next_quad = hashmap->get_or_construct(quads_2x2[ne_idx], quads_2x2[nw_idx], quads_2x2[sw_idx], quads_2x2[se_idx]);
                        next_quad->result = next_results;
                    }
                }
            }
        }
    }

    hashlife(const vector<vector<bool>>& initial_state, int log_n_shards) {

        // force initial state size to be a power of 2
        int initial_state_sidelength = initial_state.size();
        if (!IS_POWER_OF_TWO(initial_state_sidelength)) {
            cout << "Bad input shape." << endl;
            throw;
        }
        for (auto& row : initial_state) {
            if (row.size() != initial_state_sidelength) {
                cout << "Bad input shape." << endl;
                throw;
            }
        }

        // initialize hashmap
        initialize_hashmap(log_n_shards);

        // convert grid of bools to grid of (1x1) quads
        vector<vector<quad*>> initial_state_quad(initial_state_sidelength, vector<quad*>(initial_state_sidelength, nullptr));
        for (int y = 0; y < initial_state_sidelength; y++) {
            for (int x = 0; x < initial_state_sidelength; x++) {
                initial_state_quad[y][x] = initial_state[y][x] ? live_cell : dead_cell;
            }
        }

        // construct tree representing initial state
        int half_step = 1;
        tuple<quad*, quad*, quad*, quad*> key;
        while (half_step < initial_state_sidelength) {
            for (int y = 0; y < initial_state_sidelength; y += 2 * half_step) {
                for (int x = 0; x < initial_state_sidelength; x += 2 * half_step) {
                    initial_state_quad[y][x] = hashmap->get_or_construct(
                        initial_state_quad[y][x + half_step], 
                        initial_state_quad[y][x], 
                        initial_state_quad[y + half_step][x], 
                        initial_state_quad[y + half_step][x + half_step]);
                }
            }
            half_step *= 2;
        }
        top_quad = initial_state_quad[0][0];
    }

    hashlife(hashlife&&) = delete;
    hashlife& operator=(hashlife&&) = delete;

    quad* get_or_compute_result(quad* input) {
        return get_or_compute_result(input, global_n_threads);
    }

    quad* get_or_compute_result(quad* input, int thread_index) {
        if (input->result != nullptr) {
            return input->result;
        } else {
            
            // construct 5 auxillary quads
            quad* aux_n = hashmap->get_or_construct(input->ne->nw, input->nw->ne, input->nw->se, input->ne->sw, thread_index);
            quad* aux_w = hashmap->get_or_construct(input->nw->se, input->nw->sw, input->sw->nw, input->sw->ne, thread_index);
            quad* aux_s = hashmap->get_or_construct(input->se->nw, input->sw->ne, input->sw->se, input->se->sw, thread_index);
            quad* aux_e = hashmap->get_or_construct(input->ne->se, input->ne->sw, input->se->nw, input->se->ne, thread_index);
            quad* aux_c = hashmap->get_or_construct(input->ne->sw, input->nw->se, input->sw->ne, input->se->nw, thread_index);

            // first 9 "scoops"
            quad* layer2_e = get_or_compute_result(aux_e, thread_index);
            quad* layer2_ne = get_or_compute_result(input->ne, thread_index);
            quad* layer2_n = get_or_compute_result(aux_n, thread_index);
            quad* layer2_nw = get_or_compute_result(input->nw, thread_index);
            quad* layer2_w = get_or_compute_result(aux_w, thread_index);
            quad* layer2_sw = get_or_compute_result(input->sw, thread_index);
            quad* layer2_s = get_or_compute_result(aux_s, thread_index);
            quad* layer2_se = get_or_compute_result(input->se, thread_index);
            quad* layer2_c = get_or_compute_result(aux_c, thread_index);

            // construct 4 auxillary quads
            quad* layer2_aux_ne = hashmap->get_or_construct(layer2_ne, layer2_n, layer2_c, layer2_e, thread_index);
            quad* layer2_aux_nw = hashmap->get_or_construct(layer2_n, layer2_nw, layer2_w, layer2_c, thread_index);
            quad* layer2_aux_sw = hashmap->get_or_construct(layer2_c, layer2_w, layer2_sw, layer2_s, thread_index);
            quad* layer2_aux_se = hashmap->get_or_construct(layer2_e, layer2_c, layer2_s, layer2_se, thread_index);

            // next 4 "scoops"
            quad* result_ne = get_or_compute_result(layer2_aux_ne, thread_index);
            quad* result_nw = get_or_compute_result(layer2_aux_nw, thread_index);
            quad* result_sw = get_or_compute_result(layer2_aux_sw, thread_index);
            quad* result_se = get_or_compute_result(layer2_aux_se, thread_index);

            // construct, save, and return result
            quad* result = hashmap->get_or_construct(result_ne, result_nw, result_sw, result_se, thread_index);
            input->result = result;
            return result;
        }
    }

    quad* get_dead_quad(int size) {
        return get_dead_quad(size, global_n_threads);
    }

    quad* get_dead_quad(int size, int thread_index) {
        while (size >= dead_quads.size()) {
            dead_quads.push_back(hashmap->get_or_construct(dead_quads.back(), dead_quads.back(), dead_quads.back(), dead_quads.back(), thread_index));
        }
        return dead_quads[size];
    }

    void pad_top_quad() {
        return pad_top_quad(global_n_threads);
    }

    void pad_top_quad(int thread_index) {
        quad* dead_quad = get_dead_quad(top_quad->ne->log_size, thread_index);
        quad* new_ne = hashmap->get_or_construct(dead_quad, dead_quad, top_quad->ne, dead_quad, thread_index);
        quad* new_nw = hashmap->get_or_construct(dead_quad, dead_quad, dead_quad, top_quad->nw, thread_index);
        quad* new_sw = hashmap->get_or_construct(top_quad->sw, dead_quad, dead_quad, dead_quad, thread_index);
        quad* new_se = hashmap->get_or_construct(dead_quad, top_quad->se, dead_quad, dead_quad, thread_index);
        top_quad = hashmap->get_or_construct(new_ne, new_nw, new_sw, new_se, thread_index);
    }

    vector<vector<quad*>> expand_result(vector<vector<quad*>> input_grid, tuple<int, int, int, int, int, int> input_step, tuple<int, int, int, int, int, int> output_step) { 
        return expand_result(input_grid, input_step, output_step, global_n_threads);
    }


    vector<vector<quad*>> expand_result(vector<vector<quad*>> input_grid, tuple<int, int, int, int, int, int> input_step, tuple<int, int, int, int, int, int> output_step, int thread_index) {
        // macrocell size gets cut in half; time increases by half of the output macrocell sidelength

        int depth = 1 << (get<1>(output_step) - 1);
        bool shrink_east = get<4>(input_step) != get<4>(output_step) + 2 * depth;
        bool shrink_north = get<3>(input_step) != get<3>(output_step) - 2 * depth;
        bool shrink_west = get<2>(input_step) != get<2>(output_step) - 2 * depth;
        bool shrink_south = get<5>(input_step) != get<5>(output_step) + 2 * depth;
        int output_dims_y = ((input_grid.size() - 1) * 2) - (shrink_north ? 1 : 0) - (shrink_south ? 1 : 0);
        int output_dims_x = ((input_grid[0].size() - 1) * 2) - (shrink_east ? 1 : 0) - (shrink_west ? 1 : 0);

        // create a first auxillary grid (one unit wider and taller than the output)
        vector<vector<quad*>> aux_grid(output_dims_y + 1, vector<quad*>(output_dims_x + 1, nullptr));
        bool next_aux_is_combo_y = shrink_north;
        int next_input_idx_y = 0;
        bool next_aux_is_combo_x;
        int next_input_idx_x;
        for (int next_aux_idx_y = 0; next_aux_idx_y <= output_dims_y; next_aux_idx_y++) {
            next_aux_is_combo_x = shrink_west;
            next_input_idx_x = 0;
            for (int next_aux_idx_x = 0; next_aux_idx_x <= output_dims_x; next_aux_idx_x++) {
                if (next_aux_is_combo_x) {
                    if (next_aux_is_combo_y) {
                        aux_grid[next_aux_idx_y][next_aux_idx_x] = hashmap->get_or_construct(
                            input_grid[next_input_idx_y][next_input_idx_x + 1]->sw->sw,
                            input_grid[next_input_idx_y][next_input_idx_x]->se->se,
                            input_grid[next_input_idx_y + 1][next_input_idx_x]->ne->ne,
                            input_grid[next_input_idx_y + 1][next_input_idx_x + 1]->nw->nw,
                            thread_index);
                    } else {
                        aux_grid[next_aux_idx_y][next_aux_idx_x] = hashmap->get_or_construct(
                            input_grid[next_input_idx_y][next_input_idx_x + 1]->nw->sw,
                            input_grid[next_input_idx_y][next_input_idx_x]->ne->se,
                            input_grid[next_input_idx_y][next_input_idx_x]->se->ne,
                            input_grid[next_input_idx_y][next_input_idx_x + 1]->sw->nw,
                            thread_index);
                    }
                    next_input_idx_x++;
                } else {
                    if (next_aux_is_combo_y) {
                        aux_grid[next_aux_idx_y][next_aux_idx_x] = hashmap->get_or_construct(
                            input_grid[next_input_idx_y][next_input_idx_x]->se->sw,
                            input_grid[next_input_idx_y][next_input_idx_x]->sw->se,
                            input_grid[next_input_idx_y + 1][next_input_idx_x]->nw->ne,
                            input_grid[next_input_idx_y + 1][next_input_idx_x]->ne->nw,
                            thread_index);
                    } else {
                        aux_grid[next_aux_idx_y][next_aux_idx_x] = hashmap->get_or_construct(
                            input_grid[next_input_idx_y][next_input_idx_x]->ne->sw,
                            input_grid[next_input_idx_y][next_input_idx_x]->nw->se,
                            input_grid[next_input_idx_y][next_input_idx_x]->sw->ne,
                            input_grid[next_input_idx_y][next_input_idx_x]->se->nw,
                            thread_index);
                    }
                }
                next_aux_is_combo_x = !next_aux_is_combo_x;
            }
            if (next_aux_is_combo_y) {
                next_input_idx_y++;
            }
            next_aux_is_combo_y = !next_aux_is_combo_y;
        }

        // create a second auxillary grid (same size as the output)
        vector<vector<quad*>> aux_grid_2(output_dims_y, vector<quad*>(output_dims_x, nullptr));
        for (int next_aux_idx_y = 0; next_aux_idx_y < output_dims_y; next_aux_idx_y++) {
            for (int next_aux_idx_x = 0; next_aux_idx_x < output_dims_x; next_aux_idx_x++) {
                aux_grid_2[next_aux_idx_y][next_aux_idx_x] = hashmap->get_or_construct(
                    aux_grid[next_aux_idx_y][next_aux_idx_x + 1],
                    aux_grid[next_aux_idx_y][next_aux_idx_x],
                    aux_grid[next_aux_idx_y + 1][next_aux_idx_x],
                    aux_grid[next_aux_idx_y + 1][next_aux_idx_x + 1],
                    thread_index);
            }
        }

        // get the output by taking the result of every element of the second auxillary grid
        vector<vector<quad*>> output_grid(output_dims_y, vector<quad*>(output_dims_x, nullptr));
        for (int next_output_idx_y = 0; next_output_idx_y < output_dims_y; next_output_idx_y++) {
            for (int next_output_idx_x = 0; next_output_idx_x < output_dims_x; next_output_idx_x++) {
                output_grid[next_output_idx_y][next_output_idx_x] = get_or_compute_result(aux_grid_2[next_output_idx_y][next_output_idx_x], thread_index);
            }
        }
        return output_grid;
    }

    vector<vector<quad*>> expand_static(vector<vector<quad*>> input_grid, tuple<int, int, int, int, int, int> input_step, tuple<int, int, int, int, int, int> output_step) {
        // macrocell size gets cut in half; time remains unchanged

        bool shrink_east = get<4>(input_step) != get<4>(output_step);
        bool shrink_north = get<3>(input_step) != get<3>(output_step);
        bool shrink_west = get<2>(input_step) != get<2>(output_step);
        bool shrink_south = get<5>(input_step) != get<5>(output_step);
        int output_dims_y = (input_grid.size() * 2) - (shrink_north ? 1 : 0) - (shrink_south ? 1 : 0);
        int output_dims_x = (input_grid[0].size() * 2) - (shrink_east ? 1 : 0) - (shrink_west ? 1 : 0);

        // construct the output grid using children of the macrocells in the input grid
        vector<vector<quad*>> output_grid(output_dims_y, vector<quad*>(output_dims_x, nullptr));
        bool next_input_child_is_south = shrink_north;
        int next_input_idx_y = 0;
        bool next_input_child_is_east;
        int next_input_idx_x;
        for (int next_output_idx_y = 0; next_output_idx_y < output_dims_y; next_output_idx_y++) {
            next_input_child_is_east = shrink_west;
            next_input_idx_x = 0;
            for (int next_output_idx_x = 0; next_output_idx_x < output_dims_x; next_output_idx_x++) {
                if (next_input_child_is_east) {
                    if (next_input_child_is_south) {
                        output_grid[next_output_idx_y][next_output_idx_x] = input_grid[next_input_idx_y][next_input_idx_x]->se;
                    } else { 
                        output_grid[next_output_idx_y][next_output_idx_x] = input_grid[next_input_idx_y][next_input_idx_x]->ne;
                    }
                    next_input_idx_x++;
                } else {
                    if (next_input_child_is_south) {
                        output_grid[next_output_idx_y][next_output_idx_x] = input_grid[next_input_idx_y][next_input_idx_x]->sw;
                    } else {
                        output_grid[next_output_idx_y][next_output_idx_x] = input_grid[next_input_idx_y][next_input_idx_x]->nw;
                    }
                }
                next_input_child_is_east = !next_input_child_is_east;
            }
            if (next_input_child_is_south) {
                next_input_idx_y++;
            }
            next_input_child_is_south = !next_input_child_is_south;
        }
        return output_grid;
    }

    vector<vector<bool>>* show_viewport(int time, int x_min, int y_min, int x_max, int y_max) {
        // thread_index is just for doing timing profiling and isn't necessary for the actual algorithm
        return show_viewport(time, x_min, y_min, x_max, y_max, global_n_threads);
    }

    vector<vector<bool>>* show_viewport(int time, int x_min, int y_min, int x_max, int y_max, int thread_index) {
        // thread_index is just for doing timing profiling and isn't necessary for the actual algorithm

#ifdef ENABLE_TIMING
        auto planning_start = high_resolution_clock::now();
#endif
        
        // decompose the problem into a vector of intermediate steps, each of which is a grid of macrocells that must be found
        vector<tuple<int, int, int, int, int, int>> steps;  // [time, size, x_min, y_min, x_max, y_max]

        // create the first and second steps manually
        steps.push_back({time, 0, x_min, y_min, x_max, y_max});
        steps.push_back({time, 1, round_two(x_min, 1, false), round_two(y_min, 1, false), round_two(x_max, 1, true), round_two(y_max, 1, true)});

        // create the remaining steps based on the bits in time
        bool stop;
        int nonzero_bound;
        int next_size;
        int desired_depth;
        int depth;
        for(;;) {

            // stop if:
            // 1. we are at time zero AND
            // 2. we are left with one, two, or four macrocells AND
            // 3. all of them have the origin as one of their corners AND
            // 4. all of them are at least a quarter of the size of top_quad
            stop = true;
            if (get<0>(steps.back()) > 0) {
                stop = false;
            } else if ((1 << get<1>(steps.back())) < top_quad->log_size - 1) {
                stop = false;
            } else {
                nonzero_bound = 0;
                if (get<2>(steps.back()) != 0) {
                    if (nonzero_bound == 0) {
                        nonzero_bound = abs(get<2>(steps.back()));
                    } else {
                        if (nonzero_bound != abs(get<2>(steps.back()))) {
                            stop = false;
                        }
                    }
                }
                if (get<3>(steps.back()) != 0) {
                    if (nonzero_bound == 0) {
                        nonzero_bound = abs(get<3>(steps.back()));
                    } else {
                        if (nonzero_bound != abs(get<3>(steps.back()))) {
                            stop = false;
                        }
                    }
                }
                if (get<4>(steps.back()) != 0) {
                    if (nonzero_bound == 0) {
                        nonzero_bound = abs(get<4>(steps.back()));
                    } else {
                        if (nonzero_bound != abs(get<4>(steps.back()))) {
                            stop = false;
                        }
                    }
                }
                if (get<5>(steps.back()) != 0) {
                    if (nonzero_bound == 0) {
                        nonzero_bound = abs(get<5>(steps.back()));
                    } else {
                        if (nonzero_bound != abs(get<5>(steps.back()))) {
                            stop = false;
                        }
                    }
                }
                if ((1 << get<1>(steps.back())) != nonzero_bound) {
                    stop = false;
                }
            }
            if (stop) {
                break;
            }
            
            // generate an additional step and double macrocell size
            next_size = get<1>(steps.back()) + 1;
            if (get<0>(steps.back()) > 0) {
                desired_depth = get<0>(steps.back()) & -1 * get<0>(steps.back());  // smallest power of two in the previous step's time
                depth = 1 << (get<1>(steps.back()) - 1);  // depth of a scoop whose bottom has the previous step's size
            }
            if (get<0>(steps.back()) > 0 && desired_depth == depth) {
                steps.push_back({
                    get<0>(steps.back()) - depth, 
                    next_size,
                    round_two(get<2>(steps.back()) - 2 * depth, next_size, false),
                    round_two(get<3>(steps.back()) - 2 * depth, next_size, false),
                    round_two(get<4>(steps.back()) + 2 * depth, next_size, true),
                    round_two(get<5>(steps.back()) + 2 * depth, next_size, true)});
            } else {
                steps.push_back({
                    get<0>(steps.back()),
                    next_size,
                    round_two(get<2>(steps.back()), next_size, false),
                    round_two(get<3>(steps.back()), next_size, false),
                    round_two(get<4>(steps.back()), next_size, true),
                    round_two(get<5>(steps.back()), next_size, true)});
            }
        }

        // ensure our top quad is large enough to proceed and apply padding if not
        while (get<1>(steps.back()) > top_quad->log_size - 1) {
            pad_top_quad(thread_index);
        }

        // manually determine our starting point, which is comprised of one, two, or four children of top_quad
        vector<vector<quad*>> result;
        if (get<2>(steps.back()) == 0) {
            if (get<3>(steps.back()) == 0) {
                result = {{top_quad->se}};
            } else if (get<5>(steps.back()) == 0) {
                result = {{top_quad->ne}};
            } else {
                result = {{top_quad->ne}, {top_quad->se}};
            }
        } else if (get<4>(steps.back()) == 0) {
            if (get<3>(steps.back()) == 0) {
                result = {{top_quad->sw}};
            } else if (get<5>(steps.back()) == 0) {
                result = {{top_quad->nw}};
            } else {
                result = {{top_quad->nw}, {top_quad->sw}};
            }
        } else {
            if (get<3>(steps.back()) == 0) {
                result = {{top_quad->sw, top_quad->se}};
            } else if (get<5>(steps.back()) == 0) {
                result = {{top_quad->nw, top_quad->ne}};
            } else {
                result = {{top_quad->nw, top_quad->ne}, {top_quad->sw, top_quad->se}};
            }
        }

#ifdef ENABLE_TIMING
        auto planning_end = high_resolution_clock::now();
        durations_show_viewports_planning[thread_index] += planning_end - planning_start;
        auto solution_start = high_resolution_clock::now();
#endif

        // perform the steps that we planned out, from the back of the vector to the front
        for (int i = steps.size() - 1; i > 0; i--) {
            if (get<0>(steps[i]) != get<0>(steps[i-1])) {
                result = expand_result(result, steps[i], steps[i-1], thread_index);
            } else {
                result = expand_static(result, steps[i], steps[i-1]);
            }
        }

#ifdef ENABLE_TIMING
        auto solution_end = high_resolution_clock::now();
        durations_show_viewports_solution[thread_index] += solution_end - solution_start;
        auto output_start = high_resolution_clock::now();
#endif

        // convert result to a grid of booleans
        vector<vector<bool>>* result_bool = new vector<vector<bool>>(result.size(), vector<bool>(result[0].size()));
        for (int i = 0; i < result.size(); ++i) {
            for (int j = 0; j < result[i].size(); ++j) {
                (*result_bool)[i][j] = result[i][j] == live_cell;
            }
        }

#ifdef ENABLE_TIMING
        auto output_end = high_resolution_clock::now();
        durations_show_viewports_output[thread_index] += output_end - output_start;
#endif

        return result_bool;
    }

    void rehash(int n_threads) {
        concurrent_hashmap* new_hashmap = concurrent_hashmap::rehash(hashmap, n_threads);
        delete hashmap;
        hashmap = new_hashmap;
    }

    static void print_grid(const vector<vector<bool>>& grid) {
        // std::cout << '|';
        // for (size_t i = 0; i < grid[0].size(); ++i) std::cout << '-';
        // std::cout << '|' << std::endl;
        std::cout << '-';
        for (size_t i = 0; i < grid[0].size(); ++i) std::cout << ' ';
        std::cout << '-' << std::endl;
        for (const auto& row : grid) {
            std::cout << '|';
            for (bool cell : row) {
                std::cout << (cell ? 'x' : ' ');
            }
            std::cout << '|' << std::endl;
        }
        // std::cout << '|';
        // for (size_t i = 0; i < grid[0].size(); ++i) std::cout << '-';
        // std::cout << '|' << std::endl;
    }

    // vector<vector<bool>> expand_quad(quad* input) {
    //     // convert a quad to a grid of bools for debugging purposes
    //     if (input == nullptr) {
    //         return {};
    //     } else if (input == dead_cell) {
    //         return {{false}};
    //     } else if (input == live_cell) {
    //         return {{true}};
    //     }
    //     auto ne_grid = expand_quad(input->ne);
    //     auto nw_grid = expand_quad(input->nw);
    //     auto sw_grid = expand_quad(input->sw);
    //     auto se_grid = expand_quad(input->se);
    //     int half_size = ne_grid.size();
    //     vector<vector<bool>> result(2 * half_size, vector<bool>(2 * half_size));
    //     for (int i = 0; i < half_size; ++i) {
    //         for (int j = 0; j < half_size; ++j) {
    //             result[i][j] = nw_grid[i][j];
    //             result[i][j + half_size] = ne_grid[i][j];
    //             result[i + half_size][j] = sw_grid[i][j];
    //             result[i + half_size][j + half_size] = se_grid[i][j];
    //         }
    //     }
    //     return result;
    // }
};

int main() {

    vector<vector<bool>> initial_state;
    int x_center;
    int y_center;
    int period;
    int x_speed;
    int y_speed;

    if (pattern == nothing) {
        int initial_state_sidelength = 16;
        int middle = initial_state_sidelength / 2;
        initial_state = vector<vector<bool>>(initial_state_sidelength, vector<bool>(initial_state_sidelength, false));
        x_center = 0;
        y_center = 0;
        period = 1;
        x_speed = 0;
        y_speed = 0;

    } else if (pattern == r_pentomino) {
        int initial_state_sidelength = 16;
        int middle = initial_state_sidelength / 2;
        initial_state = vector<vector<bool>>(initial_state_sidelength, vector<bool>(initial_state_sidelength, false));
        initial_state[middle - 1][middle    ] = true;
        initial_state[middle    ][middle - 1] = true;
        initial_state[middle    ][middle    ] = true;
        initial_state[middle + 1][middle    ] = true;
        initial_state[middle + 1][middle + 1] = true;
        x_center = 0;
        y_center = 0;
        period = 1;
        x_speed = 0;
        y_speed = 0;

    } else if (pattern == glider) {
        int initial_state_sidelength = 16;
        int middle = initial_state_sidelength / 2;
        initial_state = vector<vector<bool>>(initial_state_sidelength, vector<bool>(initial_state_sidelength, false));
        initial_state[middle - 1][middle - 1] = true;
        initial_state[middle - 1][middle    ] = true;
        initial_state[middle - 1][middle + 1] = true;
        initial_state[middle    ][middle + 1] = true;
        initial_state[middle + 1][middle    ] = true;
        x_center = 0;
        y_center = 0;
        period = 4;
        x_speed = 1;
        y_speed = -1;

    } else if (pattern == lightweight_spaceship) {
        int initial_state_sidelength = 16;
        int middle = initial_state_sidelength / 2;
        initial_state = vector<vector<bool>>(initial_state_sidelength, vector<bool>(initial_state_sidelength, false));
        initial_state[middle - 1][middle - 1] = true;
        initial_state[middle - 1][middle] = true;
        initial_state[middle - 1][middle + 1] = true;
        initial_state[middle - 1][middle + 2] = true;
        initial_state[middle    ][middle - 1] = true;
        initial_state[middle    ][middle + 3] = true;
        initial_state[middle + 1][middle - 1] = true;
        initial_state[middle + 2][middle    ] = true;
        initial_state[middle + 2][middle + 3] = true; 
        x_center = 0;
        y_center = 0;
        period = 4;
        x_speed = -2;
        y_speed = 0;

    } else if (pattern == twenty_cell_quadratic_growth) {
        int initial_state_sidelength = 256;
        int middle = initial_state_sidelength / 2;
        initial_state = vector<vector<bool>>(initial_state_sidelength, vector<bool>(initial_state_sidelength, false));
        initial_state[middle + 32][middle     ] = true;
        initial_state[middle + 30][middle +  1] = true;
        initial_state[middle + 31][middle +  1] = true;
        initial_state[middle + 29][middle +  2] = true;
        initial_state[middle + 32][middle +  2] = true;
        initial_state[middle + 28][middle +  3] = true;
        initial_state[middle + 28][middle +  5] = true;
        initial_state[middle + 28][middle +  6] = true;
        initial_state[middle + 29][middle +  6] = true;
        initial_state[middle +  8][middle + 88] = true;
        initial_state[middle +  8][middle + 89] = true;
        initial_state[middle +  8][middle + 90] = true;
        initial_state[middle +  1][middle + 92] = true;
        initial_state[middle +  0][middle + 94] = true;
        initial_state[middle +  1][middle + 94] = true;
        initial_state[middle +  2][middle + 94] = true;
        initial_state[middle +  2][middle + 95] = true;
        initial_state[middle + 20][middle + 95] = true;
        initial_state[middle + 19][middle + 96] = true;
        initial_state[middle + 20][middle + 96] = true;
        x_center = 0;
        y_center = 0;
        period = 11136;
        x_speed = 11136 / 12;
        y_speed = 11136 / 12;

    } else if (pattern == methuselah_126932979M) {
        int initial_state_sidelength = 64;
        int middle = initial_state_sidelength / 2;
        initial_state = vector<vector<bool>>(initial_state_sidelength, vector<bool>(initial_state_sidelength, false));
        initial_state[middle     ][middle + 11] = true;
        initial_state[middle     ][middle + 13] = true;
        initial_state[middle +  1][middle + 10] = true;
        initial_state[middle +  2][middle + 11] = true;
        initial_state[middle +  2][middle + 14] = true;
        initial_state[middle +  3][middle + 13] = true;
        initial_state[middle +  3][middle + 14] = true;
        initial_state[middle +  3][middle + 15] = true;
        initial_state[middle +  5][middle +  2] = true;
        initial_state[middle +  6][middle     ] = true;
        initial_state[middle +  7][middle +  2] = true;
        initial_state[middle +  8][middle     ] = true;
        initial_state[middle +  9][middle +  2] = true;
        initial_state[middle + 10][middle +  2] = true;
        initial_state[middle + 11][middle +  3] = true;
        initial_state[middle + 11][middle +  5] = true;
        initial_state[middle + 12][middle +  5] = true;
        initial_state[middle + 16][middle + 18] = true;
        initial_state[middle + 16][middle + 19] = true;
        initial_state[middle + 16][middle + 20] = true;
        initial_state[middle + 18][middle + 14] = true;
        initial_state[middle + 18][middle + 15] = true;
        initial_state[middle + 19][middle + 12] = true;
        initial_state[middle + 19][middle + 15] = true;
        initial_state[middle + 21][middle + 12] = true;
        initial_state[middle + 21][middle + 14] = true;
        initial_state[middle + 22][middle + 13] = true;
        x_center = 0;
        y_center = 0;
        period = 1;
        x_speed = 0;
        y_speed = 0;

    } else if (pattern == lidka) {
        int initial_state_sidelength = 32;
        int middle = initial_state_sidelength / 2;
        initial_state = vector<vector<bool>>(initial_state_sidelength, vector<bool>(initial_state_sidelength, false));
        initial_state[middle     ][middle +  1] = true;
        initial_state[middle +  1][middle     ] = true;
        initial_state[middle +  1][middle +  2] = true;
        initial_state[middle +  2][middle +  1] = true;
        initial_state[middle +  4][middle + 14] = true;
        initial_state[middle +  5][middle + 12] = true;
        initial_state[middle +  5][middle + 14] = true;
        initial_state[middle +  6][middle + 11] = true;
        initial_state[middle +  6][middle + 12] = true;
        initial_state[middle +  6][middle + 14] = true;
        initial_state[middle +  8][middle + 10] = true;
        initial_state[middle +  8][middle + 11] = true;
        initial_state[middle +  8][middle + 12] = true;
        x_center = 0;
        y_center = 0;
        period = 1;
        x_speed = 0;
        y_speed = 0;
    }

    // initialization
    int n_threads = global_n_threads;
    int n_shards = global_n_shards;
    cout << "Using " << n_threads << " threads and " << n_shards << " shards." << endl;
    hashlife my_hashlife(initial_state, global_log_n_shards);

    // render some viewports
    vector<vector<vector<bool>>*> viewports(n_timesteps, nullptr); 
    int next_timestep = 0;
    auto wall_clock_start = high_resolution_clock::now();
    for(;;) {

        // do computation in parallel until we finish or need a rehash
#ifdef ENABLE_TIMING
        auto viewports_start = high_resolution_clock::now();
#endif
        #pragma omp parallel num_threads(n_threads)
        {
            int thread_index = omp_get_thread_num();
            for(;;) {

                // pause the computation if we need to run a rehash
                bool rehash_needed;
                #pragma omp atomic read
                rehash_needed = my_hashlife.hashmap->rehash_needed;
                if (rehash_needed) {
                    break;
                }

                // stop the computation if we've finished all viewports, otherwise, do another viewport
                bool done;
                int curr_timestep;
                #pragma omp critical
                {
                    if (next_timestep >= n_timesteps) {
                        done = true;
                    } else {
                        done = false;
                        curr_timestep = next_timestep;
                        next_timestep++;
#ifdef ENABLE_VERBOSE
                        if (next_timestep % (n_timesteps / 10) == 0) {
                            cout << "Thread " << thread_index << " starting viewport " << next_timestep << "." << endl;
                        }
#endif
                    }
                }
                if (done) {
                    break;
                } else {
                    int x_min = (x_center - viewport_half_width) + (x_speed * curr_timestep / period);
                    int y_min = (y_center - viewport_half_height) + (y_speed * curr_timestep / period);
                    int x_max = (x_center + viewport_half_width) + (x_speed * curr_timestep / period);
                    int y_max = (y_center + viewport_half_height) + (y_speed * curr_timestep / period);
#ifdef ENABLE_TIMING
                    auto show_viewport_start = high_resolution_clock::now();
#endif
                    viewports[curr_timestep] = my_hashlife.show_viewport(curr_timestep, x_min, y_min, x_max, y_max, thread_index);
#ifdef ENABLE_TIMING            
                    auto show_viewport_end = high_resolution_clock::now();
                    durations_show_viewports[thread_index] += show_viewport_end - show_viewport_start;
#endif
                }
            }
        }
#ifdef ENABLE_TIMING
        auto viewports_end = high_resolution_clock::now();
        duration_viewports += viewports_end - viewports_start;
#endif

        // if we've finished all viewports, exit the loop
        if (next_timestep == n_timesteps) {
            break;
        }

        // do a rehash (since that's the only other reason we could have exited the parallel block)
#ifdef ENABLE_TIMING
        auto rehashing_start = high_resolution_clock::now();
#endif
#ifdef ENABLE_VERBOSE
        cout << "Rehashing to a hashmap with capacity " << (1 << my_hashlife.hashmap->log_capacity) << "." << endl; 
#endif
        my_hashlife.rehash(n_threads);
#ifdef ENABLE_TIMING
        auto rehashing_end = high_resolution_clock::now();
        duration_rehashing += rehashing_end - rehashing_start;
#endif

    }
    auto wall_clock_end = high_resolution_clock::now();
    high_resolution_clock::duration wall_clock_total = wall_clock_end - wall_clock_start;
    cout << endl << "--- Wall-Clock Time Statistics ---" << endl;
    cout << "Whole computation took " << duration_cast<nanoseconds>(wall_clock_total).count() << " nanoseconds." << endl;

#ifdef ENABLE_TIMING
    // report timings
    auto total_durations_show_viewports = durations_show_viewports[0]; for (int i = 1; i < n_threads; i++) { total_durations_show_viewports += durations_show_viewports[i]; }
    auto total_durations_show_viewports_planning = durations_show_viewports_planning[0]; for (int i = 1; i < n_threads; i++) { total_durations_show_viewports_planning += durations_show_viewports_planning[i]; }
    auto total_durations_show_viewports_solution = durations_show_viewports_solution[0]; for (int i = 1; i < n_threads; i++) { total_durations_show_viewports_solution += durations_show_viewports_solution[i]; }
    auto total_durations_show_viewports_output = durations_show_viewports_output[0]; for (int i = 1; i < n_threads; i++) { total_durations_show_viewports_output += durations_show_viewports_output[i]; }
    auto total_durations_hashmap = durations_hashmap[0]; for (int i = 1; i < n_threads; i++) { total_durations_hashmap += durations_hashmap[i]; }
    auto total_durations_hashmap_set_lock = durations_hashmap_set_lock[0]; for (int i = 1; i < n_threads; i++) { total_durations_hashmap_set_lock += durations_hashmap_set_lock[i]; }
    auto total_durations_hashmap_unset_lock = durations_hashmap_unset_lock[0]; for (int i = 1; i < n_threads; i++) { total_durations_hashmap_unset_lock += durations_hashmap_unset_lock[i]; }
    auto total_durations_hashmap_hash_operations = durations_hashmap_hash_operations[0]; for (int i = 1; i < n_threads; i++) { total_durations_hashmap_hash_operations += durations_hashmap_hash_operations[i]; }
    auto total_durations_hashmap_real_work = total_durations_hashmap_hash_operations + total_durations_hashmap_set_lock + total_durations_hashmap_unset_lock;
    high_resolution_clock::duration task_management_overhead = wall_clock_total - (duration_viewports + duration_rehashing);
    high_resolution_clock::duration hashmap_timing_overhead = total_durations_hashmap - (total_durations_hashmap_hash_operations + total_durations_hashmap_set_lock + total_durations_hashmap_unset_lock);
    
#if ENABLE_TIMING == 1
    // cout << "Viewport calculation took " << duration_cast<nanoseconds>(duration_viewports).count() << " nanoseconds." << endl;
    // cout << "\tThis is " << 100. * duration_viewports / () << "% of the total time." << endl;
    // cout << "Rehashing operations took " << duration_cast<nanoseconds>(duration_rehashing).count() << " nanoseconds." << endl;
    // cout << "\tThis is " << 100. * duration_rehashing / (duration_rehashing + duration_rehashing + task_management_overhead) << "% of the total time." << endl;
    // cout << "This leaves a discreptancy (mostly due to parallelism overhead) of " << duration_cast<nanoseconds>(task_management_overhead).count() << " nanoseconds." << endl;
    // cout << "\tThis is " << 100. * task_management_overhead / (duration_rehashing + duration_rehashing + task_management_overhead) << "% of the total time." << endl;

    // cout << endl << "--- hashlife::show_viewport() Thread Work Statistics --" << endl;
    // cout << "\tActual computation takes up " << 100. * total_durations_show_viewports / (duration_viewports * n_threads) << "% of the work in viewport calculation." << endl;
    // cout << "\tOverhead from task distribution and syncronization takes up the remaining " << (100. - (100. * total_durations_show_viewports / (duration_viewports * n_threads))) << "%." << endl;
    // cout << "Planning part of show_viewport() took up a total of " << duration_cast<nanoseconds>(total_durations_show_viewports_planning).count() * 1e-9 << " thread-seconds." << endl;
    // cout << "\tThis is " << 100. * total_durations_show_viewports_planning / total_durations_show_viewports << "% of the work in show_viewport()." << endl;
    // cout << "\tThis is an average of " << duration_cast<nanoseconds>(total_durations_show_viewports_planning).count() * 1e-9 / n_threads << " seconds per thread." << endl;
    // cout << "Solution part of show_viewport() took up a total of " << duration_cast<nanoseconds>(total_durations_show_viewports_solution).count() * 1e-9 << " thread-seconds." << endl;
    // cout << "\tThis is " << 100. * total_durations_show_viewports_solution / total_durations_show_viewports << "% of the work in show_viewport()." << endl;
    // cout << "\tThis is an average of " << duration_cast<nanoseconds>(total_durations_show_viewports_solution).count() * 1e-9 / n_threads << " seconds per thread." << endl;
    // cout << "Output part of show_viewport() took up a total of " << duration_cast<nanoseconds>(total_durations_show_viewports_output).count() * 1e-9 << " thread-seconds." << endl;
    // cout << "\tThis is " << 100. * total_durations_show_viewports_output / total_durations_show_viewports << "% of the work in show_viewport()." << endl;
    // cout << "\tThis is an average of " << duration_cast<nanoseconds>(total_durations_show_viewports_output).count() * 1e-9 / n_threads << " seconds per thread." << endl;

    double total = duration_cast<nanoseconds>(duration_viewports + duration_rehashing + task_management_overhead).count();
    auto viewport_average_work = (total_durations_show_viewports / n_threads);
    auto barrier_waiting_overhead = (duration_viewports - viewport_average_work);
    double total_overhead = duration_cast<nanoseconds>(task_management_overhead + barrier_waiting_overhead).count();
    cout << (duration_cast<nanoseconds>(viewport_average_work).count()/total) << endl;
    cout << (duration_cast<nanoseconds>(duration_rehashing).count()/total) << endl;
    cout << (total_overhead/total) << endl;

    double total2 = duration_cast<nanoseconds>(total_durations_show_viewports_planning + total_durations_show_viewports_solution + total_durations_show_viewports_output).count();
    cout << (duration_cast<nanoseconds>(total_durations_show_viewports_planning).count() / total2) * (duration_cast<nanoseconds>(viewport_average_work)/total).count() << endl;
    cout << (duration_cast<nanoseconds>(total_durations_show_viewports_solution).count() / total2) * (duration_cast<nanoseconds>(viewport_average_work)/total).count() << endl;
    cout << (duration_cast<nanoseconds>(total_durations_show_viewports_output).count() / total2) * (duration_cast<nanoseconds>(viewport_average_work).count()/total) << endl;

#endif

#if ENABLE_TIMING >= 2
    cout << endl << "--- concurrent_hashmap::get_or_construct() Thread Work Statistics --" << endl;
    cout << "During show_viewport() calls, get_or_construct() took up a total of " << duration_cast<nanoseconds>(total_durations_hashmap).count() * 1e-9 << " thread-seconds." << endl;
    cout << "Hash operations (hash, probing, insertion, etc.) took up a total of " << duration_cast<nanoseconds>(total_durations_hashmap_hash_operations).count() * 1e-9 << " thread-seconds." << endl;
    cout << "\tThis is " << 100. * total_durations_hashmap_hash_operations / total_durations_hashmap << "% of the work in get_or_construct()." << endl;
    cout << "\tThis is an average of " << duration_cast<nanoseconds>(total_durations_hashmap_hash_operations).count() * 1e-9 / n_threads << " seconds per thread." << endl;
    cout << "omp_set_lock() operations took up a total of " << duration_cast<nanoseconds>(total_durations_hashmap_set_lock).count() * 1e-9 << " thread-seconds." << endl;
    cout << "\tThis is " << 100. * total_durations_hashmap_set_lock / total_durations_hashmap << "% of the work in get_or_construct()." << endl;
    cout << "\tThis is an average of " << duration_cast<nanoseconds>(total_durations_hashmap_set_lock).count() * 1e-9 / n_threads << " seconds per thread." << endl;
    cout << "omp_unset_lock() operations took up a total of " << duration_cast<nanoseconds>(total_durations_hashmap_unset_lock).count() * 1e-9 << " thread-seconds." << endl;
    cout << "\tThis is " << 100. * total_durations_hashmap_unset_lock / total_durations_hashmap << "% of the work in get_or_construct()." << endl;
    cout << "\tThis is an average of " << duration_cast<nanoseconds>(total_durations_hashmap_unset_lock).count() * 1e-9 / n_threads << " seconds per thread." << endl;
    cout << "This leaves a discreptancy (mostly due to timing overhead) of " << duration_cast<nanoseconds>(hashmap_timing_overhead).count() * 1e-9 << " thread-seconds." << endl;
    cout << "\tThis is " << 100. * hashmap_timing_overhead / total_durations_hashmap << "% of the work in get_or_construct()." << endl;
    cout << "\tThis is an average of " << duration_cast<nanoseconds>(hashmap_timing_overhead).count() * 1e-9 / n_threads << " seconds per thread." << endl;
    cout << "\t\tWith this overhead removed, hash operations represent " << 100. * total_durations_hashmap_hash_operations / total_durations_hashmap_real_work << "% of the work in get_or_construct()." << endl;
    cout << "\t\tWith this overhead removed, omp_set_lock() represents " << 100. * total_durations_hashmap_set_lock / total_durations_hashmap_real_work << "% of the work in get_or_construct()." << endl;
    cout << "\t\tWith this overhead removed, omp_unset_lock() represents " << 100. * total_durations_hashmap_unset_lock / total_durations_hashmap_real_work << "% of the work in get_or_construct()." << endl;
#endif

    cout << endl;
#endif

#if ENABLE_STATS >= 2
    cout << endl << "--- Hashmap Behavior Statistics ---" << endl;
    auto total_n_constructs = n_constructs[0]; for (int i = 1; i < n_threads; i++) { total_n_constructs += n_constructs[i]; }
    auto total_n_gets = n_gets[0]; for (int i = 1; i < n_threads; i++) { total_n_gets += n_gets[i]; }
    auto total_buckets_probed = thread_buckets_probed[0]; for (int i = 1; i < n_threads; i++) { total_buckets_probed += thread_buckets_probed[i]; }
    auto total_locks_set = thread_locks_set[0]; for (int i = 1; i < n_threads; i++) { total_locks_set += thread_locks_set[i]; }
    auto total_calls = total_n_constructs + total_n_gets;

    cout << "Total number of calls to get_or_construct() across all threads: " << total_calls << endl;
    cout << "\tTotal number of calls that resulted in quad construction: " << total_n_constructs << " (" << 100. * total_n_constructs / total_calls << "%)" << endl;
    cout << "\tTotal number of calls that resulted in finding existing quad: " << total_n_gets << " (" << 100. * total_n_gets / total_calls << "%)" << endl;
#if ENABLE_TIMING >= 2
    cout << "\tAverage time per call: " << duration_cast<nanoseconds>(total_durations_hashmap).count() * 1e-9 / total_calls << " seconds." << endl;
#endif
    cout << "Average number of buckets probed per call: " << total_buckets_probed / ((double) total_calls) << endl;
    cout << "Average number of shards locked per call: " << total_locks_set / ((double) total_calls) << endl;
#if ENABLE_TIMING >= 2
    cout << "\tAverage time required to lock a shard: " << duration_cast<nanoseconds>(total_durations_hashmap_set_lock).count() * 1e-9 / total_calls << " seconds." << endl;
#endif
#if ENABLE_STATS >= 3
    auto total_locks_contended = thread_locks_contended[0]; for (int i = 1; i < n_threads; i++) { total_locks_contended += thread_locks_contended[i]; }
    cout << "Total number of lock contentions: " << total_locks_contended << " (" << 100. * total_locks_contended / total_locks_set << "%)" << endl;
#endif

#endif

#ifdef ENABLE_VISUAL
    usleep(2000000);
    for (int i = 0; i < n_timesteps; ++i) {
        hashlife::print_grid(*viewports[i]);
        delete viewports[i];
        usleep(50000);
    }
#endif

    return 0;

}
