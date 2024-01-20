#define IS_POWER_OF_TWO(x) ((x) > 0 && ((x) & ((x) - 1)) == 0)
// #define ENABLE_VISUAL

#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifdef ENABLE_VISUAL
#include <unistd.h>
#endif

using namespace std;

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

class serial_hashmap {
public:

    // constants
    int log_capacity;
    int capacity;
    int rehash_threshold;

    int size;
    bool rehash_needed;
    tuple<quad*, quad*, quad*, quad*, quad*, size_t>** buckets;

    serial_hashmap(int log_capacity) {

        this->log_capacity = log_capacity;
        capacity = 1 << log_capacity;
        rehash_threshold = 1 << (log_capacity - 1);

        size = 0;
        rehash_needed = false;
        buckets = new tuple<quad*, quad*, quad*, quad*, quad*, size_t>*[capacity]();
    }

    ~serial_hashmap() {
        // tuples are reused in the rehashed hashmap, so don't delete them
        // this probably violates some programming principles regarding memory management
        // for (int i = 0; i < capacity; ++i) {
        //     if (buckets[i] != nullptr) {
        //         delete buckets[i];
        //     }
        // }
        delete[] buckets;
    }

    serial_hashmap(serial_hashmap&&) = delete;
    serial_hashmap& operator=(serial_hashmap&&) = delete;

    quad* get_or_construct(quad* ne, quad* nw, quad* sw, quad* se) {
        size_t hash_value = quad_hash(ne, nw, sw, se);
        int bucket_index = hash_value % capacity;  // hash_value & (capacity - 1);
        int first_bucket_index = bucket_index;
        quad* ptr_to_return;
        for(;;) {
            if (buckets[bucket_index] == nullptr) {
                // key not found; construct and insert a new quad
                ptr_to_return = new quad(ne, nw, sw, se);
                buckets[bucket_index] = new tuple<quad*, quad*, quad*, quad*, quad*, size_t>(ne, nw, sw, se, ptr_to_return, hash_value);
                size++;
                if (size >= rehash_threshold) {
                    rehash_needed = true;
                }
                break;
            } else if (get<0>(*buckets[bucket_index]) == ne && 
                       get<1>(*buckets[bucket_index]) == nw && 
                       get<2>(*buckets[bucket_index]) == sw && 
                       get<3>(*buckets[bucket_index]) == se) {
                // found our key; return the corresponding quad
                ptr_to_return = get<4>(*buckets[bucket_index]);
                break;
            } else {
                // hash collision; continue linearly probing, switching locks if necessary
                bucket_index = (bucket_index + 1) % capacity;  // (bucket_index + 1) % capacity;
                if (bucket_index == first_bucket_index) {
                    cout << "Attempted to construct and insert a new item, but all hashmap shards were full. This is fatal."  << endl;
                    throw;
                }
            }
        }
        return ptr_to_return;
    }

    static serial_hashmap* rehash(serial_hashmap* old_hashmap) {
        // constructs a rehashed version the old hashmap
        // caller is responsible for managing deletion of both the old and new hashmap
        serial_hashmap* new_hashmap = new serial_hashmap(old_hashmap->log_capacity + 1);
        for (int i = 0; i < old_hashmap->capacity; ++i) {
            if (old_hashmap->buckets[i] != nullptr) {
                new_hashmap->rehash_insert(old_hashmap->buckets[i]);
            }
        }
        return new_hashmap;
    }

private:
    void rehash_insert(tuple<quad*, quad*, quad*, quad*, quad*, size_t>* item) {
        size_t hash_value = get<5>(*item);
        int bucket_index = hash_value % capacity;  // hash_value & (capacity - 1);
        int first_bucket_index = bucket_index;
        for(;;) {
            if (buckets[bucket_index] == nullptr) {
                // key not found; insert the provided quad
                buckets[bucket_index] = item;
                size++;
                if (size >= rehash_threshold) {
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
                bucket_index = (bucket_index + 1) % capacity;  // (bucket_index + 1) % capacity;
                if (bucket_index == first_bucket_index) {
                    cout << "Attempted to construct and insert a new item, but all hashmap shards were full. This is fatal."  << endl;
                    throw;
                }
            }
        }
    }
};

class hashlife {
public:
    bool initialized = false;
    quad* dead_cell = new quad();
    quad* live_cell = new quad();
    serial_hashmap* hashmap;
    quad* top_quad;
    vector<quad*> dead_quads = {dead_cell};

    void initialize_hashmap() {

        // this function should be called only once
        if (initialized) {
            cout << "Hashmap cannot be initialized more than once." << endl;
            throw;
        }
        initialized = true;

        // create the hashmap
        hashmap = new serial_hashmap(18);

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

    hashlife(const vector<vector<bool>>& initial_state) {

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
        initialize_hashmap();

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
        if (input->result != nullptr) {
            return input->result;
        } else {
            
            // construct 5 auxillary quads
            quad* aux_n = hashmap->get_or_construct(input->ne->nw, input->nw->ne, input->nw->se, input->ne->sw);
            quad* aux_w = hashmap->get_or_construct(input->nw->se, input->nw->sw, input->sw->nw, input->sw->ne);
            quad* aux_s = hashmap->get_or_construct(input->se->nw, input->sw->ne, input->sw->se, input->se->sw);
            quad* aux_e = hashmap->get_or_construct(input->ne->se, input->ne->sw, input->se->nw, input->se->ne);
            quad* aux_c = hashmap->get_or_construct(input->ne->sw, input->nw->se, input->sw->ne, input->se->nw);

            // first 9 "scoops"
            quad* layer2_e = get_or_compute_result(aux_e);
            quad* layer2_ne = get_or_compute_result(input->ne);
            quad* layer2_n = get_or_compute_result(aux_n);
            quad* layer2_nw = get_or_compute_result(input->nw);
            quad* layer2_w = get_or_compute_result(aux_w);
            quad* layer2_sw = get_or_compute_result(input->sw);
            quad* layer2_s = get_or_compute_result(aux_s);
            quad* layer2_se = get_or_compute_result(input->se);
            quad* layer2_c = get_or_compute_result(aux_c);

            // construct 4 auxillary quads
            quad* layer2_aux_ne = hashmap->get_or_construct(layer2_ne, layer2_n, layer2_c, layer2_e);
            quad* layer2_aux_nw = hashmap->get_or_construct(layer2_n, layer2_nw, layer2_w, layer2_c);
            quad* layer2_aux_sw = hashmap->get_or_construct(layer2_c, layer2_w, layer2_sw, layer2_s);
            quad* layer2_aux_se = hashmap->get_or_construct(layer2_e, layer2_c, layer2_s, layer2_se);

            // next 4 "scoops"
            quad* result_ne = get_or_compute_result(layer2_aux_ne);
            quad* result_nw = get_or_compute_result(layer2_aux_nw);
            quad* result_sw = get_or_compute_result(layer2_aux_sw);
            quad* result_se = get_or_compute_result(layer2_aux_se);

            // construct, save, and return result
            quad* result = hashmap->get_or_construct(result_ne, result_nw, result_sw, result_se);
            input->result = result;
            return result;
        }
    }

    quad* get_dead_quad(int size) {
        while (size >= dead_quads.size()) {
            dead_quads.push_back(hashmap->get_or_construct(dead_quads.back(), dead_quads.back(), dead_quads.back(), dead_quads.back()));
        }
        return dead_quads[size];
    }

    void pad_top_quad() {
        quad* dead_quad = get_dead_quad(top_quad->ne->log_size);
        quad* new_ne = hashmap->get_or_construct(dead_quad, dead_quad, top_quad->ne, dead_quad);
        quad* new_nw = hashmap->get_or_construct(dead_quad, dead_quad, dead_quad, top_quad->nw);
        quad* new_sw = hashmap->get_or_construct(top_quad->sw, dead_quad, dead_quad, dead_quad);
        quad* new_se = hashmap->get_or_construct(dead_quad, top_quad->se, dead_quad, dead_quad);
        top_quad = hashmap->get_or_construct(new_ne, new_nw, new_sw, new_se);
    }

    vector<vector<quad*>> expand_result(vector<vector<quad*>> input_grid, tuple<int, int, int, int, int, int> input_step, tuple<int, int, int, int, int, int> output_step) {
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
                            input_grid[next_input_idx_y + 1][next_input_idx_x + 1]->nw->nw);
                    } else {
                        aux_grid[next_aux_idx_y][next_aux_idx_x] = hashmap->get_or_construct(
                            input_grid[next_input_idx_y][next_input_idx_x + 1]->nw->sw,
                            input_grid[next_input_idx_y][next_input_idx_x]->ne->se,
                            input_grid[next_input_idx_y][next_input_idx_x]->se->ne,
                            input_grid[next_input_idx_y][next_input_idx_x + 1]->sw->nw);
                    }
                    next_input_idx_x++;
                } else {
                    if (next_aux_is_combo_y) {
                        aux_grid[next_aux_idx_y][next_aux_idx_x] = hashmap->get_or_construct(
                            input_grid[next_input_idx_y][next_input_idx_x]->se->sw,
                            input_grid[next_input_idx_y][next_input_idx_x]->sw->se,
                            input_grid[next_input_idx_y + 1][next_input_idx_x]->nw->ne,
                            input_grid[next_input_idx_y + 1][next_input_idx_x]->ne->nw);
                    } else {
                        aux_grid[next_aux_idx_y][next_aux_idx_x] = hashmap->get_or_construct(
                            input_grid[next_input_idx_y][next_input_idx_x]->ne->sw,
                            input_grid[next_input_idx_y][next_input_idx_x]->nw->se,
                            input_grid[next_input_idx_y][next_input_idx_x]->sw->ne,
                            input_grid[next_input_idx_y][next_input_idx_x]->se->nw);
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
                    aux_grid[next_aux_idx_y + 1][next_aux_idx_x + 1]);
            }
        }

        // get the output by taking the result of every element of the second auxillary grid
        vector<vector<quad*>> output_grid(output_dims_y, vector<quad*>(output_dims_x, nullptr));
        for (int next_output_idx_y = 0; next_output_idx_y < output_dims_y; next_output_idx_y++) {
            for (int next_output_idx_x = 0; next_output_idx_x < output_dims_x; next_output_idx_x++) {
                output_grid[next_output_idx_y][next_output_idx_x] = get_or_compute_result(aux_grid_2[next_output_idx_y][next_output_idx_x]);
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
            pad_top_quad();
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

        // perform the steps that we planned out, from the back of the vector to the front
        for (int i = steps.size() - 1; i > 0; i--) {
            if (get<0>(steps[i]) != get<0>(steps[i-1])) {
                result = expand_result(result, steps[i], steps[i-1]);
            } else {
                result = expand_static(result, steps[i], steps[i-1]);
            }
        }

        // convert result to a grid of booleans
        vector<vector<bool>>* result_bool = new vector<vector<bool>>(result.size(), vector<bool>(result[0].size()));
        for (int i = 0; i < result.size(); ++i) {
            for (int j = 0; j < result[i].size(); ++j) {
                (*result_bool)[i][j] = result[i][j] == live_cell;
            }
        }
        return result_bool;
    }

    void rehash() {
        serial_hashmap* new_hashmap = serial_hashmap::rehash(hashmap);
        delete hashmap;
        hashmap = new_hashmap;
    }

    static void print_grid(const vector<vector<bool>>& grid) {
        std::cout << '|';
        for (size_t i = 0; i < grid[0].size(); ++i) std::cout << '-';
        std::cout << '|' << std::endl;
        for (const auto& row : grid) {
            std::cout << '|';
            for (bool cell : row) {
                std::cout << (cell ? 'x' : ' ');
            }
            std::cout << '|' << std::endl;
        }
        std::cout << '|';
        for (size_t i = 0; i < grid[0].size(); ++i) std::cout << '-';
        std::cout << '|' << std::endl;
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

    // empty grid
    int initial_state_sidelength = 256;
    int middle = initial_state_sidelength / 2;
    vector<vector<bool>> initial_state(initial_state_sidelength, vector<bool>(initial_state_sidelength, false));

    // r-pentomino
    // initial_state[middle - 1][middle    ] = true;
    // initial_state[middle    ][middle - 1] = true;
    // initial_state[middle    ][middle    ] = true;
    // initial_state[middle + 1][middle    ] = true;
    // initial_state[middle + 1][middle + 1] = true;

    // // glider
    // initial_state[middle - 1][middle - 1] = true;
    // initial_state[middle - 1][middle    ] = true;
    // initial_state[middle - 1][middle + 1] = true;
    // initial_state[middle    ][middle + 1] = true;
    // initial_state[middle + 1][middle    ] = true;

    // // lighweight spaceship
    // initial_state[middle - 1][middle - 1] = true;
    // initial_state[middle - 1][middle] = true;
    // initial_state[middle - 1][middle + 1] = true;
    // initial_state[middle - 1][middle + 2] = true;
    // initial_state[middle    ][middle - 1] = true;
    // initial_state[middle    ][middle + 3] = true;
    // initial_state[middle + 1][middle - 1] = true;
    // initial_state[middle + 2][middle    ] = true;
    // initial_state[middle + 2][middle + 3] = true; 

    // 20-cell quadratic growth
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

    // initialization
    hashlife my_hashlife(initial_state);

    // viewport parameters
    int n_timesteps = 200000;
    int x_padding = 96;
    int y_padding = 19;
    int x_min = 0 - x_padding;
    int y_min = 0 - y_padding;
    int x_max = 97 + x_padding;
    int y_max = 33 + y_padding;

    // render some viewports
    vector<vector<vector<bool>>*> viewports(n_timesteps, nullptr); 
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < n_timesteps; i++) {
        if (my_hashlife.hashmap->rehash_needed) {
            my_hashlife.rehash();
        }
        viewports[i] = my_hashlife.show_viewport(i, x_min, y_min, x_max, y_max);
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Computation took " << elapsed_seconds.count() << " seconds." << std::endl;

#ifdef ENABLE_VISUAL
    usleep(5000000);
    for (int i = 0; i < n_timesteps; ++i) {
        hashlife::print_grid(*viewports[i]);
        delete viewports[i];
        usleep(50000);
    }
#endif

    return 0;

}
