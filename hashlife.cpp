// #define DEBUG_HASH
// #define DEBUG_RESULT
// #define DEBUG_EXPAND
// #define DEBUG_PROGRESS
#define ENABLE_SLEEP

#include <cmath>
#include <functional>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <vector>
#ifdef ENABLE_SLEEP
    #include <unistd.h>
#endif

using namespace std;

int round_two(int number, int exponent, bool round_up) {
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
    int size; // square macrocell with side lengths 2^size
    quad* ne;
    quad* nw;
    quad* sw;
    quad* se;
    quad* result;

    quad() 
        : size(0), ne(nullptr), nw(nullptr), sw(nullptr), se(nullptr), result(nullptr) {}

    quad(quad* ne, quad* nw, quad* sw, quad* se) 
        : size(ne->size + 1), ne(ne), nw(nw), sw(sw), se(se), result(nullptr) {}

    quad(quad* ne, quad* nw, quad* sw, quad* se, quad* result) 
        : size(ne->size + 1), ne(ne), nw(nw), sw(sw), se(se), result(result) {}

    bool operator==(const quad& other) const {
        return ne == other.ne && nw == other.nw && sw == other.sw && se == other.se;
    }
};

struct quad_hash {
    // hash function that takes four quad* as input
    size_t operator()(const tuple<quad*, quad*, quad*, quad*>& t) const {
        size_t h1 = hash<quad*>{}(get<0>(t));
        size_t h2 = hash<quad*>{}(get<1>(t));
        size_t h3 = hash<quad*>{}(get<2>(t));
        size_t h4 = hash<quad*>{}(get<3>(t));
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
    }
};

class hashlife {
public:
    quad* dead_cell = new quad();
    quad* live_cell = new quad();
    unordered_map<tuple<quad*, quad*, quad*, quad*>, quad*, quad_hash> hashmap;
    quad* top_quad;
    vector<quad*> dead_quads = {dead_cell};

    unordered_map<tuple<quad*, quad*, quad*, quad*>, quad*, quad_hash> initialize_hashmap() {

        // enumerate both (1x1) macrocells; these aren't memoized
        quad* quads_1x1[] = {dead_cell, live_cell};

        // create the empty hashmap
        unordered_map<tuple<quad*, quad*, quad*, quad*>, quad*, quad_hash> hashmap;

        // generate and memoize all 16 (2x2) macrocells; they don't have results
        vector<quad*> quads_2x2;
        tuple<quad*, quad*, quad*, quad*> next_key;
        quad* next_quad;
        for (int ne_idx = 0; ne_idx < 2; ne_idx++) {
            for (int nw_idx = 0; nw_idx < 2; nw_idx++) {
                for (int sw_idx = 0; sw_idx < 2; sw_idx++) {
                    for (int se_idx = 0; se_idx < 2; se_idx++) {
                        next_key = make_tuple(quads_1x1[ne_idx], quads_1x1[nw_idx], quads_1x1[sw_idx], quads_1x1[se_idx]);
                        next_quad = new quad(quads_1x1[ne_idx], quads_1x1[nw_idx], quads_1x1[sw_idx], quads_1x1[se_idx]);
                        hashmap[next_key] = next_quad;
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
                        next_results = hashmap[make_tuple(results_ne, results_nw, results_sw, results_se)];
                        
                        // store in hashmap
                        next_key = make_tuple(quads_2x2[ne_idx], quads_2x2[nw_idx], quads_2x2[sw_idx], quads_2x2[se_idx]);
                        next_quad = new quad(quads_2x2[ne_idx], quads_2x2[nw_idx], quads_2x2[sw_idx], quads_2x2[se_idx], next_results);
                        hashmap[next_key] = next_quad;
                    }
                }
            }
        }
        return hashmap;
    }

    quad* get_or_add_quad(quad* ne, quad* nw, quad* sw, quad* se) {
        auto key = make_tuple(ne, nw, sw, se);
        auto it = hashmap.find(key);
        if (it != hashmap.end()) {
#ifdef DEBUG_HASH
            cout << "Quad of size " << it->second->size << " is already hashed." << endl;
#endif
            return it->second;
        } else {
            quad* new_quad = new quad(ne, nw, sw, se);
            hashmap[key] = new_quad;
#ifdef DEBUG_HASH
            cout << "Quad of size " << new_quad->size << " was created. ";
            if (top_quad == ne || top_quad == nw || top_quad == sw || top_quad == se) {
                cout << "The newly created quad uses top_quad as a child. This should only occur in edge cases where everything is dead.";
            }
            cout << endl;
#endif
            return new_quad;
        }
    }

    hashlife(const vector<vector<bool>>& initial_state) {

        // force initial state size to be a power of 2
        int initial_state_sidelength = initial_state.size();
        if (!((initial_state_sidelength) > 0 && ((initial_state_sidelength) & ((initial_state_sidelength) - 1)) == 0)) {
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
        hashmap = initialize_hashmap();
#ifdef DEBUG_HASH
        cout << "Hashmap initialization complete." << endl;
#endif

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
                    initial_state_quad[y][x] = get_or_add_quad(
                        initial_state_quad[y][x + half_step], 
                        initial_state_quad[y][x], 
                        initial_state_quad[y + half_step][x], 
                        initial_state_quad[y + half_step][x + half_step]);
                }
            }
            half_step *= 2;
        }
        top_quad = initial_state_quad[0][0];
#ifdef DEBUG_PROGRESS
        cout << "Construction of initial state representation complete." << endl;
#endif
    }

    quad* get_or_compute_result(quad* input) {
        if (input->result != nullptr) {
#ifdef DEBUG_RESULT
            cout << "Result already available for quad of size " << input->size << "." << endl;
#endif
            return input->result;
        } else {
#ifdef DEBUG_RESULT
            cout << "Computing result for quad of size " << input->size << "." << endl;
#endif
            
            // construct 5 auxillary quads
            quad* aux_n = get_or_add_quad(input->ne->nw, input->nw->ne, input->nw->se, input->ne->sw);
            quad* aux_w = get_or_add_quad(input->nw->se, input->nw->sw, input->sw->nw, input->sw->ne);
            quad* aux_s = get_or_add_quad(input->se->nw, input->sw->ne, input->sw->se, input->se->sw);
            quad* aux_e = get_or_add_quad(input->ne->se, input->ne->sw, input->se->nw, input->se->ne);
            quad* aux_c = get_or_add_quad(input->ne->sw, input->nw->se, input->sw->ne, input->se->nw);

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
            quad* layer2_aux_ne = get_or_add_quad(layer2_ne, layer2_n, layer2_c, layer2_e);
            quad* layer2_aux_nw = get_or_add_quad(layer2_n, layer2_nw, layer2_w, layer2_c);
            quad* layer2_aux_sw = get_or_add_quad(layer2_c, layer2_w, layer2_sw, layer2_s);
            quad* layer2_aux_se = get_or_add_quad(layer2_e, layer2_c, layer2_s, layer2_se);

            // next 4 "scoops"
            quad* result_ne = get_or_compute_result(layer2_aux_ne);
            quad* result_nw = get_or_compute_result(layer2_aux_nw);
            quad* result_sw = get_or_compute_result(layer2_aux_sw);
            quad* result_se = get_or_compute_result(layer2_aux_se);

            // construct, save, and return result
            quad* result = get_or_add_quad(result_ne, result_nw, result_sw, result_se);
            input->result = result;
            return result;
        }
    }

    quad* get_dead_quad(int size) {
        while (size >= dead_quads.size()) {
            dead_quads.push_back(get_or_add_quad(dead_quads.back(), dead_quads.back(), dead_quads.back(), dead_quads.back()));
        }
        return dead_quads[size];
    }

    void pad_top_quad() {
        quad* dead_quad = get_dead_quad(top_quad->ne->size);
        quad* new_ne = get_or_add_quad(dead_quad, dead_quad, top_quad->ne, dead_quad);
        quad* new_nw = get_or_add_quad(dead_quad, dead_quad, dead_quad, top_quad->nw);
        quad* new_sw = get_or_add_quad(top_quad->sw, dead_quad, dead_quad, dead_quad);
        quad* new_se = get_or_add_quad(dead_quad, top_quad->se, dead_quad, dead_quad);
        top_quad = get_or_add_quad(new_ne, new_nw, new_sw, new_se);
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
                        aux_grid[next_aux_idx_y][next_aux_idx_x] = get_or_add_quad(
                            input_grid[next_input_idx_y][next_input_idx_x + 1]->sw->sw,
                            input_grid[next_input_idx_y][next_input_idx_x]->se->se,
                            input_grid[next_input_idx_y + 1][next_input_idx_x]->ne->ne,
                            input_grid[next_input_idx_y + 1][next_input_idx_x + 1]->nw->nw);
                    } else {
                        aux_grid[next_aux_idx_y][next_aux_idx_x] = get_or_add_quad(
                            input_grid[next_input_idx_y][next_input_idx_x + 1]->nw->sw,
                            input_grid[next_input_idx_y][next_input_idx_x]->ne->se,
                            input_grid[next_input_idx_y][next_input_idx_x]->se->ne,
                            input_grid[next_input_idx_y][next_input_idx_x + 1]->sw->nw);
                    }
                    next_input_idx_x++;
                } else {
                    if (next_aux_is_combo_y) {
                        aux_grid[next_aux_idx_y][next_aux_idx_x] = get_or_add_quad(
                            input_grid[next_input_idx_y][next_input_idx_x]->se->sw,
                            input_grid[next_input_idx_y][next_input_idx_x]->sw->se,
                            input_grid[next_input_idx_y + 1][next_input_idx_x]->nw->ne,
                            input_grid[next_input_idx_y + 1][next_input_idx_x]->ne->nw);
                    } else {
                        aux_grid[next_aux_idx_y][next_aux_idx_x] = get_or_add_quad(
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
                aux_grid_2[next_aux_idx_y][next_aux_idx_x] = get_or_add_quad(
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
        #ifdef DEBUG_EXPAND
            cout << "Expanding using result method. "
                 << "Input grid has height " << input_grid.size() << " and width " << input_grid[0].size() << ". "
                 << "Output grid has height " << output_grid.size() << " and width " << output_grid[0].size() << ". " << endl;
        #endif
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
        #ifdef DEBUG_EXPAND
            cout << "Expanding using static method. "
                 << "Input grid has height " << input_grid.size() << " and width " << input_grid[0].size() << ". "
                 << "Output grid has height " << output_grid.size() << " and width " << output_grid[0].size() << ". " << endl;
        #endif
        return output_grid;
    }

    vector<vector<bool>> show_viewport(int time, int x_min, int y_min, int x_max, int y_max) {
        
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
        while (true) {

            // stop if:
            // 1. we are at time zero
            // 2. we are left with one, two, or four macrocells
            // 3. all of which have the origin as one of their corners
            // 4. all of which are at least a quarter of the size of top_quad
            stop = true;
            if (get<0>(steps.back()) > 0) {
                stop = false;
            } else if ((1 << get<1>(steps.back())) < top_quad->size - 1) {
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

        #ifdef DEBUG_EXPAND
            for (const auto& step : steps) {
                cout << "(" 
                          << get<0>(step) << ", " 
                          << get<1>(step) << ", " 
                          << get<2>(step) << ", " 
                          << get<3>(step) << ", " 
                          << get<4>(step) << ", " 
                          << get<5>(step) 
                          << ")" << endl;
            }
        #endif

        // ensure our top quad is large enough to proceed and apply padding if not
        while (get<1>(steps.back()) > top_quad->size - 1) {
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
        vector<vector<bool>> result_bool(result.size(), vector<bool>(result[0].size()));
        for (int i = 0; i < result.size(); ++i) {
            for (int j = 0; j < result[i].size(); ++j) {
                result_bool[i][j] = result[i][j] == live_cell;
            }
        }
        return result_bool;
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

    vector<vector<bool>> expand_quad(quad* input) {
        // convert a quad to a grid of bools for inspection
        if (input == nullptr) {
            return {};
        } else if (input == dead_cell) {
            return {{false}};
        } else if (input == live_cell) {
            return {{true}};
        }
        auto ne_grid = expand_quad(input->ne);
        auto nw_grid = expand_quad(input->nw);
        auto sw_grid = expand_quad(input->sw);
        auto se_grid = expand_quad(input->se);
        int half_size = ne_grid.size();
        vector<vector<bool>> result(2 * half_size, vector<bool>(2 * half_size));
        for (int i = 0; i < half_size; ++i) {
            for (int j = 0; j < half_size; ++j) {
                result[i][j] = nw_grid[i][j];
                result[i][j + half_size] = ne_grid[i][j];
                result[i + half_size][j] = sw_grid[i][j];
                result[i + half_size][j + half_size] = se_grid[i][j];
            }
        }
        return result;
    }

    bool verify_hashmap(quad* node) {
        // verify correctness of the hashmap
        if (node == dead_cell || node == live_cell) {
            return true;
        }
        auto key = make_tuple(node->ne, node->nw, node->sw, node->se);
        if (hashmap.find(key) == hashmap.end() || hashmap[key] != node) {
            return false;
        }
        return verify_hashmap(node->ne) && verify_hashmap(node->nw) && verify_hashmap(node->sw) && verify_hashmap(node->se);
    }

};

int main() {

    // empty grid
    int initial_state_sidelength = 16;
    int middle = initial_state_sidelength / 2;
    vector<vector<bool>> initial_state(initial_state_sidelength, vector<bool>(initial_state_sidelength, false));

    // r-pentomino
    initial_state[middle - 1][middle    ] = true;
    initial_state[middle    ][middle - 1] = true;
    initial_state[middle    ][middle    ] = true;
    initial_state[middle + 1][middle    ] = true;
    initial_state[middle + 1][middle + 1] = true;

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

    // initialization
    hashlife my_hashlife(initial_state);

    // // print stuff
    // cout << "Number of items in hashmap: " << my_hashlife.hashmap.size() << endl;
    // if (my_hashlife.verify_hashmap(my_hashlife.top_quad)) {
    //     cout << "Hashmap validation passed." << endl;
    // } else {
    //     cout << "Hashmap validation failed." << endl;
    // }
    // hashlife::print_grid(my_hashlife.expand_quad(my_hashlife.top_quad));

    // // computing a result
    // quad* my_result = my_hashlife.get_or_compute_result(my_hashlife.top_quad);

    // // print stuff
    // cout << "Number of items in hashmap: " << my_hashlife.hashmap.size() << endl; 
    // if (my_hashlife.verify_hashmap(my_hashlife.top_quad)) {
    //     cout << "Hashmap validation passed." << endl;
    // } else {
    //     cout << "Hashmap validation failed." << endl;
    // }
    // hashlife::print_grid(my_hashlife.expand_quad(my_result));

    // render some viewports
    for (int i = 0; i < 2000; i++) {
        hashlife::print_grid(my_hashlife.show_viewport(i, -51, -28, 52, 29));
        usleep(50000);
    }

    return 0;
}