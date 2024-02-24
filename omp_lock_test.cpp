#include <chrono>
#include <ctime>
#include <iostream>
#include <vector>

#include "omp.h"

using namespace std;

int main() {

    for (int i = 0; i < 6; i++) {

        int n_threads = 1 << i;

        omp_lock_t* locks = new omp_lock_t[n_threads]();
        for (int j = 0; j < n_threads; j++) {
            omp_init_lock(&locks[j]);
        }

        auto start_time = chrono::high_resolution_clock::now();
        #pragma omp parallel num_threads(n_threads)
        {
            int thread_index = omp_get_thread_num();
            int blah = 2309451;
            for (int k = 0; k < 1000000; k++) {
                omp_set_lock(&locks[thread_index]);
                blah = blah ^ (blah * 3) + 7;
                omp_unset_lock(&locks[thread_index]);
            }
        }
        auto end_time = chrono::high_resolution_clock::now();
        cout << n_threads << " threads: " << chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count() << endl;

        for (int j = 0; j < n_threads; j++) {
            omp_destroy_lock(&locks[j]);
        }
        delete locks;
    }
}