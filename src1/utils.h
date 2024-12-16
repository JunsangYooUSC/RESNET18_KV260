/******************************************************************************
 * Filename: utils.h
 * Author: Junsang Yoo
 *
 * Description:
 * util functions
 *
 * Functions:
 * - 
 ******************************************************************************/

#ifndef UTILS_H
#define UTILS_H

// Include C++ headers
#include <iostream>
#include <string>
#include <cmath>
#include <random>
// Include project headers
#include "conv_config.h"

// gen_rand: generate random array
template<typename DTYPE, unsigned int LEN>
void gen_rand(DTYPE arr[LEN], float min_val, float max_val, unsigned int seed=1) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> dist(min_val, max_val);
	for (int i = 0; i < LEN; i++) {
		// Generate random float within range
		float rand_val = dist(rng);
		// Convert to fixed point
		arr[i] = static_cast<DTYPE>(rand_val);
	}
}


#endif
