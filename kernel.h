#ifndef KERNEL_H
#define KERNEL_H
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <cstring>
#include <cmath>
#include <limits>
#include "device_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <cuda_occupancy.h>
#include <unordered_map>

using namespace std;

#ifdef _MSC_VER
#include <ctime>
inline double get_time()
{
	return static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
}
#else
#include <sys/time.h>
inline double get_time()
{
	timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec + 1e-6 * tv.tv_usec;
}
#endif


//github https://github.com/seung-lab/connected-components-3d CPU CCL 3D
namespace cc3d {

	template <typename T>
	class DisjointSet {
	public:
		T *ids;
		size_t length;

		DisjointSet() {
			length = 65536;
			ids = new T[length]();
		}

		DisjointSet(size_t len) {
			length = len;
			ids = new T[length]();
		}

		DisjointSet(const DisjointSet &cpy) {
			length = cpy.length;
			ids = new T[length]();

			for (int i = 0; i < length; i++) {
				ids[i] = cpy.ids[i];
			}
		}

		~DisjointSet() {
			delete[]ids;
		}

		T root(T n) {
			T i = ids[n];
			while (i != ids[i]) {
				ids[i] = ids[ids[i]]; // path compression
				i = ids[i];
			}

			return i;
		}

		bool find(T p, T q) {
			return root(p) == root(q);
		}

		void add(T p) {
			if (p >= length) {
				printf("Connected Components Error: Label %d cannot be mapped to union-find array of length %lu.\n", p, length);
				throw "maximum length exception";
			}

			if (ids[p] == 0) {
				ids[p] = p;
			}
		}

		void unify(T p, T q) {
			if (p == q) {
				return;
			}

			T i = root(p);
			T j = root(q);

			if (i == 0) {
				add(p);
				i = p;
			}

			if (j == 0) {
				add(q);
				j = q;
			}

			ids[i] = j;
		}

		void print() {
			for (int i = 0; i < 15; i++) {
				printf("%d, ", ids[i]);
			}
			printf("\n");
		}

		// would be easy to write remove. 
		// Will be O(n).
	};

	// This is the original Wu et al decision tree but without
	// any copy operations, only union find. We can decompose the problem
	// into the z - 1 problem unified with the original 2D algorithm.
	// If literally none of the Z - 1 are filled, we can use a faster version
	// of this that uses copies.
	template <typename T, typename OUT = uint32_t>
	inline void unify2d(
		const int64_t loc, const T cur,
		const int64_t x, const int64_t y,
		const int64_t sx, const int64_t sy,
		const T* in_labels, const OUT* out_labels,
		DisjointSet<uint32_t> &equivalences
	) {

		if (y > 0 && cur == in_labels[loc - sx]) {
			equivalences.unify(out_labels[loc], out_labels[loc - sx]);
		}
		else if (x > 0 && cur == in_labels[loc - 1]) {
			equivalences.unify(out_labels[loc], out_labels[loc - 1]);

			if (x < sx - 1 && y > 0 && cur == in_labels[loc + 1 - sx]) {
				equivalences.unify(out_labels[loc], out_labels[loc + 1 - sx]);
			}
		}
		else if (x > 0 && y > 0 && cur == in_labels[loc - 1 - sx]) {
			equivalences.unify(out_labels[loc], out_labels[loc - 1 - sx]);

			if (x < sx - 1 && y > 0 && cur == in_labels[loc + 1 - sx]) {
				equivalences.unify(out_labels[loc], out_labels[loc + 1 - sx]);
			}
		}
		else if (x < sx - 1 && y > 0 && cur == in_labels[loc + 1 - sx]) {
			equivalences.unify(out_labels[loc], out_labels[loc + 1 - sx]);
		}
	}

	template <typename T, typename OUT = uint32_t>
	inline void unify2d_rt(
		const int64_t loc, const T cur,
		const int64_t x, const int64_t y,
		const int64_t sx, const int64_t sy,
		const T* in_labels, const OUT* out_labels,
		DisjointSet<uint32_t> &equivalences
	) {

		if (x < sx - 1 && y > 0 && cur == in_labels[loc + 1 - sx]) {
			equivalences.unify(out_labels[loc], out_labels[loc + 1 - sx]);
		}
	}

	template <typename T, typename OUT = uint32_t>
	inline void unify2d_lt(
		const int64_t loc, const T cur,
		const int64_t x, const int64_t y,
		const int64_t sx, const int64_t sy,
		const T* in_labels, const OUT* out_labels,
		DisjointSet<uint32_t> &equivalences
	) {

		if (x > 0 && cur == in_labels[loc - 1]) {
			equivalences.unify(out_labels[loc], out_labels[loc - 1]);
		}
		else if (x > 0 && y > 0 && cur == in_labels[loc - 1 - sx]) {
			equivalences.unify(out_labels[loc], out_labels[loc - 1 - sx]);
		}
	}

	// This is the second raster pass of the two pass algorithm family.
	// The input array (output_labels) has been assigned provisional 
	// labels and this resolves them into their final labels. We
	// modify this pass to also ensure that the output labels are
	// numbered from 1 sequentially.
	template <typename OUT = uint32_t>
	OUT* relabel(
		OUT* out_labels, const int64_t voxels,
		const int64_t num_labels, DisjointSet<uint32_t> &equivalences
	) {

		OUT label;
		OUT* renumber = new OUT[num_labels + 1]();
		OUT next_label = 1;

		// Raster Scan 2: Write final labels based on equivalences
		for (int64_t loc = 0; loc < voxels; loc++) {
			if (!out_labels[loc]) {
				continue;
			}

			label = equivalences.root(out_labels[loc]);

			if (renumber[label]) {
				out_labels[loc] = renumber[label];
			}
			else {
				renumber[label] = next_label;
				out_labels[loc] = next_label;
				next_label++;
			}
		}

		delete[] renumber;

		return out_labels;
	}

	template <typename T, typename OUT = uint32_t>
	OUT* connected_components3d_26(
		T* in_labels,
		const int64_t sx, const int64_t sy, const int64_t sz,
		int64_t max_labels, OUT *out_labels = NULL
	) {

		const int64_t sxy = sx * sy;
		const int64_t voxels = sxy * sz;

		max_labels = std::max(std::min(max_labels, voxels), static_cast<int64_t>(1L)); // can't allocate 0 arrays

		DisjointSet<uint32_t> equivalences(max_labels);

		if (out_labels == NULL) {
			out_labels = new OUT[voxels]();
		}

		/*
		Layout of forward pass mask (which faces backwards).
		N is the current location.
		z = -1     z = 0
		A B C      J K L   y = -1
		D E F      M N     y =  0
		G H I              y = +1
		-1 0 +1    -1 0   <-- x axis
		*/

		// Z - 1
		const int64_t A = -1 - sx - sxy;
		const int64_t B = -sx - sxy;
		const int64_t C = +1 - sx - sxy;
		const int64_t D = -1 - sxy;
		const int64_t E = -sxy;
		const int64_t F = +1 - sxy;
		const int64_t G = -1 + sx - sxy;
		const int64_t H = +sx - sxy;
		const int64_t I = +1 + sx - sxy;

		// Current Z
		const int64_t J = -1 - sx;
		const int64_t K = -sx;
		const int64_t L = +1 - sx;
		const int64_t M = -1;
		// N = 0;

		OUT next_label = 0;
		int64_t loc = 0;

		// Raster Scan 1: Set temporary labels and 
		// record equivalences in a disjoint set.
		for (int32_t z = 0; z < sz; z++) {
			for (int32_t y = 0; y < sy; y++) {
				for (int32_t x = 0; x < sx; x++) {
					loc = x + sx * (y + sy * z);
					const T cur = in_labels[loc];

					if (cur == 0) {
						continue;
					}

					if (z > 0 && cur == in_labels[loc + E]) {
						out_labels[loc] = out_labels[loc + E];
						// unify2d<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);
					}
					else if (z > 0 && y > 0 && cur == in_labels[loc + B]) {
						out_labels[loc] = out_labels[loc + B];
						// unify2d<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);

						if (y < sy - 1 && z > 0 && cur == in_labels[loc + H]) {
							equivalences.unify(out_labels[loc], out_labels[loc + H]);
						}
						else if (x > 0 && y < sy - 1 && z > 0 && cur == in_labels[loc + G]) {
							equivalences.unify(out_labels[loc], out_labels[loc + G]);

							if (x < sx - 1 && y < sy - 1 && z > 0 && cur == in_labels[loc + I]) {
								equivalences.unify(out_labels[loc], out_labels[loc + I]);
							}
						}
						else if (x < sx - 1 && y < sy - 1 && z > 0 && cur == in_labels[loc + I]) {
							equivalences.unify(out_labels[loc], out_labels[loc + I]);
						}
					}
					else if (x > 0 && z > 0 && cur == in_labels[loc + D]) {
						out_labels[loc] = out_labels[loc + D];
						unify2d_rt<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);

						if (x < sx - 1 && z > 0 && cur == in_labels[loc + F]) {
							equivalences.unify(out_labels[loc], out_labels[loc + F]);
						}
						else if (x < sx - 1 && y > 0 && z > 0 && cur == in_labels[loc + C]) {
							equivalences.unify(out_labels[loc], out_labels[loc + C]);

							if (x < sx - 1 && y < sy - 1 && z > 0 && cur == in_labels[loc + I]) {
								equivalences.unify(out_labels[loc], out_labels[loc + I]);
							}
						}
						else if (x < sx - 1 && y < sy - 1 && z > 0 && cur == in_labels[loc + I]) {
							equivalences.unify(out_labels[loc], out_labels[loc + I]);
						}
					}
					else if (y < sy - 1 && z > 0 && cur == in_labels[loc + H]) {
						out_labels[loc] = out_labels[loc + H];
						unify2d<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);

						if (x > 0 && y > 0 && z > 0 && cur == in_labels[loc + A]) {
							equivalences.unify(out_labels[loc], out_labels[loc + A]);
						}
						if (x < sx - 1 && y > 0 && z > 0 && cur == in_labels[loc + C]) {
							equivalences.unify(out_labels[loc], out_labels[loc + C]);
						}
					}
					else if (x < sx - 1 && z > 0 && cur == in_labels[loc + F]) {
						out_labels[loc] = out_labels[loc + F];
						unify2d_lt<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);

						if (x > 0 && y > 0 && z > 0 && cur == in_labels[loc + A]) {
							equivalences.unify(out_labels[loc], out_labels[loc + A]);
						}
						if (x > 0 && y < sy - 1 && z > 0 && cur == in_labels[loc + G]) {
							equivalences.unify(out_labels[loc], out_labels[loc + G]);
						}
					}
					else if (x > 0 && y > 0 && z > 0 && cur == in_labels[loc + A]) {
						out_labels[loc] = out_labels[loc + A];
						unify2d_rt<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);

						if (x < sx - 1 && y > 0 && z > 0 && cur == in_labels[loc + C]) {
							equivalences.unify(out_labels[loc], out_labels[loc + C]);
						}
						if (x > 0 && y < sy - 1 && z > 0 && cur == in_labels[loc + G]) {
							equivalences.unify(out_labels[loc], out_labels[loc + G]);
						}
						if (x < sx - 1 && y < sy - 1 && z > 0 && cur == in_labels[loc + I]) {
							equivalences.unify(out_labels[loc], out_labels[loc + I]);
						}
					}
					else if (x < sx - 1 && y > 0 && z > 0 && cur == in_labels[loc + C]) {
						out_labels[loc] = out_labels[loc + C];
						unify2d_lt<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);

						if (x > 0 && y < sy - 1 && z > 0 && cur == in_labels[loc + G]) {
							equivalences.unify(out_labels[loc], out_labels[loc + G]);
						}
						if (x < sx - 1 && y < sy - 1 && z > 0 && cur == in_labels[loc + I]) {
							equivalences.unify(out_labels[loc], out_labels[loc + I]);
						}
					}
					else if (x > 0 && y < sy - 1 && z > 0 && cur == in_labels[loc + G]) {
						out_labels[loc] = out_labels[loc + G];
						unify2d<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);

						if (x < sx - 1 && y < sy - 1 && z > 0 && cur == in_labels[loc + I]) {
							equivalences.unify(out_labels[loc], out_labels[loc + I]);
						}
					}
					else if (x < sx - 1 && y < sy - 1 && z > 0 && cur == in_labels[loc + I]) {
						out_labels[loc] = out_labels[loc + I];
						unify2d<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);
					}
					// It's the original 2D problem now
					else if (y > 0 && cur == in_labels[loc + K]) {
						out_labels[loc] = out_labels[loc + K];
					}
					else if (x > 0 && cur == in_labels[loc + M]) {
						out_labels[loc] = out_labels[loc + M];

						if (x < sx - 1 && y > 0 && cur == in_labels[loc + L]) {
							equivalences.unify(out_labels[loc], out_labels[loc + L]);
						}
					}
					else if (x > 0 && y > 0 && cur == in_labels[loc + J]) {
						out_labels[loc] = out_labels[loc + J];

						if (x < sx - 1 && y > 0 && cur == in_labels[loc + L]) {
							equivalences.unify(out_labels[loc], out_labels[loc + L]);
						}
					}
					else if (x < sx - 1 && y > 0 && cur == in_labels[loc + L]) {
						out_labels[loc] = out_labels[loc + L];
					}
					else {
						next_label++;
						out_labels[loc] = next_label;
						equivalences.add(out_labels[loc]);
					}
				}
			}
		}

		return relabel<OUT>(out_labels, voxels, next_label, equivalences);
	}

	template <typename T, typename OUT = uint32_t>
	OUT* connected_components3d_18(
		T* in_labels,
		const int64_t sx, const int64_t sy, const int64_t sz,
		int64_t max_labels, OUT *out_labels = NULL
	) {

		const int64_t sxy = sx * sy;
		const int64_t voxels = sxy * sz;

		max_labels = std::max(std::min(max_labels, voxels), static_cast<int64_t>(1L)); // can't allocate 0 arrays

		DisjointSet<uint32_t> equivalences(max_labels);

		if (out_labels == NULL) {
			out_labels = new OUT[voxels]();
		}

		/*
		Layout of forward pass mask (which faces backwards).
		N is the current location.
		z = -1     z = 0
		A B C      J K L   y = -1
		D E F      M N     y =  0
		G H I              y = +1
		-1 0 +1    -1 0   <-- x axis
		*/

		// Z - 1
		const int64_t B = -sx - sxy;
		const int64_t D = -1 - sxy;
		const int64_t E = -sxy;
		const int64_t F = +1 - sxy;
		const int64_t H = +sx - sxy;

		// Current Z
		const int64_t J = -1 - sx;
		const int64_t K = -sx;
		const int64_t L = +1 - sx;
		const int64_t M = -1;
		// N = 0;

		OUT next_label = 0;
		int64_t loc = 0;

		// Raster Scan 1: Set temporary labels and 
		// record equivalences in a disjoint set.
		for (int64_t z = 0; z < sz; z++) {
			for (int64_t y = 0; y < sy; y++) {
				for (int64_t x = 0; x < sx; x++) {
					loc = x + sx * (y + sy * z);
					const T cur = in_labels[loc];

					if (cur == 0) {
						continue;
					}

					if (z > 0 && cur == in_labels[loc + E]) {
						out_labels[loc] = out_labels[loc + E];

						if (x > 0 && y > 0 && cur == in_labels[loc + J]) {
							equivalences.unify(out_labels[loc], out_labels[loc + J]);
						}
						if (x < sx - 1 && y > 0 && cur == in_labels[loc + L]) {
							equivalences.unify(out_labels[loc], out_labels[loc + L]);
						}
					}
					else if (y > 0 && z > 0 && cur == in_labels[loc + B]) {
						out_labels[loc] = out_labels[loc + B];

						if (x > 0 && cur == in_labels[loc + M]) {
							equivalences.unify(out_labels[loc], out_labels[loc + M]);
						}
						if (y < sy - 1 && z > 0 && cur == in_labels[loc + H]) {
							equivalences.unify(out_labels[loc], out_labels[loc + H]);
						}
					}
					else if (x > 0 && z > 0 && cur == in_labels[loc + D]) {
						out_labels[loc] = out_labels[loc + D];

						if (x < sx - 1 && y > 0 && cur == in_labels[loc + L]) {
							equivalences.unify(out_labels[loc], out_labels[loc + L]);
						}
						else {
							if (y > 0 && cur == in_labels[loc + K]) {
								equivalences.unify(out_labels[loc], out_labels[loc + K]);
							}
							if (x < sx - 1 && z > 0 && cur == in_labels[loc + F]) {
								equivalences.unify(out_labels[loc], out_labels[loc + F]);
							}
						}
					}
					else if (x < sx - 1 && z > 0 && cur == in_labels[loc + F]) {
						out_labels[loc] = out_labels[loc + F];

						if (x > 0 && y > 0 && cur == in_labels[loc + J]) {
							equivalences.unify(out_labels[loc], out_labels[loc + J]);
						}
						else {
							if (x > 0 && cur == in_labels[loc + M]) {
								equivalences.unify(out_labels[loc], out_labels[loc + M]);
							}
							if (y > 0 && cur == in_labels[loc + K]) {
								equivalences.unify(out_labels[loc], out_labels[loc + K]);
							}
						}
					}
					else if (y < sy - 1 && z > 0 && cur == in_labels[loc + H]) {
						out_labels[loc] = out_labels[loc + H];
						unify2d<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);
					}
					// It's the original 2D problem now
					else if (y > 0 && cur == in_labels[loc + K]) {
						out_labels[loc] = out_labels[loc + K];
					}
					else if (x > 0 && cur == in_labels[loc + M]) {
						out_labels[loc] = out_labels[loc + M];

						if (x < sx - 1 && y > 0 && cur == in_labels[loc + L]) {
							equivalences.unify(out_labels[loc], out_labels[loc + L]);
						}
					}
					else if (x > 0 && y > 0 && cur == in_labels[loc + J]) {
						out_labels[loc] = out_labels[loc + J];

						if (x < sx - 1 && y > 0 && cur == in_labels[loc + L]) {
							equivalences.unify(out_labels[loc], out_labels[loc + L]);
						}
					}
					else if (x < sx - 1 && y > 0 && cur == in_labels[loc + L]) {
						out_labels[loc] = out_labels[loc + L];
					}
					else {
						next_label++;
						out_labels[loc] = next_label;
						equivalences.add(out_labels[loc]);
					}
				}
			}
		}

		return relabel<OUT>(out_labels, voxels, next_label, equivalences);
	}

	template <typename T, typename OUT = uint32_t>
	OUT* connected_components3d_6(
		T* in_labels,
		const int64_t sx, const int64_t sy, const int64_t sz,
		int64_t max_labels, OUT *out_labels = NULL
	) {

		const int64_t sxy = sx * sy;
		const int64_t voxels = sxy * sz;

		max_labels = std::max(std::min(max_labels, voxels), static_cast<int64_t>(1L)); // can't allocate 0 arrays

		DisjointSet<uint32_t> equivalences(max_labels);

		if (out_labels == NULL) {
			out_labels = new OUT[voxels]();
		}

		/*
		Layout of forward pass mask (which faces backwards).
		N is the current location.
		z = -1     z = 0
		A B C      J K L   y = -1
		D E F      M N     y =  0
		G H I              y = +1
		-1 0 +1    -1 0   <-- x axis
		*/

		// Z - 1
		const int64_t E = -sxy;

		// Current Z
		const int64_t K = -sx;
		const int64_t M = -1;
		// N = 0;

		int64_t loc = 0;
		OUT next_label = 0;

		// Raster Scan 1: Set temporary labels and 
		// record equivalences in a disjoint set.

		for (int64_t z = 0; z < sz; z++) {
			for (int64_t y = 0; y < sy; y++) {
				for (int64_t x = 0; x < sx; x++) {
					loc = x + sx * (y + sy * z);

					const T cur = in_labels[loc];

					if (cur == 0) {
						continue;
					}

					if (x > 0 && cur == in_labels[loc + M]) {
						out_labels[loc] = out_labels[loc + M];

						if (y > 0 && cur == in_labels[loc + K]) {
							equivalences.unify(out_labels[loc], out_labels[loc + K]);
						}
						if (z > 0 && cur == in_labels[loc + E]) {
							equivalences.unify(out_labels[loc], out_labels[loc + E]);
						}
					}
					else if (y > 0 && cur == in_labels[loc + K]) {
						out_labels[loc] = out_labels[loc + K];

						if (z > 0 && cur == in_labels[loc + E]) {
							equivalences.unify(out_labels[loc], out_labels[loc + E]);
						}
					}
					else if (z > 0 && cur == in_labels[loc + E]) {
						out_labels[loc] = out_labels[loc + E];
					}
					else {
						next_label++;
						out_labels[loc] = next_label;
						equivalences.add(out_labels[loc]);
					}
				}
			}
		}

		return relabel<OUT>(out_labels, voxels, next_label, equivalences);
	}


	template <typename T, typename OUT = uint32_t>
	OUT* connected_components3d(
		T* in_labels,
		const int64_t sx, const int64_t sy, const int64_t sz,
		int64_t max_labels, const int64_t connectivity,
		OUT *out_labels = NULL
	) {

		if (connectivity == 26) {
			return connected_components3d_26<T, OUT>(
				in_labels, sx, sy, sz,
				max_labels, out_labels
				);
		}
		else if (connectivity == 18) {
			return connected_components3d_18<T, OUT>(
				in_labels, sx, sy, sz,
				max_labels, out_labels
				);
		}
		else if (connectivity == 6) {
			return connected_components3d_6<T, OUT>(
				in_labels, sx, sy, sz,
				max_labels, out_labels
				);
		}
		else {
			throw std::runtime_error("Only 6, 18, and 26 3D connectivities are supported.");
		}
	}

	template <typename T, typename OUT = uint32_t>
	OUT* connected_components3d(
		T* in_labels,
		const int64_t sx, const int64_t sy, const int64_t sz,
		const int64_t connectivity = 26
	) {
		const int64_t voxels = sx * sy * sz;
		return connected_components3d<T, OUT>(in_labels, sx, sy, sz, voxels, connectivity);
	}

};

class CCL {
public:
	CCL() = default;


	
public:
	
	std::shared_ptr<cv::Mat> binary_mat(const char *file_path,int *slice_num);

	unordered_map<int, int> GPU_conn_26(std::shared_ptr<cv::Mat> b_m, 
					    const int slice_num, 
					    std::unordered_map<int, int> &conn_map);

	void CPU_conn_26(std::shared_ptr<cv::Mat> mat_data, vector<vector<cv::Vec3i>> &result);

	void conn_check(unordered_map<int, int> &gpu_conn, unordered_map<int, int> &cpu_conn);

	void git_3d_conn(uchar* label_data, unordered_map<int, int> &cpu_conn, int size[3]);

private:

	int * cuda_ccl(unsigned char * image,
		int degree_of_connectivity,
		int threshold,
		const int WIDTH,
		const int HEIGHT,
		const int SLICE);

	std::shared_ptr<cv::Mat> Load_itk_series(const char *path);
	
	//void  load_Series_dicom(const char *file_path);
	
	std::shared_ptr<cv::Mat> binary_matrix(
			const cv::Mat &matr,
			signed int isolevel);
};













#endif // !KERNEL_H

