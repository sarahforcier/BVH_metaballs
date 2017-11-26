#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "utilities.h"
#include "scene.h"

#define NUM_BUCKETS 4

struct less_than_axis {
	less_than_axis(int x, float val) : axis(x), val(val) {}
	__host__ __device__ bool operator()(const Metaball & ball) const { return ball.translation[axis] < val; }

private:
	int axis;
	float val;
};

struct get_maxb {
	__host__ __device__
		glm::vec3 operator()(const Metaball& g) {
		return g.translation + glm::vec3(g.radius);
	}
};

struct get_max_vec {
	__host__ __device__
		glm::vec3 operator()(const glm::vec3 g1,
			const glm::vec3 g2) {
		glm::vec3 ret;
		g1.x > g2.x ? ret.x = g1.x : ret.x = g2.x;
		g1.y > g2.y ? ret.y = g1.y : ret.y = g2.y;
		g1.z > g2.z ? ret.z = g1.z : ret.z = g2.z;
		return ret;
	}
};

struct get_min_vec {
	__host__ __device__
		glm::vec3 operator()(const glm::vec3 g1,
			const glm::vec3 g2) {
		glm::vec3 ret;
		g1.x < g2.x ? ret.x = g1.x : ret.x = g2.x;
		g1.y < g2.y ? ret.y = g1.y : ret.y = g2.y;
		g1.z < g2.z ? ret.z = g1.z : ret.z = g2.z;
		return ret;
	}
};

struct get_minb {
	__host__ __device__
		glm::vec3 operator()(const Metaball& g) {
		return g.translation - glm::vec3(g.radius);
	}
};

struct get_sa {
	__host__ __device__
		float operator()(const Metaball& g) {
		float a = g.radius;
		return 4.0f * PI*powf(((powf(a*a, 1.6) + powf(a*a, 1.6) + powf(a*a, 1.6)) / 3.0f), 1.0f / 1.6f);
	}
};

struct sum_sa {
	__host__ __device__
		float operator()(const float g1,
			const float g2) {
		return g1 + g2;
	}
};

void constructBVHTree(int bvh_depth, Metaball * dev_metaballs, Scene * hst_scene) {
	int num_BVHnodes = (1 << (bvh_depth + 1)) - 1;
	int num_geoms = hst_scene->metaballs.size();
	/*int * BVHstart = new int[num_BVHnodes];
	int * BVHend = new int[num_BVHnodes];*/
	std::vector<int> BVHstart;
	std::vector<int> BVHend;
	BVHstart.resize(num_BVHnodes);
	BVHend.resize(num_BVHnodes);
	BVHstart[0] = 0;
	BVHend[0] = num_geoms - 1;
	//BVHNode * BVHnodes = new BVHNode[num_BVHnodes];
	std::vector<BVHNode> BVHnodes;
	BVHnodes.resize(num_BVHnodes);
	//printf("test: %i %i", BVHstart[0], BVHend[0]);
	glm::vec3 maxb = thrust::transform_reduce(thrust::device, dev_metaballs, dev_metaballs + num_geoms, get_maxb(), glm::vec3(-1.0f*INFINITY), get_max_vec());
	glm::vec3 minb = thrust::transform_reduce(thrust::device, dev_metaballs, dev_metaballs + num_geoms, get_minb(), glm::vec3(1.0f*INFINITY), get_min_vec());
	BVHnodes[0] = BVHNode();
	BVHnodes[0].maxB = maxb;
	BVHnodes[0].minB = minb;
	int curr_axis = 0;
	for (int depth = 0; depth <= bvh_depth - 1; depth++) { //depth <= BVH_DEPTH
		int num_nodes_at_depth = 1 << depth;
		int offset = (1 << depth) - 1;
		for (int n = 0; n < num_nodes_at_depth; n++) { // n < num_nodes_at_depth
			int curr_bvh = offset + n;
			int axis = curr_axis;
			float val;
			float repeat = true;
			int start_idx = BVHstart[curr_bvh];
			int end_idx = BVHend[curr_bvh];
			if (start_idx == end_idx) {
				continue;
			}
			if (repeat) {
				int buck_off = 0;
				float buck_width = (BVHnodes[curr_bvh].maxB[axis] - BVHnodes[curr_bvh].minB[axis]) / (float)NUM_BUCKETS;
				val = BVHnodes[curr_bvh].minB[axis];
				float buckCost[NUM_BUCKETS];
				float buckIdx[NUM_BUCKETS];
				float total_sa = thrust::transform_reduce(thrust::device, dev_metaballs + start_idx, dev_metaballs + end_idx + 1, get_sa(), 0.0f, sum_sa());
				for (int buck = 0; buck < NUM_BUCKETS; buck++) {
					val += buck_width;
					Metaball* end = thrust::partition(thrust::device, dev_metaballs + start_idx + buck_off, dev_metaballs + end_idx + 1, less_than_axis(axis, val));
					int num_left = end - (dev_metaballs + start_idx + buck_off);
					float left_sa = thrust::transform_reduce(thrust::device, dev_metaballs + start_idx + buck_off, dev_metaballs + start_idx + buck_off + num_left, get_sa(), 0.0f, sum_sa());
					buck_off += num_left;
					//printf("left_sa %f\n", left_sa);
					buckIdx[buck] = buck_off;
					buckCost[buck] = left_sa;
				}
				float total_cost = 0;
				float min_cost = INFINITY;
				int min_idx = -1;
				for (int buck = 0; buck < NUM_BUCKETS; buck++) {
					total_cost += buckCost[buck];
					float cost = fabsf((total_sa / 2.0f) - total_cost);
					if (cost < min_cost) {
						min_cost = cost;
						min_idx = buckIdx[buck];
					}
				}
				int next_offset = (1 << (depth + 1)) - 1;
				int l_idx = next_offset + (2 * n);
				BVHstart[l_idx] = start_idx;
				BVHend[l_idx] = start_idx + min_idx - 1;
				int r_idx = next_offset + (2 * n) + 1;
				BVHstart[r_idx] = start_idx + min_idx;
				BVHend[r_idx] = end_idx;
				glm::vec3 lmaxb = thrust::transform_reduce(thrust::device, dev_metaballs + start_idx, dev_metaballs + BVHend[l_idx] + 1, get_maxb(), glm::vec3(-1.0f*INFINITY), get_max_vec());
				glm::vec3 lminb = thrust::transform_reduce(thrust::device, dev_metaballs + start_idx, dev_metaballs + BVHend[l_idx] + 1, get_minb(), glm::vec3(1.0f*INFINITY), get_min_vec());

				glm::vec3 rmaxb = thrust::transform_reduce(thrust::device, dev_metaballs + BVHstart[r_idx], dev_metaballs + end_idx + 1, get_maxb(), glm::vec3(-1.0f*INFINITY), get_max_vec());
				glm::vec3 rminb = thrust::transform_reduce(thrust::device, dev_metaballs + BVHstart[r_idx], dev_metaballs + end_idx + 1, get_minb(), glm::vec3(1.0f*INFINITY), get_min_vec());
				BVHnodes[l_idx] = BVHNode();
				BVHnodes[r_idx] = BVHNode();
				BVHnodes[l_idx].id = l_idx;
				BVHnodes[r_idx].id = r_idx;
				BVHnodes[l_idx].startM = BVHstart[l_idx];
				BVHnodes[l_idx].endM = BVHend[l_idx];
				BVHnodes[r_idx].startM = BVHstart[r_idx];
				BVHnodes[r_idx].endM = BVHend[r_idx];
				BVHnodes[l_idx].maxB = lmaxb;
				BVHnodes[l_idx].minB = lminb;
				BVHnodes[r_idx].maxB = rmaxb;
				BVHnodes[r_idx].minB = rminb;
				BVHnodes[curr_bvh].isLeaf = false;
				BVHnodes[curr_bvh].child1id = l_idx;
				BVHnodes[curr_bvh].child2id = r_idx;

				repeat = false;
			}

		}
		curr_axis = (curr_axis + 1) % 3;
	}

	const int blockSize1d1 = 128;
	//num_geoms
								 //dim3 numblocksGeoms = (num_geoms + blockSize1d1 - 1) / blockSize1d1;
								 //printf("after BVH construct \n");
								 //kernCheckBounds << < numblocksGeoms, blockSize1d1 >> > (num_geoms, dev_geoms);

								 //const int blockSize1d = 128; //num_BVHnodes
								 //cudaMemcpy(dev_BVHnodes, BVHnodes.data(), num_BVHnodes * sizeof(BVHNode), cudaMemcpyHostToDevice);
								 //cudaMemcpy(dev_BVHstart, BVHstart.data(), num_BVHnodes * sizeof(int), cudaMemcpyHostToDevice);
								 //cudaMemcpy(dev_BVHend, BVHend.data(), num_BVHnodes * sizeof(int), cudaMemcpyHostToDevice);
								 //dim3 numblocksBVH = (num_BVHnodes + blockSize1d - 1) / blockSize1d;
								 //kernSetBVHTransform << < numblocksBVH, blockSize1d >> > (num_BVHnodes, dev_BVHnodes);

}