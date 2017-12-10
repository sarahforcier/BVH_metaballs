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

#define BVH 1
#define MAX_BVH_DEPTH 12
#define MAXSPLITNODES 150
#define NUM_BUCKETS 4
#define THRESHOLD 0.2
#define MAXSECANTSTEPS 100
#define MAXBVHSTACK 50
#define MAXBVHINTERSECTS 100

__device__ glm::vec3 indexToColor(int i, int max) {
	int comp_range = max / 3;
	int z = i % (comp_range);
	int y = (i / z ) % (comp_range);
	int x = i / (y*z);
	return glm::vec3((float)x / (float)comp_range, (float)y / (float)comp_range, (float)z / (float)comp_range);
}

__device__ bool metaBallsOverlap(Metaball & a, Metaball & b) {
	return glm::distance(a.translation, b.translation) < (a.radius + b.radius);
}

//METABALL FUNCTIONS
__device__ float calculateDensity(Metaball * metaballs, int first_node_idx, LLNode * nodeBuffer, glm::vec3 x) {
	float density = 0.f;
	int node_idx = first_node_idx;
	LLNode * node;
	Metaball * ball;
	while (node_idx > 0) {
		node = &nodeBuffer[node_idx];
		//ball = &metaballs[node->metaballid];
		ball = (node->metaball);
		float dist = glm::distance(x, ball->translation);
		if (dist < ball->radius) {
			float val = 1.0f - dist * dist / (ball->radius * ball->radius);
			density += val * val;
		}
		node_idx = node->next;
	}
	density -= THRESHOLD;
	return density;
}

__device__ glm::vec3 calculateNormals(int count, Metaball * metaballs, glm::vec3 x) {
	glm::vec3 normal(0.f);
	for (int j = 0; j < count; ++j) {
		glm::vec3 diff = x - metaballs[j].translation;
		normal += 2.f * diff / (glm::length2(diff) * glm::length2(diff));
	}
	return glm::normalize(normal);
}

__device__ glm::vec3 calculateColor(int count, Metaball * metaballs, glm::vec3 x) {
	glm::vec3 normal(0.f);
	for (int j = 0; j < count; ++j) {
		Metaball ball = metaballs[j];
		glm::vec3 diff = x - ball.translation;
		normal += ball.velocity / (glm::length2(diff) * glm::length2(diff));
	}
	return glm::abs(glm::normalize(normal));
}

__device__ glm::mat4 dev_buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) {
	float pi = 3.14159f;
	glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
	glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotation.x * (float)pi / 180, glm::vec3(1, 0, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y * (float)pi / 180, glm::vec3(0, 1, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z * (float)pi / 180, glm::vec3(0, 0, 1));
	glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
	return translationMat * rotationMat * scaleMat;
}

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

struct set_bvh_id {
	set_bvh_id(int id) : id(id) {}
	__host__ __device__ void operator()(Metaball & ball) const { ball.bvh_id = id; }

private:
	int id;
};

#if BVH

void constructBVHTreeBasic(int bvh_depth, Metaball * dev_metaballs, BVHNode * dev_BVHNodes, Scene * hst_scene) {
	int num_BVHnodes = (1 << (bvh_depth + 1)) - 1;
	int num_geoms = hst_scene->metaballs.size();
	//std::vector<int> BVHstart;
	//std::vector<int> BVHend;
	//BVHstart.resize(num_BVHnodes);
	//BVHend.resize(num_BVHnodes);
	//BVHstart[0] = 0;
	//BVHend[0] = num_geoms - 1;
	std::vector<BVHNode> BVHnodes;
	BVHnodes.resize(num_BVHnodes); 
	BVHnodes[0] = BVHNode();
	BVHnodes[0].startM = 0;
	BVHnodes[0].endM = num_geoms - 1;
	glm::vec3 maxb = thrust::transform_reduce(thrust::device, dev_metaballs, dev_metaballs + num_geoms, get_maxb(), glm::vec3(-1.0f*INFINITY), get_max_vec());
	glm::vec3 minb = thrust::transform_reduce(thrust::device, dev_metaballs, dev_metaballs + num_geoms, get_minb(), glm::vec3(1.0f*INFINITY), get_min_vec());
	
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
			int start_idx = BVHnodes[curr_bvh].startM;
			int end_idx = BVHnodes[curr_bvh].endM;
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
				//BVHstart[l_idx] = start_idx;
				//BVHend[l_idx] = start_idx + min_idx - 1;
				int r_idx = next_offset + (2 * n) + 1;
				//BVHstart[r_idx] = start_idx + min_idx;
				//BVHend[r_idx] = end_idx;

				BVHnodes[l_idx] = BVHNode();
				BVHnodes[r_idx] = BVHNode();
				BVHnodes[l_idx].id = l_idx;
				BVHnodes[r_idx].id = r_idx;
				BVHnodes[l_idx].startM = start_idx;
				BVHnodes[l_idx].endM = start_idx + min_idx - 1;
				BVHnodes[r_idx].startM = start_idx + min_idx;
				BVHnodes[r_idx].endM = end_idx;
				glm::vec3 lmaxb = thrust::transform_reduce(thrust::device, dev_metaballs + start_idx, dev_metaballs + BVHnodes[l_idx].endM + 1, get_maxb(), glm::vec3(-1.0f*INFINITY), get_max_vec());
				glm::vec3 lminb = thrust::transform_reduce(thrust::device, dev_metaballs + start_idx, dev_metaballs + BVHnodes[l_idx].endM + 1, get_minb(), glm::vec3(1.0f*INFINITY), get_min_vec());

				glm::vec3 rmaxb = thrust::transform_reduce(thrust::device, dev_metaballs + BVHnodes[r_idx].startM, dev_metaballs + end_idx + 1, get_maxb(), glm::vec3(-1.0f*INFINITY), get_max_vec());
				glm::vec3 rminb = thrust::transform_reduce(thrust::device, dev_metaballs + BVHnodes[r_idx].startM, dev_metaballs + end_idx + 1, get_minb(), glm::vec3(1.0f*INFINITY), get_min_vec());
				
				//BVHnodes[l_idx].startM = BVHstart[l_idx];
				//BVHnodes[l_idx].endM = BVHend[l_idx];
				//BVHnodes[r_idx].startM = BVHstart[r_idx];
				//BVHnodes[r_idx].endM = BVHend[r_idx];
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
	cudaMemcpy(dev_BVHNodes, BVHnodes.data(), num_BVHnodes * sizeof(BVHNode), cudaMemcpyHostToDevice);
								 //cudaMemcpy(dev_BVHstart, BVHstart.data(), num_BVHnodes * sizeof(int), cudaMemcpyHostToDevice);
								 //cudaMemcpy(dev_BVHend, BVHend.data(), num_BVHnodes * sizeof(int), cudaMemcpyHostToDevice);
								 //dim3 numblocksBVH = (num_BVHnodes + blockSize1d - 1) / blockSize1d;
								 //kernSetBVHTransform << < numblocksBVH, blockSize1d >> > (num_BVHnodes, dev_BVHnodes);

}

bool metaballIntersectsBBox(Metaball ball, BBox bbox) {
	float R = ball.radius;
	glm::vec3 S = ball.translation;
	glm::vec3 C1 = bbox.minB;
	glm::vec3 C2 = bbox.maxB;
	float dist_squared = R * R;
	/* assume C1 and C2 are element-wise sorted, if not, do that now */
	if (S.x < C1.x) dist_squared -= (S.x - C1.x) * (S.x - C1.x);
	else if (S.x > C2.x) dist_squared -= (S.x - C2.x) * (S.x - C2.x);
	if (S.y < C1.y) dist_squared -= (S.y - C1.y) * (S.y - C1.y);
	else if (S.y > C2.y) dist_squared -= (S.y - C2.y) * (S.y - C2.y);
	if (S.z < C1.z) dist_squared -= (S.z - C1.z) * (S.z - C1.z);
	else if (S.z > C2.z) dist_squared -= (S.z - C2.z) * (S.z - C2.z);
	return dist_squared > 0.f;
}

void BBoxSplitbyPlane(BBox bbox, BBox & lBBox, BBox & rBBox, int axis, float val) {
	lBBox = bbox;
	rBBox = bbox;
	lBBox.maxB[axis] = val;
	rBBox.minB[axis] = val;
}

void BBoxMetaballUnion(BBox & bbox, const Metaball ball) {

	glm::vec3 umax;
	glm::vec3 umin;
	glm::vec3 g1 = bbox.maxB;
	glm::vec3 g2 = ball.translation + glm::vec3(ball.radius);
	g1.x > g2.x ? bbox.maxB.x = g1.x : bbox.maxB.x = g2.x;
	g1.y > g2.y ? bbox.maxB.y = g1.y : bbox.maxB.y = g2.y;
	g1.z > g2.z ? bbox.maxB.z = g1.z : bbox.maxB.z = g2.z;

	g1 = bbox.minB;
	g2 = ball.translation - glm::vec3(ball.radius);
	g1.x < g2.x ? bbox.minB.x = g1.x : bbox.minB.x = g2.x;
	g1.y < g2.y ? bbox.minB.y = g1.y : bbox.minB.y = g2.y;
	g1.z < g2.z ? bbox.minB.z = g1.z : bbox.minB.z = g2.z;

}

BBox BBoxIntersection(BBox b1, BBox b2) {
	glm::vec3 retmax;
	glm::vec3 retmin;
	glm::vec3 g1 = b1.minB;
	glm::vec3 g2 = b2.minB;
	g1.x > g2.x ? retmin.x = g1.x : retmin.x = g2.x;
	g1.y > g2.y ? retmin.y = g1.y : retmin.y = g2.y;
	g1.z > g2.z ? retmin.z = g1.z : retmin.z = g2.z;

	g1 = b1.maxB;
	g2 = b2.maxB;
	g1.x < g2.x ? retmax.x = g1.x : retmax.x = g2.x;
	g1.y < g2.y ? retmax.y = g1.y : retmax.y = g2.y;
	g1.z < g2.z ? retmax.z = g1.z : retmax.z = g2.z;
	BBox retBBox;
	retBBox.maxB = retmax;
	retBBox.minB = retmin;
	return retBBox;
}

void printBBoxInfo(BBox bbox, char * str) {
	glm::vec3 maxb = bbox.maxB;
	glm::vec3 minb = bbox.minB;
	printf(str);
	printf("max %f %f %f\n", maxb[0], maxb[1], maxb[2]);
	printf("min %f %f %f\n", minb[0], minb[1], minb[2]);
}

BBox calculateMetaballBBox(Metaball* metaballs, int num_metaballs) {
	glm::vec3 maxB = thrust::transform_reduce(thrust::host, metaballs, metaballs + num_metaballs, get_maxb(), glm::vec3(-1.0f*INFINITY), get_max_vec());
	glm::vec3 minB = thrust::transform_reduce(thrust::host, metaballs, metaballs + num_metaballs, get_minb(), glm::vec3(1.0f*INFINITY), get_min_vec());
	BBox bbox;
	bbox.maxB = maxB;
	bbox.minB = minB;
	return bbox;
}

void buildBVHNode(int depth, 
				int max_depth, 
				int axis,
				int bvh_idx, 
				std::vector<BVHNode> & BVHNodes, 
				BBox bbox,
				Metaball * metaballs,
				int start_idx,
				int num_metaballs,
				int max_num_split_balls,
				std::vector<Metaball> & splitBalls,
				std::vector<Metaball> & leafSplitBalls) {

	// Find the best splitting plane s
	// for all metaballs in metaballs to metaballs + num_metaballs
	// distribute to left or right child node

	//printf("bvh idx %i, num balls %i \n", bvh_idx, num_metaballs);

	if (depth == max_depth || num_metaballs <= 1) {
		if (depth != max_depth) {
			int breakpoint;
		}
		BVHNodes[bvh_idx].isLeaf = true;
		//TAKE CARE OF THROWING SPLIT NODES
		int depth_offset = (1 << (depth)) - 1;
		//depth++;
		int row_idx = bvh_idx - depth_offset;
		int max_depth_diff = max_depth - depth;
		int idx_mult = 1 << max_depth_diff;
		int leaf_offset = row_idx * idx_mult * max_num_split_balls;
		//leaf_offset = row_idx * max_num_split_balls;
		int num_splitballs = (splitBalls.size() < MAXSPLITNODES) ? splitBalls.size() : MAXSPLITNODES;
		for (int i = 0; i < num_splitballs; i++) {
			leafSplitBalls[leaf_offset + i] = splitBalls[i];
		}
		BVHNodes[bvh_idx].startS = leaf_offset;
		BVHNodes[bvh_idx].endS = leaf_offset + splitBalls.size() - 1;
		return;
	}

	// TEMPORARY TEST VAL
	float val = bbox.minB[axis] + (bbox.maxB[axis] - bbox.minB[axis]) / 2.0f ;

	Metaball* end = thrust::partition(thrust::host, metaballs, metaballs + num_metaballs, less_than_axis(axis, val));
	//printf("partition idx %i %f %i\n", axis, val, end - metaballs);

	// assume "left" child is < than "val" in "axis"

	int rmetaballs_start = end - metaballs;
	if (rmetaballs_start == 0 || rmetaballs_start == num_metaballs + 1) {
		BVHNodes[bvh_idx].isLeaf = true;
		//TAKE CARE OF THROWING SPLIT NODES
		int depth_offset = (1 << (depth)) - 1;
		//depth++;
		int row_idx = bvh_idx - depth_offset;
		int max_depth_diff = max_depth - depth;
		int idx_mult = 1 << max_depth_diff;
		int leaf_offset = row_idx * idx_mult * max_num_split_balls;
		//leaf_offset = row_idx * max_num_split_balls;
		int num_splitballs = (splitBalls.size() < MAXSPLITNODES) ? splitBalls.size() : MAXSPLITNODES;
		for (int i = 0; i < num_splitballs; i++) {
			leafSplitBalls[leaf_offset + i] = splitBalls[i];
		}
		BVHNodes[bvh_idx].startS = leaf_offset;
		BVHNodes[bvh_idx].endS = leaf_offset + splitBalls.size() - 1;
		return;
	}

	BBox lMetaballBbox = calculateMetaballBBox(metaballs, end - metaballs);
	BBox rMetaballBbox = calculateMetaballBBox(end, num_metaballs - (rmetaballs_start));

	BBox lChildBBox;
	BBox rChildBBox;
	BBoxSplitbyPlane(bbox, lChildBBox, rChildBBox, axis, val);
	std::vector<Metaball> lSplitBalls;
	std::vector<Metaball> rSplitBalls;

	// LEFT SIDE
	// for each ball in splitBalls and rChildMetaballs
	for (int i = rmetaballs_start; i < num_metaballs; i++) {
		// if ball overlaps lChildBbox
		if (metaballIntersectsBBox(metaballs[i], lChildBBox)) {
			// add ball to lSplitBalls
			lSplitBalls.push_back(metaballs[i]);
			// left mBallBbox U bbox(ball)
			BBoxMetaballUnion(lMetaballBbox, metaballs[i]);
		}
	}
	for (int i = 0; i < splitBalls.size(); i++) {
		if (metaballIntersectsBBox(splitBalls[i], lChildBBox)) {
			lSplitBalls.push_back(splitBalls[i]);
			BBoxMetaballUnion(lMetaballBbox, splitBalls[i]);
		}
	}
	// lChildBbox <- intersection of lChildBbox and lBallBbox
	lChildBBox = BBoxIntersection(lMetaballBbox, lChildBBox);

	// DO SAME FOR RIGHT SIDE
	for (int i = 0; i < rmetaballs_start; i++) {
		// if ball overlaps lChildBbox
		if (metaballIntersectsBBox(metaballs[i], rChildBBox)) {
			// add ball to lSplitBalls
			rSplitBalls.push_back(metaballs[i]);
			// left mBallBbox U bbox(ball)
			BBoxMetaballUnion(rMetaballBbox, metaballs[i]);
		}
	}
	for (int i = 0; i < splitBalls.size(); i++) {
		if (metaballIntersectsBBox(splitBalls[i], rChildBBox)) {
			rSplitBalls.push_back(splitBalls[i]);
			BBoxMetaballUnion(rMetaballBbox, splitBalls[i]);
		}
	}

	// rChildBbox <- intersection of rChildBbox and rBallBbox
	rChildBBox = BBoxIntersection(rMetaballBbox, rChildBBox);

	int depth_offset = (1 << (depth)) - 1;
	depth++;
	int row_idx = bvh_idx - depth_offset;
	int child_depth_offset = (1 << (depth)) - 1;

	BVHNode lChild;
	lChild.id_col = BVHNodes[bvh_idx].id_col + glm::vec3((1.0f / (float)MAX_BVH_DEPTH), 0.f, 0.f);
	BVHNode rChild;
	rChild.id_col = BVHNodes[bvh_idx].id_col + glm::vec3(0.0f, 1.0f / (float)MAX_BVH_DEPTH, 0.f);
	lChild.id = child_depth_offset + (2 * row_idx);
	rChild.id = child_depth_offset + (2 * row_idx) + 1;
	lChild.Bbox = lChildBBox;
	rChild.Bbox = rChildBBox;
	lChild.startM = start_idx;
	lChild.endM = start_idx + rmetaballs_start - 1;
	rChild.startM = start_idx + rmetaballs_start;
	rChild.endM = start_idx + num_metaballs - 1;
	BVHNodes[lChild.id] = lChild;
	BVHNodes[rChild.id] = rChild;
	BVHNodes[bvh_idx].isLeaf = false;
	BVHNodes[bvh_idx].child1id = lChild.id;
	BVHNodes[bvh_idx].child2id = rChild.id;
	axis = (axis + 1) % 3;

	//recurse left side
	buildBVHNode(depth, max_depth, axis, lChild.id, BVHNodes, lChildBBox, 
		metaballs, start_idx, rmetaballs_start, max_num_split_balls, lSplitBalls, leafSplitBalls);
	//recurse right side
	buildBVHNode(depth, max_depth, axis, rChild.id, BVHNodes, rChildBBox, 
		end, start_idx + rmetaballs_start, num_metaballs - (rmetaballs_start), max_num_split_balls, rSplitBalls, leafSplitBalls);
}

void constructBVHTree(int bvh_depth, Metaball * metaballs, std::vector<BVHNode> & BVHnodes, std::vector<Metaball> & allSplitBalls, int num_split_balls, Scene * hst_scene) {
	int num_BVHnodes = (1 << (bvh_depth + 1)) - 1;
	int num_geoms = hst_scene->metaballs.size();

	std::vector<Metaball> splitBalls;

	BVHnodes[0] = BVHNode();
	BVHnodes[0].startM = 0;
	BVHnodes[0].endM = num_geoms;
	BVHnodes[0].id_col = glm::vec3(0.f);

	BBox bbox = calculateMetaballBBox(metaballs, num_geoms);
	buildBVHNode(0, bvh_depth, 0, 0, BVHnodes, bbox , metaballs, 0, num_geoms, num_split_balls, splitBalls,allSplitBalls);
	//printf("partition test %i %i \n", num_geoms, end - metaballs);

	
}

__global__ void kernSetBVHTransform(int n, BVHNode * BVHnodes) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("%i \n", idx);
	if (idx >= n) {
		return;
	}
	BVHNode & node = BVHnodes[idx];
	//printf("node: %i %f %f\n", idx, node.minB.x ,node.maxB.x);
	glm::vec3 translate = (node.Bbox.maxB + node.Bbox.minB) / 2.0f;
	glm::vec3 scale = (node.Bbox.maxB - node.Bbox.minB);
	node.transform = dev_buildTransformationMatrix(
		translate, glm::vec3(0.0f), scale);
	node.inverseTransform = glm::inverse(node.transform);
	node.invTranspose = glm::inverseTranspose(node.transform);
	/*printf("idx: %i, trans: %f %f %f, scale %f %f %f, isleaf %i\n", idx, translate.x, translate.y, translate.z,
	scale.x, scale.y, scale.z, node.isLeaf);*/
}

__host__ __device__ float BVHIntersectionTest(BVHNode box, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
	Ray q;
	q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
	q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

	float tmin = -1e38f;
	float tmax = 1e38f;
	glm::vec3 tmin_n;
	glm::vec3 tmax_n;
	for (int xyz = 0; xyz < 3; ++xyz) {
		float qdxyz = q.direction[xyz];
		/*if (glm::abs(qdxyz) > 0.00001f)*/ {
			float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
			float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
			float ta = glm::min(t1, t2);
			float tb = glm::max(t1, t2);
			glm::vec3 n;
			n[xyz] = t2 < t1 ? +1 : -1;
			if (ta > 0 && ta > tmin) {
				tmin = ta;
				tmin_n = n;
			}
			if (tb < tmax) {
				tmax = tb;
				tmax_n = n;
			}
		}
	}

	if (tmax >= tmin && tmax > 0) {
		outside = true;
		if (tmin <= 0) {
			tmin = tmax;
			tmin_n = tmax_n;
			outside = false;
		}
		intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
		normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f)));
		return glm::length(r.origin - intersectionPoint);
	}
	return -1;
}


__global__ void computeLinkedListBVH(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, ShadeableIntersection * intersections
	, BVHNode * BVHNodes
	, int num_bvh
	, Geom* geoms
	, Metaball * metaballs
	, Metaball * splitmetaballs
	, int ball_size
	, int iter
	, int * LLcounter,
	int * headPtrBuffer,
	LLNode * nodeBuffer
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		glm::vec3 geom_ranges[MAXBVHINTERSECTS];
		glm::vec3 split_ranges[MAXBVHINTERSECTS];
		int geom_idx = 0;
		PathSegment pathSegment = pathSegments[path_index];
		BVHNode* stack[MAXBVHSTACK];
		BVHNode* * stackPtr = stack;
		*stackPtr++ = NULL; // push

		bool loutside = true;
		bool routside = true;

		glm::vec3 ltmp_intersect;
		glm::vec3 ltmp_normal;
		glm::vec3 rtmp_intersect;
		glm::vec3 rtmp_normal;

		// naive parse through global geoms

		BVHNode * curr_bvh = &BVHNodes[0];
		bool loop = true;

		while (curr_bvh != NULL) {
			BVHNode *lchild = &BVHNodes[curr_bvh->child1id];
			BVHNode *rchild = &BVHNodes[curr_bvh->child2id];
			float t1 = BVHIntersectionTest(*lchild, pathSegment.ray, ltmp_intersect, ltmp_normal, loutside);
			float t2 = BVHIntersectionTest(*rchild, pathSegment.ray, rtmp_intersect, rtmp_normal, routside);
			/*BVHNode * child1 = lchild;
			BVHNode * child2 = rchild;*/
			BVHNode * child1 = (t2 > t1) ? lchild : rchild;
			BVHNode * child2 = (t2 > t1) ? rchild : lchild;
			if (t2 <= t1) {
				float temp = t1;
				t1 = t2;
				t2 = temp;
			}

			//printf("left isleaf %i, right isleaf %i", child1->isLeaf, child2->isLeaf);

			if (t1 > 0.0f && child1->isLeaf) {
				//geom_t[geom_idx] = t1;
				//split_t[split_idx] = t1;
				geom_ranges[geom_idx] = glm::vec3(child1->startM, child1->endM, child1->id);
				split_ranges[geom_idx++] = glm::vec3(child1->startS, child1->endS, child1->id);
				//printf("%i %i,", path_index,child1->id);
				//queue up geoms for intersection test
			}
			if (t2 > 0.0f && child2->isLeaf) {
				//geom_t[geom_idx] = t2;
				//split_t[split_idx] = t2;
				geom_ranges[geom_idx] = glm::vec3(child2->startM, child2->endM, child2->id);
				split_ranges[geom_idx++] = glm::vec3(child2->startS, child2->endS, child2->id);
				//printf("%i %i,", path_index,child2->id);
				//queue up geoms for intersection test
			}

			bool traverseL = (t1 > 0.0f && !child1->isLeaf);
			bool traverseR = (t2 > 0.0f && !child2->isLeaf);

			if (!traverseL && !traverseR) {
				curr_bvh = *--stackPtr; // pop
			}
			else
			{
				int r_id = curr_bvh->child2id;
				curr_bvh = ((t1 > 0.0f && !child1->isLeaf)) ? child1 : child2;
				if ((t1 > 0.0f && !child1->isLeaf) && (t2 > 0.0f && !child2->isLeaf)) {
					*stackPtr++ = child2; // push
				}
			}
		}

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal; 

		int offset = path_index * ball_size;
		int count = 0;

		for (int i = 0; i < geom_idx; i++) { 
			int start = geom_ranges[i][0];
			int end = geom_ranges[i][1];

			for (int m = start; m <= end; m++) {
				Metaball * ball = &metaballs[m];
				float t = rayMarchTest(*ball, iter, pathSegment.ray, intersect_point, normal, outside);

				if (t > 0.0f) {
					int count = atomicAdd(LLcounter, 1); // returns before add
					LLNode &node = nodeBuffer[count];
					//node.metaballid = m;
					node.metaball = ball;
					node.bvh_id = geom_ranges[i][2];
					node.next = headPtrBuffer[path_index];
					headPtrBuffer[path_index] = count;
				}
			}
			int startS = split_ranges[i][0];
			int endS = split_ranges[i][1];

			for (int m = startS; m <= endS; m++) {
				Metaball * ball = &splitmetaballs[m];
				float t = rayMarchTest(*ball, iter, pathSegment.ray, intersect_point, normal, outside);

				if (t > 0.0f) {
					int count = atomicAdd(LLcounter, 1); // returns before add
					LLNode &node = nodeBuffer[count];
					//node.metaballid = m;
					node.metaball = ball;
					node.bvh_id = split_ranges[i][2];
					node.next = headPtrBuffer[path_index];
					headPtrBuffer[path_index] = count;
				}
			}
		}

	}
}

__device__ glm::vec3 calculateBVHNormals(int count, BVHNode * bvhnode, Metaball * metaballs, Metaball *splitballs, glm::vec3 x) {
	int startM = bvhnode->startM;
	int endM = bvhnode->endM;
	int startS = bvhnode->startS;
	int endS = bvhnode->endS;
	glm::vec3 normal(0.f);
	for (int j = startM; j <= endM; ++j) {
		glm::vec3 diff = x - metaballs[j].translation;
		normal += 2.f * diff / (glm::length2(diff) * glm::length2(diff));
	}
	for (int j = startS; j <= endS; ++j) {
		glm::vec3 diff = x - splitballs[j].translation;
		normal += 2.f * diff / (glm::length2(diff) * glm::length2(diff));
	}
	return glm::normalize(normal);
}

__device__ float calculateBVHDensity(Metaball * metaballs, int first_node_idx, LLNode * nodeBuffer, glm::vec3 x,
	BVHNode * BVHNode, Metaball * splitMetaballs) {
	float density = 0.f;
	int node_idx = first_node_idx;
	LLNode * node;
	Metaball * ball;
	//while (node_idx > 0) {
	//	node = &nodeBuffer[node_idx];
	//	//ball = &metaballs[node->metaballid];
	//	if (node->bvh_id == BVHNode->id) {
	//		ball = node->metaball;
	//		float dist = glm::distance(x, ball->translation);
	//		if (dist < ball->radius) {
	//			float val = 1.0f - dist * dist / (ball->radius * ball->radius);
	//			density += val * val;
	//		}
	//	}
	//	node_idx = node->next;
	//}

	int start = BVHNode->startM;
	int end = BVHNode->startS;
	for (int i = BVHNode->startM; i <= BVHNode->endM; i++) {
		ball = &metaballs[i];
		float dist = glm::distance(x, ball->translation);
		if (dist < ball->radius) {
			float val = 1.0f - dist * dist / (ball->radius * ball->radius);
			density += val * val;
		}
	}
	for (int i = BVHNode->startS; i <= BVHNode->endS; i++) {
		ball = &splitMetaballs[i];
		float dist = glm::distance(x, ball->translation);
		if (dist < ball->radius) {
			float val = 1.0f - dist * dist / (ball->radius * ball->radius);
			density += val * val;
		}
	}
	density -= THRESHOLD;
	return density;
}

__global__
void computeBVHIntersections(
	int depth,
	PathSegment * pathSegments,
	int num_paths,
	int * indices,
	Geom * geoms,
	int geoms_size,
	ShadeableIntersection * intersections,
	Metaball * metaballs,
	int num_balls,
	int * LLcounter,
	int * headPtrBuffer,
	LLNode * nodeBuffer,
	BVHNode * BVHNodes,
	Metaball * splitMetaballs)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		// break if no intersections
		int first_node_idx = headPtrBuffer[path_index];
		if (first_node_idx < 0) {
			intersections[path_index].t = -1.0f;
			return;
		}

		// find first positive influence
		float density = 0.f;
		float s;
		float first_s = FLT_MAX;
		float first_density = 0.f;
		glm::vec3 x;
		int node_idx = first_node_idx;
		BVHNode * bvhnode;
		while (node_idx >= 0) {
			// calculate influence
			//s = glm::dot(pathSegment.ray.direction, metaballs[nodeBuffer[node_idx].metaballid].translation - pathSegment.ray.origin);
			
			//int bvh_id = nodeBuffer[node_idx].metaball.bvh_id;

			int bvh_id = nodeBuffer[node_idx].bvh_id;

			//printf("metaball id, %i, split %i, bvh id, %i\n", nodeBuffer[node_idx].metaball.id, nodeBuffer[node_idx].metaball.split, bvh_id);
			BVHNode * tempbvhnode = &BVHNodes[bvh_id];
			//printf("bvhid %i \n", tempbvhnode->id);
			s = glm::dot(pathSegment.ray.direction, nodeBuffer[node_idx].metaball->translation - pathSegment.ray.origin);
			x = pathSegment.ray.origin + s * pathSegment.ray.direction;

			density = calculateBVHDensity(metaballs, first_node_idx, nodeBuffer, x, tempbvhnode, splitMetaballs);
			if (density > 0.f && first_s > (s + EPSILON)) {
				first_s = s;
				first_density = density;
				bvhnode = tempbvhnode;
			}
			node_idx = nodeBuffer[node_idx].next;
		}



		// Secant method (root finding)
		float t0 = 0;
		float t1 = first_s;

		float f1 = first_density;
		float f0;
		if (first_s != FLT_MAX) {
			f0 =  calculateBVHDensity(metaballs, first_node_idx, nodeBuffer, pathSegment.ray.origin, bvhnode, splitMetaballs);
		}
		float t2 = 0;
		float f2 = 0;
		int steps = 0;
		glm::vec3 x2;
		while (first_s != FLT_MAX && (t1 - t0 > 0.0001) && steps < MAXSECANTSTEPS) {
			t2 = t1 - f1 * (t1 - t0) / (f1 - f0);
			x2 = pathSegment.ray.origin + t2 * pathSegment.ray.direction;
			f2 = calculateBVHDensity(metaballs, first_node_idx, nodeBuffer, x2, bvhnode, splitMetaballs);
			if (f2 > 0) {
				t1 = t2;
				f1 = f2;
			}
			else {
				t0 = t2;
				f0 = f2;
			}
			steps++;
		}

		glm::vec3 normal;
		glm::vec3 color_test;
		if (first_s != FLT_MAX) {
			//normal = calculateBVHNormals(num_balls, bvhnode, metaballs, splitMetaballs, x2);
			normal = calculateNormals(num_balls,metaballs, x2);
			//color_test = calculateColor(num_balls, metaballs, x2);
			//printf("%i", bvhnode->id);
			//color_test = bvhnode->id_col;
			color_test = normal;
			//color_test = glm::vec3(1.0f, 1.0f, 1.0f);
		}
		
		// TODO dichotomic method

		// TODO other geom

		//The ray hits something
		//intersections[path_index].t = t_min;
		intersections[path_index].t = (first_s != FLT_MAX) ? t2 : -1.f;

		intersections[path_index].debug = color_test;
#if SECANTSTEPDEBUG
		intersections[path_index].debug = glm::vec3(1.f, 0.f, 0.f);
		if (steps >= MAXSECANTSTEPS) {
			intersections[path_index].debug = glm::vec3(0.0f, 0.f, 1.f);
		}
#endif
		//intersections[path_index].materialId = geoms[hit_geom_index].materialid;
		intersections[path_index].materialId = 0; // TODO
		intersections[path_index].wo = pathSegment.ray.direction;
		intersections[path_index].surfaceNormal = normal;
		intersections[path_index].surfacePoint = x2;
	}
}
#endif


__global__
void computeBVHIntersectionsDebug(
	int depth,
	PathSegment * pathSegments,
	int num_paths,
	int * indices,
	Geom * geoms,
	int geoms_size,
	ShadeableIntersection * intersections,
	Metaball * metaballs,
	int num_balls,
	int * LLcounter,
	int * headPtrBuffer,
	LLNode * nodeBuffer,
	BVHNode * BVHNodes,
	Metaball * splitMetaballs)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		// break if no intersections
		int first_node_idx = headPtrBuffer[path_index];
		if (first_node_idx < 0) {
			intersections[path_index].t = -1.0f;
			return;
		}

		// find first positive influence
		float density = 0.f;
		float s;
		float first_s = FLT_MAX;
		float first_density = 0.f;
		glm::vec3 x;
		int node_idx = first_node_idx;
		BVHNode * bvhnode;
		glm::vec3 tempnorm;
		bool outside;
		glm::vec3 normal;
		glm::vec3 x2;
		while (node_idx >= 0) {
			// calculate influence
			//s = glm::dot(pathSegment.ray.direction, metaballs[nodeBuffer[node_idx].metaballid].translation - pathSegment.ray.origin);

			//int bvh_id = nodeBuffer[node_idx].metaball.bvh_id;

			int bvh_id = nodeBuffer[node_idx].bvh_id;

			//printf("metaball id, %i, split %i, bvh id, %i\n", nodeBuffer[node_idx].metaball.id, nodeBuffer[node_idx].metaball.split, bvh_id);
			BVHNode * tempbvhnode = &BVHNodes[bvh_id];
			//s = glm::dot(pathSegment.ray.direction, nodeBuffer[node_idx].metaball->translation - pathSegment.ray.origin);
			s = rayMarchTest(*nodeBuffer[node_idx].metaball, 0, pathSegment.ray, x, tempnorm, outside);
			//x = pathSegment.ray.origin + s * pathSegment.ray.direction;

			//density = calculateBVHDensity(metaballs, first_node_idx, nodeBuffer, x, tempbvhnode, splitMetaballs);
			if (first_s > s) {
				normal = tempnorm;
				x2 = x;
				first_s = s;
				first_density = density;
				bvhnode = tempbvhnode;
			}
			node_idx = nodeBuffer[node_idx].next;
		}



		// Secant method (root finding)
		float t0 = 0;
		float t1 = first_s;

		float f1 = first_density;
		float f0;
		//if (first_s != FLT_MAX) {
		//	f0 = calculateBVHDensity(metaballs, first_node_idx, nodeBuffer, pathSegment.ray.origin, bvhnode, splitMetaballs);
		//}
		//float t2 = 0;
		//float f2 = 0;
		//int steps = 0;
		//glm::vec3 x2;
		//while (first_s != FLT_MAX && (t1 - t0 > 0.0001) && steps < MAXSECANTSTEPS) {
		//	t2 = t1 - f1 * (t1 - t0) / (f1 - f0);
		//	x2 = pathSegment.ray.origin + t2 * pathSegment.ray.direction;
		//	f2 = calculateBVHDensity(metaballs, first_node_idx, nodeBuffer, x2, bvhnode, splitMetaballs);
		//	if (f2 > 0) {
		//		t1 = t2;
		//		f1 = f2;
		//	}
		//	else {
		//		t0 = t2;
		//		f0 = f2;
		//	}
		//	steps++;
		//}

		//glm::vec3 color_test = calculateColor(num_balls, metaballs, x2);
		glm::vec3 color_test = normal;
		
		/*if (first_s != FLT_MAX) {
			normal = calculateBVHNormals(num_balls, bvhnode, metaballs, splitMetaballs, x2);
		}*/

		// TODO dichotomic method

		// TODO other geom

		//The ray hits something
		//intersections[path_index].t = t_min;
		//intersections[path_index].t = (first_s != FLT_MAX) ? t2 : -1.f;
		intersections[path_index].t = (first_s != FLT_MAX) ? first_s : -1.f;

		intersections[path_index].debug = bvhnode->id_col;
		intersections[path_index].debug = color_test;
#if SECANTSTEPDEBUG
		intersections[path_index].debug = glm::vec3(1.f, 0.f, 0.f);
		if (steps >= MAXSECANTSTEPS) {
			intersections[path_index].debug = glm::vec3(0.0f, 0.f, 1.f);
		}
#endif
		//intersections[path_index].materialId = geoms[hit_geom_index].materialid;
		intersections[path_index].materialId = 0; // TODO
		intersections[path_index].wo = pathSegment.ray.direction;
		intersections[path_index].surfaceNormal = normal;
		intersections[path_index].surfacePoint = x2;
	}
}