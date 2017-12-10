#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include "stream_compaction/efficient_shared.h"
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "bvh.h"
#include "timer.h"

#define ERRORCHECK 1

#define NORMALSDEBUG 1
#define NUM_BVH_NODES (1 << (MAX_BVH_DEPTH + 1)) - 1
#define NUM_BVH_LEAVES (1 << MAX_BVH_DEPTH)
#define SECANTSTEPDEBUG 0
#define MAXDICHOTOMICSTEPS 30
#define MAXLISTSIZE 300
#define SCENEBOUND 8

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) 
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

// get pixel value from spherical direction
__host__ __device__
glm::vec3 getColor(float * dev_data, int width, int height, glm::vec3& w) {
	float phi = std::atan2(w.z, w.x);
	float u = (phi < 0.f ? (phi + TWO_PI) : phi) / TWO_PI;
	float v = 1.f - std::acos(w.y) / PI;

	int x = glm::min((float)width * u, (float)width - 1.f);
	int y = glm::min((float)height * (1.f - v), (float)height - 1.f);

	int index = y * width + x;
	return glm::vec3(dev_data[0], dev_data[0], dev_data[0]);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ 
void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image) 
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);
		// color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
		// color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
		// color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static Texture * dev_environment = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static int * dev_indices = NULL;

static Metaball * dev_metaballs = NULL;
static Metaball * dev_metaballs_translated = NULL;
static Metaball * dev_splitMetaballs = NULL;
static float * dev_ballDist = NULL;
static int * dev_LLcounter = NULL;
static int * dev_headPtrBuffer = NULL;
static LLNode * dev_nodeBuffer = NULL;
static BVHNode * dev_bvhTree = NULL;

Metaball * metaballsCPU = NULL;

std::vector<BVHNode> BVHnodes;
std::vector<Metaball> allSplitBalls;

void pathtraceInit(Scene *scene) 
{
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	metaballsCPU = new Metaball[hst_scene->metaballs.size()];

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	if (scene->environmentMap.size() > 0) {
  		cudaMalloc(&(scene->environmentMap[0].dev_data), scene->environmentMap[0].imagesize * sizeof(float)); // environment map image
  		cudaMemcpy(scene->environmentMap[0].dev_data, scene->environmentMap[0].host_data, scene->environmentMap[0].imagesize * sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc(&dev_environment, sizeof(Texture)); // environment map struct
		cudaMemcpy(dev_environment, scene->environmentMap.data(), sizeof(Texture), cudaMemcpyHostToDevice);
  	}
  	
  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_indices, pixelcount * sizeof(int));

	cudaMalloc(&dev_metaballs, scene->metaballs.size() * sizeof(Metaball)); // metaball memory
	cudaMemcpy(dev_metaballs, scene->metaballs.data(), scene->metaballs.size() * sizeof(Metaball), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_metaballs_translated, scene->metaballs.size() * sizeof(Metaball)); // metaball memory
	cudaMemcpy(dev_metaballs_translated, scene->metaballs.data(), scene->metaballs.size() * sizeof(Metaball), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_ballDist, scene->metaballs.size() * sizeof(float)); 

	cudaMalloc(&dev_headPtrBuffer, pixelcount * sizeof(int)); // initialize Linked List
	cudaMemset(dev_headPtrBuffer, -1, pixelcount * sizeof(int));

	cudaMalloc(&dev_nodeBuffer, MAXLISTSIZE * pixelcount * sizeof(LLNode));
	cudaMemset(dev_nodeBuffer, -1, MAXLISTSIZE * pixelcount * sizeof(LLNode));

	cudaMalloc(&dev_LLcounter, sizeof(int));
	cudaMemset(dev_LLcounter, 0, sizeof(int));
	
	cudaMalloc(&dev_bvhTree, ((1 << (MAX_BVH_DEPTH + 1)) - 1) * sizeof(BVHNode)); // initialize for Split BVH

	cudaMalloc(&dev_splitMetaballs, (1 << MAX_BVH_DEPTH) * MAXSPLITNODES * sizeof(Metaball)); //splitmetaballs

	cudaMemcpy(metaballsCPU, dev_metaballs_translated, hst_scene->metaballs.size() * sizeof(Metaball), cudaMemcpyDeviceToHost);
	int num_bvh_nodes = (1 << (MAX_BVH_DEPTH + 1)) - 1;
	int num_bvh_leaves = (1 << MAX_BVH_DEPTH);
	BVHnodes.clear();
	BVHnodes.resize(NUM_BVH_NODES);
	allSplitBalls.clear();
	allSplitBalls.resize(MAXSPLITNODES * num_bvh_leaves);
	constructBVHTree(MAX_BVH_DEPTH, metaballsCPU, BVHnodes, allSplitBalls, MAXSPLITNODES, hst_scene);
	cudaMemcpy(dev_metaballs, metaballsCPU, hst_scene->metaballs.size() * sizeof(Metaball), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_bvhTree, BVHnodes.data(), num_bvh_nodes * sizeof(BVHNode), cudaMemcpyHostToDevice);
	checkCUDAError("bvh error");
	cudaDeviceSynchronize();
	cudaMemcpy(dev_splitMetaballs, allSplitBalls.data(), MAXSPLITNODES * num_bvh_leaves * sizeof(Metaball), cudaMemcpyHostToDevice);
	checkCUDAError("splitmetaballs error");
	cudaDeviceSynchronize();

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() 
{
	delete metaballsCPU;

    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_environment); 
  	cudaFree(dev_intersections);
	cudaFree(dev_indices);

	cudaFree(dev_metaballs);
	cudaFree(dev_metaballs_translated);
	cudaFree(dev_ballDist);
	cudaFree(dev_LLcounter);
	cudaFree(dev_headPtrBuffer);
	cudaFree(dev_nodeBuffer);
	cudaFree(dev_bvhTree);
	cudaFree(dev_splitMetaballs);
    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ 
void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, int* indices)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);

		segment.pixelIndex = index;
		indices[index] = index + 1;
		segment.remainingBounces = traceDepth;
	}
}

#if !BVH
__global__ 
void generateLinkedList(
	PathSegment * pathSegments, 
	int num_paths, 
	Metaball * metaballs, 
	int num_balls, 
	int * LLcounter,
	int * headPtrBuffer,
	LLNode * nodeBuffer,
	int iter)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths) {
		PathSegment pathSegment = pathSegments[path_index];
		glm::vec3 intersect_point;
		glm::vec3 normal;
		bool outside = true;
		for (int i = 0; i < num_balls; i++) {
			Metaball * ball = &metaballs[i];
			float t = rayMarchTest(*ball, iter, pathSegment.ray, intersect_point, normal, outside);
			if (t > 0.0f) {
				int count = atomicAdd(LLcounter, 1); // returns before add
				LLNode &node = nodeBuffer[count];
				node.metaball = ball;
				//node.metaballid = i;
				node.next = headPtrBuffer[path_index];
				headPtrBuffer[path_index] = count;
			}
		}
	}
}

// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
__global__ 
void computeIntersections(
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
	LLNode * nodeBuffer)
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
		while (node_idx >= 0) {
			// calculate influence
			//s = glm::dot(pathSegment.ray.direction, metaballs[nodeBuffer[node_idx].metaballid].translation - pathSegment.ray.origin);
			s = glm::dot(pathSegment.ray.direction, nodeBuffer[node_idx].metaball->translation - pathSegment.ray.origin);
			x = pathSegment.ray.origin + s * pathSegment.ray.direction;

			density = calculateDensity(metaballs, first_node_idx, nodeBuffer, x);
			if (density > 0 && first_s > s) {
				first_s = s;
				first_density = density;
			}
			node_idx = nodeBuffer[node_idx].next;
		}

		

		// Secant method (root finding)
		float t0 = 0;
		float t1 = first_s;
		
		float f1 = first_density;
		float f0 = calculateDensity(metaballs, first_node_idx, nodeBuffer, pathSegment.ray.origin);
		float t2 = 0;
		float f2 = 0;
		int steps = 0;
		glm::vec3 x2;
		while (first_s != FLT_MAX && (t1 - t0 > 0.0001) && steps < MAXSECANTSTEPS) {
			t2 = t1 - f1 * (t1 - t0) / (f1 - f0);
			x2 = pathSegment.ray.origin + t2 * pathSegment.ray.direction;
			f2 = calculateDensity(metaballs, first_node_idx, nodeBuffer, x2);
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

		glm::vec3 normal = calculateNormals(num_balls, metaballs, x2);
		glm::vec3 color_test = normal;

		// TODO dichotomic method

		// TODO other geom

		//The ray hits something
		//intersections[path_index].t = t_min;
		intersections[path_index].t = (first_s != FLT_MAX) ? t2 : -1.f;

		intersections[path_index].debug = color_test;
#if SECANTSTEPDEBUG
		intersections[path_index].debug = glm::vec3(1.f,0.f,0.f);
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
void translateMetaballs(int num_balls, Metaball * metaballs, Metaball * metaballs_translated)
{
	int ball_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (ball_index < num_balls)
	{
		float b = SCENEBOUND;
		metaballs_translated[ball_index] = metaballs[ball_index];
		metaballs_translated[ball_index].translation += metaballs[ball_index].velocity / 10.f;
		if (metaballs_translated[ball_index].translation.y > b) {
			metaballs_translated[ball_index].velocity.y *= -1.f;
			metaballs_translated[ball_index].translation.y = b;
		}
		if (metaballs_translated[ball_index].translation.y < -b) {
			metaballs_translated[ball_index].velocity.y *= -1.f;
			metaballs_translated[ball_index].translation.y = -b;
		}
		if (metaballs_translated[ball_index].translation.z > b) {
			metaballs_translated[ball_index].velocity.z *= -1.f;
			metaballs_translated[ball_index].translation.z = b;
		}
		if (metaballs_translated[ball_index].translation.z < -b) {
			metaballs_translated[ball_index].velocity.z *= -1.f;
			metaballs_translated[ball_index].translation.z = -b;
		}
		if (metaballs_translated[ball_index].translation.x > b) {
			metaballs_translated[ball_index].velocity.x *= -1.f;
			metaballs_translated[ball_index].translation.x = b;
		}
		if (metaballs_translated[ball_index].translation.x < -b) {
			metaballs_translated[ball_index].velocity.x *= -1.f;
			metaballs_translated[ball_index].translation.x = -b;
		}

	}
}

__global__ 
void computeBallDist(int num_balls, glm::vec3 cam_pos, Metaball * metaballs, float * ballDist)
{
	int ball_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (ball_index < num_balls)
	{
		ballDist[ball_index] = glm::distance(cam_pos, metaballs[ball_index].translation) - metaballs->radius;
	}
}

__global__ 
void shadeMetaball(
	int iter,
	int num_paths,
	ShadeableIntersection * shadeableIntersections,
	PathSegment * pathSegments,
	int * indices,
	Material * materials,
	Texture * environment)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
									 // Set up the RNG
									 // LOOK: this is how you use thrust's RNG! Please look at
									 // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
			}

			// TODO : lighting computation
			else {
				scatterRay(pathSegments[idx], intersection.surfacePoint, intersection.surfaceNormal, material, rng);
			}
		}
		// If there was no intersection, color the ray black.
		// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
		// used for opacity, in which case they can indicate "no opacity".
		// This can be useful for post-processing and image compositing.
		else {
			if (environment) {
				pathSegments[idx].color = environment->getColor(pathSegments[idx].ray.direction);
			}
			else {
				pathSegments[idx].color = glm::vec3(0.f);
			}
		}
	}
}

__global__ void shadeDebug(
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, int * indices
	, Material * materials
	, Texture * environment

)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...

			glm::vec3 camdir = intersection.wo;
			glm::vec3 lightpos = glm::vec3(0, 5, 7);
			float NdotH = glm::dot(-camdir, intersection.surfaceNormal);
			float specular = glm::pow(NdotH, 10.f);
			pathSegments[idx].color = glm::dot(intersection.surfaceNormal, -camdir) * intersection.debug;
#if SECANTSTEPDEBUG == 0
			pathSegments[idx].color += specular * glm::vec3(0.8f, 0.8f, 0.8f);
			//pathSegments[idx].color = glm::vec3(1.0f, 1.0f, 1.0f);
#endif
			//pathSegments[idx].color = camdir;
		}
		else {
			pathSegments[idx].color = glm::vec3(0.1f);
		}
	}
}

// Add the current iteration's output to the overall image
__global__ 
void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		//image[iterationPath.pixelIndex] = iterationPath.color;
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) 
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths, dev_indices);
	checkCUDAError("generate camera ray");

	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

	// move metaballs
	dim3 numblocksMetaballs = (hst_scene->metaballs.size() + blockSize1d - 1) / blockSize1d;
	//translateMetaballs << <numblocksMetaballs, blockSize1d >> > (hst_scene->metaballs.size(), dev_metaballs);
	//checkCUDAError("translate metaballs");

	//// calculate ball distance from camera
	//computeBallDist << <numblocksMetaballs, blockSize1d >> > (hst_scene->metaballs.size(), hst_scene->state.camera.position, dev_metaballs, dev_ballDist);
	//checkCUDAError("ball distance computation");
	//cudaDeviceSynchronize();

	//// sort metaballs based on distance from camera
	//// so when added to linked list, already in order
	//thrust::sort_by_key(thrust::device, dev_ballDist, dev_ballDist + hst_scene->metaballs.size(), dev_metaballs, thrust::greater<float>());
	//checkCUDAError("sort metaball distance");
	//cudaDeviceSynchronize();

#if BVH
	int num_bvh_nodes = (1 << (MAX_BVH_DEPTH + 1)) - 1;
	int num_bvh_leaves = (1 << MAX_BVH_DEPTH);
#else
	cudaMemcpy(dev_metaballs, dev_metaballs_translated, hst_scene->metaballs.size() * sizeof(Metaball), cudaMemcpyDeviceToDevice;
#endif

	// Concurrent Linked List Construction
	// head pointer buffer contains index into node buffer for each path
	// of closest metaball from camera (since previous sort descending)



	bool iterationComplete = false;
	bool build_bvh = true;

	int depth = 0;
#if NORMALSDEBUG
	int max_depth = 1;
#else
	int max_depth = 3;
#endif
	while (depth < max_depth) {
		cudaMemset(dev_LLcounter, 0, sizeof(int));
		cudaMemset(dev_headPtrBuffer, -1, pixelcount * sizeof(int));
		cudaMemset(dev_nodeBuffer, -1, MAXLISTSIZE * pixelcount * sizeof(LLNode));
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

#if !BVH
		generateLinkedList << <numblocksPathSegmentTracing, blockSize1d >> >(
			dev_paths, num_paths,
			dev_metaballs, hst_scene->metaballs.size(),
			dev_LLcounter, dev_headPtrBuffer, dev_nodeBuffer, iter);
		checkCUDAError("generate Linked List");
		cudaDeviceSynchronize();

		// tracing
		computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
			depth,
			dev_paths, num_paths,
			dev_indices,
			dev_geoms, hst_scene->geoms.size(),
			dev_intersections,
			dev_metaballs, hst_scene->metaballs.size(),
			dev_LLcounter, dev_headPtrBuffer, dev_nodeBuffer);
		checkCUDAError("compute Intersection");
		cudaDeviceSynchronize();
		depth++;

#else 

		int num_BVHnodes = (1 << (MAX_BVH_DEPTH + 1)) - 1;
		dim3 numblocksBVH = (num_BVHnodes + blockSize1d - 1) / blockSize1d;

		kernSetBVHTransform << < numblocksBVH, blockSize1d >> > (num_BVHnodes, dev_bvhTree);
		checkCUDAError("setBVHTransform");
		cudaDeviceSynchronize();

		computeLinkedListBVH << <numblocksPathSegmentTracing, blockSize1d >> >(
			depth
			, num_paths
			, dev_paths
			, dev_intersections
			, dev_bvhTree
			, num_bvh_nodes
			, dev_geoms
			, dev_metaballs
			, dev_splitMetaballs
			, hst_scene->metaballs.size()
			, iter
			, dev_LLcounter, dev_headPtrBuffer, dev_nodeBuffer
		);

		checkCUDAError("compute Linked List with BVH");
		cudaDeviceSynchronize();
		startCpuTimer();
		computeBVHIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth,
			dev_paths, num_paths,
			dev_indices,
			dev_geoms, hst_scene->geoms.size(),
			dev_intersections,
			dev_metaballs, hst_scene->metaballs.size(),
			dev_LLcounter, dev_headPtrBuffer, dev_nodeBuffer,
			dev_bvhTree,dev_splitMetaballs);	
		if (build_bvh) {
			constructBVHTree(MAX_BVH_DEPTH, metaballsCPU, BVHnodes, allSplitBalls, MAXSPLITNODES, hst_scene);
			cudaMemcpy(dev_metaballs, metaballsCPU, hst_scene->metaballs.size() * sizeof(Metaball), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_bvhTree, BVHnodes.data(), num_bvh_nodes * sizeof(BVHNode), cudaMemcpyHostToDevice);
			//checkCUDAError("bvh error");
			//cudaDeviceSynchronize();
			cudaMemcpy(dev_splitMetaballs, allSplitBalls.data(), MAXSPLITNODES * num_bvh_leaves * sizeof(Metaball), cudaMemcpyHostToDevice);
			//checkCUDAError("splitmetaballs error");
			//cudaDeviceSynchronize();
			build_bvh = false;
		}
		endCpuTimer();
		printf("intersection time");
		printCPUTime();
		
		//checkCUDAError("compute Intersection with BVH");
		//cudaDeviceSynchronize();
		depth++;

#endif

#if NORMALSDEBUG
		shadeDebug<<<numblocksPathSegmentTracing, blockSize1d>>> (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_indices,
			dev_materials,
			dev_environment
			);
		checkCUDAError("shading");
#else
		shadeMetaball << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_indices,
			dev_materials,
			dev_environment
			);
		checkCUDAError("shading");
#endif
		//iterationComplete = true; // TODO: should be based off stream compaction results.
	}

	// what is counter? is it ever greater than size of dev_nodeBuffer
	int count;
	cudaMemcpy(&count, dev_LLcounter, sizeof(int), cudaMemcpyDeviceToHost);
	printf("LLcount: %i\n", count);

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);
	checkCUDAError("finalGather");

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
	checkCUDAError("sendImageToPBO");

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

void pathtraceReset()
{
	// move metaballs
	const int blockSize1d = 128;
	dim3 numblocksMetaballs = (hst_scene->metaballs.size() + blockSize1d - 1) / blockSize1d;
	translateMetaballs << <numblocksMetaballs, blockSize1d >> > (hst_scene->metaballs.size(), dev_metaballs, dev_metaballs_translated);
	checkCUDAError("translate metaballs");
	cudaMemcpy(metaballsCPU, dev_metaballs_translated, hst_scene->metaballs.size() * sizeof(Metaball), cudaMemcpyDeviceToHost);
	int num_bvh_nodes = (1 << (MAX_BVH_DEPTH + 1)) - 1;
	int num_bvh_leaves = (1 << MAX_BVH_DEPTH);
	BVHnodes.clear();
	BVHnodes.resize(NUM_BVH_NODES);
	allSplitBalls.clear();
	allSplitBalls.resize(MAXSPLITNODES * num_bvh_leaves);

	// clear image
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	checkCUDAError("pathtraceReset");
}
