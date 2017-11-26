#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

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

#define THRESHOLD 0.2
#define MAX_BVH_DEPTH 5

#define SECANTSTEPDEBUG 0
#define MAXSECANTSTEPS 30
#define MAXDICHOTOMICSTEPS 30

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
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        //color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        //color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        //color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);
		color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

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
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;

// TODO: static variables for device memory, any extra info you need, etc
static Metaball * dev_metaballs = NULL;
static Metaball * dev_ballHits = NULL;
static float * dev_ballDist = NULL;
static BVHNode * dev_bvhTree = NULL;



void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
	cudaMalloc(&dev_metaballs, scene->metaballs.size() * sizeof(Metaball));
	cudaMemcpy(dev_metaballs, scene->metaballs.data(), scene->metaballs.size() * sizeof(Metaball), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_ballHits, scene->metaballs.size() * pixelcount * sizeof(Metaball));
	cudaMalloc(&dev_ballDist, scene->metaballs.size() * pixelcount * sizeof(float));

	cudaMalloc(&dev_bvhTree, ((1 << (MAX_BVH_DEPTH + 1)) - 1) * sizeof(BVHNode));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
	cudaFree(dev_metaballs);
	cudaFree(dev_ballHits);
	cudaFree(dev_ballDist);
	cudaFree(dev_bvhTree);
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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
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
		segment.remainingBounces = traceDepth;
	}
}

__device__ float calculateDensity(int count, Metaball * ballHits, int offset, glm::vec3 x) {
	float density = 0.f;
	for (int j = 0; j < count; ++j) {
		float dist = glm::distance(x, ballHits[offset + j].translation);
		if (dist < ballHits[offset + j].radius) {
			float val = 1.0f - dist * dist / (ballHits[offset + j].radius * ballHits[offset + j].radius);
			density += val * val;
		}
	}
	density -= THRESHOLD;
	return density;
}

__device__ glm::vec3 calculateNormals(int count, Metaball * ballHits, int offset, glm::vec3 x) {
	glm::vec3 normal(0.f);
	for (int j = 0; j < count; ++j) {
		glm::vec3 diff = x - ballHits[j].translation;
		normal += 2.f * diff / (glm::length2(diff) * glm::length2(diff));
	}
	return glm::normalize(normal);
}

__device__ glm::vec3 calculateColor(int count, Metaball * ballHits, int offset, glm::vec3 x) {
	glm::vec3 normal(0.f);
	for (int j = 0; j < count; ++j) {
		glm::vec3 diff = x - ballHits[j].translation;
		normal += ballHits[j].velocity / (glm::length2(diff) * glm::length2(diff));
	}
	return glm::abs(glm::normalize(normal));
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
	, Metaball * metaballs
	, int ball_size
	, Metaball * ballHits
	, float * ballDist
	, int iter
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		//TODO 
		//find metaball intersection
		//loop over all metaballs and add to intersections along ray
		int count = 0;
		int offset = path_index * ball_size;
		for (int i = 0; i < ball_size; i++) {
			Metaball & ball = metaballs[i];
			t = rayMarchTest(ball, iter, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			if (t > 0.0f)
			{
				ballHits[offset + count] = ball;
				ballDist[offset + count] = t;
				count++;
	
				if (t_min > t) {
					t_min = t;
					hit_geom_index = count; // NOT ACTUALLY INDEX, CHECK NUMBER OF INTERSECTIONS FOR DEBUG
					intersect_point = tmp_intersect;
					normal = tmp_normal;
				}
			}
		}
		//store all metaballs that ray intersections - naive way: keep an array and populate it

		thrust::sort_by_key(thrust::seq, ballDist + offset, ballDist + offset + count, ballHits + offset);

		float final_t = ballDist[offset];

		glm::vec3 debug_vector = ballHits[offset].translation;

		// find first positive influence

		int first;
		float density = 0.f;
		float s;
		glm::vec3 x;
		for (first = 0; first < count; ++first) {
			// calculate influence
			s = glm::dot(pathSegment.ray.direction, ballHits[offset + first].translation - pathSegment.ray.origin);
			x = pathSegment.ray.origin + s * pathSegment.ray.direction;

			density = calculateDensity(count, ballHits, offset, x);
			if (density > 0) {
				break;
			}
		}

		//do the secant method
		float t0 = 0;
		float t1 = s;
		
		float f1 = density;
		float f0 = calculateDensity(count, ballHits, offset, pathSegment.ray.origin);
		float t2 = 0;
		float f2 = 0;
		int steps = 0;
		glm::vec3 x2;
		while (first != count && (t1 - t0 > 0.0001) && steps < MAXSECANTSTEPS) {
			t2 = t1 - f1 * (t1 - t0) / (f1 - f0);
			x2 = pathSegment.ray.origin + t2 * pathSegment.ray.direction;
			f2 = calculateDensity(count, ballHits, offset, x2);
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

		normal = calculateNormals(ball_size, metaballs, offset, x2);
		glm::vec3 color_test = calculateColor(ball_size, metaballs, offset, x2);
		final_t = (first != count) ? t2 : -1.f;

		// TODO do dichotomic method


		// naive parse through global geoms

		for (int i = 0; i < 0; i++) // SET TO 0 TO IGNORE GEOMS
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			//intersections[path_index].t = t_min;
			intersections[path_index].t = final_t;

			intersections[path_index].debug = color_test;
#if SECANTSTEPDEBUG
			intersections[path_index].debug = glm::vec3(1.f,0.f,0.f);
			if (steps >= MAXSECANTSTEPS) {
				intersections[path_index].debug = glm::vec3(0.0f, 0.f, 1.f);
			}
#endif
			//intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].materialId = count;
			intersections[path_index].wo = pathSegment.ray.direction;
			intersections[path_index].surfaceNormal = normal;
		}

		//delete[] ballDist;
		//delete[] ballHits;

	}
}

__global__ void translateMetaballs(int num_balls, Metaball * metaballs)
{
	int ball_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (ball_index < num_balls)
	{
		metaballs[ball_index].translation += metaballs[ball_index].velocity / 10.f;
		if (metaballs[ball_index].translation.y > 5.f) {
			metaballs[ball_index].velocity.y *= -1.f;
			metaballs[ball_index].translation.y = 5.f;
		}
		if (metaballs[ball_index].translation.y < -5.f) {
			metaballs[ball_index].velocity.y *= -1.f;
			metaballs[ball_index].translation.y = -5.f;
		}
		if (metaballs[ball_index].translation.z > 5.f) {
			metaballs[ball_index].velocity.z *= -1.f;
			metaballs[ball_index].translation.z = 5.f;
		}
		if (metaballs[ball_index].translation.z < -5.f) {
			metaballs[ball_index].velocity.z *= -1.f;
			metaballs[ball_index].translation.z = -5.f;
		}
		if (metaballs[ball_index].translation.x > 5.f) {
			metaballs[ball_index].velocity.x *= -1.f;
			metaballs[ball_index].translation.x = 5.f;
		}
		if (metaballs[ball_index].translation.x < -5.f) {
			metaballs[ball_index].velocity.x *= -1.f;
			metaballs[ball_index].translation.x = -5.f;
		}

	}

}

__global__ void shadeFakeMaterial (
  int iter
  , int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	)
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
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {
        float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
        pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
        pathSegments[idx].color *= u01(rng); // apply some noise because why not
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
    }
  }
}

__global__ void shadeMetaballs(
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials

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
#endif
			//pathSegments[idx].color = camdir;
		}
		else {
			pathSegments[idx].color = glm::vec3(0.1f);
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] = iterationPath.color;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
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

    // TODO: perform one iteration of path tracing

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
  startCpuTimer();
	while (!iterationComplete) {

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// move metaballs
	dim3 numblocksMetaballs = (hst_scene->metaballs.size() + blockSize1d - 1) / blockSize1d;
	translateMetaballs << <numblocksMetaballs, blockSize1d >> > (hst_scene->metaballs.size(), dev_metaballs);
	checkCUDAError("translate metaballs");
	cudaDeviceSynchronize();

	constructBVHTree(3, dev_metaballs, dev_bvhTree, hst_scene);

	const int blockSize1d1 = 128;//num_geoms
	int num_BVHnodes = (1 << (MAX_BVH_DEPTH + 1)) - 1;
	dim3 numblocksBVH = (num_BVHnodes + blockSize1d - 1) / blockSize1d;

	kernSetBVHTransform << < numblocksBVH, blockSize1d >> > (num_BVHnodes, dev_bvhTree);


	// tracing
	dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
	computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
		depth
		, num_paths
		, dev_paths
		, dev_geoms
		, hst_scene->geoms.size()
		, dev_intersections
		, dev_metaballs
		, hst_scene->metaballs.size()
		, dev_ballHits
		, dev_ballDist
		, iter
		);
	checkCUDAError("trace one bounce");
	cudaDeviceSynchronize();
	depth++;


	// TODO:
	// --- Shading Stage ---
	// Shade path segments based on intersections and generate new rays by
  // evaluating the BSDF.
  // Start off with just a big kernel that handles all the different
  // materials you have in the scenefile.
  // TODO: compare between directly shading the path segments and shading
  // path segments that have been reshuffled to be contiguous in memory.

  shadeMetaballs<<<numblocksPathSegmentTracing, blockSize1d>>> (
    iter,
    num_paths,
    dev_intersections,
    dev_paths,
    dev_materials
  );
  iterationComplete = true; // TODO: should be based off stream compaction results.
	}


	printf("Iteration Done\n");
	endCpuTimer();
	printTime();

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
