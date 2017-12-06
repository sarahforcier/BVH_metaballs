#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "stb_image.h"
#include "glm/glm.hpp"
#include "utilities.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom {
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Metaball {
	int id;
	int materialid;
	float radius;
	glm::vec3 translation;
	glm::vec3 velocity;
	int bvh_id;
	int split;
};

struct BBox {
	glm::vec3 maxB;
	glm::vec3 minB;
};

struct LLNode {
	//int metaballid;
	Metaball metaball;
	int next;
};

struct BVHNode {
	glm::vec3 minB;
	glm::vec3 maxB;
	BBox Bbox;
	int startM;
	int endM;
	int startS;
	int endS;
	bool isLeaf = true;
	int id;
	int child1id;
	int child2id;
	glm::mat4 transform;
	glm::mat4 inverseTransform;
	glm::mat4 invTranspose;
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct Texture {
    int width;
    int height;
    int imagesize;
    float * host_data = nullptr;
	float * dev_data = nullptr;

    // see ../external/include/stb_image.h for usage
    Texture(int w, int h, char const *file) : width(w), height(h) {
		int comp = 3;
		imagesize = width * height * 3;
        host_data = stbi_loadf(file, &width, &height, &comp, 0); // 3 components per pixel
        dev_data = NULL;
    }

	Texture(const Texture& other) = delete;
	Texture(Texture&& other)
		: host_data(other.host_data),
		dev_data(other.dev_data),
		width(other.width),
		height(other.height),
		imagesize(other.imagesize)
	{
		other.host_data = nullptr;
		other.dev_data = nullptr;
	}

    ~Texture() {
		if (host_data) {
			stbi_image_free(host_data);
		}
    }

    // get pixel value from spherical direction
    __host__ __device__
    glm::vec3 getColor(glm::vec3& w) {   
        float phi = std::atan2(w.z, w.x);
        float u = (phi < 0.f ? (phi + TWO_PI) : phi) / TWO_PI;
        float v = 1.f - std::acos(w.y) / PI;
        
        int x = glm::min((float)width * u, (float)width - 1.f);
        int y = glm::min((float)height * (1.f - v), (float)height - 1.f);

        int index = y * width + x;
        return glm::vec3(dev_data[index * 3], dev_data[index * 3 + 1], dev_data[index * 3 + 2]);
    }
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
	Ray ray;
	glm::vec3 color;
	int pixelIndex;
	int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  glm::vec3 surfacePoint;
  int materialId;
  glm::vec3 wo;
  glm::vec3 debug;
};
