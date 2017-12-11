# Cloudy with a Chance of Metaballs

Authors:
[Sarah Forcier](https://github.com/sarahforcier), 
[Charles Wang](https://github.com/charlesliwang)

## Overview
Metaballs are a fun way to create interesting geometries or simulate deformation or fluids. However, their isosurface representation is difficult to calculate in real time. One such method uses the marching cube algorithm to tessellate the surface, but this scales poorly for an increasing number of metaballs or for high resolution because it requires voxelization of the space. We aim to achieve speedup by keeping the metaballs in implicit form and using a modified bounding volume hierarchy as described by Gourmel et al. (Siggraph 2009). Along with a BVH structure, this method describes a fast approach to finding a ray-isosurface intersection that avoids slow ray marching. The BVH is computed on the CPU, but the rendering will be performed with CUDA on the GPU with BVH nodes and metaball information stored as textures. Once we render metaballs in real time, we want to demonstrate the benefits of this method by implementing fresnel reflection and refraction and ambient occlusion. 

## Features
* Secant Method for ray-isosurface intersection
* Fitted BVH with Split Nodes
* Environment Map with Reflection
* Cool Tornado Demo Scene!

### Secant Method

Rather than raymarching through the scene, we use the secant method to find a "root" at the surface of the metaball. In order to find the first interval, we find the first metaball intersection projection with a positive (f(x)) value. Because we can assume that the metaball density at the camera is negative, we iterate along a slope to coverge towards a density of 0.

![](img/secant.png) \
Example diagram illustrating the secant method (from wikipedia)

**Concurrent Linked List**

When using the secant method, we must find the first intersection projection with a positive metaball density. This allows us to keep track of all of the metaballs intersected, such that for each ray, we have a list of metaballs that contribute to the overall density.

### Fitted BVH with Split Nodes

The general structure of our modified BVH structure for metaballs follows a standard BVH for geometry with a couple major differences. First, each BVH leaf node contains not only the metaballs contained in its bounding box, but also a list of "split metaballs" - metaballs that overlap the bounding box but are not completely contained. This means that they may have some influence on the metaballs inside the bounding box, but are not fully encapsulated. Our BVH structure is also a fitted BVH structure where no two BVH nodes overlap. This helps to reduce the number of redundant bounding box intersection tests.

![](img/splitbvh.png) \
Example diagram illustrating split metaballs in blue

**Indexing Metaballs and BVH**

With a GPU implementation, BVH nodes cannot contain a dynamic array of pointers to metaballs. Instead, we reorder metaballs in memory by their BVH leaf node IDs and each BVH keeps track of a start and end index to access their metaballs.

![](img/bvh_array_example.PNG) \
The metaball and bvh data may be ordered something like this

**Indexing Split Metaballs**

Metaballs must be duplicated for split metaballs because they may affect metaballs in other bvh leaf nodes. To store these, we have to put a cap on the total number of split metaballs. We know that only leaf nodes contain split metaballs and that the maximum number of leaf nodes is the number of nodes at max depth. The total number of split nodes is then MAX_PER_LEAF * NUM_NODES_AT_MAX_DEPTH. Some nodes are leaf nodes, but not at the max depths, so we index them in the final array by cascading it down and finding that index in the max depth only array.

![](img/split_array_example.PNG) \
The leaf node in red will be indexed to a corresponding node at max depth for storing split metaballs

### Optimizations

**Streams CPU and GPU Overlap**

Our BVH tree construction takes place on the CPU, because using dynamic arrays to pass down split metaballs data to child nodes is more convenient. Though a CPU construction algorithm may be slower than a GPU construction algorithm, we can take advantage of CPU and GPU overlap to hide the cost of our construction process. By executing the BVH construction command immediately after our intersection test kernel (our most expensive kernel) and synchronizing afterwards, the CPU and GPU commands run in parallel, reducing the overall computation time per frame.

## Performance Analysis

## Milestone November 20, 2017
Path tracing and ray-isosurface intersection (visualization of metaballs)

![](img/secant.gif)

![](img/secant.png)

![](img/secant_iter_debug.gif)


## Milestone November 27, 2017
![](img/LinkedList.gif)

![](img/splitbvh.png)

## Milestone December 4, 2017
![](img/internal.png)

## Milestone December 11, 2017
TODO: add ambient occlusion and design snazzy scenes

## References
1. O. Gourmel, A. Pajot, M. Paulin, L. Barthe, P. Poulin. [Fitted BVH for Fast Raytracing of Metaballs](http://www.ligum.umontreal.ca/Gourmel-2010-FBVH/Gourmel-2010-FBVH.pdf). EUROGRAPHICS, 2009
2. O. Gourmel, A. Pajot, L. Barthe, M. Paulin, P. Poulin. [BVH for efficient raytracing of dynamic metaballs on GPU](https://dl.acm.org/citation.cfm?id=1598041). SIGGRAPH, 2009
3. Y. Kanamori, Z. Szego, T. Nishita. [GPU-based Fast Ray Casting for a Large Number of Metaballs](http://kanamori.cs.tsukuba.ac.jp/projects/metaball/eg08_metaballs.pdf). EUROGRAPHICS, 2008
4. L. Szecsi, D. Illes. [Real-Time Metaball Ray Casting with Fragment Lists](http://cg.iit.bme.hu/~szecsi/cikkek/metaball12/meta.pdf). EUROGRAPHICS, 2012
5. Jay McKee, AMD, [Real-Time Concurrent Linked List Construction on the GPU](http://developer.amd.com/wordpress/media/2013/06/2041_final.pdf)