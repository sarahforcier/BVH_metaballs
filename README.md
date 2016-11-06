# Final Project

[CIS 565](https://cis565-fall-2016.github.io/): GPU Programming and Architecture

University of Pennsylvania

Fall 2016

The final project gives you an opportunity to embark on a large GPU programming endeavor of your choice.  You are free to select an area in graphics, GPU computing, or both.  You can reproduce the results of recent research, add a novel extension to existing work, fulfill a real need in the community, or implement something completely original.

Expect this to be 2-3x more work than the hardest project this semester.  The best projects will require a time commitment of at least 100 hours per student.  It will be worth it.

## Guidelines

* Form teams of two.  Each team member will receive the same grade.  Teams of three will be considered, but will be expected to build something truly amazing like [this](https://github.com/ishaan13/PhotonMapper).  Teams of one are not encouraged, but can be accommodated on a case-by-case basis.
* Use GitHub.  We encourage, but do not require, you to develop in a public repo as open source.
* Programming language, graphics and compute APIs, and target platforms are up to you.
* You are allowed to use existing code and libraries.  These should be prominently credited and your code should be easily identifiable.  Be incredibly clear about what is your work and what is not.

## Project Ideas

### WebGL

#### Extend the [WebGL 2 Samples Pack](https://github.com/WebGLSamples/WebGL2Samples)

See https://github.com/WebGLSamples/WebGL2Samples/issues/91

#### WebGL vs. OpenGL ES Mobile Power Usage

On mobile, how does the power usage of JavaScript/WebGL compare to C++/OpenGL ES 2 (or Java/ES or Objective C/ES)?  For example, if the same app is developed in WebGL and OpenGL ES, which one drains the battery first - and why?  What type of benchmarks need to be developed, e.g., CPU bound, GPU compute bound, GPU memory bound, etc.?

#### WebGL for Graphics Education

Create a tutorial using WebGL that teaches either basic _GPU architecture_ (parallelism, branches, multithreading, SIMD, etc.) or _Tile-Based Architectures_ like those used in mobile GPUs.  For inspiration, see [Making WebGL Dance](http://acko.net/files/fullfrontal/fullfrontal/webglmath/online.html) by Steven Wittens.

#### WebGL-Next

Propose and prototype a next-generation WebGL API.  See [Thoughts about a WebGL-Next](http://floooh.github.io/2016/08/13/webgl-next.html).

### glTF

Disclosure: I am one of the glTF spec editors.

![](images/gltf.png)

#### Various Projects

glTF, the GL Transmission Format (glTF), is a new runtime asset delivery format for WebGL.  It needs an ecosystem of tools, documentation, and extensions.  See [these ideas](https://github.com/KhronosGroup/glTF/issues/456).  Ideas:

* Contribute to [assimp](http://www.assimp.org/), [notes](https://github.com/KhronosGroup/glTF/issues/726#issuecomment-249858688).
* Contribute to the [Blender glTF exporter](https://github.com/Kupoman/blendergltf).
* Write a detailed size/performance analysis of glTF compared to other 3D model formats

#### GPU-accelerated prebaked AO

GPU-accelerate the [prebaking AO stage](http://cesiumjs.org/2016/08/08/ambient-occlusion/) in the glTF Pipeline.  See the [AO Roadmap](https://github.com/AnalyticalGraphicsInc/gltf-pipeline/issues/125).

Use a uniform grid; see [A Memory Efficient Uniform Grid Build Process for GPUs](http://jcgt.org/published/0005/03/04/).

### Vulkan

#### renderdoc

Contribute to [renderdoc](https://github.com/baldurk/renderdoc/wiki/Vulkan).

#### In Defense of Batching

Does batching still help with Vulkan?  If so, when it is worth it?

#### Domain-specific shading languages

Create a domain-specific shading language targeting SPIR-V.  Perhaps a language tuned for CSG, voxels, or ray marching.

#### Utility library

Implement a new abstraction layer using Vulkan that is higher level than Vulkan, but lower level than, for example, Unity.

#### Multithreaded Engine

Prototype a small engine with multithreading for LOD and culling.

#### Tutorial Series

Write a _Vulkan for OpenGL developers_ tutorial series.

### CUDA / GPU Computing

#### Point Cloud Processing

GPU accelerate filters in [PDAL](http://www.pdal.io/) or [PCL](http://pointclouds.org/).

##### Autonomous Cars

GPU accelerate parts of [autonomous cars](http://www.nvidia.com/object/drive-automotive-technology.html).

#### Alternative Rendering Pipelines

Use CUDA or compute shaders to build a custom or hybrid rendering pipeline, e.g., instead of creating a rasterization pipeline for triangles, create a graphics pipeline optimizations for [points](http://graphics.ucsd.edu/~matthias/Papers/Surfels.pdf), [voxels](https://research.nvidia.com/publication/voxelpipe-programmable-pipeline-3d-voxelization), or [vectors](http://w3.impa.br/~diego/projects/GanEtAl14/).

[![](images/points.png)](http://graphics.ucsd.edu/~matthias/Papers/Surfels.pdf)

Surfels: Surface Elements as Rendering Primitives by Hanspeter Pfister et al.

[![](images/voxels.png)](https://research.nvidia.com/publication/voxelpipe-programmable-pipeline-3d-voxelization)

VoxelPipe: A Programmable Pipeline for 3D Voxelization by Jacopo Pantaleoni.

[![](images/vectors.png)](http://w3.impa.br/~diego/projects/GanEtAl14/)

Massively-Parallel Vector Graphics by Francisco Ganacim.

### VR

#### Efficient multiview rendering

For example, see [Multiview Rendering for VR](https://community.arm.com/events/1272) and [Efficient Stereoscopic Rendering of Building Information Models (BIM)](http://jcgt.org/published/0005/03/01/).

### Other

#### Google Tango

Anything that runs in real-time using the GPU and the [Tango API](https://developers.google.com/tango/).

#### Embedded Systems

Anything using the NVIDIA [Jetson TK1](http://www.nvidia.com/object/jetson-tk1-embedded-dev-kit.html).

### Previous Semesters

For inspiration, browse the CIS 565 final projects from previous semesters: [Fall 2015](http://cis565-fall-2015.github.io/studentwork.html), [Fall 2014](http://cis565-fall-2014.github.io/studentwork.html), [Fall 2013](http://cis565-fall-2013.github.io/studentwork.html), [Fall 2012](http://cis565-fall-2012.github.io/studentwork.html), [Spring 2012](http://cis565-spring-2012.github.com/studentwork.html), and [Spring 2011](http://www.seas.upenn.edu/~cis565/StudentWork-2011S.htm).

A guideline is that your project should be better than last semester's projects; that is how we move the field forward.

#### Selected Projects

* **Fall 2015**
   * [Forward+ Renderer using OpenGL/Compute Shaders](https://github.com/bcrusco/Forward-Plus-Renderer) by Bradley Crusco and Megan Moore
   * [WebGL Fragment Shader Profiler](https://github.com/terrynsun/WebGL-Fragment-Shader-Profiler) by Sally Kong and Terry Sun
   * [GPU Cloth with OpenGL Compute Shaders](https://github.com/likangning93/GPU_cloth) by Gary Li
* **Fall 2014**
   * [Bidirectional Path Tracer in CUDA](https://github.com/paula18/Photon-Mapping) by Paula Huelin Merino and Robbie Cassidy
   * [GPU-Accelerated Dynamic Fracture in the Browser with WebCL](https://github.com/kainino0x/cis565final) by Kai Ninomiya and Jiatong He
   * [Uniform grid and kd-tree in CUDA](https://github.com/jeremynewlin/Accel) by Jeremy Newlin and Danny Rerucha
* **Fall 2013**
   * [Surface Mesh Reconstruction from RGBD Images](https://github.com/cboots/RGBD-to-Mesh) by Collin Boots and Dalton Banks
   * [Sparse Voxel Octree](https://github.com/otaku690/SparseVoxelOctree) by Cheng-Tso Lin
   * [Terrain tessellation](https://github.com/mchen15/Gaia) by Mikey Chen and Vimanyu Jain
   * [GPU Photon Mapper](https://github.com/ishaan13/PhotonMapper) by Ishaan Singh, Yingting Xiao, and Xiaoyan Zhu
* **Fall 2012**
   * [Non-photorealistic Rendering](http://gpuprojects.blogspot.com/) by Kong Ma
   * [Procedural Terrain](http://gputerrain.blogspot.com/) by Tiju Thomas
   * [KD Trees on the GPU](http://www.colorseffectscode.com/Projects/FinalProject.html) by Zakiuddin Shehzan Mohammed
* **Spring 2012**
   * [Single Pass Order Independent Transparency](http://gamerendering.blogspot.com/) by Sean Lilley
   * [GPU-Accelerated Logo Detection](http://erickboke.blogspot.com/) by Yu Luo
   * [GPU-Accelerated Simplified General Perturbation No. 4 (SGP4) Model](http://www.matthewahn.com/blog/sgp4-14558-satellites-in-orbit/) by Matthew Ahn
* **Spring 2011**
   * [Fast Pedestrian Recognition on the GPU](http://spevis.blogspot.com/) by Fan Deng
   * [Screen Space Fluid Rendering](http://fastfluids.blogspot.com/) by Terry Kaleas
   * [Deferred Shader with Screen Space Classification](http://smt565.blogspot.com/) by Sean Thomas

### Conferences and Journals

Browse these for ideas galore!

* [Journal of Computer Graphics Techniques](http://jcgt.org/read.html)
* [Advances in Real-Time Rendering](http://advances.realtimerendering.com/) SIGGRAPH courses
* [Ke-Sen Huang's conference pages](http://kesen.realtimerendering.com/) - papers from SIGGRAPH, Eurographics, I3D, and elsewhere
* [Real-Time Rendering Portal](http://www.realtimerendering.com/portal.html) and [WebGL Resources](http://www.realtimerendering.com/webgl.html) - links to an amazing amount of content

## Timeline

### **Wednesday 11/16** - Project Pitch

Sign up for a time slot ASAP.

Your project pitch is a 15-minute meeting with Patrick, Shrek, and Gary and a write-up no longer than one page that includes an overview of your approach with specific goals.  First, focus on why there is a need for your project.  Then describe what exactly you are going to do.  In addition to your write-up, provide supplemental figures, images, or videos.

Think of your pitch as if you are trying to get a startup funded, convincing your company to start a new project, or responding to a grant.  For an example, see [Forward+ Renderer using OpenGL/Compute Shaders](https://github.com/bcrusco/Forward-Plus-Renderer/blob/master/Final%20Project%20Pitch.pdf) by Bradley Crusco and Megan Moore

**Before the meeting**:
* Email your one page pitch and any supplemental material to Patrick, Shrek, and Gary by end of Tuesday 11/15.

**After the meeting**:
* Push your pitch to a new GitHub repo for your project
* Email the repo link to  cis-565-fall-2016@googlegroups.com (if the project is open source)

### **Monday 11/21** - Milestone 1

Your first presentation should be 7-10 minutes long.  Present your work-in-progress.  Your presentation should include a few slides, plus videos, screenshots, or demos if possible.  Be sure to
* Demonstrate working code (videos and screenshots are OK; it doesn’t have to be live).
* Provide a roadmap with future milestones (11/28 and 12/12), and the final result (date TBA).  Set goals for each.

See the Cesium [Presenter's Guide](https://github.com/AnalyticalGraphicsInc/cesium/tree/master/Documentation/Contributors/PresentersGuide#presenters-guide) for tips on presenting.  Be sure to present as a team; for a great example, see http://www.youtube.com/watch?v=OTCuYzAw31Y

After class, push your presentation to your GitHub repo.

### **Monday 11/28** - Milestone 2

A 5-7 minute presentation on your progress over the past week.  Demonstrate how you reached or exceeded the goals for this milestone.  If you didn't reach them, explain why.  Remind us of your upcoming milestones.

After class, push your presentation to your GitHub repo.

### **Monday 12/12** - Milestone 3

Same format as Milestone 2.

### **Date TBA** - Final Presentation

10-minute final presentation and demo.

By midnight the day before:
* Push the following to GitHub
   * Final presentation slides
   * Final code - should be clean, documented, and tested
* A detailed README.md including:
   * Name of your project
   * Your names and links to your website/LinkedIn/twitter/whatever
   * Choice screenshots including debug views
   * Link to demo if possible.  WebGL demos should include your names and a link back to your github repo
   * Overview of technique and links to references
   * Link to video
      * Two to four minutes in length to show off your work.  Your video should complement your paper and clarify anything that is difficult to describe in just words and images.  Your video should both make us excited about your work and help us if we were to implement it
   * Detailed performance analysis
   * Install and build instructions

As always, see [How to make an attractive GitHub repo](https://github.com/pjcozzi/Articles/blob/master/CIS565/GitHubRepo/README.md#how-to-make-an-attractive-github-repo) for tips on your README.md.

## Grading

The final project is worth 50% of your final grade.  The breakdown is:

* Milestone 1: 25%
* Milestone 2: 25%
* Milestone 3: 25%
* Final Presentation: 25%
