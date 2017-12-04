#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadEnvironment();
    int loadGeom(string objectid);
	int loadMetaballs(int num);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
	std::vector<Metaball> metaballs;
    std::vector<Material> materials;
    std::vector<Texture> environmentMap;
    RenderState state;
};
