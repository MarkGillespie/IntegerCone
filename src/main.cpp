#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "ConesFlattening.h"
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

#include "args/args.hxx"

// Polyscope visualization handle, to quickly add data to the surface
polyscope::SurfaceMesh *psMesh;

void saveCones(const VectorX &conesK, std::string conesPath,
               double eps = 1e-3) {
  std::ofstream conesFile(conesPath);
  if (conesFile.fail()) {
    std::cout << "Open " << conesPath << "failed\n";
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < conesK.size(); ++i) {
    if (conesK[i] > -eps && conesK[i] < eps)
      continue;
    conesFile << i + 1 << " " << conesK[i] << std::endl;
  }
  conesFile.close();
}

void saveU(const VectorX &u, std::string uPath) {
  std::ofstream uFile(uPath);
  if (uFile.fail()) {
    std::cout << "Open " << uPath << "failed\n";
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < u.size(); ++i) {
    uFile << u[i] << std::endl;
  }
  uFile.close();
}

void saveInfo(double factor, double t, double l2norm, int iter, double gamma,
              std::string infoPath) {
  std::ofstream infoFile(infoPath);
  if (infoFile.fail()) {
    std::cout << "Open " << infoPath << "failed\n";
    exit(EXIT_FAILURE);
  }

  infoFile << "u L2 norm : \t" << l2norm << std::endl;
  infoFile << "factor : \t" << factor << std::endl;
  infoFile << "iter : \t" << iter << std::endl;
  infoFile << "gamma : \t" << gamma << std::endl;
  infoFile << "cost time : \t" << t << " s\n";
  infoFile.close();
}

// https://github.com/vijaiaeroastro/openMeshPolyscope
// This function unpacks openmesh data to a simpler cpp containers suitable for
// polyscope
std::pair<std::vector<std::array<double, 3>>,
          std::vector<std::array<unsigned int, 3>>>
unpack_open_mesh(Mesh &reference_mesh) {
  std::vector<std::array<double, 3>> vertexPositions;
  std::vector<std::array<unsigned int, 3>> meshConnectivity;
  for (auto v : reference_mesh.vertices()) {
    auto current_point = reference_mesh.point(v);
    std::array<double, 3> current_array_point;
    current_array_point[0] = current_point[0];
    current_array_point[1] = current_point[1];
    current_array_point[2] = current_point[2];
    vertexPositions.push_back(current_array_point);
  }
  for (auto f : reference_mesh.faces()) {
    std::array<unsigned int, 3> current_triangle;
    unsigned int temp_index = 0;
    for (auto fv_it = reference_mesh.fv_iter(f); fv_it.is_valid(); ++fv_it) {
      current_triangle[temp_index] = fv_it->idx();
      temp_index++;
    }
    meshConnectivity.push_back(current_triangle);
    temp_index = 0;
  }
  return std::make_pair(vertexPositions, meshConnectivity);
}

// A user-defined callback, for creating control panels (etc)
// Use ImGUI commands to build whatever you want here, see
// https://github.com/ocornut/imgui/blob/master/imgui.h
void myCallback() {

  // ImGui::InputInt("nSites", &nSites);
  // ImGui::SliderFloat("tCoef", &tCoef, 0.0001, 1);
  // ImGui::Checkbox("useDelaunay", &useDelaunay);

  // if (ImGui::Button("Trivial Connection")) {
  //   findParallelField();
  // }
}

int main(int argc, const char *argv[]) {

  // Configure the argument parser
  args::ArgumentParser parser("Integer cones");
  args::Positional<std::string> meshFilename(
      parser, "input_obj_file", "Mesh to be processed (required).");
  args::Positional<std::string> outputBase(parser, "output_file_basename",
                                           {"scaleFactors"});
  args::ValueFlag<double> normBound(parser, "double", "normBound",
                                    {"normBound"});
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

  // Parse args
  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help &h) {
    std::cout << parser;
    return 0;
  } catch (args::ParseError &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  std::string objPath = args::get(meshFilename);
  std::string outPath = args::get(outputBase);
  double sigma = (normBound) ? args::get(normBound) : 0.15;
  int cN = 8;
  int iter = 0;
  double factor = 1;
  double gamma;
  bool seamless = true;

  Mesh mesh;

  std::cout << "load mesh from " << objPath << std::endl;
  if (!MeshTools::ReadMesh(mesh, objPath)) {
    std::cout << "load failed!\n";
    exit(EXIT_FAILURE);
  }

  // Initialize polyscope
  polyscope::init();

  // Set the callback function
  polyscope::state::userCallback = myCallback;

  // // Load mesh
  // std::tie(mesh, geom) = readManifoldSurfaceMesh(filename);
  auto unpackOpenMesh = unpack_open_mesh(mesh);

  // Register the mesh with polyscope
  // psMesh = polyscope::registerSurfaceMesh("mesh", geom->vertexPositions,
  //                                         mesh->getFaceVertexList(),
  //                                         polyscopePermutations(*mesh));
  psMesh = polyscope::registerSurfaceMesh("mesh", unpackOpenMesh.first,
                                          unpackOpenMesh.second);
  polyscope::show();

  VectorX conesK, u, A;

  clock_t start, end;
  start = clock();

  ConesFlattening::initCoef(mesh, 2, sigma);
  ConesFlattening::geneCone(conesK, u, A, cN, factor, iter, gamma);

  VectorX u_l2 = u;
  end = clock();

  double costTime = (double)(end - start) / CLOCKS_PER_SEC;
  std::cout << "Cost time : " << costTime << " s\n";

  std::string conesPath = outPath + "-cones.txt";
  std::string uPath = outPath + "-u.txt";
  std::string infoPath = outPath + "-info.txt";

  saveCones(conesK, conesPath);
  saveU(u, uPath);
  saveInfo(factor, costTime, sqrt(A.transpose() * u_l2.cwiseAbs2()), iter,
           gamma, infoPath);

  psMesh->addVertexScalarQuantity("K", conesK);
  psMesh->addVertexScalarQuantity("u", u);
  polyscope::show();

  return 0;
}
