#include "ConesFlattening.h"
#include "Opt/AlgOpt.h"
#include <fstream>
#include <omp.h>

namespace ConesFlattening {
ColMajorSparseMatrix L, A, P, PP;
VectorX K;

int lp = -1;
double sigma;

SparseMatrix<double> lpMassMatrix(ManifoldSurfaceMesh &mesh,
                                  IntrinsicGeometryInterface &geom,
                                  int lp = -1) {
  SparseMatrix<double> A(mesh.nVertices(), mesh.nVertices());
  if (lp < 0) {
    A.setIdentity();
    return A;
  }

  geom.requireVertexDualAreas();
  geom.requireVertexIndices();
  const VertexData<size_t> &vIdx = geom.vertexIndices;
  const VertexData<double> &vArea = geom.vertexDualAreas;

  double surfaceArea = 0;
  for (Vertex v : mesh.vertices())
    surfaceArea += vArea[v];
  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(mesh.nVertices());

  for (Vertex v : mesh.vertices()) {
    trips.emplace_back(vIdx[v], vIdx[v], pow(vArea[v] / surfaceArea, 1. / lp));
  }
  geom.unrequireVertexIndices();
  geom.unrequireVertexDualAreas();
  A.setFromTriplets(trips.begin(), trips.end());

  return A;
}

SparseMatrix<double> interiorMat(ManifoldSurfaceMesh &mesh,
                                 IntrinsicGeometryInterface &geom) {
  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(mesh.nVertices());

  geom.requireVertexIndices();
  const VertexData<size_t> &vIdx = geom.vertexIndices;

  int rowId = 0;
  for (Vertex v : mesh.vertices()) {
    if (v.isBoundary())
      continue;
    trips.emplace_back(rowId, vIdx[v], 1);
    rowId++;
  }

  geom.unrequireVertexIndices();

  SparseMatrix<double> P(rowId, mesh.nVertices());
  P.setFromTriplets(trips.begin(), trips.end());
  return P;
}

SparseMatrix<double> interiorMatP(ManifoldSurfaceMesh &mesh,
                                  IntrinsicGeometryInterface &geom) {
  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(mesh.nVertices());

  geom.requireVertexIndices();
  const VertexData<size_t> &vIdx = geom.vertexIndices;

  int rowId = 0;
  for (Vertex v : mesh.vertices()) {
    if (v.isBoundary())
      continue;
    trips.emplace_back(rowId, vIdx[v], 1);
    rowId++;

    if (rowId + 1 == (int)mesh.nVertices()) {
      break;
    }
  }
  geom.unrequireVertexIndices();

  SparseMatrix<double> PP(rowId, mesh.nVertices());
  PP.setFromTriplets(trips.begin(), trips.end());
  return PP;
}

// Gaussian curvature at interior vertices, geodesic curvature at boundary
// vertices
VertexData<double> vertexCurvatures(ManifoldSurfaceMesh &mesh,
                                    IntrinsicGeometryInterface &geom) {
  VertexData<double> K(mesh);
  geom.requireVertexAngleSums();
  for (Vertex v : mesh.vertices()) {
    double flatSum = v.isBoundary() ? M_PI : 2 * M_PI;
    K[v] = flatSum - geom.vertexAngleSums[v];
  }
  geom.unrequireVertexAngleSums();
  return K;
}

void initCoef(ManifoldSurfaceMesh &mesh, IntrinsicGeometryInterface &geom,
              int lp_, double sigma_) {
  std::cout << "initializing with geometry-central" << std::endl;

  lp = lp_;
  sigma = sigma_;

  geom.requireCotanLaplacian();
  L = geom.cotanLaplacian;
  geom.unrequireCotanLaplacian();

  A = lpMassMatrix(mesh, geom, lp);
  K = vertexCurvatures(mesh, geom).raw();
  P = interiorMat(mesh, geom);
  PP = interiorMatP(mesh, geom);
}

struct pairCmpLess {
  inline bool operator()(std::pair<int, double> &x, std::pair<int, double> &y) {
    return fabs(x.second - M_PI_2 * round(x.second * M_2_PI)) <
           fabs(y.second - M_PI_2 * round(y.second * M_2_PI));
  }
};

void geneCone(VectorX &conesK, VectorX &phi, VectorX &MA, int coneMax,
              double &factorn, int &itern, double &gamma_) {
  ColMajorSparseMatrix LN = P * L * A.cwiseInverse() * P.transpose();
  VectorX KN = P * K;
  VectorX Aphi = VectorX::Zero(LN.cols());
  VectorX iphi(LN.cols());
  MA = A.diagonal();
  MA = MA.cwiseAbs2();

  // VectorX init_K = VectorX::Zero(LN.cols());
  // for (size_t i = 0; i < 20; i++)
  //{
  //	int id = 1 + (int)(LN.cols() * rand() / (RAND_MAX + 1.0));
  //	init_K[id] = M_PI_2;
  //}
  // for (size_t i = 0; i < 10; i++)
  //{
  //	int id = 1 + (int)(LN.cols() * rand() / (RAND_MAX + 1.0));
  //	init_K[id] = -M_PI_2;
  //}
  // for (size_t i = 0; i < 5; i++)
  //{
  //	int id = 1 + (int)(LN.cols() * rand() / (RAND_MAX + 1.0));
  //	init_K[id] = -M_PI;
  //}
  // Eigen::SimplicialLDLT<ColMajorSparseMatrix> solver;
  // solver.compute(PP*L*PP.transpose());
  ////Aphi = solver.solve(P*init_K - P * K);
  ////solver.compute(P*L*P.transpose());
  // iphi = solver.solve(PP*init_K - PP * K);
  // std::cout << "error ====  " << (PP*L*PP.transpose()*iphi - PP * init_K + PP
  // * K).norm() << std::endl; iphi = PP.transpose()*iphi; iphi = iphi -
  // (MA.cwiseProduct(iphi)).sum() / MA.sum() * VectorX::Ones(iphi.size());
  // std::cout << "init bound ==  " << sqrt(MA.transpose() * iphi.cwiseAbs2())
  // << std::endl;
  // ----- different initial ------
  // iphi = VectorX::Random(LN.cols());
  // iphi = VectorX::Ones(LN.cols());
  // iphi = iphi * 1e6 / iphi.norm();
  // Aphi = A * iphi;
  // std::ofstream uFile("init-u.txt");
  // if (uFile.fail())
  //{
  //	std::cout << "Open " << "init-u.txt" << "failed\n";
  //	exit(EXIT_FAILURE);
  //}

  // for (int i = 0; i < Aphi.size(); ++i)
  //{
  //	uFile << iphi[i] << std::endl;
  //}
  // uFile.close();

  // double eps = 1e-3;
  // std::ofstream conesFile("ran_cone-cones.txt");
  // if (conesFile.fail())
  //{
  //	std::cout << "Open " << "ran_cone-cones.txt" << "failed\n";
  //	exit(EXIT_FAILURE);
  //}

  // for (int i = 0; i < init_K.size(); ++i)
  //{
  //	if (init_K[i] > -eps && init_K[i] < eps) continue;
  //	conesFile << i + 1 << " " << init_K[i] << std::endl;
  //}
  // conesFile.close();

  // ----- different initial ------

  int para_num = 1;
  double gamma = 1e-2;
  double lm_eps = 0.12;
  volatile bool is_success = false;
  std::vector<int> cn(para_num, coneMax);
  std::vector<double> factor(para_num);
  std::vector<VectorX> s(para_num);
  for (int i = 0; i < para_num; i++) {
    factor[i] = 1.0 - 0.04 * i;
  }

  std::vector<OptCandidates *> candidate(para_num, NULL);
  std::vector<IntegerL0DR<false> *> IL(para_num, NULL);

  for (int i = 0; i < para_num; i++) {
    candidate[i] = new OptCandidates(P * L * P.transpose() * M_2_PI, MA,
                                     KN * M_2_PI, coneMax, sigma);
    // IL[i] = new IntegerL0DR<false>(2, LN * M_2_PI * factor[i] * sigma, KN *
    // M_2_PI, *candidate[i], is_success); s[i] = IL[i]->initOri(Aphi);
    // IL[i]->init(1e-2, 5e-7, 2e4, 0);
    // candidate[i]->factor = factor[i];
  }
  printf("Collect integer candidates\n");
  omp_set_num_threads(para_num);
  for (size_t iter = 0; iter < 8; iter++) {
    for (int i = 0; i < para_num; i++) {
      IL[i] = new IntegerL0DR<false>(2, LN * M_2_PI * factor[i] * sigma,
                                     KN * M_2_PI, *candidate[i], is_success,
                                     lm_eps);
      s[i] = IL[i]->initOri(Aphi);
      IL[i]->init(gamma, 5e-7, 2e4, 0);
    }
#pragma omp parallel for
    for (int i = 0; i < para_num; i++) {
      s[i] = IL[i]->runAll(s[i]);
    }
    int is_candidate_void = true;
    for (int i = 0; i < para_num; i++) {
      delete IL[i];
      if (candidate[i]->is_candidate_void == false) {
        iter = 10;
        break;
      }
    }
    gamma *= 0.1;
    lm_eps *= 1.5;
  }

  double umin, cmin, bmin = DBL_MAX;
  umin = DBL_MAX;
  cmin = DBL_MAX;

  int opt_i;
  for (int i = 0; i < para_num; i++) {
    if (candidate[i]->qualified() && candidate[i]->E_min < cmin) {
      opt_i = i;
      cmin = candidate[i]->E_min;
      bmin = candidate[i]->BB_min;
    }
    if (candidate[i]->qualified() && candidate[i]->E_min == cmin &&
        candidate[i]->BB_min < bmin) {
      opt_i = i;
      cmin = candidate[i]->E_min;
      bmin = candidate[i]->BB_min;
    }
  }
  if (!(cmin < DBL_MAX)) {
    for (int i = 0; i < para_num; i++) {
      if (candidate[i]->B_min < umin) {
        opt_i = i;
        umin = candidate[i]->B_min;
      }
    }
  }
  phi = P.transpose() * candidate[opt_i]->optOne();
  // conesK = L * phi + P.transpose() * K;
  conesK =
      P.transpose() * (P * L * P.transpose() * candidate[opt_i]->optOne() + KN);
  factorn = factor[opt_i];
  itern = candidate[opt_i]->opt_iter;
  gamma_ = gamma;
}
} // namespace ConesFlattening
