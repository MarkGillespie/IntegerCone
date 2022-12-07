#pragma once

#include "MeshDefinition.h"
#include "Opt/Types.h"

class ConeGene {
public:
  ConeGene();
  ~ConeGene();

  void initCoef(const Mesh &mesh, int lp_, double sigma_);
  void geneCone(VectorX &conesK, VectorX &phi, VectorX &MA, int coneMax,
                double &factor, int &iter);

private:
};
