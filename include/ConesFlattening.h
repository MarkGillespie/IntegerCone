#include "MeshDefinition.h"
#include "Opt/Types.h"

namespace ConesFlattening {
void initCoef(const Mesh &mesh, int lp_, double sigma_);
void geneCone(VectorX &conesK, VectorX &phi, VectorX &MA, int coneMax,
              double &factor, int &iter, double &gamma_);
} // namespace ConesFlattening
