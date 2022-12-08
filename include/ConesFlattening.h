#pragma once

#include "geometrycentral/surface/intrinsic_geometry_interface.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"

#include "Opt/Types.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace ConesFlattening {
void geneCone(VectorX &conesK, VectorX &phi, VectorX &MA, int coneMax,
              double &factor, int &iter, double &gamma_);

void initCoef(ManifoldSurfaceMesh &mesh, IntrinsicGeometryInterface &geom,
              int lp_, double sigma_);
} // namespace ConesFlattening
