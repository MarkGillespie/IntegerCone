#pragma once
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

struct MeshTraits : public OpenMesh::DefaultTraits {
  typedef OpenMesh::Vec3d Point;
  typedef OpenMesh::Vec3d Normal;
  typedef OpenMesh::Vec2d TexCoord2D;
  VertexAttributes(OpenMesh::Attributes::Status | OpenMesh::Attributes::Normal);
  FaceAttributes(OpenMesh::Attributes::Status | OpenMesh::Attributes::Normal);
  EdgeAttributes(OpenMesh::Attributes::Status);
  HalfedgeAttributes(OpenMesh::Attributes::Status);
};
typedef OpenMesh::TriMesh_ArrayKernelT<MeshTraits> Mesh;

class MeshTools {
public:
  static bool ReadMesh(Mesh &mesh, const std::string &filename);
  static bool ReadOBJ(Mesh &mesh, const std::string &filename);
  // static bool ReadOFF(Mesh & mesh, const std::string & filename);
  static bool WriteMesh(const Mesh &mesh, const std::string &filename,
                        const std::streamsize &precision = 6);
  static bool WriteOBJ(const Mesh &mesh, const std::string &filename,
                       const std::streamsize &precision = 6);
  static bool WriteVertexTextutedOBJ(const Mesh &mesh,
                                     const std::string &filename,
                                     const std::streamsize &precision = 6);
  static bool WriteFaceTextutedOBJ(const Mesh &mesh,
                                   const std::string &filename,
                                   const std::streamsize &precision = 6);
  static double Area(const Mesh &mesh);
  static double AverageEdgeLength(const Mesh &mesh);
  static bool HasBoundary(const Mesh &mesh);
  static bool HasOneComponent(const Mesh &mesh);
  static int Genus(const Mesh &mesh);
  static void BoundingBox(const Mesh &mesh, Mesh::Point &bmax,
                          Mesh::Point &bmin);
  static void Reassign(const Mesh &mesh1, Mesh &mesh2);
};
