#pragma once

#ifdef GC_HAVE_MKL
#define EIGEN_USE_BLAS
#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif
#include <Eigen/PardisoSupport>
#endif


#include "geometrycentral/numerical/linear_algebra_types.h"
#include "geometrycentral/numerical/linear_algebra_utilities.h"
#include "geometrycentral/numerical/linear_solvers.h"

#ifdef GC_HAVE_SUITESPARSE
#include "geometrycentral/numerical/suitesparse_utilities.h"
#endif

using namespace geometrycentral;

enum class SolverType {
    eigenLDLT,
    eigenDenseLDLT,
    eigenLU,
    eigenQR,
    pardisoLDLT,
    pardisoLU,
    suitesparseLLT,
    suitesparseLU,
    suitesparseQR
};

template <typename T>
class ConfigurableSolver {
  public:
    // blank constructor
    ConfigurableSolver();

    // constructor
    ConfigurableSolver(const SolverType& type, SparseMatrix<T>& A);

    void setSolverType(const SolverType& type);

    // destructor
    ~ConfigurableSolver();

    // computes symbolic factorization
    void computeSymbolic(SparseMatrix<T>& A, const std::vector<int>& perm = {});

    // computes numeric factorization
    void computeNumeric(SparseMatrix<T>& A);

    // computes factorization
    void compute(SparseMatrix<T>& A);

    // solve
    void solve(DenseMatrix<T>& x, const DenseMatrix<T>& b,
               bool transposed = false);
    DenseMatrix<T> solve(const DenseMatrix<T>& b, bool transposed = false);


    // clears factorization
    void clear();

    // members
    bool validSymbolic;
    bool validNumeric;

    const SolverType& solverType;
    const std::string& solverName;

  protected:
    SolverType internalSolverType;
    std::string internalSolverName;

    // initializes pardiso parameters
    void initPardisoParameters();

    // members
    Eigen::SimplicialLDLT<SparseMatrix<T>> eigenLDLT;
    Eigen::LDLT<DenseMatrix<T>> eigenDenseLDLT;
    Eigen::SparseLU<SparseMatrix<T>> eigenLU;
    Eigen::SparseQR<SparseMatrix<T>, Eigen::COLAMDOrdering<int>> eigenQR;
#ifdef GC_HAVE_MKL
    Eigen::PardisoLDLT<SparseMatrix<T>> pardisoLDLT;
    Eigen::PardisoLU<SparseMatrix<T>> pardisoLU;
#endif

    // suitesparse if available
#ifdef GC_HAVE_SUITESPARSE
    std::unique_ptr<LinearSolver<T>> suitesparseQR;
    std::unique_ptr<SquareSolver<T>> suitesparseLU;
    std::unique_ptr<PositiveDefiniteSolver<T>> suitesparseLLT;
#endif

    // LDL metadata
    std::vector<int> Lp, Parent, Lnz, Li, Pattern, Flag, P, Pinv;
    std::vector<T> Lx, D, Y;

    int rows = -1, cols = -1;
};
