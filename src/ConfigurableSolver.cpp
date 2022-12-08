#include "ConfigurableSolver.h"

template <typename T>
ConfigurableSolver<T>::ConfigurableSolver()
    : validSymbolic(false), validNumeric(false), solverType(internalSolverType),
      solverName(internalSolverName) {
    initPardisoParameters();
}

template <typename T>
ConfigurableSolver<T>::ConfigurableSolver(const SolverType& type,
                                          SparseMatrix<T>& A)
    : validSymbolic(false), validNumeric(false), solverType(internalSolverType),
      solverName(internalSolverName) {
    setSolverType(type);
    initPardisoParameters();
    compute(A);
}

template <typename T>
ConfigurableSolver<T>::~ConfigurableSolver() {
    clear();
}

template <typename T>
void ConfigurableSolver<T>::setSolverType(const SolverType& type) {
    if (internalSolverType != type) {
        internalSolverType = type;

        switch (type) {
        case SolverType::eigenLDLT:
            internalSolverName = "Eigen LDLT";
            break;
        case SolverType::eigenDenseLDLT:
            internalSolverName = "Eigen Dense LDLT";
            break;
        case SolverType::eigenLU:
            internalSolverName = "Eigen LU";
            break;
        case SolverType::eigenQR:
            internalSolverName = "Eigen QR";
            break;
#ifdef GC_HAVE_MKL
        case SolverType::pardisoLDLT:
            internalSolverName = "Pardiso Supernodal LDLT";
            break;
        case SolverType::pardisoLU:
            internalSolverName = "Pardiso LU";
            break;
#else
        case SolverType::pardisoLDLT:
        case SolverType::pardisoLU:
            throw std::invalid_argument(
                "Cannot use Pardiso solvers since CMake was unable to find "
                "Intel MKL");
            break;
#endif
#ifdef GC_HAVE_SUITESPARSE
        case SolverType::suitesparseQR:
            internalSolverName = "suitesparseQR";
            break;
        case SolverType::suitesparseLU:
            internalSolverName = "suitesparseLU";
            break;
        case SolverType::suitesparseLLT:
            internalSolverName = "suitesparseLLT";
            break;
#else
        case SolverType::suitesparseQR:
        case SolverType::suitesparseLU:
        case SolverType::suitesparseLLT:
            throw std::invalid_argument(
                "Cannot use Suitesparse solvers since CMake was unable to find "
                "Suitesparse");
            break;

#endif
        }
        clear();
    }
}

template <typename T>
void ConfigurableSolver<T>::initPardisoParameters() {
#ifdef GC_HAVE_MKL
    // clang-format off
  	pardisoLDLT.pardisoParameterArray()[1]  = 0; // minimum degree ordering
    pardisoLDLT.pardisoParameterArray()[7]  = 0; // default 2 iterative refinement steps
    pardisoLDLT.pardisoParameterArray()[9]  = 8; // pivot perturbation
    pardisoLDLT.pardisoParameterArray()[10] = 0; // scale entries close to diagonal
    pardisoLDLT.pardisoParameterArray()[12] = 0; // weighted matching
    pardisoLDLT.pardisoParameterArray()[20] = 3; // 2x2 pivoting
    pardisoLDLT.pardisoParameterArray()[23] = 0; // split forward and backward solve // TODO: setting to 10 causes solution to explode
    pardisoLDLT.pardisoParameterArray()[24] = 0; // sequential forward and backward solve
  	pardisoLDLT.pardisoParameterArray()[27] = 0; // double precision
    pardisoLDLT.pardisoParameterArray()[30] = 0; // sparse right hand side
    pardisoLDLT.pardisoParameterArray()[38] = 0; // low rank update
    pardisoLDLT.pardisoParameterArray()[1]  = 0; // minimum degree ordering

    // Intel recommends pushing 0 diagonal elements to end of matrix
    // For KKT systems, its recommended to set 10 & 12 to 1 - Increases number of nonzeros, speeds up symbolic but slows down numeric & backsubstitution
    // But, pivot perturbation of 1e-8 with 1x1 and 2x2 seems to provide sufficient accuracy (iterative refinement is set to 0 with this option)
    // Low rank updates and sparse right hand sides cannot be used together; 23 needs to be set to 10 for updates
    // Sparse backsubstitution will not be any faster than dense backsubstitution since solution is dense
    // clang-format on

#endif
}

template <typename T>
void ConfigurableSolver<T>::computeSymbolic(SparseMatrix<T>& E,
                                            const std::vector<int>& perm) {
    rows = E.rows();
    cols = E.cols();

    Eigen::ComputationInfo info;
    std::string solverName;

    switch (solverType) {
    case SolverType::eigenLDLT:
        eigenLDLT.analyzePattern(E);
        info = eigenLDLT.info();
        break;
    case SolverType::eigenDenseLDLT:
        eigenDenseLDLT.compute(DenseMatrix<T>(E));
        info = eigenDenseLDLT.info();
        break;
    case SolverType::eigenLU:
        eigenLU.analyzePattern(E);
        info = eigenLU.info();
        break;
    case SolverType::eigenQR:
        eigenQR.analyzePattern(E);
        info = eigenQR.info();
        break;
#ifdef GC_HAVE_MKL
    case SolverType::pardisoLDLT:
        if (perm.size() > 0) {
            int n = (int)perm.size();
            Eigen::VectorXi EPerm(n);
            for (int i = 0; i < n; i++) EPerm(i) = perm[i];

            // pardisoLDLT.pardisoParameterArray()[4] = 1;
            // pardisoLDLT.setPerm(EPerm);

            std::cout << "setting a permutation not supported yet" << std::endl;
            pardisoLDLT.pardisoParameterArray()[4] = 0;

        } else {
            pardisoLDLT.pardisoParameterArray()[4] = 0;
        }

        pardisoLDLT.analyzePattern(E);
        info = pardisoLDLT.info();
        break;
    case SolverType::pardisoLU:
        pardisoLU.analyzePattern(E);
        info = pardisoLU.info();
        break;
#else
    case SolverType::pardisoLDLT:
    case SolverType::pardisoLU:
        throw std::invalid_argument(
            "cannot compute symbolic factorization with Pardiso since it was "
            "not found during compilation");
        break;
#endif
#ifdef GC_HAVE_SUITESPARSE
    case SolverType::suitesparseLLT:
    case SolverType::suitesparseLU:
    case SolverType::suitesparseQR:
        // TODO: suitesparse symbolic factorization
        info = Eigen::Success;
        break;
#else
    case SolverType::suitesparseLLT:
    case SolverType::suitesparseLU:
    case SolverType::suitesparseQR:
        throw std::invalid_argument(
            "cannot compute symbolic factorization with "
            "Suitesparse since it was not found during compilation");
        break;
#endif
    }


    if (info == Eigen::Success)
        validSymbolic = true;
    else
        std::cout << "symbolic factorization is invalid" << std::endl;
    validNumeric = false;
}

template <typename T>
void ConfigurableSolver<T>::computeNumeric(SparseMatrix<T>& E) {
    if (!validSymbolic) {
        computeSymbolic(E);
    }
    Eigen::ComputationInfo info;
    std::string solverName;

    switch (solverType) {
    case SolverType::eigenLDLT:
        eigenLDLT.factorize(E);
        info = eigenLDLT.info();
        break;
    case SolverType::eigenDenseLDLT:
        // already computed in computeSymbolic
        info = eigenDenseLDLT.info();
        break;
    case SolverType::eigenLU:
        eigenLU.factorize(E);
        info = eigenLU.info();
        break;
    case SolverType::eigenQR:
        eigenQR.factorize(E);
        info = eigenQR.info();
        break;
#ifdef GC_HAVE_MKL
    case SolverType::pardisoLDLT:
        pardisoLDLT.factorize(E);
        info = pardisoLDLT.info();
        break;
    case SolverType::pardisoLU:
        pardisoLU.factorize(E);
        info = pardisoLU.info();
        break;
#else
    case SolverType::pardisoLDLT:
    case SolverType::pardisoLU:
        throw std::invalid_argument(
            "cannot compute numeric factorization with Pardiso since it "
            "was not found during compilation");
        break;
#endif
#ifdef GC_HAVE_SUITESPARSE
    case SolverType::suitesparseLLT:
        suitesparseLLT.reset(new PositiveDefiniteSolver<T>(E));
        // TODO: get info from standard GC solvers
        info = Eigen::Success;
        break;
    case SolverType::suitesparseLU:
        suitesparseLU.reset(new SquareSolver<T>(E));
        // TODO: get info from standard GC solvers
        info = Eigen::Success;
        break;
    case SolverType::suitesparseQR:
        suitesparseQR.reset(new Solver<T>(E));
        // TODO: get info from standard GC solvers
        info = Eigen::Success;
        break;
#else
    case SolverType::suitesparseLLT:
    case SolverType::suitesparseLU:
    case SolverType::suitesparseQR:
        throw std::invalid_argument(
            "cannot compute numeric factorization with Suitesparse since "
            "it was not found during compilation");
        break;
#endif
    }

    if (info == Eigen::Success)
        validNumeric = true;
    else
        std::cout << "numeric factorization is invalid" << std::endl;
}

template <typename T>
void ConfigurableSolver<T>::compute(SparseMatrix<T>& A) {
    computeSymbolic(A);
    computeNumeric(A);
}

template <typename T>
void ConfigurableSolver<T>::solve(DenseMatrix<T>& x, const DenseMatrix<T>& b,
                                  bool transposed) {

    GC_SAFETY_ASSERT(b.rows() == rows,
                     "right hand side has " + std::to_string(b.rows()) +
                         " but matrix has " + std::to_string(rows) + " rows");
    if (validNumeric) {
        switch (solverType) {
        case SolverType::eigenLDLT:
            x = eigenLDLT.solve(b);
            break;
        case SolverType::eigenDenseLDLT:
            x = eigenDenseLDLT.solve(b);
            break;
        case SolverType::eigenLU:
            x = eigenLU.solve(b);
            break;
        case SolverType::eigenQR:
            x = eigenQR.solve(b);
            break;
#ifdef GC_HAVE_MKL
        case SolverType::pardisoLDLT:
            x = pardisoLDLT.solve(b);
            break;
        case SolverType::pardisoLU:
            x = pardisoLU.solve(b);
            break;
#else
        case SolverType::pardisoLDLT:
        case SolverType::pardisoLU:
            throw std::invalid_argument(
                "cannot solve system with Pardiso since it was not found "
                "during compilation");
            break;
#endif
#ifdef GC_HAVE_SUITESPARSE
        case SolverType::suitesparseLLT:
            x = suitesparseLLT->solve(b);
            break;
        case SolverType::suitesparseLU:
            x = suitesparseLU->solve(b);
            break;
        case SolverType::suitesparseQR:
            x = suitesparseQR->solve(b);
            break;
#else
        case SolverType::suitesparseLLT:
        case SolverType::suitesparseLU:
        case SolverType::suitesparseQR:
            throw std::invalid_argument(
                "cannot solve system with Suitesparse since it was not found "
                "during compilation");
            break;
#endif
        }
    } else {
        x = DenseMatrix<T>::Zero(b.rows(), b.cols());
    }
}

template <typename T>
DenseMatrix<T> ConfigurableSolver<T>::solve(const DenseMatrix<T>& b,
                                            bool transposed) {
    DenseMatrix<T> x;
    solve(x, b, transposed);
    return x;
}

template <typename T>
void ConfigurableSolver<T>::clear() {
    validSymbolic = false;
    validNumeric  = false;

    Lp.clear();
    Parent.clear();
    Lnz.clear();
    Li.clear();
    Pattern.clear();
    Flag.clear();
    P.clear();
    Pinv.clear();
    Lx.clear();
    D.clear();
    Y.clear();

    rows = -1;
    cols = -1;
}


// Explicit instantiations
template class ConfigurableSolver<double>;
template class ConfigurableSolver<float>;
template class ConfigurableSolver<std::complex<double>>;
