#include "init.h"
#include <iostream>
#include <chrono>

Eigen::SparseMatrix<double> MatrixDot(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B);
// Eigen::SparseMatrix<double> MatrixDot(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B, Eigen::Index blockSize);
// Eigen::SparseMatrix<double> MatrixDotSymmetric(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B) ;
// Eigen::SparseMatrix<double> ComputePTAP(const Eigen::SparseMatrix<double>& P, const Eigen::SparseMatrix<double>& A) ;
// Eigen::SparseMatrix<double> ManualSparseMatrixMultiply(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B);
int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <path>" << std::endl;
        return 1;
    }
    std::string path = argv[1];
    std::string APath = path + ".A.spm";
    std::string PPath = path + ".P.spm";
    Eigen::SparseMatrix<double, Eigen::RowMajor> A;
    Eigen::SparseMatrix<double, Eigen::RowMajor> P;

    zcy::io::read_spm(APath.c_str(), A);
    zcy::io::read_spm(PPath.c_str(), P);

    std::cout << "A: " << A.rows() << "x" << A.cols() << ", " << A.nonZeros() << " nonzeros" << std::endl;
    std::cout << "P: " << P.rows() << "x" << P.cols() << ", " << P.nonZeros() << " nonzeros" << std::endl;
    // Eigen::SparseMatrix<double, Eigen::RowMajor> AL = A.triangularView<Eigen::Lower>();
    // for(int i = 0; i < AL.outerSize(); ++i)
    // {
    //     AL.coeffRef(i, i) /= 2;
    // }
    Eigen::SparseMatrix<double, Eigen::RowMajor> PT = P.transpose();
    // 计算PTALP
    // auto start = std::chrono::high_resolution_clock::now();
    // auto result = MatrixDot(PT, A);
    Eigen::SparseMatrix<double, Eigen::RowMajor> result = A*P;
    result.makeCompressed();
    // write_txt("A_MyEigen.txt", A.valuePtr(), A.nonZeros());
    // write_txt("A_MyEigen_offset.txt", A.outerIndexPtr(), A.rows() + 1);
    // write_txt("A_MyEigen_idx.txt",A.innerIndexPtr(), A.nonZeros());
    std::cout << "result: " << result.rows() << "x" << result.cols() << ", " << result.nonZeros() << " nonzeros" << std::endl;
    write_txt("AP_MyEigen.txt", result.valuePtr(), result.nonZeros());
    // write_txt("AP_MyEigen_offset.txt", result.outerIndexPtr(), result.rows() + 1);
    // write_txt("AP_MyEigen_idx.txt",result.innerIndexPtr(), result.nonZeros());
    // result = MatrixDot(result, P);
    // std::cout << "result: " << result.rows() << "x" << result.cols() << ", " << result.nonZeros() << " nonzeros" << std::endl;
    // result = result+result.transpose();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "Time: " << duration.count() << "ms" << std::endl;
    // Eigen::Index blockSize = 2;  // Example block size
    // start = std::chrono::high_resolution_clock::now();
    // result = MatrixDotSymmetric(PT, A);
    // result = MatrixDot(result, P);
    // result = ComputePTAP(P, A);
    // end = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "Time: " << duration.count() << "ms" << std::endl;
    

    return 0;
}

Eigen::SparseMatrix<double> MatrixDot(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B) 
{
    typedef typename Eigen::internal::remove_all<Eigen::SparseMatrix<double>>::type::Scalar ValueScalar;
    typedef typename Eigen::internal::remove_all<Eigen::SparseMatrix<double>>::type::StorageIndex StorageIndex;

    // The final size of the result matrix
    Eigen::Index rows = A.innerSize();
    Eigen::Index cols = B.outerSize();
    assert(A.outerSize() == B.innerSize());

    Eigen::internal::AmbiVector<ValueScalar, StorageIndex> Temp(rows);

    Eigen::SparseMatrix<double> result(rows, cols);
    Eigen::Index nonZeros = A.nonZeros() + B.nonZeros();
    result.reserve(nonZeros);   //  预留空间减少内存分配次数

    Eigen::internal::evaluator<Eigen::SparseMatrix<double>> evalA(A), evalB(B);
    double register density = double(nonZeros) / (double(cols) * double(rows));
    #pragma omp parallel for
    for(Eigen::Index i = 0; i < cols; ++i)
    {
        Temp.init(density);
        Temp.setZero();
        #pragma omp parallel for
        for(typename Eigen::internal::evaluator<Eigen::SparseMatrix<double>>::InnerIterator itB(evalB, i); itB; ++itB)
        {
            Temp.restart();
            ValueScalar value = itB.value();
            #pragma omp parallel for
            for(typename Eigen::internal::evaluator<Eigen::SparseMatrix<double>>::InnerIterator itA(evalA, itB.index()); itA; ++itA)
            {
                Temp.coeffRef(itA.index()) += itA.value() * value;
            }
        }
        result.startVec(i);
        for(typename Eigen::internal::AmbiVector<ValueScalar, StorageIndex>::Iterator it(Temp, 1e-10); it; ++it)
        {
            result.insertBackByOuterInner(i, it.index()) = it.value();
        }
    }
    result.finalize();
    return result;
}

// Eigen::SparseMatrix<double> MatrixDot(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B, Eigen::Index blockSize) {
//     typedef typename Eigen::internal::remove_all<Eigen::SparseMatrix<double>>::type::Scalar ValueScalar;
//     typedef typename Eigen::internal::remove_all<Eigen::SparseMatrix<double>>::type::StorageIndex StorageIndex;

//     // The final size of the result matrix
//     Eigen::Index rows = A.rows();
//     Eigen::Index cols = B.cols();
//     assert(A.cols() == B.rows());

//     Eigen::SparseMatrix<double> result(rows, cols);
//     Eigen::Index nonZeros = A.nonZeros() + B.nonZeros();
//     result.reserve(nonZeros);

//     Eigen::internal::evaluator<Eigen::SparseMatrix<double>> evalA(A), evalB(B);

//     // Iterate over blocks
//     for (Eigen::Index blockCol = 0; blockCol < cols; blockCol += blockSize) {
//         for (Eigen::Index blockRow = 0; blockRow < rows; blockRow += blockSize) {
//             // Process each block
//             for (Eigen::Index j = blockCol; j < std::min(blockCol + blockSize, cols); ++j) {
//                 Eigen::internal::AmbiVector<ValueScalar, StorageIndex> Temp(rows);
//                 Temp.init(double(nonZeros) / (double(cols) * double(rows)));
//                 Temp.setZero();
//                 for (typename Eigen::internal::evaluator<Eigen::SparseMatrix<double>>::InnerIterator itB(evalB, j); itB; ++itB) {
//                     if (itB.index() >= blockRow && itB.index() < blockRow + blockSize) {
//                         Temp.restart();
//                         ValueScalar value = itB.value();
//                         for (typename Eigen::internal::evaluator<Eigen::SparseMatrix<double>>::InnerIterator itA(evalA, itB.index()); itA; ++itA) {
//                             Temp.coeffRef(itA.index()) += itA.value() * value;
//                         }
//                     }
//                 }
//                 result.startVec(j); // Ensure startVec is called for each column in order
//                 for (typename Eigen::internal::AmbiVector<ValueScalar, StorageIndex>::Iterator it(Temp, 1e-10); it; ++it) {
//                     result.insertBackByOuterInner(j, it.index()) = it.value();
//                 }
//             }
//         }
//     }

//     result.finalize();
//     return result;
// }
// Eigen::SparseMatrix<double> MatrixDotSymmetric(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B) {
//     typedef typename Eigen::internal::remove_all<Eigen::SparseMatrix<double>>::type::Scalar ValueScalar;
//     typedef typename Eigen::internal::remove_all<Eigen::SparseMatrix<double>>::type::StorageIndex StorageIndex;

//     // The final size of the result matrix
//     Eigen::Index rows = A.rows();
//     Eigen::Index cols = B.cols();
//     assert(A.cols() == B.rows());

//     Eigen::SparseMatrix<double> result(rows, cols);
//     Eigen::Index nonZeros = A.nonZeros() + B.nonZeros();
//     result.reserve(nonZeros);

//     Eigen::internal::evaluator<Eigen::SparseMatrix<double>> evalA(A), evalB(B);

//     // Iterate over columns of the result matrix
//     for (Eigen::Index j = 0; j < cols; ++j) {
//         Eigen::internal::AmbiVector<ValueScalar, StorageIndex> Temp(rows);
//         Temp.init(double(nonZeros) / (double(cols) * double(rows)));
//         Temp.setZero();
        
//         for (typename Eigen::internal::evaluator<Eigen::SparseMatrix<double>>::InnerIterator itB(evalB, j); itB; ++itB) {
//             Temp.restart();
//             ValueScalar value = itB.value();
//             for (typename Eigen::internal::evaluator<Eigen::SparseMatrix<double>>::InnerIterator itA(evalA, itB.index()); itA; ++itA) {
//                 Temp.coeffRef(itA.index()) += itA.value() * value;
//                 if (itB.index() != j) { // Use symmetry to reduce computation
//                     Temp.coeffRef(itA.index()) += itA.value() * evalB.coeff(j, itB.index());
//                 }
//             }
//         }
//         result.startVec(j);
//         for (typename Eigen::internal::AmbiVector<ValueScalar, StorageIndex>::Iterator it(Temp, 1e-10); it; ++it) {
//             result.insertBackByOuterInner(j, it.index()) = it.value();
//         }
//     }

//     result.finalize();
//     return result;
// }
// Eigen::SparseMatrix<double> ManualSparseMatrixMultiply(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B) {
//     typedef typename Eigen::internal::remove_all<Eigen::SparseMatrix<double>>::type::Scalar ValueScalar;
//     typedef typename Eigen::internal::remove_all<Eigen::SparseMatrix<double>>::type::StorageIndex StorageIndex;

//     // The final size of the result matrix
//     Eigen::Index rows = A.rows();
//     Eigen::Index cols = B.cols();
//     assert(A.cols() == B.rows());

//     Eigen::SparseMatrix<double> result(rows, cols);
//     Eigen::Index nonZeros = A.nonZeros() + B.nonZeros();
//     result.reserve(nonZeros);

//     Eigen::internal::evaluator<Eigen::SparseMatrix<double>> evalA(A), evalB(B);

//     for (Eigen::Index j = 0; j < cols; ++j) {
//         Eigen::internal::AmbiVector<ValueScalar, StorageIndex> Temp(rows);
//         Temp.init(double(nonZeros) / (double(cols) * double(rows)));
//         Temp.setZero();
        
//         for (typename Eigen::internal::evaluator<Eigen::SparseMatrix<double>>::InnerIterator itB(evalB, j); itB; ++itB) {
//             Temp.restart();
//             ValueScalar value = itB.value();
//             for (typename Eigen::internal::evaluator<Eigen::SparseMatrix<double>>::InnerIterator itA(evalA, itB.index()); itA; ++itA) {
//                 Temp.coeffRef(itA.index()) += itA.value() * value;
//                 if (itB.index() != j) {
//                     Temp.coeffRef(itA.index()) += itA.value() * evalB.coeff(j, itB.index());
//                 }
//             }
//         }
//         result.startVec(j);
//         for (typename Eigen::internal::AmbiVector<ValueScalar, StorageIndex>::Iterator it(Temp, 1e-10); it; ++it) {
//             result.insertBackByOuterInner(j, it.index()) = it.value();
//         }
//     }

//     result.finalize();
//     return result;
// }

// Eigen::SparseMatrix<double> ComputePTAP(const Eigen::SparseMatrix<double>& P, const Eigen::SparseMatrix<double>& A) {
//     Eigen::SparseMatrix<double> PT = P.transpose();
//     Eigen::SparseMatrix<double> PTA = ManualSparseMatrixMultiply(PT, A);
//     PTA.makeCompressed();
//     return ManualSparseMatrixMultiply(PTA, P);
// }