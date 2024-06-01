#include "save_spm.hpp"
#include <chrono>
#include <vector>

Eigen::SparseMatrix<double> MatrixDot(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B) {
    int rowsA = A.rows();
    int colsB = B.cols();

    Eigen::SparseMatrix<double> C(rowsA, colsB);

    // 遍历A的每一行
    for (int i = 0; i < rowsA; ++i) {
        // 遍历B的每一列
        for (int j = 0; j < colsB; ++j) {
            double sum = 0.0;
            // 计算A的第i行与B的第j列的点积
            for (int k = 0; k < A.cols(); ++k) {
                sum += A.coeff(i, k) * B.coeff(k, j);
            }
            if (sum != 0.0) {
                C.insert(i, j) = sum; // 将结果存入C中
            }
        }
    }

    C.makeCompressed(); // 压缩结果矩阵C
    return C;
}

Eigen::SparseMatrix<double> MatrixDot2(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B) {
    int rowsA = A.rows();
    int colsA = A.cols();
    int colsB = B.cols();

    // Result matrix in CSR format
    std::vector<Eigen::Triplet<double>> tripletList;

    // Iterate through each row of A
    for (int row = 0; row < rowsA; ++row) {
        // Iterate through each column of B
        for (int colB = 0; colB < colsB; ++colB) {
            double sum = 0.0;

            // Iterate through non-zero elements of the current row of A
            for (int kA = A.outerIndexPtr()[row]; kA < A.outerIndexPtr()[row + 1]; ++kA) {
                int colA = A.innerIndexPtr()[kA]; // Column index of current non-zero element in A
                // Find corresponding non-zero element in B
                for (int kB = B.outerIndexPtr()[colA]; kB < B.outerIndexPtr()[colA + 1]; ++kB) {
                    if (B.innerIndexPtr()[kB] == colB) {
                        sum += A.valuePtr()[kA] * B.valuePtr()[kB];
                        break; // Found corresponding non-zero element in B, move to next column
                    }
                }
            }

            // If sum is non-zero, add it to the result matrix
            if (sum != 0.0) {
                tripletList.push_back(Eigen::Triplet<double>(row, colB, sum));
            }
        }
    }

    // Construct the result matrix from the triplet list
    Eigen::SparseMatrix<double> C(rowsA, colsB);
    C.setFromTriplets(tripletList.begin(), tripletList.end());
    C.makeCompressed();
    return C;
}

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
    
    Eigen::SparseMatrix<double, Eigen::RowMajor> PT = P.transpose();
    // 计算PTALP
    auto start = std::chrono::high_resolution_clock::now();
    auto result = MatrixDot2(PT, A);
    std::cout << "result: " << result.rows() << "x" << result.cols() << ", " << result.nonZeros() << " nonzeros" << std::endl;
    result = MatrixDot2(result, P);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time: " << duration.count() << "ms" << std::endl;
    

    return 0;
}