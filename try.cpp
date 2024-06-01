#include <Eigen/SparseCore>
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include "init.h"



int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <path>" << std::endl;
        return 1;
    }
    std::string path = argv[1];
    std::string APath = path + ".A.spm";
    std::string PPath = path + ".P.spm";
    MatrixInfo A;
    MatrixInfo P;

    ReadSpm(APath.c_str(), A);
    ReadSpm(PPath.c_str(), P);

    std::cout << "A: " << A.rows << "x" << A.cols << ", " << A.nonZeros << " nonzeros" << std::endl;
    std::cout << "P: " << P.rows << "x" << P.cols << ", " << P.nonZeros << " nonzeros" << std::endl;
    MatrixInfo PT = transpose(P);
    write_txt("A.txt",A.val,A.nonZeros);

    // 单位矩阵
    MatrixInfo E(A.rows, A.cols, A.rows);
    for (int i = 0; i < A.rows; ++i)
    {
        E.offset[i] = i;
        E.idx[i] = i;
        E.val[i] = 1.0;
    }
    E.offset[A.rows] = A.rows;
    // 进行矩阵乘法

    // write_txt("PT.txt", PT.val, PT.nonZeros);
    // write_txt("PT_offset.txt", PT.offset, PT.rows + 1);
    // write_txt("PT_idx.txt", PT.idx, PT.nonZeros);


    // write_txt("A.txt", A.val, A.nonZeros);
    // write_txt("A_offset.txt", A.offset, A.rows + 1);
    // write_txt("A_idx.txt", A.idx, A.nonZeros);
    // auto start = std::chrono::high_resolution_clock::now();
    // MatrixInfo PTA= SSM(PT, A);
    auto start = std::chrono::high_resolution_clock::now();
    
    MatrixInfo C= SSM(A,P);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Mine time costs: " << elapsed.count() << "s" << std::endl;
    Eigen::SparseMatrix<double, Eigen::RowMajor> PA = ToEigen(C);
    
    Eigen::SparseMatrix<double, Eigen::RowMajor> A1;
    Eigen::SparseMatrix<double, Eigen::RowMajor> P1;
    zcy::io::read_spm(APath.c_str(), A1);
    zcy::io::read_spm(PPath.c_str(), P1);
    start = std::chrono::high_resolution_clock::now();
    Eigen::SparseMatrix<double, Eigen::RowMajor> result = A1*P1;
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Eigen time costs: " << elapsed.count() << "s" << std::endl;
    
    Eigen::SparseMatrix<double, Eigen::RowMajor>result2 = PA-result;
    std::cout << "result: " << result2.rows() << "x" << result2.cols() << ", " << result2.nonZeros() << " nonzeros" << std::endl;
    write_txt("result.txt",result2.valuePtr(),result2.nonZeros());
    return 0;
}
