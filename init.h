#pragma once
#include "save_spm.hpp"
#include <unordered_map>
class MatrixInfo
{
public:
    int64_t rows;
    int64_t cols;
    int64_t nonZeros;
    // int major;
    int *offset;
    int *idx;
    double *val;
    MatrixInfo()
    {
    }
    MatrixInfo(int64_t r, int64_t c, int64_t nz)
        : rows(r), cols(c), nonZeros(nz)
    {
        offset = new int[r + 1];
        idx = new int[nz];
        val = new double[nz];
    }

    ~MatrixInfo()
    {
        delete[] offset;
        delete[] idx;
        delete[] val;
    }
};
struct HashFunc
{
	template<typename T, typename U>
	size_t operator()(const std::pair<T, U>& p) const {
		return std::hash<T>()(p.first) ^ std::hash<U>()(p.second);
	}
};

// 键值比较，哈希碰撞的比较定义，需要直到两个自定义对象是否相等
struct EqualKey {
	template<typename T, typename U>
	bool operator ()(const std::pair<T, U>& p1, const std::pair<T, U>& p2) const {
		return p1.first == p2.first && p1.second == p2.second;
	}
};
                        
class CalRelation
{
public:
    std::unordered_map<int,std::vector<std::pair<int,int>>> CalMap;
    std::unordered_map<int,std::pair<int,int>> Index2Pos;
    std::unordered_map<std::pair<int,int>,int,HashFunc,EqualKey> Pos2Index;

    CalRelation(){

    }


};
MatrixInfo transpose(const MatrixInfo &A)
{
    MatrixInfo A_transpose(A.cols, A.rows, A.nonZeros);
    // 初始化A^T的行偏移为0
    for (int i = 0; i <= A_transpose.rows; ++i)
    {
        A_transpose.offset[i] = 0;
    }

    // 计算A^T的行偏移
    for (int i = 0; i < A.rows; ++i)
    {
        for (int j = A.offset[i]; j < A.offset[i + 1]; ++j)
        {
            A_transpose.offset[A.idx[j] + 1]++;
        }
    }

    // 转置行偏移为数组
    for (int i = 0; i < A_transpose.rows; ++i)
    {
        A_transpose.offset[i + 1] += A_transpose.offset[i];
    }

    // 初始化转置后的列索引和值数组
    std::vector<int> new_col_indices(A_transpose.nonZeros);
    std::vector<double> new_values(A_transpose.nonZeros);

    // 计算转置后的列索引和值
    for (int i = 0; i < A.rows; ++i)
    {
        for (int j = A.offset[i]; j < A.offset[i + 1]; ++j)
        {
            int col = A.idx[j];
            int dest = A_transpose.offset[col]++;
            new_col_indices[dest] = i;
            new_values[dest] = A.val[j];
        }
    }
    for (int i = A_transpose.rows; i > 0; --i)
    {
        A_transpose.offset[i] = A_transpose.offset[i - 1];
    }
    A_transpose.offset[0] = 0;

    // 将临时数组复制回A_transpose
    std::copy(new_col_indices.begin(), new_col_indices.end(), A_transpose.idx);
    std::copy(new_values.begin(), new_values.end(), A_transpose.val);

    return A_transpose;
}

// 两个稀疏矩阵乘法
MatrixInfo SSM(const MatrixInfo &A, const MatrixInfo &B, CalRelation* Relation = nullptr)
{

    if (A.cols != B.rows)
    {
        throw std::invalid_argument("Error: Matrix dimensions do not match, cannot multiply");
    }

    MatrixInfo C(A.rows, B.cols, 0);

    // 临时存储空间
    std::vector<int> cOffsets(A.rows + 1, 0);
    std::vector<int> cIndex;
    std::vector<double> cValues;

    // 用于跟踪每一列的非零元素位置
    std::vector<int> colOffsets(B.cols, -1);
    // 遍历矩阵A的每一行
    for (int i = 0; i < A.rows; ++i)
    {
        for (int k = A.offset[i]; k < A.offset[i + 1]; ++k)
        {
            int col = A.idx[k];     // A中当前元素所在列
            double aVal = A.val[k]; // A中当前元素的值

            // 遍历矩阵B的第col行的每个非零元素
            for (int l = B.offset[col]; l < B.offset[col + 1]; ++l)
            {
                int j = B.idx[l];       // B中当前非零元素的列
                double bVal = B.val[l]; // B中当前非零元素的值

                // 如果当前列j在当前行中还没有被访问过
                if (colOffsets[j] == -1)
                {
                    Relation->CalMap[cValues.size()].push_back(std::make_pair(k,l));

                    colOffsets[j] = cValues.size(); // 记录位置
                    cIndex.push_back(j);            // 存储列索引
                    cValues.push_back(aVal * bVal); // 存储乘积值
                }
                else
                {
                    // 如果当前列j已经被访问过，累加值
                    Relation->CalMap[colOffsets[j]].push_back(std::make_pair(k,l));

                    cValues[colOffsets[j]] += aVal * bVal;
                }
            }
        }

        // 更新C的行偏移量
        cOffsets[i + 1] = cValues.size();

        // 重置colOffsets以便处理下一行
        for (int idx = cOffsets[i]; idx < cOffsets[i + 1]; ++idx)
        {
            colOffsets[cIndex[idx]] = -1;
        }
    }
    // 将临时存储中的值复制到矩阵C中
    C.nonZeros = cValues.size();
    C.offset = new int[A.rows + 1];
    C.idx = new int[C.nonZeros];
    C.val = new double[C.nonZeros];

    std::copy(cOffsets.begin(), cOffsets.end(), C.offset);
    std::copy(cIndex.begin(), cIndex.end(), C.idx);
    std::copy(cValues.begin(), cValues.end(), C.val);
    return C;
}
// B对称，取BL
MatrixInfo SSM_Modified(const MatrixInfo &A, const MatrixInfo &B, CalRelation* Relation = nullptr)
{

    if (A.cols != B.rows)
    {
        throw std::invalid_argument("Error: Matrix dimensions do not match, cannot multiply");
    }

    MatrixInfo C(A.rows, B.cols, 0);

    // 临时存储空间
    std::vector<int> cOffsets(A.rows + 1, 0);
    std::vector<int> cIndex;
    std::vector<double> cValues;

    // 用于跟踪每一列的非零元素位置
    std::vector<int> colOffsets(B.cols, -1);
    // 遍历矩阵A的每一行
    for (int i = 0; i < A.rows; ++i)
    {
        for (int k = A.offset[i]; k < A.offset[i + 1]; ++k)
        {
            int col = A.idx[k];     // A中当前元素所在列
            double aVal = A.val[k]; // A中当前元素的值

            // 遍历矩阵B的第col行的每个非零元素
            for (int l = B.offset[col]; l < B.offset[col + 1] && B.idx[l] <= col; ++l)
            {
                int j = B.idx[l];       // B中当前非零元素的列
                double bVal = j == col ?B.val[l]/2 : B.val[l]; // B中当前非零元素的值

                // 如果当前列j在当前行中还没有被访问过
                if (colOffsets[j] == -1)
                {
                    Relation->CalMap[cValues.size()].push_back(std::make_pair(k,l));
                    Relation->Index2Pos[cValues.size()] = std::make_pair(i,j);
                    Relation->Pos2Index[std::make_pair(i,j)] = cValues.size();

                    colOffsets[j] = cValues.size(); // 记录位置
                    cIndex.push_back(j);            // 存储列索引
                    cValues.push_back(aVal * bVal); // 存储乘积值
                }
                else
                {
                    // 如果当前列j已经被访问过，累加值
                    Relation->CalMap[colOffsets[j]].push_back(std::make_pair(k,l));

                    cValues[colOffsets[j]] += aVal * bVal;
                }
            }
        }

        // 更新C的行偏移量
        cOffsets[i + 1] = cValues.size();

        // 重置colOffsets以便处理下一行
        for (int idx = cOffsets[i]; idx < cOffsets[i + 1]; ++idx)
        {
            colOffsets[cIndex[idx]] = -1;
        }
    }
    // 将临时存储中的值复制到矩阵C中
    C.nonZeros = cValues.size();
    C.offset = new int[A.rows + 1];
    C.idx = new int[C.nonZeros];
    C.val = new double[C.nonZeros];

    std::copy(cOffsets.begin(), cOffsets.end(), C.offset);
    std::copy(cIndex.begin(), cIndex.end(), C.idx);
    std::copy(cValues.begin(), cValues.end(), C.val);
    return C;
}
void PTAP1(const MatrixInfo &PT, const MatrixInfo &A, MatrixInfo *PTA)
{
    if (PT.cols != A.rows)
    {
        throw std::invalid_argument("Error: Matrix dimensions do not match, cannot multiply");
    }
    std::unordered_map<int, double> rowMaps(A.rows + 1);
    rowMaps[A.rows] = -1;
    for (int i = 0; i < PTA->rows; ++i)
    {
        // std::cout << "i = " << i << std::endl;
        for (int CIdx = PTA->offset[i]; CIdx < PTA->offset[i + 1]; ++CIdx)
        {
            // std::cout << "  CIdx = " << CIdx <<std::endl;
            int Col = PTA->idx[CIdx];
            double Cval = 0;
            for (int PTIdx = A.offset[i]; PTIdx < PT.offset[i + 1]; ++PTIdx)
            {
                // std::cout << "      PTIdx = " << PTIdx <<std::endl;
                double PTval = PT.val[PTIdx];
                int PTCol = PT.idx[PTIdx];
                // TODO 感觉这里还能被优化，目前只是扫一遍，感觉可以在预处理加入类似于hash表的东西进行O(1)的查找
                if (rowMaps[A.rows] == -1)
                {
                    for (int IdxA = A.offset[PTCol]; IdxA < A.offset[PTCol + 1]; ++IdxA)
                    {
                        rowMaps[A.idx[IdxA]] = A.val[IdxA];
                    }
                    rowMaps[A.rows] = 0;
                }
                double AVal = rowMaps[Col];
                Cval += PTval * AVal;
            }
            PTA->val[CIdx] = Cval;
            for(auto& pair : rowMaps)
            {
                pair.second = double();
            }
            rowMaps[A.rows] = -1;
            // std::cout << "PTA Row" << i << "Col " << Col << "Value " << Cval << std::endl;
        }
    }
}
void PTA2(const MatrixInfo &PT, const MatrixInfo &A, MatrixInfo *PTA,CalRelation * relation)
{
    #pragma omp parallel for
    for(int i = 0; i< PTA->nonZeros;++i)
    {
        double sum = 0.0;
        for(auto& pair : relation->CalMap[i])
        {
            sum += PT.val[pair.first] * A.val[pair.second];
        }
        PTA->val[i] = sum;
    }
}
void PTALP2(const MatrixInfo &PTALP, const MatrixInfo &PTALPT, MatrixInfo *PTAP)
{
    std::cout << PTALP.nonZeros << " " << PTALPT.nonZeros << " "<<PTAP->nonZeros<<std::endl;
    #pragma omp parallel for
    for(int i = 0; i< PTALP.nonZeros;++i)
    {
        PTAP->val[i] = PTALP.val[i] + PTALPT.val[i];
    }
}
void PTAP2(const MatrixInfo &PTA, const MatrixInfo &P, MatrixInfo *PTAP,CalRelation * relation, MatrixInfo *Complete = nullptr, CalRelation * CompleteCal =nullptr)
{
    #pragma omp parallel for
    for(int i = 0; i< PTAP->nonZeros;++i)
    {
        double sum = 0.0;
        for(auto& pair : relation->CalMap[i])
        {
            sum += PTA.val[pair.first] * P.val[pair.second];
        }
        int row = relation->Index2Pos[i].first;
        int col = relation->Index2Pos[i].second;
        int idx1 = CompleteCal->Pos2Index[std::make_pair(row,col)];
        int idx2 = CompleteCal->Pos2Index[std::make_pair(col,row)];
        Complete->val[idx1] = sum;
        
        if(row<=col)
        {
            Complete->val[idx1] += Complete->val[idx2];
            Complete->val[idx2] =Complete->val[idx1];
        }
    }
}
Eigen::SparseMatrix<double, Eigen::RowMajor> ToEigen(const MatrixInfo &C)
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> A;
    A.resize(C.rows, C.cols);
    for (int i = 0; i < C.rows; ++i)
    {
        for (int j = C.offset[i]; j < C.offset[i + 1]; ++j)
        {
            // std::cout << "Row " << i << " Column " << C.idx[j] << " Value " << C.val[j] << std::endl;
            A.insert(i, C.idx[j]) = C.val[j];
        }
    }
    return A;
}

int ReadSpm(const char *path, MatrixInfo &A)
{
    std::ifstream ifs(path, std::ios::binary);
    if (ifs.fail())
    {
        return __LINE__;
    }
    int64_t mat_size[4];

    ifs.read((char *)&mat_size[0], 4 * sizeof(int64_t));
    // A.resize(mat_size[0], mat_size[1]);
    for (int i = 0; i < 4; i++)
    {
        std::cout << mat_size[i] << " ";
    }
    std::cout << std::endl;
    A.rows = mat_size[0];
    A.cols = mat_size[1];
    A.nonZeros = mat_size[2];
    A.val = new double[mat_size[2]];
    A.idx = new int[mat_size[2]];
    int size;

    A.offset = new int[mat_size[0] + 1];
    size = mat_size[0] + 1;
    ifs.read((char *)A.val, mat_size[2] * sizeof(double));
    ifs.read((char *)A.idx, mat_size[2] * sizeof(int));
    ifs.read((char *)A.offset, size * sizeof(int));
    ifs.close();
    return 0;
}
template <class T>
int write_txt(const char *path, T val[], int size)
{
    std::ofstream ofs(path);
    if (ofs.fail())
    {
        std::cerr << "open file fail." << std::endl;
        return __LINE__;
    }
    for (int i = 0; i < size; i++)
    {
        ofs << val[i] << " ";
    }
    ofs.close();
    return 0;
}