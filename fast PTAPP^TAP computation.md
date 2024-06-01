# Fast $P^TAP$ Computation



## Backgrounds

在一些科学计算领域如有限元分析(FEA)中 我们经常需要求解一个大型稀疏线性系统$Ax=b$，$A\in R^{n \times n}，b\in R^n$，多数情况下，$A$为实对称矩阵。在实际求解过程中，由于$n$非常大，求解这个线性系统的计算开销极其巨大，一种名为Galerkin Projection的方法把$x$的解空间限制在子空间$\text{span}\{ p_1,...p_m \}, m \ll n $上，原来的问题就被转换为求解下面的线性系统:
$$
\begin{align}
A^hy &= b^h\\
A^h &= P^TAP\\
b^h &=P^Tb \\
x &= Py
\end{align}
$$
其中$P=(p_1,p_2,...,p_m)$也是一个稀疏矩阵。这样我们就只需要求解一个规模远小于原问题的$ m\times m$的稀疏线性系统，大大降低了其求解开销和难度。



## Your Work

Galerkin Porjection中的一个重点就是计算$A^h=P^TAP$。

你的任务是设计算法尽可能地高效和快速地完成这一过程。常见的想法有外积法和分块法。二者都有较高的并行性可以挖掘。

由于在实际应用过程中，$A$和$P$的 sparsity pattern总是不变的，因此可以通过预计算的方式根据这样的sparsity pattern调整计算策略。预计算的时间开销不计入$P^TAP$的计算开销中。



## Environment

我们使用`Eigen`库提供的数据结构来储存稀疏矩阵。系数矩阵的序列化和反序列化的程序已经提供在工程目录中，供你使用和参考。

工程目录中还有一个可执行文件`test_outerProduct`，它的调用方式如下所示。

四个子命令，子命令`gen`按照输入参数生成稀疏矩阵；子命令`[eig|row|two]`分别调用三种算法去计算$P^TAP$。

```shell
./test_outerProduct [gen|eig|row|two] 
```

+ 稀疏矩阵生成

  ```shell
  ./test_outerProduct gen n m densityA densityB name
  ```

  指令生成稀疏程度为 densityA的稀疏矩阵$A\in R^{n\times n}$、稀疏程度为densityB的稀疏矩阵$P$，和右手项$b \in R^n$。三者分别序列化输出为`name.A.spm`, `name.P.spm`和`name.C.spm`

+ 稀疏矩阵求解

  ```shell
  time ./test_outerProduct [eig|row|two] name 
  ```

  调用三种方法中的一种读入`name`标识的稀疏矩阵系统开始进行$P^TAP$的计算和求解。统计预计算`init`和计算$P^TAP$的时长。



你可以比对自己实现的$P^TAP$算法和这个可执行文件程序的三种算法的速度，判断是否有更好的实现方法。