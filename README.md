# MagmaSharp
.NET High Level API for MAGMA - Matrix Algebra for GPU and Multicore Architectures.
The project is supposed to be High Level API which means not all methods and capabilities will be implemented. In fact only selected and most important method are going to be exposed on .NET platform. The Library can run regardless of the CUDA present. In case the CUDA is not detected, the coresponded Lapack routine will be executed. On this way, the library can be execution engine for other .NET High Level APIs and libraries.

## Implementation
Currently the library supports MAGMA driver routines for general rectangular matrix:

1. ```gesv``` - solve linear system, AX = B, A is general non-symetric matrix,
2. ```gels``` - least square solve, AX = B, A is rectangular,
3. ```geev``` - eigen value solver for non-symetric matrix, AX = X lambda
4. ```gesvd```- singular value decomposition (SVD), A) U SIgma V^2H.

The library suppors `float` and `double` values types.

# Software requirements

The project is build on .NET Core 3.1 and .NET Standard 2.1. 

# Software (Native Libraries) requirements
In order to install and use the library the following Native library is needed to be installed. 

- Intel MKL which can be downloaded at https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library/choose-download.html


# How to Build MagmaSharp from the source

1. Download Intel MKL Math library from the official site and install it to default location: C:/Program Files (x86)/IntelSWTools
2. Build Magma library and copy it to you local development environment at (MagmaSharp/MagmaLib)

![Magma runtime location](img/magma_lib_location.jpg)

3. Open 'MagmaSharp.Sln' with Visual Studio 2019.
4. Make Sure the building architecture is x64.
5. Build the Solution.

