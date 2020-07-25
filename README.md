# MagmaSharp
C# implementation High level API of Linear Algebra based on of Magma - linear algebra library for for hybrid manycore and GPU systems that can enable applications to fully exploit the power that each of the hybrid components offers.
C# implementation of Magma - linear algebra library for for hybrid manycore and GPU systems that can enable applications to fully exploit the power that each of the hybrid components offers.


# Software (Native Libraries) requirements
- Intel MKL which can be downloaded at https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library/choose-download.html
- Magma 3.5.2 which can be  build from source code at https://bitbucket.org/icl/magma

# How to Build MagmaSharp from the source

1. Download Intel MKL Math library from the official site and install it to default location: C:/Program Files (x86)/IntelSWTools
2. Build Magma library and copy it to you local development environment at (c:/Libs/Magmav2.2.0/)
![Magma runtime location](img/magma_lib_location.jpg)
3. Open 'MagmaSharp.Sln' with Visual Studio 2019 and build it.