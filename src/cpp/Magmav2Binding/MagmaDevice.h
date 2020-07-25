
#pragma once

#include "pch.h"
/*
	MAGMA  2.3.0, https://icl.utk.edu/magma/software/index.html
	Matrix Algebra for GPU and Multicore Architectures


	Device management
	Utilities

	
*/

namespace MagmaBinding
{
	//
	extern "C" MAGMABINDINGS_API void printInfo();
	extern "C" MAGMABINDINGS_API int start();
	extern "C" MAGMABINDINGS_API int end();
	extern "C" MAGMABINDINGS_API int mbv2getdevice_arch();
	extern "C" MAGMABINDINGS_API void mbv2getdevices(int* devices, int size, int* num_dev);
	extern "C" MAGMABINDINGS_API void mbv2getdevice(int* device);
	extern "C" MAGMABINDINGS_API void mbv2setdevice(int device);
}
