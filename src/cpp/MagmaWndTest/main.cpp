#include <iostream>
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "HelperTest.h"
#include "MagmaDevice.h"
#include "mbmagma.h"

#include "magma_v2.h"
#include "magma_lapack.h"
#include "magmablas_d.h"

#include "mkl_lapacke.h"
using namespace MagmaBinding;
#include "GSVTests.h"
#include "SVDTests.h"
#include "LSSTests.h"
#include "EIGENTests.h"



int main(int argc, char** argv)
{
	mbv2getdevice_arch();
	//EIGEN
	mv2sgeevs_cpu_test();
	mv2sgeevs_test();
	mv2sgeev_cpu_test();
	mv2sgeev_test();

	mv2dgeevs_cpu_test();
	mv2dgeevs_test();
	mv2dgeev_cpu_test();
	mv2dgeev_test();

	//LSS
	mv2sgels_cpu_test();
	mv2sgels_test();
	mv2sgels_gpu_test();

	mv2dgels_cpu_test();
	mv2dgels_test();
	mv2dgels_gpu_test();
	
	//GSV tests
	mv2dgesv_cpu_test();
	mv2dgesv_test();
	mv2dgesv_gpu_test();
	mv2sgesv_cpu_test();
	mv2sgesv_test();
	mv2sgesv_gpu_test();

	//SVD
	mv2sgesvd_cpu_test();
	mv2sgesvds_cpu_test();
	mv2sgesvds_test();
	mv2sgesvd_test();


	mv2dgesvd_cpu_test();
	mv2dgesvds_cpu_test();
	mv2dgesvds_test();
	mv2dgesvd_test();

	return 0;

}
