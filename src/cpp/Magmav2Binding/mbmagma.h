
#pragma once

#include "lblapack.h"
#include "pch.h"
/*
	MAGMA  2.3.0, https://icl.utk.edu/magma/software/index.html
	Matrix Algebra for GPU and Multicore Architectures


	Driver routines
	Driver routines solve an entire problem.

	Name	Description
	------------------------------------------------------------------
	gesv	solve linear system, AX = B, A is general(non - symmetric)
	gels	least squares solve, AX = B, A is rectangular
	geev	non - symmetric eigenvalue solver, AX = X Lambda
	gesvd	singular value decomposition(SVD), A = U Sigma V^ H
	
*/
using namespace LapackBinding;

namespace MagmaBinding
{
	
	
	//AX=B - solver
	extern "C" MAGMABINDINGS_API int mbv2sgesv(int n, int nrhs, float* A, int lda, int* ipiv, float* B, int lbd);
	extern "C" MAGMABINDINGS_API int mbv2sgesv_gpu(int n, int nrhs, float* A, int lda, int* ipiv, float* B, int lbd);
	//double
	extern "C" MAGMABINDINGS_API int mbv2dgesv(int n, int nrhs, double* A, int lda, int* ipiv, double* B, int lbd);
	extern "C" MAGMABINDINGS_API int mbv2dgesv_gpu(int n, int nrhs, double* A, int lda, int* ipiv, double* B, int lbd);

	//SVD
	extern "C" MAGMABINDINGS_API int mbv2sgesvds(int m, int n, float* A, float* s, float* U, float* VT);
	extern "C" MAGMABINDINGS_API int mbv2sgesvd(mbv2vector jobu, mbv2vector jobv, int m, int n,
		float* A, int lda, float* s, float* U, int ldu, float* VT, int ldvt);

	//SVD
	extern "C" MAGMABINDINGS_API int mbv2dgesvds(int m, int n, double* A, double* s, double* U, double* VT);
	extern "C" MAGMABINDINGS_API int mbv2dgesvd(mbv2vector jobu, mbv2vector jobv, int m, int n,
		double* A, int lda, double* s, double* U, int ldu, double* VT, int ldvt);


	//LSS - least squares solver
	extern "C" MAGMABINDINGS_API int mbv2sgels(int m, int n, int nrhs, float* A, int lda, float* B, int lbd);
	extern "C" MAGMABINDINGS_API int mbv2sgels_gpu(int m, int n, int nrhs, float* A, int lda, float* B, int lbd);

	extern "C" MAGMABINDINGS_API int mbv2dgels(int m, int n, int nrhs, double* A, int lda, double* B, int lbd);
	extern "C" MAGMABINDINGS_API int mbv2dgels_gpu(int m, int n, int nrhs, double* A, int lda, double* B, int lbd);

	//EIGEN
	extern "C" MAGMABINDINGS_API int mbv2sgeevs(int n, float* A, int lda, float* wr, float* wi, float* VL, int ldvl, float* VR, int ldvr);
	extern "C" MAGMABINDINGS_API int mbv2sgeev(mbv2vector jobvl, mbv2vector jobvr, int n, float* A, int lda, float* wr, float* wi, float* Vl, int ldvl, float* Vr, int ldvr);
	
	extern "C" MAGMABINDINGS_API int mbv2dgeevs(int n, double* A, int lda, double* wr, double* wi, double* VL, int ldvl, double* VR, int ldvr);
	extern "C" MAGMABINDINGS_API int mbv2dgeev(mbv2vector jobvl, mbv2vector jobvr, int n, double* A, int lda, double* wr, double* wi, double* Vl, int ldvl, double* Vr, int ldvr);


}
