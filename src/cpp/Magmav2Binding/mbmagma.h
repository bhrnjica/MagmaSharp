
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
	extern "C" MAGMABINDINGS_API int mbv2sgesv(bool rowmajor, int n, int nrhs, float* A, int lda, float* B, int ldb);
	extern "C" MAGMABINDINGS_API int mbv2sgesv_gpu(bool rowmajor, int n, int nrhs, float* A, int lda,  float* B, int lbd);
	//double
	extern "C" MAGMABINDINGS_API int mbv2dgesv(bool rowmajor, int n, int nrhs, double* A, int lda,  double* B, int lbd);
	extern "C" MAGMABINDINGS_API int mbv2dgesv_gpu(bool rowmajor, int n, int nrhs, double* A, int lda, double* B, int lbd);

	//SVD
	extern "C" MAGMABINDINGS_API int mbv2sgesvds(bool rowmajor, int m, int n, float* A, float* s, float* U, bool calcU, float* VT, bool calcV);
	extern "C" MAGMABINDINGS_API int mbv2sgesvd(bool rowmajor, mbv2vector jobu, mbv2vector jobv, int m, int n, float* A, int lda, float* s, float* U, int ldu, float* VT, int ldvt);

	//SVD
	extern "C" MAGMABINDINGS_API int mbv2dgesvds(bool rowmajor, int m, int n, double* A, double* s, double* U, bool calcU, double* VT, bool calcV);
	extern "C" MAGMABINDINGS_API int mbv2dgesvd(bool rowmajor, mbv2vector jobu, mbv2vector jobv, int m, int n, double* A, int lda, double* s, double* U, int ldu, double* VT, int ldvt);


	//LSS - least squares solver
	extern "C" MAGMABINDINGS_API int mbv2sgels(bool rowmajor, int m, int n, int nrhs, float* A, int lda, float* B, int lbd);
	extern "C" MAGMABINDINGS_API int mbv2sgels_gpu(bool rowmajor, int m, int n, int nrhs, float* A, int lda, float* B, int lbd);

	extern "C" MAGMABINDINGS_API int mbv2dgels(bool rowmajor, int m, int n, int nrhs, double* A, int lda, double* B, int lbd);
	extern "C" MAGMABINDINGS_API int mbv2dgels_gpu(bool rowmajor, int m, int n, int nrhs, double* A, int lda, double* B, int lbd);

	//EIGEN
	extern "C" MAGMABINDINGS_API int mbv2sgeevs(bool rowmajor, int n, float* A, int lda, float* wr, float* wi, float* Vl, bool computeLeft, float* Vr, bool computeRight);
	extern "C" MAGMABINDINGS_API int mbv2sgeev(bool rowmajor, mbv2vector jobvl, mbv2vector jobvr, int n, float* A, int lda, float* wr, float* wi, float* Vl, int ldvl, float* Vr, int ldvr);
	
	extern "C" MAGMABINDINGS_API int mbv2dgeevs(bool rowmajor, int n, double* A, int lda, double* wr, double* wi, double* Vl, bool computeLeft, double* Vr, bool computeRight);
	extern "C" MAGMABINDINGS_API int mbv2dgeev(bool rowMajor, mbv2vector jobvl, mbv2vector jobvr, int n, double* A, int lda, double* wr, double* wi, double* Vl, int ldvl, double* Vr, int ldvr);

	//Matrix-Matrix operations
	extern "C" MAGMABINDINGS_API void mbv2sgemm(bool rowmajor, mbv2trans opA, mbv2trans opB, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);
	extern "C" MAGMABINDINGS_API void mbv2dgemm(bool rowmajor, mbv2trans opA, mbv2trans opB, int m, int n, int k, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc);

	//Inversion Matrix
	extern "C" MAGMABINDINGS_API int mbv2sgetri_gpu(bool rowmajor, int n, float* dA, int ldda);
	extern "C" MAGMABINDINGS_API int mbv2sgetri(bool rowmajor, int n, float* dA, int ldda);
	extern "C" MAGMABINDINGS_API int mbv2dgetri_gpu(bool rowmajor, int n, double* dA, int ldda);
	extern "C" MAGMABINDINGS_API int mbv2dgetri(bool rowmajor, int n, double* dA, int ldda);
}
