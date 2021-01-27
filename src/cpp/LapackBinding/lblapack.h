
#pragma once

#include "pch.h"
/*
	Intel® Math Kernel Library LAPACK


	Driver routines
	Driver routines solve an entire problem.

	Name	Description
	------------------------------------------------------------------
	gesv	solve linear system, AX = B, A is general(non - symmetric)
	gels	least squares solve, AX = B, A is rectangular
	geev	non - symmetric eigenvalue solver, AX = X Lambda
	gesvd	singular value decomposition(SVD), A = U Sigma V^ H
	
*/

namespace LapackBinding
{
	
	enum class mbv2vector
	{
		MagmaNoVec = 301,  /* geev, syev, gesvd */
		MagmaVec = 302,  /* geev, syev */
		MagmaIVec = 303,  /* stedc */
		MagmaAllVec = 304,  /* gesvd, trevc */
		MagmaSomeVec = 305,  /* gesvd, trevc */
		MagmaOverwriteVec = 306,  /* gesvd */
		MagmaBacktransVec = 307   /* trevc */
	};

	enum class mbv2trans
	{
		NoTrans,//none
		Trans,//transpose the matrix before operations
		ConjTrans,//not supported yet
	};
	//
	char convertToChar(mbv2vector vec);
	void transpose(const float* src, float* dst, const int N, const int M);
	void transpose(const double* src, double* dst, const int N, const int M);

	//AX=B - solver
	extern "C" MAGMABINDINGS_API int mbv2sgesv_cpu(bool rowmajor, int n, int nrhs, float* A, int lda, float* B, int ldb);
	//double
	extern "C" MAGMABINDINGS_API int mbv2dgesv_cpu(bool rowmajor, int n, int nrhs, double* A, int lda,  double* B, int ldb);
	
	//SVD
	extern "C" MAGMABINDINGS_API int mbv2sgesvd_cpu(bool rowmajor, mbv2vector jobu, mbv2vector jobv, int m, int n, float* A, int lda, float* s, float* U, int ldu, float* VT, int ldvt);
	extern "C" MAGMABINDINGS_API int mbv2sgesvds_cpu(bool rowmajor, int m, int n, float* A, float* s, float* U, bool calcU, float* VT, bool calcV);
	
	//SVD
	extern "C" MAGMABINDINGS_API int mbv2dgesvd_cpu(bool rowmajor, mbv2vector jobu, mbv2vector jobv, int m, int n, double* A, int lda, double* s, double* U, int ldu, double* VT, int ldvt);
	extern "C" MAGMABINDINGS_API int mbv2dgesvds_cpu(bool rowmajor, int m, int n, double* A, double* s, double* U, bool calcU, double* VT, bool calcV);
	
	

	//LSS - least squares solver
	extern "C" MAGMABINDINGS_API int mbv2sgels_cpu(bool rowmajor, int m, int n, int nrhs, float* A, int lda, float* B, int lbd);
	extern "C" MAGMABINDINGS_API int mbv2dgels_cpu(bool rowmajor, int m, int n, int nrhs, double* A, int lda, double* B, int lbd);
	
	//EIGEN
	extern "C" MAGMABINDINGS_API int mbv2sgeevs_cpu(bool rowmajor, int n, float* A, int lda, float* wr, float* wi, float* VL, bool computeLeft, float* Vr, bool computeRight);
	extern "C" MAGMABINDINGS_API int mbv2sgeev_cpu(bool rowmajor, char jobvl, char jobvr, int n, float* A, int lda, float* wr, float* wi, float* Vl, int ldvl, float* Vr, int ldvr);
	
	extern "C" MAGMABINDINGS_API int mbv2dgeevs_cpu(bool rowmajor, int n, double* A, int lda, double* wr, double* wi, double* VL, bool computeLeft, double* VR, bool computeRight);
	extern "C" MAGMABINDINGS_API int mbv2dgeev_cpu(bool rowmajor, char jobvl, char jobvr, int n, double* A, int lda, double* wr, double* wi, double* Vl, int ldvl, double* Vr, int ldvr);

	//Matrix-Matrix multiplication
	extern "C" MAGMABINDINGS_API void mbv2sgemm_cpu(bool rowmajor, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);
	extern "C" MAGMABINDINGS_API void mbv2dgemm_cpu(bool rowmajor, int m, int n, int k, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc);

	//Transpose 
	extern "C" MAGMABINDINGS_API void mbv2stranspose_cpu(bool rowmajor, int m, int n, const float* A, int lda, float* At, int ldat);
	extern "C" MAGMABINDINGS_API void mbv2dtranspose_cpu(bool rowmajor, int m, int n, const double* A, int lda, double* At, int ldat);

	//Inverse matrix
	extern "C" MAGMABINDINGS_API int mbv2dinverse_cpu(bool rowmajor, int n, double* A, int lda);
	extern "C" MAGMABINDINGS_API int mbv2sinverse_cpu(bool rowmajor, int n, float* A, int lda);
}
