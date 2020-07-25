
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

	char convertToChar(mbv2vector vec);

	//AX=B - solver
	extern "C" MAGMABINDINGS_API int mbv2sgesv_cpu(int n, int nrhs, float* A, int lda, int* ipiv, float* B, int lbd);
	//double
	extern "C" MAGMABINDINGS_API int mbv2dgesv_cpu(int n, int nrhs, double* A, int lda, int* ipiv, double* B, int lbd);
	
	//SVD
	extern "C" MAGMABINDINGS_API int mbv2sgesvd_cpu(mbv2vector jobu, mbv2vector jobv, int m, int n,
		float* A, int lda, float* s, float* U, int ldu, float* VT, int ldvt);
	extern "C" MAGMABINDINGS_API int mbv2sgesvds_cpu(int m, int n, float* A, float* s, float* U, float* VT);
	
	//SVD
	extern "C" MAGMABINDINGS_API int mbv2dgesvd_cpu(mbv2vector jobu, mbv2vector jobv, int m, int n,
		double* A, int lda, double* s, double* U, int ldu, double* VT, int ldvt);
	extern "C" MAGMABINDINGS_API int mbv2dgesvds_cpu(int m, int n, double* A, double* s, double* U, double* VT);
	

	//LSS - least squares solver
	extern "C" MAGMABINDINGS_API int mbv2sgels_cpu(int m, int n, int nrhs, float* A, int lda, float* B, int lbd);
	
	extern "C" MAGMABINDINGS_API int mbv2dgels_cpu(int m, int n, int nrhs, double* A, int lda, double* B, int lbd);
	
	//EIGEN
	extern "C" MAGMABINDINGS_API int mbv2sgeevs_cpu(int n, float* A, int lda, float* wr, float* wi, float* VL, int ldvl, float* VR, int ldvr);
	extern "C" MAGMABINDINGS_API int mbv2sgeev_cpu(mbv2vector jobvl, mbv2vector jobvr, int n, float* A, int lda, float* wr, float* wi, float* Vl, int ldvl, float* Vr, int ldvr);
	
	extern "C" MAGMABINDINGS_API int mbv2dgeevs_cpu(int n, double* A, int lda, double* wr, double* wi, double* VL, int ldvl, double* VR, int ldvr);
	extern "C" MAGMABINDINGS_API int mbv2dgeev_cpu(mbv2vector jobvl, mbv2vector jobvr, int n, double* A, int lda, double* wr, double* wi, double* Vl, int ldvl, double* Vr, int ldvr);

}
