
#include "pch.h"
#include "lblapack.h"
#include <stdio.h>
#include <mkl_lapacke.h>
#include <crtdbg.h>

namespace LapackBinding
{
	
	int mbv2dgesv_cpu(int n, int nrhs, double* A, int lda, int* ipiv, double* B, int lbd)
	{
		int info;
		info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, A, lda, ipiv, B, lbd);
		return info;

	}

	int mbv2sgesv_cpu(int n, int nrhs, float* A, int lda, int* ipiv, float* B, int lbd)
	{
		int info;
		info = LAPACKE_sgesv(LAPACK_COL_MAJOR, n, nrhs, A, lda, ipiv, B, lbd);
		return info;

	}

	//SVD
	int mbv2sgesvds_cpu(int m, int n, float* A, float* s, float* U, float* VT)
	{
		//U and V matrices
		mbv2vector jobU = mbv2vector::MagmaAllVec;
		mbv2vector jobV = mbv2vector::MagmaAllVec;
		int lda = m;
		int ldu = m;
		int ldvt = n;
		return mbv2sgesvd_cpu(jobU, jobV, m, n, A, lda, s, U, ldu, VT, ldvt);
	}
	
	int mbv2sgesvd_cpu(mbv2vector jobu, mbv2vector jobv, int m, int n,
		float* A, int lda, float* s, float* U, int ldu, float* VT, int ldvt)
	{
		
		int info;
		//convert job
		char jobU = convertToChar(jobu);
		char jobV = convertToChar(jobv);

		//query for workspace size
		const int dim = min(m, n) - 1;
		float* superb = (float*)malloc(dim * sizeof(float));
		
		/* Compute SVD */
		info = LAPACKE_sgesvd(LAPACK_COL_MAJOR, jobU, jobV, m, n, A, lda, s, U, ldu, VT, ldvt, superb);

		/* Check for convergence */
		if (info > 0)
		{
			printf("The algorithm computing SVD failed to converge.\n");
		}

		free(superb); // free host memory
		return info;
	}
	
	int mbv2dgesvd_cpu(mbv2vector jobu, mbv2vector jobv, int m, int n,
		double* A, int lda, double* s, double* U, int ldu, double* VT, int ldvt)
	{
		int info;
		//convert job
		char jobU = convertToChar(jobu);
		char jobV = convertToChar(jobv);

		//query for workspace size
		int dim = min(m, n) - 1;
		double* superb = (double*)malloc(dim * sizeof(double));

		/* Compute SVD */
		info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, jobU, jobV, m, n, A, lda, s, U, ldu, VT, ldvt, superb);

		/* Check for convergence */
		if (info > 0) {
			printf("The algorithm computing SVD failed to converge.\n");
		}

		free(superb); // free host memory
		return info;
	}
	int mbv2dgesvds_cpu(int m, int n, double* A, double* s, double* U, double* VT)
	{
		//U and V matrices
		mbv2vector jobU = mbv2vector::MagmaAllVec;
		mbv2vector jobV = mbv2vector::MagmaAllVec;

		return mbv2dgesvd_cpu(jobU, jobV, m, n, A, m, s, U, m, VT, n);
	}
	
	//LSS
	int mbv2sgels_cpu(int m, int n, int nrhs, float* A, int lda, float* B, int ldb)
	{
		//declare helpers
		int info;
		
		/* Solve the equations A*X = B */
		info = LAPACKE_sgels(LAPACK_COL_MAJOR,'N', m, n, nrhs, A, lda, B, ldb);


		/* Check for the full rank */
		if (info > 0) {
			printf("The diagonal element %i of the triangular factor ", info);
			printf("of A is zero, so that A does not have full rank;\n");
			printf("the least squares solution could not be computed.\n");
		}

		return info;

	}

	int mbv2dgels_cpu(int m, int n, int nrhs, double* A, int lda, double* B, int ldb)
	{
		//declare helpers
		int info;

		/* Solve the equations A*X = B */
		info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', m, n, nrhs, A, lda, B, ldb);


		/* Check for the full rank */
		if (info > 0) {
			printf("The diagonal element %i of the triangular factor ", info);
			printf("of A is zero, so that A does not have full rank;\n");
			printf("the least squares solution could not be computed.\n");
		}

		return info;

	}

	//EIGEN
	int mbv2sgeevs_cpu(int n, float* A, int lda, float* wr, float* wi, float* Vl, int ldvl, float* Vr, int ldvr)
	{
		//left and right matrices
		mbv2vector jjobvl = mbv2vector::MagmaNoVec;
		mbv2vector jjobvr = mbv2vector::MagmaNoVec;

		return mbv2sgeev_cpu(jjobvl, jjobvr, n, A, lda, wr, wi, Vl, ldvl, Vr, ldvr);
	}
	
	int mbv2sgeev_cpu(mbv2vector jobvl, mbv2vector jobvr, int n, float* A, int lda, float* wr, float* wi, float* Vl, int ldvl, float* Vr, int ldvr)
	{
		int info;
		//convert job
		char jjobvl = convertToChar(jobvl);
		char jjobvr = convertToChar(jobvr);

		/* Compute SVD */
		info = LAPACKE_sgeev(LAPACK_COL_MAJOR,jjobvl, jjobvr, n, A, lda, wr, wi, Vl, ldvr, Vr, ldvr);

		/* Check for convergence */
		if (info > 0) 
		{
			printf("The algorithm computing Eigen values failed to converge.\n");
		}

		//
		return info;
	}
	
	int mbv2dgeevs_cpu(int n, double* A, int lda, double* wr, double* wi, double* Vl, int ldvl, double* Vr, int ldvr)
	{
		//left and right matrices
		mbv2vector jjobvl = mbv2vector::MagmaNoVec;
		mbv2vector jjobvr = mbv2vector::MagmaNoVec;

		return mbv2dgeev_cpu(jjobvl, jjobvr, n, A, lda, wr, wi, Vl, ldvl, Vr, ldvr);
	}

	int mbv2dgeev_cpu(mbv2vector jobvl, mbv2vector jobvr, int n, double* A, int lda, double* wr, double* wi, double* Vl, int ldvl,double* Vr, int ldvr)
	{
		int info;
		//convert job
		char jjobvl = convertToChar(jobvl);
		char jjobvr = convertToChar(jobvr);

		/* Compute SVD */
		info = LAPACKE_dgeev(LAPACK_COL_MAJOR, jjobvl, jjobvr, n, A, lda, wr, wi, Vl, ldvr, Vr, ldvr);

		/* Check for convergence */
		if (info > 0) {
			printf("The algorithm computing Eigen values failed to converge.\n");
		}

		//
		return info;
	}

	//Util
	char convertToChar(mbv2vector vec)
	{
		switch (vec)
		{
			case mbv2vector::MagmaNoVec:
				return 'N';
				break;
			case mbv2vector::MagmaAllVec:
				return 'A';
			case mbv2vector::MagmaSomeVec:
				return 'S';
			case mbv2vector::MagmaOverwriteVec:
				return 'O';
			default:
				return 'N';
		}
		return 'N';
	}


}

