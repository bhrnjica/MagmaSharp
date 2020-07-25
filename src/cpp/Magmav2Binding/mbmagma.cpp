
#include "pch.h"
#include "mbmagma.h"

#include <stdio.h>
#include <cuda.h>
#include "magma_v2.h"
#include "magma_lapack.h"
#include "mkl_lapacke.h"

namespace MagmaBinding
{
	//Helper
	extern magma_vec_t convertToVec(mbv2vector vec);

	/// <summary>
	/// DGESV solves a system of linear equations A * X = B where A is a general N-by-N matrix and X and B are N-by-NRHS matrices.
	/// The LU decomposition with partial pivoting and row interchanges is used to factor A as A = P * L * U, where P is a permutation matrix, 
	///  L is unit lower triangular, and U is upper triangular. The factored form of A is then used to solve the system of equations A * X = B.
	/// </summary>
	/// <param name="n">The number of linear equations, that is, the order of the matrix A; n≥ 0.</param>
	/// <param name="nrhs">The number of right-hand sides, that is, the number of columns of the matrix B; nrhs≥ 0.</param>
	/// <param name="A"> Array, dimension (LDA,N). On entry, the M-by-N matrix to be factored. On exit, the factors L and U from the factorization 
	///  A = P*L*U; the unit diagonal elements of L are not stored.</param>
	/// <param name="lda">The leading dimension of the array A. LDA >= max(1,N).</param>
	/// <param name="ipiv"> Array, dimension (min(M,N)) The pivot indices; for 1 <= i <= min(M,N), row i of the matrix was interchanged with row IPIV(i).</param>
	/// <param name="B">array, dimension (LDB,NRHS) On entry, the right hand side matrix B. On exit, the solution matrix X.</param>
	/// <param name="lbd">The leading dimension of the array B. LDB >= max(1,N).</param>
	/// <param name="info">		= 0: successful exit; 
	/// 						< 0: if INFO = -i, the i - th argument had an illegal value.</param>
	/// <returns></returns>
	int mbv2dgesv(int n, int nrhs, double* A, int lda, int* ipiv, double* B, int lbd)
	{
		magma_init(); // initialize Magma
		int info;
		// solve the linear system a*x=c
		// c -mxn matrix , a -mxm matrix ;
		// c is overwritten by the solution
		magma_dgesv(n, nrhs, A, lda, ipiv, B, lbd, &info);

		//printf(" magma_dgesv time : %7.5 f sec .\n", gpu_time); // time

		magma_finalize(); // finalize Magma

		return info;
	}

	/// <summary>
	/// DGESV solves a system of linear equations A * X = B where A is a general N-by-N matrix and X and B are N-by-NRHS matrices.
	/// The LU decomposition with partial pivoting and row interchanges is used to factor A as A = P * L * U, where P is a permutation matrix, 
	/// L is unit lower triangular, and U is upper triangular. The factored form of A is then used to solve the system of equations A * X = B.
	/// </summary>
	/// <param name="n">The number of linear equations, that is, the order of the matrix A; n≥ 0.</param>
	/// <param name="nrhs">The number of right-hand sides, that is, the number of columns of the matrix B; nrhs≥ 0.</param>
	/// <param name="A"> Array, dimension (LDA,N). On entry, the M-by-N matrix to be factored. On exit, the factors L and U from the factorization 
	///  A = P*L*U; the unit diagonal elements of L are not stored.</param>
	/// <param name="lda">The leading dimension of the array A. LDA >= max(1,N).</param>
	/// <param name="ipiv"> Array, dimension (min(M,N)) The pivot indices; for 1 <= i <= min(M,N), row i of the matrix was interchanged with row IPIV(i).</param>
	/// <param name="B">array, dimension (LDB,NRHS) On entry, the right hand side matrix B. On exit, the solution matrix X.</param>
	/// <param name="lbd">The leading dimension of the array B. LDB >= max(1,N).</param>
	/// <param name="info">		= 0: successful exit; 
	/// 						< 0: if INFO = -i, the i - th argument had an illegal value.</param>
	/// <returns></returns>
	int mbv2dgesv_gpu(int n, int nrhs, double* A, int ldda, int* ipiv, double* B, int lddb)
	{
		magma_init();
		int info;
		magma_queue_t queue = NULL;
		magma_int_t dev = 0;
		magma_queue_create(dev, &queue);
		//
		int nxn = n * n;//squared matrix of the system
		int nxm = nrhs * n;//mostly B matrix is n x 1

		/* Copy matrix C from the CPU to the GPU */
		double* dA, * dX;
		magma_dmalloc(&dA, nxn);
		magma_dmalloc(&dX, nxm);
		assert(dA != nullptr);
		assert(dX != nullptr);

		// ... fill in dA and dX (on GPU)
		magma_dsetmatrix(n, n, A, ldda, dA, ldda, queue); // copy A -> dA
		magma_dsetmatrix(n, nrhs, B, lddb, dX, lddb, queue); // copy B -> dX

		// solve AX = B where B is in X
		magma_dgesv_gpu(n, nrhs, dA, ldda, ipiv, dX, lddb, &info);

		//retrieve the results matrix X from GPU
		magma_dgetmatrix(n, nrhs, dX, lddb, B, lddb, queue);
		magma_dgetmatrix(n, n, dA, ldda, A, ldda, queue);

		magma_free(dA);// free host memory
		magma_free(dX);// free host memory
		magma_queue_destroy(queue); // destroy queue
		magma_finalize();
		return info;
	}

	
	/// <summary>
	/// DGESV solves a system of linear equations A * X = B where A is a general N-by-N matrix and X and B are N-by-NRHS matrices.
	/// The LU decomposition with partial pivoting and row interchanges is used to factor A as A = P * L * U, where P is a permutation matrix, 
	///  L is unit lower triangular, and U is upper triangular. The factored form of A is then used to solve the system of equations A * X = B.
	/// </summary>
	/// <param name="n">The number of linear equations, that is, the order of the matrix A; n≥ 0.</param>
	/// <param name="nrhs">The number of right-hand sides, that is, the number of columns of the matrix B; nrhs≥ 0.</param>
	/// <param name="A"> Array, dimension (LDA,N). On entry, the M-by-N matrix to be factored. On exit, the factors L and U from the factorization 
	///  A = P*L*U; the unit diagonal elements of L are not stored.</param>
	/// <param name="lda">The leading dimension of the array A. LDA >= max(1,N).</param>
	/// <param name="ipiv"> Array, dimension (min(M,N)) The pivot indices; for 1 <= i <= min(M,N), row i of the matrix was interchanged with row IPIV(i).</param>
	/// <param name="B">array, dimension (LDB,NRHS) On entry, the right hand side matrix B. On exit, the solution matrix X.</param>
	/// <param name="lbd">The leading dimension of the array B. LDB >= max(1,N).</param>
	/// <param name="info">		= 0: successful exit; 
	/// 						< 0: if INFO = -i, the i - th argument had an illegal value.</param>
	/// <returns></returns>
	int mbv2sgesv(int n, int nrhs, float* A, int lda, int* ipiv, float* B, int lbd)
	{
		magma_init(); // initialize Magma
		//real_Double_t gpu_time;
		int info;
		// solve the linear system a*x=c
		// c -mxn matrix , a -mxm matrix ;
		// c is overwritten by the solution
		//gpu_time = magma_sync_wtime(NULL);

		magma_sgesv(n, nrhs, A, lda, ipiv, B, lbd, &info);

		//gpu_time = magma_sync_wtime(NULL) - gpu_time;


		//printf(" magma_dgesv time : %7.5 f sec .\n", gpu_time); // time

		magma_finalize(); // finalize Magma

		return info;
	}

	/// <summary>
	/// DGESV solves a system of linear equations A * X = B where A is a general N-by-N matrix and X and B are N-by-NRHS matrices.
	/// The LU decomposition with partial pivoting and row interchanges is used to factor A as A = P * L * U, where P is a permutation matrix, 
	/// L is unit lower triangular, and U is upper triangular. The factored form of A is then used to solve the system of equations A * X = B.
	/// </summary>
	/// <param name="n">The number of linear equations, that is, the order of the matrix A; n≥ 0.</param>
	/// <param name="nrhs">The number of right-hand sides, that is, the number of columns of the matrix B; nrhs≥ 0.</param>
	/// <param name="A"> Array, dimension (LDA,N). On entry, the M-by-N matrix to be factored. On exit, the factors L and U from the factorization 
	///  A = P*L*U; the unit diagonal elements of L are not stored.</param>
	/// <param name="lda">The leading dimension of the array A. LDA >= max(1,N).</param>
	/// <param name="ipiv"> Array, dimension (min(M,N)) The pivot indices; for 1 <= i <= min(M,N), row i of the matrix was interchanged with row IPIV(i).</param>
	/// <param name="B">array, dimension (LDB,NRHS) On entry, the right hand side matrix B. On exit, the solution matrix X.</param>
	/// <param name="lbd">The leading dimension of the array B. LDB >= max(1,N).</param>
	/// <param name="info">		= 0: successful exit; 
	/// 						< 0: if INFO = -i, the i - th argument had an illegal value.</param>
	/// <returns></returns>
	int mbv2sgesv_gpu(int n, int nrhs, float* A, int ldda, int* ipiv, float* B, int lddb)
	{
		magma_init();
		int info;
		magma_queue_t queue = NULL;
		magma_int_t dev = 0;
		magma_queue_create(dev, &queue);
		//
		int nxn = n * n;//squared matrix of the system
		int nxm = nrhs * n;//mostly B matrix is n x 1

		/* Copy matrix C from the CPU to the GPU */
		float* dA, * dX;
		magma_smalloc(&dA, nxn);
		magma_smalloc(&dX, nxm);
		assert(dA != nullptr);
		assert(dX != nullptr);

		// ... fill in dA and dX (on GPU)
		magma_ssetmatrix(n, n, A, ldda, dA, ldda, queue); // copy A -> dA
		magma_ssetmatrix(n, nrhs, B, lddb, dX, lddb, queue); // copy B -> dX

		// solve AX = B where B is in X
		magma_sgesv_gpu(n, nrhs, dA, ldda, ipiv, dX, lddb, &info);

		//retrieve the results matrix X from GPU
		magma_sgetmatrix(n, nrhs, dX, lddb, B, lddb, queue);
		magma_sgetmatrix(n, n, dA, ldda, A, ldda, queue);

		magma_free(dA);// free host memory
		magma_free(dX);// free host memory
		magma_queue_destroy(queue); // destroy queue
		magma_finalize();
		return 0;
	}

	//SVD
	int mbv2sgesvds(int m, int n, float* A, float* s, float* U, float* VT)
	{
		//U and V matrices
		mbv2vector jobU = mbv2vector::MagmaAllVec;
		mbv2vector jobV = mbv2vector::MagmaAllVec;
		
		int info =  mbv2sgesvd(jobU, jobV, m, n, A, m, s, U, m, VT, n);
		return info;
	}

	
	int mbv2sgesvd(mbv2vector jobu, mbv2vector jobv, int m, int n, float* A, int lda, float* s, float* U, int ldu, float* VT, int ldvt)
	{
		//magma_print_environment();
		magma_init();

		// Constants
		const magma_int_t ione = 1;
		const magma_int_t ineg_one = -1;
		const double d_neg_one = -1;
		const double nan = MAGMA_D_NAN;
		int info;
		//convert job
		magma_vec_t jobU = convertToVec(jobu);
		magma_vec_t jobV = convertToVec(jobv);

		//query for workspace size
		magma_int_t query_magma;
		float dummy[1];
		float* h_work; // h_work - workspace
		int inf = magma_sgesvd(jobU, jobV, m, n, NULL, lda, NULL, NULL, ldu, NULL, ldvt, dummy, ineg_one, &info);

		assert(inf == 0);
		query_magma = (magma_int_t)MAGMA_D_REAL(dummy[0]);
		magma_smalloc_pinned(&h_work, query_magma); // host memory for h_work

		/* Compute SVD */
		inf = magma_sgesvd(jobU, jobV, m, n, A, lda, s, U, ldu, VT, ldvt, h_work, query_magma, &info);

		/* Check for convergence */
		if (inf > 0) {
			printf("The algorithm computing SVD failed to converge.\n");
		}

		// Free memory
		magma_free_pinned(h_work); // free host memory
		magma_finalize(); // finalize Magma

		return info;
	}

	int mbv2dgesvds(int m, int n, double* A, double* s, double* U, double* VT)
	{
		//U and V matrices
		mbv2vector jobU = mbv2vector::MagmaAllVec;
		mbv2vector jobV = mbv2vector::MagmaAllVec;
		
		return mbv2dgesvd(jobU, jobV, m, n, A, m, s, U, m, VT, n);
	}

	int mbv2dgesvd(mbv2vector jobu, mbv2vector jobv, int m, int n,
		double* A, int lda, double* s, double* U, int ldu, double* VT, int ldvt)
	{
		magma_init();
		int info;
		// Constants
		const magma_int_t ione = 1;
		const magma_int_t ineg_one = -1;
		const double d_neg_one = -1;
		
		//convert job
		magma_vec_t jobU = convertToVec(jobu);
		magma_vec_t jobV = convertToVec(jobv);

		//query for workspace size
		magma_int_t query_magma;
		double dummy[1];
		double* h_work; // h_work - workspace
		int inf = magma_dgesvd(jobU, jobV, m, n, NULL, lda, NULL, NULL, ldu, NULL, ldvt, dummy, ineg_one, &info);

		assert(inf == 0);
		query_magma = (magma_int_t)MAGMA_D_REAL(dummy[0]);
		magma_dmalloc_pinned(&h_work, query_magma); // host memory for h_work

		/* Compute SVD */
		inf = magma_dgesvd(jobU, jobV, m, n, A, lda, s, U, ldu, VT, ldvt, h_work, query_magma, &info);
		
		/* Check for convergence */
		if (inf > 0) {
			printf("The algorithm computing SVD failed to converge.\n");
		}

		//Free memory
		magma_free_pinned(h_work); // free host memory
		magma_finalize(); // finalize Magma

		return info;
	}



	//LSS

	int mbv2sgels(int m, int n, int nrhs, float* A, int lda, float* B, int ldb)
	{
		magma_init();
		//declare helpers
		int info, lwork;
		float wkopt;
		float* work;

		/* Query and allocate the optimal workspace */
		lwork = -1;
		magma_sgels(magma_trans_t::MagmaNoTrans, m, n, nrhs, A, lda, B, ldb, &wkopt, lwork, &info);

		lwork = (int)wkopt;
		work = (float*)malloc(lwork * sizeof(float));


		/* Solve the equations A*X = B */
		magma_sgels(magma_trans_t::MagmaNoTrans, m, n, nrhs, A, lda, B, ldb, work, lwork, &info);


		/* Check for the full rank */
		if (info > 0) {
			printf("The diagonal element %i of the triangular factor ", info);
			printf("of A is zero, so that A does not have full rank;\n");
			printf("the least squares solution could not be computed.\n");
		}

		/* Free workspace */
		free((void*)work);
		magma_finalize();
		return info;

	}

	int mbv2sgels_gpu(int m, int n, int nrhs, float* A, int lda, float* B, int ldb)
	{
		magma_init();
		//declare helpers
		const magma_int_t ineg_one = -1;
		int info, lwork;
		float wkopt[1];
		float* work;
		magma_queue_t queue = NULL;
		magma_int_t dev = 0;
		magma_queue_create(dev, &queue);

		/* Query and allocate the optimal workspace */
		magma_sgels_gpu(magma_trans_t::MagmaNoTrans, m, n, nrhs, NULL, lda, NULL, ldb, wkopt, ineg_one, &info);

		lwork = (magma_int_t)MAGMA_D_REAL(wkopt[0]);
		magma_smalloc_pinned(&work, lwork); // host memory for h_work

		/* Copy matrix C from the CPU to the GPU */
		float* dA, * dB;
		magma_smalloc(&dA, (const int)(lda * n));
		magma_smalloc(&dB, (const int)(ldb * nrhs));
		assert(dA != nullptr);
		assert(dB != nullptr);

		// ... fill in dA and dX (on GPU)
		magma_ssetmatrix(lda, n, A, lda, dA, lda, queue); // copy A -> dA
		magma_ssetmatrix(ldb, nrhs, B, ldb, dB, ldb, queue); // copy B -> dB

		/* Solve the equations A*X = B */
		magma_sgels_gpu(magma_trans_t::MagmaNoTrans, m, n, nrhs, dA, lda, dB, ldb, work, lwork, &info);


		/* Check for the full rank */
		if (info > 0) {
			printf("The diagonal element %i of the triangular factor ", info);
			printf("of A is zero, so that A does not have full rank;\n");
			printf("the least squares solution could not be computed.\n");
		}

		//retrieve the results matrix X from GPU
		magma_sgetmatrix(ldb, nrhs, dB, ldb, B, ldb, queue);
		magma_sgetmatrix(lda, n, dA, lda, A, lda, queue);

		magma_free(dA);// free host memory
		magma_free(dB);// free host memory
		magma_queue_destroy(queue); // destroy queue
		magma_free_pinned(work); // free host memory
		magma_finalize();
		return info;

	}
	
	int mbv2dgels(int m, int n, int nrhs, double* A, int lda, double* B, int ldb)
	{
		magma_init();
		//declare helpers
		int info;
		int lwork = -1;
		double wkopt;
		double* work;

		/* Query and allocate the optimal workspace */
		magma_dgels(magma_trans_t::MagmaNoTrans, m, n, nrhs, A, lda, B, ldb, &wkopt, lwork, &info);

		lwork = (int)wkopt;
		work = (double*)malloc(lwork * sizeof(double));


		/* Solve the equations A*X = B */
		magma_dgels(magma_trans_t::MagmaNoTrans, m, n, nrhs, A, lda, B, ldb, work, lwork, &info);


		/* Check for the full rank */
		if (info > 0) {
			printf("The diagonal element %i of the triangular factor ", info);
			printf("of A is zero, so that A does not have full rank;\n");
			printf("the least squares solution could not be computed.\n");
		}

		/* Free workspace */
		free((void*)work);
		magma_finalize();
		return info;
	}

	int mbv2dgels_gpu(int m, int n, int nrhs, double* A, int lda, double* B, int ldb)
	{
		magma_init();
		//declare helpers
		int ineg_one = -1;
		int info;
		double wkopt[1];
		double* work;
		magma_queue_t queue = NULL;
		magma_int_t dev = 0;
		magma_queue_create(dev, &queue);

		/* Query and allocate the optimal workspace */
		magma_dgels_gpu(magma_trans_t::MagmaNoTrans, m, n, nrhs, NULL, lda, NULL, ldb, wkopt, ineg_one, &info);

		int lwork = (int)MAGMA_D_REAL(wkopt[0]);
		magma_dmalloc_pinned(&work, lwork); // host memory for h_work

		/* Copy matrix C from the CPU to the GPU */
		double* dA, * dB;
		magma_dmalloc(&dA, (const int)(lda * n));
		magma_dmalloc(&dB, (const int)(ldb * nrhs));
		assert(dA != nullptr);
		assert(dB != nullptr);

		// ... fill in dA and dX (on GPU)
		magma_dsetmatrix(m, n, A, lda, dA, lda, queue); // copy A -> dA
		magma_dsetmatrix(ldb, nrhs, B, ldb, dB, ldb, queue); // copy B -> dB

		/* Solve the equations A*X = B */
		magma_dgels_gpu(magma_trans_t::MagmaNoTrans, m, n, nrhs, dA, lda, dB, ldb, work, lwork, &info);


		/* Check for the full rank */
		if (info > 0) {
			printf("The diagonal element %i of the triangular factor ", info);
			printf("of A is zero, so that A does not have full rank;\n");
			printf("the least squares solution could not be computed.\n");
		}

		//retrieve the results matrix X from GPU
		magma_dgetmatrix(ldb, nrhs, dB, ldb, B, ldb, queue);
		magma_dgetmatrix(lda, n, dA, lda, A, lda, queue);

		magma_free(dA);// free host memory
		magma_free(dB);// free host memory
		magma_queue_destroy(queue); // destroy queue
		magma_free_pinned(work); // free host memory
		magma_finalize();
		return info;

	}

	//EIGEN
	int mbv2sgeevs(int n, float* A, int lda, float* wr, float* wi, float* Vl, int ldvl, float* Vr, int ldvr)
	{
		//left and right matrices
		mbv2vector jjobvl = mbv2vector::MagmaNoVec;
		mbv2vector jjobvr = mbv2vector::MagmaNoVec;

		return mbv2sgeev(jjobvl, jjobvr, n, A, lda, wr, wi, Vl, ldvl, Vr, ldvr);
	}
	
	int mbv2sgeev(mbv2vector jobvl, mbv2vector jobvr, int n, float* A, int lda, float* wr, float* wi, float* Vl, int ldvl,float* Vr, int ldvr)
	{
		magma_init();

		// Constants
		const magma_int_t ineg_one = -1;
		int info;
		//convert job
		magma_vec_t jjobvl = convertToVec(jobvl);
		magma_vec_t jjobvr = convertToVec(jobvr);

		//query for workspace size
		magma_int_t query_magma;
		float dummy[1];
		float* h_work; // h_work - workspace
		int inf = magma_sgeev(jjobvl, jjobvr, n, NULL, lda, NULL, NULL, NULL, ldvl, NULL, ldvr, dummy, ineg_one, &info);

		assert(inf == 0);
		query_magma = (magma_int_t)MAGMA_D_REAL(dummy[0]);
		magma_smalloc_pinned(&h_work, query_magma); // host memory for h_work

		/* Compute EIGEN */
		inf = magma_sgeev(jjobvl, jjobvr, n, A, lda, wr, wi, Vl, ldvr, Vr, ldvr, h_work, query_magma, &info);

		/* Check for convergence */
		if (inf > 0) {
			printf("The algorithm computing Eigen values failed to converge.\n");
		}

		// Free memory
		magma_free_pinned(h_work); // free host memory
		magma_finalize(); // finalize Magma

		return info;
	}

	int mbv2dgeevs(int n, double* A, int lda, double* wr, double* wi, double* Vl, int ldvl, double* Vr, int ldvr)
	{
		//left and right matrices
		mbv2vector jjobvl = mbv2vector::MagmaNoVec;
		mbv2vector jjobvr = mbv2vector::MagmaNoVec;

		return mbv2dgeev(jjobvl, jjobvr, n, A, lda, wr, wi, Vl, ldvl, Vr, ldvr);
	}

	int mbv2dgeev(mbv2vector jobvl, mbv2vector jobvr, int n, double* A, int lda, double* wr, double* wi, double* Vl, int ldvl,double* Vr, int ldvr)
	{
		magma_init();

		// Constants
		const magma_int_t ineg_one = -1;
		int info;
		//convert job
		magma_vec_t jjobvl = convertToVec(jobvl);
		magma_vec_t jjobvr = convertToVec(jobvr);

		//query for workspace size
		magma_int_t query_magma;
		double dummy[1];
		double* h_work; // h_work - workspace
		int inf = magma_dgeev(jjobvl, jjobvr, n, NULL, lda, NULL, NULL, NULL, ldvl, NULL, ldvr, dummy, ineg_one, &info);

		assert(inf == 0);
		query_magma = (magma_int_t)MAGMA_D_REAL(dummy[0]);
		magma_dmalloc_pinned(&h_work, query_magma); // host memory for h_work

		/* Compute SVD */
		inf = magma_dgeev(jjobvl, jjobvr, n, A, lda, wr, wi, Vl, ldvr, Vr, ldvr, h_work, query_magma, &info);

		/* Check for convergence */
		if (inf > 0) {
			printf("The algorithm computing Eigen values failed to converge.\n");
		}

		// Free memory
		magma_free_pinned(h_work); // free host memory
		magma_finalize(); // finalize Magma

		return info;
	}
	

	//Util
	magma_vec_t convertToVec(mbv2vector vec)
	{
		switch (vec)
		{
		case mbv2vector::MagmaNoVec:
			return magma_vec_t::MagmaNoVec;
			break;
		case mbv2vector::MagmaVec:
			return magma_vec_t::MagmaVec;
		case mbv2vector::MagmaIVec:
			return magma_vec_t::MagmaIVec;
		case mbv2vector::MagmaAllVec:
			return magma_vec_t::MagmaAllVec;
		case mbv2vector::MagmaSomeVec:
			return magma_vec_t::MagmaSomeVec;
		case mbv2vector::MagmaOverwriteVec:
			return magma_vec_t::MagmaOverwriteVec;
		case mbv2vector::MagmaBacktransVec:
			return magma_vec_t::MagmaBacktransVec;
		default:
			return magma_vec_t::MagmaNoVec;
		}
		return magma_vec_t::MagmaNoVec;
	}

}

