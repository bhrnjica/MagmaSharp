
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
	extern void transpose(float* src, float* dst, const int N, const int M);
	extern void transpose(double* src, double* dst, const int N, const int M);
	extern void print_matrix(char* desc, int m, int n, float* a, const int lda);
	extern void print_matrix(char* desc, int m, int n, double* a, const int lda);
	extern void reverseVector(float* arr, int start, int end);
	extern void reverseVector(double* arr, int start, int end);
	
	int mbv2dgesv(bool rowmajor, int n, int nrhs, double* A, int lda, int* ipiv, double* B, int ldb)
	{
		magma_init(); // initialize Magma

		int info;
		double* At = NULL, * Bt = NULL;

		if (rowmajor)
		{
			magma_dmalloc_pinned(&At, (const int)(lda * n));
			magma_dmalloc_pinned(&Bt, (const int)(n * nrhs));

			//
			transpose(A, At, lda, n);
			transpose(B, Bt, ldb, nrhs);
			magma_dgesv(n, nrhs, At, lda, ipiv, Bt, ldb, &info);

			//transpose results into row major matrix
			transpose(At, A, n, lda);
			transpose(Bt, B, nrhs, ldb);
		}
		else
			magma_dgesv(n, nrhs, A, lda, ipiv, B, ldb, &info);

		//free memory
		magma_free_pinned(At);
		magma_free_pinned(Bt);

		magma_finalize(); // finalize Magma

		return info;
	}

	
	int mbv2dgesv_gpu(bool rowmajor, int n, int nrhs, double* A, int ldda, int* ipiv, double* B, int lddb)
	{
		magma_init();

		int info;
		magma_queue_t queue = NULL;
		magma_int_t dev = 0;
		magma_queue_create(dev, &queue);

		//
		int nxn = ldda * n;//squared matrix of the system
		int nxm = nrhs * lddb;//mostly B matrix is n x 1

		/* Copy matrix C from the CPU to the GPU */
		double* dA, * dX, * At = NULL, * Bt = NULL;
		if (rowmajor)
		{
			magma_dmalloc_pinned(&At, (const int)(nxn));
			magma_dmalloc_pinned(&Bt, (const int)(nxm));
			//
			transpose(A, At, ldda, n);
			transpose(B, Bt, lddb, nrhs);
		}

		magma_dmalloc(&dA, nxn);
		magma_dmalloc(&dX, nxm);

		if (rowmajor)
		{
			// ... fill in dA and dX (on GPU)
			magma_dsetmatrix(n, n, At, ldda, dA, ldda, queue); // copy A -> dA
			magma_dsetmatrix(n, nrhs, Bt, lddb, dX, lddb, queue); // copy B -> dX
		}
		else
		{
			// ... fill in dA and dX (on GPU)
			magma_dsetmatrix(n, n, A, ldda, dA, ldda, queue); // copy A -> dA
			magma_dsetmatrix(n, nrhs, B, lddb, dX, lddb, queue); // copy B -> dX
		}

		// solve AX = B where B is in X
		magma_dgesv_gpu(n, nrhs, dA, ldda, ipiv, dX, lddb, &info);

		if (rowmajor)
		{
			//retrieve the results matrix X from GPU
			magma_dgetmatrix(n, nrhs, dX, lddb, Bt, lddb, queue);
			magma_dgetmatrix(n, n, dA, ldda, At, ldda, queue);

			//
			transpose(At, A, n, ldda);
			transpose(Bt, B, nrhs, lddb);
		}
		else
		{
			//retrieve the results matrix X from GPU
			magma_dgetmatrix(n, nrhs, dX, lddb, B, lddb, queue);
			magma_dgetmatrix(n, n, dA, ldda, A, ldda, queue);
		}


		magma_free(dA);// free host memory
		magma_free(dX);// free host memory
		magma_free_pinned(At);
		magma_free_pinned(Bt);
		magma_queue_destroy(queue); // destroy queue
		magma_finalize();
		return 0;
	}

	
	int mbv2sgesv(bool rowmajor, int n, int nrhs, float* A, int lda, int* ipiv, float* B, int ldb)
	{
		magma_init(); // initialize Magma
		
		int info;
		float * At = NULL, *Bt = NULL;

		if (rowmajor)
		{
			magma_smalloc_pinned(&At, (const int)(lda * n));
			magma_smalloc_pinned(&Bt, (const int)(n * nrhs));

			//
			transpose(A, At, lda, n);
			transpose(B, Bt, ldb, nrhs);
			magma_sgesv(n, nrhs, At, lda, ipiv, Bt, ldb, &info);

			//transpose results into row major matrix
			transpose(At, A, n, lda);
			transpose(Bt, B, nrhs, ldb);
		}
		else
			magma_sgesv(n, nrhs, A, lda, ipiv, B, ldb, &info);
		
		//free memory
		magma_free_pinned(At);
		magma_free_pinned(Bt);

		magma_finalize(); // finalize Magma

		return info;
	}

	int mbv2sgesv_gpu(bool rowmajor, int n, int nrhs, float* A, int ldda, int* ipiv, float* B, int lddb)
	{
		magma_init();

		int info;
		magma_queue_t queue = NULL;
		magma_int_t dev = 0;
		magma_queue_create(dev, &queue);

		//
		int nxn = ldda * n;//squared matrix of the system
		int nxm = nrhs * lddb;//mostly B matrix is n x 1

		/* Copy matrix C from the CPU to the GPU */
		float* dA, * dX, *At = NULL, * Bt = NULL;
		if (rowmajor)
		{
			magma_smalloc_pinned(&At, (const int)(nxn));
			magma_smalloc_pinned(&Bt, (const int)(nxm));
			//
			transpose(A, At, ldda, n);
			transpose(B, Bt, lddb, nrhs);
		}

		magma_smalloc(&dA, nxn);
		magma_smalloc(&dX, nxm);

		if (rowmajor)
		{
			// ... fill in dA and dX (on GPU)
			magma_ssetmatrix(n, n, At, ldda, dA, ldda, queue); // copy A -> dA
			magma_ssetmatrix(n, nrhs, Bt, lddb, dX, lddb, queue); // copy B -> dX
		}
		else
		{
			// ... fill in dA and dX (on GPU)
			magma_ssetmatrix(n, n, A, ldda, dA, ldda, queue); // copy A -> dA
			magma_ssetmatrix(n, nrhs, B, lddb, dX, lddb, queue); // copy B -> dX
		}

		// solve AX = B where B is in X
		magma_sgesv_gpu(n, nrhs, dA, ldda, ipiv, dX, lddb, &info);

		if (rowmajor)
		{
			//retrieve the results matrix X from GPU
			magma_sgetmatrix(n, nrhs, dX, lddb, Bt, lddb, queue);
			magma_sgetmatrix(n, n, dA, ldda, At, ldda, queue);

			//
			transpose(At, A, n, ldda);
			transpose(Bt, B, nrhs, lddb);
		}
		else
		{
			//retrieve the results matrix X from GPU
			magma_sgetmatrix(n, nrhs, dX, lddb, B, lddb, queue);
			magma_sgetmatrix(n, n, dA, ldda, A, ldda, queue);
		}
		

		magma_free(dA);// free host memory
		magma_free(dX);// free host memory
		magma_free_pinned(At);
		magma_free_pinned(Bt);
		magma_queue_destroy(queue); // destroy queue
		magma_finalize();
		return 0;
	}

	//SVD
	int mbv2sgesvds(bool rowmajor, int m, int n, float* A, float* s, float* U, bool calcU, float* VT, bool calcV)
	{
		//U and V matrices
		mbv2vector jobV;
		mbv2vector jobU;

		if (calcU)
			jobU = mbv2vector::MagmaAllVec;
		else
			jobU = mbv2vector::MagmaNoVec;
		//
		if (calcV)
			jobV = mbv2vector::MagmaAllVec;
		else
			jobV = mbv2vector::MagmaNoVec;
		//
		int lda = m;
		int ldu = m;
		int ldvt = n;

		return mbv2sgesvd(rowmajor, jobU, jobV, m, n, A, lda, s, U, ldu, VT, ldvt);
	}

	int mbv2sgesvd(bool rowmajor, mbv2vector jobu, mbv2vector jobv, int m, int n, float* A, int lda, float* s, float* U, int ldu, float* VT, int ldvt)
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
		float* h_work, *At = NULL, *Ut = NULL, *VTt = NULL; 

		//calculate query
		int inf = magma_sgesvd(jobU, jobV, m, n, NULL, lda, NULL, NULL, ldu, NULL, ldvt, dummy, ineg_one, &info);

		assert(inf == 0);
		query_magma = (magma_int_t)MAGMA_D_REAL(dummy[0]);
		magma_smalloc_pinned(&h_work, query_magma); // host memory for h_work

		if(rowmajor)
			magma_smalloc_pinned(&At , (const int) (lda * n));

		//allocate left and right orthonormal matrices
		if(jobU== magma_vec_t::MagmaAllVec && rowmajor)
			magma_smalloc_pinned(&Ut, (const int)(m * m));
		if(jobV == magma_vec_t::MagmaAllVec && rowmajor)
			magma_smalloc_pinned(&VTt, (const int)(n * n));
		
		//define queue
		magma_queue_t queue = NULL;
		magma_int_t dev = 0;
		magma_queue_create(dev, &queue);

		/* Compute SVD */
		if (rowmajor)
		{
			//before compute transpose the matrix into col major 
			transpose(A, At, m, n);
			inf = magma_sgesvd(jobU, jobV, m, n, At, lda, s, Ut, ldu, VTt, ldvt, h_work, query_magma, &info);

			//transpose results into row major matrix
			transpose(At, A, m, n);

			//transpose left orthonormal matrix
			if (jobU == magma_vec_t::MagmaAllVec)
				transpose(Ut, U, m, m);

			//transpose left orthonormal matrix
			if (jobV == magma_vec_t::MagmaAllVec)
				transpose(VTt, VT, n, n);
		}
		else
			inf = magma_sgesvd(jobU, jobV, m, n, A, lda, s, U, ldu, VT, ldvt, h_work, query_magma, &info);

		/* Check for convergence */
		if (inf > 0) {
			printf("The algorithm computing SVD failed to converge.\n");
		}

		
		/*print_matrix((char*)"Left Matrix U=",m,m,U, m);
		print_matrix((char*)"RIGHT Matrix L=", n, n, VT, n);*/

		// Free memory
		magma_free_pinned(At); // free host memory
		magma_free_pinned(Ut); // free host memory
		magma_free_pinned(VTt); // free host memory
		magma_free_pinned(h_work); // free host memory
		magma_finalize(); // finalize Magma

		return info;
	}

	int mbv2dgesvds(bool rowmajor, int m, int n, double* A, double* s, double* U, bool calcU, double* VT, bool calcV)
	{
		//U and V matrices
		mbv2vector jobV;
		mbv2vector jobU;

		if (calcU)
			jobU = mbv2vector::MagmaAllVec;
		else
			jobU = mbv2vector::MagmaNoVec;
		//
		if (calcV)
			jobV = mbv2vector::MagmaAllVec;
		else
			jobV = mbv2vector::MagmaNoVec;
		//
		int lda = m;
		int ldu = m;
		int ldvt = n;

		return mbv2dgesvd(rowmajor, jobU, jobV, m, n, A, lda, s, U, ldu, VT, ldvt);

	}

	int mbv2dgesvd(bool rowmajor, mbv2vector jobu, mbv2vector jobv, int m, int n, double* A, int lda, double* s, double* U, int ldu, double* VT, int ldvt)
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
		double* h_work, * At=NULL, * Ut = NULL, * VTt = NULL; // h_work - workspace
		int inf = magma_dgesvd(jobU, jobV, m, n, NULL, lda, NULL, NULL, ldu, NULL, ldvt, dummy, ineg_one, &info);

		assert(inf == 0);
		query_magma = (magma_int_t)MAGMA_D_REAL(dummy[0]);
		magma_dmalloc_pinned(&h_work, query_magma); // host memory for h_work
		magma_dmalloc_pinned(&At, (const int)(lda * n));

		//allocate left and right orthonormal matrices
		if (jobU == magma_vec_t::MagmaAllVec && rowmajor)
			magma_dmalloc_pinned(&Ut, (const int)(m * m));

		if (jobV == magma_vec_t::MagmaAllVec && rowmajor)
			magma_dmalloc_pinned(&VTt, (const int)(n * n));

		//define queue
		magma_queue_t queue = NULL;
		magma_int_t dev = 0;
		magma_queue_create(dev, &queue);

		/* Compute SVD */
		if (rowmajor)
		{
			//before compute transpose the matrix into col major 
			transpose(A, At, m, n);
			inf = magma_dgesvd(jobU, jobV, m, n, At, lda, s, Ut, ldu, VTt, ldvt, h_work, query_magma, &info);
		}
		else
			inf = magma_dgesvd(jobU, jobV, m, n, A, lda, s, U, ldu, VT, ldvt, h_work, query_magma, &info);
		
		/* Check for convergence */
		if (inf > 0) {
			printf("The algorithm computing SVD failed to converge.\n");
		}

		// Free memory
		if (rowmajor)
		{
			//transpose results into row major matrix
			if (rowmajor)
				transpose(At, A, m, n);

			//transpose left orthonormal matrix
			if (jobU == magma_vec_t::MagmaAllVec && rowmajor)
				transpose(Ut, U, m, m);

			//transpose left orthonormal matrix
			if (jobV == magma_vec_t::MagmaAllVec)
				transpose(VTt, VT, n, n);

			/*print_matrix((char*)"Left Matrix U=",m,m,U, m);
			print_matrix((char*)"RIGHT Matrix L=", n, n, VT, n);*/

			magma_free_pinned(At); // free host memory
			if (jobU == magma_vec_t::MagmaAllVec && rowmajor)
				magma_free_pinned(Ut); // free host memory
			if (jobV == magma_vec_t::MagmaAllVec && rowmajor)
				magma_free_pinned(VTt); // free host memory
		}
		magma_free_pinned(h_work); // free host memory

		magma_finalize(); // finalize Magma

		return info;
	}



	//LSS

	int mbv2sgels(bool rowmajor, int m, int n, int nrhs, float* A, int lda, float* B, int ldb)
	{
		magma_init();
		//declare helpers
		int info, lwork;
		float wkopt;
		float* work, *At = NULL, *Bt = NULL;

		/* Query and allocate the optimal workspace */
		lwork = -1;
		magma_sgels(magma_trans_t::MagmaNoTrans, m, n, nrhs, A, lda, B, ldb, &wkopt, lwork, &info);

		lwork = (int)wkopt;
		work = (float*)malloc(lwork * sizeof(float));

		if (rowmajor)
		{
			magma_smalloc_pinned(&At, (const int)(lda * n));
			magma_smalloc_pinned(&Bt, (const int)(nrhs * m));

			//before compute transpose the matrix into col major 
			transpose(A, At, m, n);
			transpose(B, Bt, m, nrhs);

			/* Solve the equations A*X = B */
			magma_sgels(magma_trans_t::MagmaNoTrans, m, n, nrhs, At, lda, Bt, ldb, work, lwork, &info);

			//transpose results into row major matrix
			transpose(Bt, B, nrhs, m);	
		}
		else
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
		if (rowmajor)
		{
			magma_free_pinned(At); // free host memory
			magma_free_pinned(Bt); // free host memory
		}
		//
		magma_finalize();
		return info;

	}

	int mbv2sgels_gpu(bool rowmajor, int m, int n, int nrhs, float* A, int lda, float* B, int ldb)
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
		float* dA, * dB, * At = NULL, * Bt = NULL;;
		if (rowmajor)
		{
			magma_smalloc_pinned(&At, (const int)(m*n));
			magma_smalloc_pinned(&Bt, (const int)(nrhs*m));
			//
			transpose(A, At, lda, n);
			transpose(B, Bt, ldb, nrhs);
		}

		magma_smalloc(&dA, (const int)(lda * n));
		magma_smalloc(&dB, (const int)(ldb * nrhs));
		
		if (rowmajor)
		{
			// ... fill in dA and dX (on GPU)
			magma_ssetmatrix(lda, n, At, lda, dA, lda, queue); // copy A -> dA
			magma_ssetmatrix(ldb, nrhs, Bt, ldb, dB, ldb, queue); // copy B -> dX
		}
		else
		{
			// ... fill in dA and dX (on GPU)
			magma_ssetmatrix(lda, n, A, lda, dA, lda, queue); // copy A -> dA
			magma_ssetmatrix(ldb, nrhs, B, ldb, dB, ldb, queue); // copy B -> dB
		}

		

		/* Solve the equations A*X = B */
		magma_sgels_gpu(magma_trans_t::MagmaNoTrans, m, n, nrhs, dA, lda, dB, ldb, work, lwork, &info);


		/* Check for the full rank */
		if (info > 0) {
			printf("The diagonal element %i of the triangular factor ", info);
			printf("of A is zero, so that A does not have full rank;\n");
			printf("the least squares solution could not be computed.\n");
		}


		if (rowmajor)
		{
			//retrieve the results matrix X from GPU
			magma_sgetmatrix(ldb, nrhs, dB, ldb, Bt, ldb, queue);
			magma_sgetmatrix(lda, n, dA, lda, At, lda, queue);

			//
			transpose(At, A, n, lda);
			transpose(Bt, B, nrhs, ldb);
		}
		else
		{
			//retrieve the results matrix X from GPU
			magma_sgetmatrix(ldb, nrhs, dB, ldb, B, ldb, queue);
			magma_sgetmatrix(lda, n, dA, lda, A, lda, queue);

		}

		//Free memory
		magma_free(dA);// free host memory
		magma_free(dB);// free host memory
		if (rowmajor)
		{
			magma_free(At);// free host memory
			magma_free(Bt);// free host memory
		}
		magma_queue_destroy(queue); // destroy queue
		magma_free_pinned(work); // free host memory
		magma_finalize();
		return info;

	}
	
	int mbv2dgels(bool rowmajor, int m, int n, int nrhs, double* A, int lda, double* B, int ldb)
	{
		magma_init();
		//declare helpers
		int info;
		int lwork = -1;
		double wkopt;
		double* work, *At, *Bt;

		/* Query and allocate the optimal workspace */
		magma_dgels(magma_trans_t::MagmaNoTrans, m, n, nrhs, A, lda, B, ldb, &wkopt, lwork, &info);

		lwork = (int)wkopt;
		work = (double*)malloc(lwork * sizeof(double));

		if (rowmajor)
		{
			magma_dmalloc_pinned(&At, (const int)(lda * n));
			magma_dmalloc_pinned(&Bt, (const int)(nrhs * m));

			//before compute transpose the matrix into col major 
			transpose(A, At, m, n);
			transpose(B, Bt, m, nrhs);

			/* Solve the equations A*X = B */
			magma_dgels(magma_trans_t::MagmaNoTrans, m, n, nrhs, At, lda, Bt, ldb, work, lwork, &info);

			//transpose results into row major matrix
			transpose(Bt, B, nrhs, m);
		}
		else
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

	int mbv2dgels_gpu(bool rowmajor, int m, int n, int nrhs, double* A, int lda, double* B, int ldb)
	{
		magma_init();
		//declare helpers
		const magma_int_t ineg_one = -1;
		int info, lwork;
		double wkopt[1];
		double* work;
		magma_queue_t queue = NULL;
		magma_int_t dev = 0;
		magma_queue_create(dev, &queue);

		/* Query and allocate the optimal workspace */
		magma_dgels_gpu(magma_trans_t::MagmaNoTrans, m, n, nrhs, NULL, lda, NULL, ldb, wkopt, ineg_one, &info);

		lwork = (magma_int_t)MAGMA_D_REAL(wkopt[0]);
		magma_dmalloc_pinned(&work, lwork); // host memory for h_work

		/* Copy matrix C from the CPU to the GPU */
		double* dA, * dB, * At = NULL, * Bt = NULL;;
		if (rowmajor)
		{
			magma_dmalloc_pinned(&At, (const int)(m * n));
			magma_dmalloc_pinned(&Bt, (const int)(nrhs * m));
			//
			transpose(A, At, lda, n);
			transpose(B, Bt, ldb, nrhs);
		}

		magma_dmalloc(&dA, (const int)(lda * n));
		magma_dmalloc(&dB, (const int)(ldb * nrhs));

		if (rowmajor)
		{
			// ... fill in dA and dX (on GPU)
			magma_dsetmatrix(lda, n, At, lda, dA, lda, queue); // copy A -> dA
			magma_dsetmatrix(ldb, nrhs, Bt, ldb, dB, ldb, queue); // copy B -> dX
		}
		else
		{
			// ... fill in dA and dX (on GPU)
			magma_dsetmatrix(lda, n, A, lda, dA, lda, queue); // copy A -> dA
			magma_dsetmatrix(ldb, nrhs, B, ldb, dB, ldb, queue); // copy B -> dB
		}



		/* Solve the equations A*X = B */
		magma_dgels_gpu(magma_trans_t::MagmaNoTrans, m, n, nrhs, dA, lda, dB, ldb, work, lwork, &info);


		/* Check for the full rank */
		if (info > 0) {
			printf("The diagonal element %i of the triangular factor ", info);
			printf("of A is zero, so that A does not have full rank;\n");
			printf("the least squares solution could not be computed.\n");
		}


		if (rowmajor)
		{
			//retrieve the results matrix X from GPU
			magma_dgetmatrix(ldb, nrhs, dB, ldb, Bt, ldb, queue);
			magma_dgetmatrix(lda, n, dA, lda, At, lda, queue);

			//
			transpose(At, A, n, lda);
			transpose(Bt, B, nrhs, ldb);
		}
		else
		{
			//retrieve the results matrix X from GPU
			magma_dgetmatrix(ldb, nrhs, dB, ldb, B, ldb, queue);
			magma_dgetmatrix(lda, n, dA, lda, A, lda, queue);

		}

		//Free memory
		magma_free(dA);// free host memory
		magma_free(dB);// free host memory
		if (rowmajor)
		{
			magma_free(At);// free host memory
			magma_free(Bt);// free host memory
		}
		magma_queue_destroy(queue); // destroy queue
		magma_free_pinned(work); // free host memory
		magma_finalize();
		return info;

	}

	//EIGEN
	int mbv2sgeevs(bool rowmajor, int n, float* A, int lda, float* wr, float* wi, float* Vl, bool computeLeft, float* Vr, bool computeRight)
	{
		//left and right matrices
		mbv2vector jjobvl = mbv2vector::MagmaNoVec;
		mbv2vector jjobvr = mbv2vector::MagmaNoVec;

		return mbv2sgeev(rowmajor, jjobvl, jjobvr, n, A, lda, wr, wi, Vl, n, Vr, n);
	}
	
	int mbv2sgeev(bool rowmajor, mbv2vector jobvl, mbv2vector jobvr, int n, float* A, int lda, float* wr, float* wi, float* Vl, int ldvl,float* Vr, int ldvr)
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
		float* h_work, * At = NULL, * Vlt = NULL, * Vrt = NULL; // h_work - workspace
		int inf = magma_sgeev(jjobvl, jjobvr, n, NULL, lda, NULL, NULL, NULL, ldvl, NULL, ldvr, dummy, ineg_one, &info);

		assert(inf == 0);
		query_magma = (magma_int_t)MAGMA_D_REAL(dummy[0]);
		magma_smalloc_pinned(&h_work, query_magma); // host memory for h_work

		if (rowmajor)
		{
			magma_smalloc_pinned(&At, (const int)(lda * n));
			if (jjobvl == magma_vec_t::MagmaVec)
				magma_smalloc_pinned(&Vlt, (const int)(lda * n));

			if (jjobvr == magma_vec_t::MagmaVec)
				magma_smalloc_pinned(&Vrt, (const int)(lda * n));

			//before compute transpose the matrix into col major 
			transpose(A, At, n, n);

			/* Compute SVD */
			inf = magma_sgeev(jjobvl, jjobvr, n, At, lda, wr, wi, Vlt, ldvr, Vrt, ldvr, h_work, query_magma, &info);

			transpose(A, At, n, n);

			if (jjobvl == magma_vec_t::MagmaVec)
				transpose(Vlt, Vl, n, n);

			if (jjobvr == magma_vec_t::MagmaVec)
				transpose(Vrt, Vr, n, n);
		}
		else
			/* Compute SVD */
			inf = magma_sgeev(jjobvl, jjobvr, n, A, lda, wr, wi, Vl, ldvr, Vr, ldvr, h_work, query_magma, &info);


		/* Check for convergence */
		if (inf > 0) {
			printf("The algorithm computing Eigen values failed to converge.\n");
		}

		// Free memory
		magma_free_pinned(h_work); // free host memory
		magma_free_pinned(At); // free host memory
		magma_free_pinned(Vlt); // free host memory
		magma_free_pinned(Vrt); // free host memory
		magma_finalize(); // finalize Magma

		return info;
	}

	int mbv2dgeevs(bool rowmajor, int n, double* A, int lda, double* wr, double* wi, double* Vl, bool computeLeft, double* Vr, bool computeRight)
	{
		//left and right matrices
		mbv2vector jjobvl = mbv2vector::MagmaNoVec;
		mbv2vector jjobvr = mbv2vector::MagmaNoVec;
		if(computeLeft)
			jjobvl = mbv2vector::MagmaVec;
		if(computeRight)
			jjobvl = mbv2vector::MagmaVec;
		//
		return mbv2dgeev(rowmajor, jjobvl, jjobvr, n, A, lda, wr, wi, Vl, n, Vr, n);
	}

	int mbv2dgeev(bool rowmajor, mbv2vector jobvl, mbv2vector jobvr, int n, double* A, int lda, double* wr, double* wi, double* Vl, int ldvl,double* Vr, int ldvr)
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
		double* h_work, * At=NULL, * Vlt= NULL, * Vrt= NULL; // h_work - workspace
		int inf = magma_dgeev(jjobvl, jjobvr, n, NULL, lda, NULL, NULL, NULL, ldvl, NULL, ldvr, dummy, ineg_one, &info);

		assert(inf == 0);
		query_magma = (magma_int_t)MAGMA_D_REAL(dummy[0]);
		magma_dmalloc_pinned(&h_work, query_magma); // host memory for h_work

		if (rowmajor)
		{
			magma_dmalloc_pinned(&At, (const int)(lda * n));
			if (jjobvl == magma_vec_t::MagmaVec)
				magma_dmalloc_pinned(&Vlt, (const int)(lda * n));

			if (jjobvr == magma_vec_t::MagmaVec)
				magma_dmalloc_pinned(&Vrt, (const int)(lda * n));

			//before compute transpose the matrix into col major 
			transpose(A, At, n, n);

			/* Compute SVD */
			inf = magma_dgeev(jjobvl, jjobvr, n, At, lda, wr, wi, Vlt, ldvr, Vrt, ldvr, h_work, query_magma, &info);

			transpose(A, At, n, n);
			
			if(jjobvl== magma_vec_t::MagmaVec)
				transpose(Vlt, Vl, n, n);

			if(jjobvr == magma_vec_t::MagmaVec)
				transpose(Vrt, Vr, n, n);
		}
		else
			/* Compute SVD */
			inf = magma_dgeev(jjobvl, jjobvr, n, A, lda, wr, wi, Vl, ldvr, Vr, ldvr, h_work, query_magma, &info);
		

		/* Check for convergence */
		if (inf > 0) {
			printf("The algorithm computing Eigen values failed to converge.\n");
		}

		// Free memory
		magma_free_pinned(h_work); // free host memory
		magma_free_pinned(At); // free host memory
		magma_free_pinned(Vlt); // free host memory
		magma_free_pinned(Vrt); // free host memory
		magma_finalize(); // finalize Magma

		return info;
	}
	
	//Matrix-Matrix operations
	void mbv2sgemm(bool rowmajor, mbv2trans opA, mbv2trans opB, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc)
	{
		magma_init();

		magma_trans_t trA = magma_trans_t::MagmaNoTrans;
		magma_trans_t trB = magma_trans_t::MagmaNoTrans;
		//
		magma_queue_t queue = NULL;
		magma_int_t dev = 0;
		magma_queue_create(dev, &queue);

		//
		if (opA == mbv2trans::Trans)
			trA = magma_trans_t::MagmaTrans;
		else if (opA == mbv2trans::ConjTrans)
			throw WS_E_NOT_SUPPORTED;

		if (opB == mbv2trans::Trans)
			trB = magma_trans_t::MagmaTrans;
		else if (opB == mbv2trans::ConjTrans)
			throw WS_E_NOT_SUPPORTED;


		/* Copy matrices A and B from the CPU to the GPU */
		float* dA, * dB, *dC, * At = NULL, * Bt = NULL, * Ct = NULL;
		if (rowmajor)
		{
			magma_smalloc_pinned(&At, (const int)(k * m));
			magma_smalloc_pinned(&Bt, (const int)(k * n));
			magma_smalloc_pinned(&Ct, (const int)(n * m));
			//
			transpose((float*)A, At, m, k);
			transpose((float*)B, Bt, k, n);
			transpose((float*)C, Ct, m, n);

			magma_smalloc(&dA, (const int)(k * m));
			magma_smalloc(&dB, (const int)(n * k));
			magma_smalloc(&dC, (const int)(n * m));

			// ... fill in dA and dX (on GPU)
			magma_ssetmatrix(m, k, At, lda, dA, lda, queue); // copy A -> dA
			magma_ssetmatrix(k, n, Bt, ldb, dB, ldb, queue); // copy B -> dX
			magma_ssetmatrix(m, n, Ct, ldc, dC, ldc, queue); // copy C -> dC

		}
		else
		{
			magma_smalloc(&dA, (const int)(m * k));
			magma_smalloc(&dB, (const int)(k * n));
			magma_smalloc(&dC, (const int)(m * n));

			// ... fill in dA and dX (on GPU)
			magma_ssetmatrix(m, k, A, lda, dA, lda, queue); // copy A -> dA
			magma_ssetmatrix(k, n, B, ldb, dB, ldb, queue); // copy B -> dB
			magma_ssetmatrix(m, n, C, ldc, dC, ldc, queue); // copy C -> dC
		}

		//
		magmablas_sgemm(trA, trB, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue);


		if (rowmajor)
		{
			magma_sgetmatrix(m, n, dC, ldc, Ct, ldc, queue);
			transpose(Ct,         C, n,ldc);
		}
		else
		{
			//retrieve the results matrix C from GPU
			magma_sgetmatrix(m, n, dC, ldc,         C, ldc, queue);
		}


		magma_free(dA);// free host memory
		magma_free(dB);// free host memory
		magma_free(dC);// free host memory

		magma_free_pinned(At);
		magma_free_pinned(Bt);

		magma_queue_destroy(queue); // destroy queue
		magma_finalize();
		return;
	}

	void mbv2dgemm(bool rowmajor, mbv2trans opA, mbv2trans opB, int m, int n, int k, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc)
	{
		magma_init();

		magma_trans_t trA = magma_trans_t::MagmaNoTrans;
		magma_trans_t trB = magma_trans_t::MagmaNoTrans;
		//
		magma_queue_t queue = NULL;
		magma_int_t dev = 0;
		magma_queue_create(dev, &queue);

		//
		if (opA == mbv2trans::Trans)
			trA = magma_trans_t::MagmaTrans;
		else if (opA == mbv2trans::ConjTrans)
			throw WS_E_NOT_SUPPORTED;

		if (opB == mbv2trans::Trans)
			trB = magma_trans_t::MagmaTrans;
		else if (opB == mbv2trans::ConjTrans)
			throw WS_E_NOT_SUPPORTED;


		/* Copy matrices A and B from the CPU to the GPU */
		double* dA, * dB, * dC, * At = NULL, * Bt = NULL, * Ct = NULL;
		if (rowmajor)
		{
			magma_dmalloc_pinned(&At, (const int)(k * m));
			magma_dmalloc_pinned(&Bt, (const int)(k * n));
			magma_dmalloc_pinned(&Ct, (const int)(n * m));
			//
			transpose((double*)A, At, m, k);
			transpose((double*)B, Bt, k, n);
			transpose((double*)C, Ct, m, n);

			magma_dmalloc(&dA, (const int)(k * m));
			magma_dmalloc(&dB, (const int)(n * k));
			magma_dmalloc(&dC, (const int)(n * m));

			// ... fill in dA and dX (on GPU)
			magma_dsetmatrix(m, k, At, lda, dA, lda, queue); // copy A -> dA
			magma_dsetmatrix(k, n, Bt, ldb, dB, ldb, queue); // copy B -> dX
			magma_dsetmatrix(m, n, Ct, ldc, dC, ldc, queue); // copy C -> dC

		}
		else
		{
			magma_dmalloc(&dA, (const int)(m * k));
			magma_dmalloc(&dB, (const int)(k * n));
			magma_dmalloc(&dC, (const int)(m * n));

			// ... fill in dA and dX (on GPU)
			magma_dsetmatrix(m, k, A, lda, dA, lda, queue); // copy A -> dA
			magma_dsetmatrix(k, n, B, ldb, dB, ldb, queue); // copy B -> dB
			magma_dsetmatrix(m, n, C, ldc, dC, ldc, queue); // copy C -> dC
		}

		//
		magmablas_dgemm(trA, trB, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue);


		if (rowmajor)
		{
			magma_dgetmatrix(m, n, dC, ldc, Ct, ldc, queue);
			transpose(Ct, C, n, ldc);
		}
		else
		{
			//retrieve the results matrix C from GPU
			magma_dgetmatrix(m, n, dC, ldc, C, ldc, queue);
		}


		magma_free(dA);// free host memory
		magma_free(dB);// free host memory
		magma_free(dC);// free host memory

		magma_free_pinned(At);
		magma_free_pinned(Bt);

		magma_queue_destroy(queue); // destroy queue
		magma_finalize();
		return;
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


	void transpose(float* src, float* dst, const int N, const int M)
	{
#pragma omp parallel for
		for (int n = 0; n < N * M; n++)
		{
			int i = n / N;
			int j = n % N;
			dst[n] = src[M * j + i];
		}
	}

	void transpose(double* src, double* dst, const int N, const int M)
	{
#pragma omp parallel for
		for (int n = 0; n < N * M; n++)
		{
			int i = n / N;
			int j = n % N;
			dst[n] = src[M * j + i];
		}
	}

	void print_matrix(char* desc, int m, int n, float* a, const int lda) {
		int i, j;
		printf("\n %s\n", desc);
		for (i = 0; i < m; i++)
		{
			for (j = 0; j < n; j++)
				printf(" %6.2f", a[i * lda + j]);

			printf("\n");
		}
	}


	void print_matrix(char* desc, int m, int n, double* a, const int lda) {
		int i, j;
		printf("\n %s\n", desc);
		for (i = 0; i < m; i++)
		{
			for (j = 0; j < n; j++)
				printf(" %6.2f", a[i * lda + j]);

			printf("\n");
		}
	}

	/* Function to reverse arr[] from start to end*/
	void reverseVector(float* arr, int start, int end)
	{
		while (start < end)
		{
			float temp = arr[start];
			arr[start] = arr[end];
			arr[end] = temp;
			start++;
			end--;
		}
	}

	void reverseVector(double* arr, int start, int end)
	{
		while (start < end)
		{
			double temp = arr[start];
			arr[start] = arr[end];
			arr[end] = temp;
			start++;
			end--;
		}
	}
}

