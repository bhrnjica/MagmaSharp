#include <iostream>
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include "mbmagma.h"

#include "magma_v2.h"
#include "magma_lapack.h"
#include "magmablas_d.h"

#include "mkl_lapacke.h"
#include "HelperTest.h"

using namespace MagmaBinding;

void mv2dgesvds_cpu_test()
{
	/* Locals */
	const int M = 6;//number of rows
	const int N = 5;//number of columns

	// Constants
	const magma_int_t ione = 1;
	const magma_int_t ineg_one = -1;
	const double d_neg_one = -1;
	const double nan = MAGMA_D_NAN;

	const int LDA = M;//Leading dimension for A
	const int LDU = M;//Leading dimensions for U
	const int LDVT = N;//Leading dimensions for N

	int n = N, m = M, lda = LDA, ldu = LDU, ldvt = LDVT, info;

	/* Local arrays (Col major matrix)*/
	double s[N], u[LDU * M], vt[LDVT * N];

	double a[LDA * M] = {
		8.79, 6.11, -9.15, 9.57, -3.49, 9.84,
		9.93, 6.91, -7.93, 1.64, 4.02, 0.15,
		9.83, 5.04, 4.86, 8.83, 9.8, -8.99,
		5.45, -0.27, 4.85, 0.74, 10, -6.02,
		3.16, 7.98, 3.01, 5.8, 4.27, -5.31
	};

	/* Executable statements */
	printf("mbv2dgesvds_cpu test\n");

	/* Compute SVD */
	info = mbv2dgesvds_cpu( m, n, a, s, u, vt);

	/* Check for convergence */
	if (info > 0) {
		printf("The algorithm computing SVD failed to converge.\n");

	}

	assert(27.47 == round_up(s[0], 2));
	assert(22.64 == round_up(s[1], 2));
	assert(8.56 == round_up(s[2], 2));
	assert(5.99 == round_up(s[3], 2));
	assert(2.01 == round_up(s[4], 2));

	/* Print singular values */
	print_matrix((char*)"Singular values", 1, n, s, 1);

	/* Print left singular vectors */
	print_matrix((char*)"Left singular vectors (stored columnwise)", m, n, u, ldu);

	/* Print right singular vectors */
	print_matrix((char*)"Right singular vectors (stored rowwise)", n, n, vt, ldvt);
}
void mv2dgesvd_cpu_test()
{

	/* Locals */
	const int M = 6;//number of rows
	const int N = 5;//number of columns

	// Constants
	const magma_int_t ione = 1;
	const magma_int_t ineg_one = -1;
	const double d_neg_one = -1;
	const double nan = MAGMA_D_NAN;

	//U and V matrices
	mbv2vector jobU = mbv2vector::MagmaAllVec;
	mbv2vector jobV = mbv2vector::MagmaAllVec;

	const int LDA = M;//Leadning dimension for A
	const int LDU = M;//Leading dimensions for U
	const int LDVT = N;//Leading dimensions for N

	int n = N, m = M, lda = LDA, ldu = LDU, ldvt = LDVT, info;

	/* Local arrays (Col major matrix)*/
	double s[N], u[LDU * M], vt[LDVT * N];

	double a[LDA * M] = {
		8.79, 6.11, -9.15, 9.57, -3.49, 9.84,
		9.93, 6.91, -7.93, 1.64, 4.02, 0.15,
		9.83, 5.04, 4.86, 8.83, 9.8, -8.99,
		5.45, -0.27, 4.85, 0.74, 10, -6.02,
		3.16, 7.98, 3.01, 5.8, 4.27, -5.31
	};

	/* Executable statements */
	printf("mbv2dgesvd_cpu test\n");

	/* Compute SVD */
	info = mbv2dgesvd_cpu(jobU, jobV, m, n, a, lda, s, u, ldu, vt, ldvt);

	/* Check for convergence */
	if (info > 0) {
		printf("The algorithm computing SVD failed to converge.\n");
		
	}
	assert(0 == info);
	assert(27.47 == round_up(s[0], 2));
	assert(22.64 == round_up(s[1], 2));
	assert(8.56 == round_up(s[2], 2));
	assert(5.99 == round_up(s[3], 2));
	assert(2.01 == round_up(s[4], 2));

	/* Print singular values */
	print_matrix((char*)"Singular values", 1, n, s, 1);

	/* Print left singular vectors */
	print_matrix((char*)"Left singular vectors (stored columnwise)", m, n, u, ldu);

	/* Print right singular vectors */
	print_matrix((char*)"Right singular vectors (stored rowwise)", n, n, vt, ldvt);


}
void mv2dgesvds_test()
{
	/* Locals */
	const int M = 6;//number of rows
	const int N = 5;//number of columns

	// Constants
	const magma_int_t ione = 1;
	const magma_int_t ineg_one = -1;
	const double d_neg_one = -1;
	const double nan = MAGMA_D_NAN;

	const int LDA = M;//Leadning dimension for A
	const int LDU = M;//Leading dimensions for U
	const int LDVT = N;//Leading dimensions for N

	int n = N, m = M, lda = LDA, ldu = LDU, ldvt = LDVT, info;

	/* Local arrays (Col major matrix)*/
	double s[N], u[LDU * M], vt[LDVT * N];

	double a[LDA * M] = {
		8.79, 6.11, -9.15, 9.57, -3.49, 9.84,
		9.93, 6.91, -7.93, 1.64, 4.02, 0.15,
		9.83, 5.04, 4.86, 8.83, 9.8, -8.99,
		5.45, -0.27, 4.85, 0.74, 10, -6.02,
		3.16, 7.98, 3.01, 5.8, 4.27, -5.31
	};

	/* Executable statements */
	printf("mbv2dgesvds test\n");

	/* Compute SVD */
	info = mbv2dgesvds(m, n, a, s, u, vt);

	/* Check for convergence */
	if (info > 0) {
		printf("The algorithm computing SVD failed to converge.\n");

	}
	assert(0 == info);
	assert(27.47 == round_up(s[0], 2));
	assert(22.64 == round_up(s[1], 2));
	assert(8.56 == round_up(s[2], 2));
	assert(5.99 == round_up(s[3], 2));
	assert(2.01 == round_up(s[4], 2));

	/* Print singular values */
	print_matrix((char*)"Singular values", 1, n, s, 1);

	/* Print left singular vectors */
	print_matrix((char*)"Left singular vectors (stored columnwise)", m, n, u, ldu);

	/* Print right singular vectors */
	print_matrix((char*)"Right singular vectors (stored rowwise)", n, n, vt, ldvt);

}
void mv2dgesvd_test()
{
	/* Locals */
	const int M = 6;//number of rows
	const int N = 5;//number of columns

	// Constants
	const magma_int_t ione = 1;
	const magma_int_t ineg_one = -1;
	const double d_neg_one = -1;
	const double nan = MAGMA_D_NAN;

	//U and V matrices
	mbv2vector jobU = mbv2vector::MagmaAllVec;
	mbv2vector jobV = mbv2vector::MagmaAllVec;

	const int LDA = M;//Leadning dimension for A
	const int LDU = M;//Leading dimensions for U
	const int LDVT = N;//Leading dimensions for N

	int n = N, m = M, lda = LDA, ldu = LDU, ldvt = LDVT, info;

	/* Local arrays (Col major matrix)*/
	double s[N], u[LDU * M], vt[LDVT * N];

	double a[LDA * M] = {
		8.79, 6.11, -9.15, 9.57, -3.49, 9.84,
		9.93, 6.91, -7.93, 1.64, 4.02, 0.15,
		9.83, 5.04, 4.86, 8.83, 9.8, -8.99,
		5.45, -0.27, 4.85, 0.74, 10, -6.02,
		3.16, 7.98, 3.01, 5.8, 4.27, -5.31
	};

	/* Executable statements */
	printf("mbv2dgesvd test\n");

	/* Compute SVD */
	info = mbv2dgesvd(jobU, jobV, m, n, a, lda, s, u, ldu, vt, ldvt);

	/* Check for convergence */
	if (info > 0) {
		printf("The algorithm computing SVD failed to converge.\n");

	}
	assert(0 == info);
	assert(27.47 == round_up(s[0], 2));
	assert(22.64 == round_up(s[1], 2));
	assert(8.56 == round_up(s[2], 2));
	assert(5.99 == round_up(s[3], 2));
	assert(2.01 == round_up(s[4], 2));

	/* Print singular values */
	print_matrix((char*)"Singular values", 1, n, s, 1);

	/* Print left singular vectors */
	print_matrix((char*)"Left singular vectors (stored columnwise)", m, n, u, ldu);

	/* Print right singular vectors */
	print_matrix((char*)"Right singular vectors (stored rowwise)", n, n, vt, ldvt);

}


void mv2sgesvds_cpu_test()
{
	/* Locals */
	const int M = 6;//number of rows
	const int N = 5;//number of columns

	// Constants
	const magma_int_t ione = 1;
	const magma_int_t ineg_one = -1;
	const double d_neg_one = -1;
	const double nan = MAGMA_D_NAN;

	const int LDA = M;//Leading dimension for A
	const int LDU = M;//Leading dimensions for U
	const int LDVT = N;//Leading dimensions for N

	int n = N, m = M, lda = LDA, ldu = LDU, ldvt = LDVT, info;

	/* Local arrays (Col major matrix)*/
	float s[N], u[LDU * M], vt[LDVT * N];

	float a[LDA * M] = {
		8.79f,  6.11f, -9.15f, 9.57f, -3.49f, 9.84f,
		9.93f,  6.91f, -7.93f, 1.64f,  4.02f, 0.15f,
		9.83f,  5.04f,  4.86f, 8.83f,   9.8f, -8.99f,
		5.45f, -0.27f,  4.85f, 0.74f,  10.0f, -6.02f,
		3.16f,  7.98f,  3.01f,  5.8f,  4.27f, -5.31f
	};

	/* Executable statements */
	printf("mdv2sgesvds_cpu test\n");

	/* Compute SVD */
	info = mbv2sgesvds_cpu(m, n, a, s, u, vt);

	/* Check for convergence */
	if (info > 0) {
		printf("The algorithm computing SVD failed to converge.\n");

	}
	assert(0 == info);
	assert(27.47f == round_up(s[0], 2));
	assert(22.64f == round_up(s[1], 2));
	assert(8.56f == round_up(s[2], 2));
	assert(5.99f == round_up(s[3], 2));
	assert(2.01f == round_up(s[4], 2));

	/* Print singular values */
	print_matrix((char*)"Singular values", 1, n, s, 1);

	/* Print left singular vectors */
	print_matrix((char*)"Left singular vectors (stored columnwise)", m, n, u, ldu);

	/* Print right singular vectors */
	print_matrix((char*)"Right singular vectors (stored rowwise)", n, n, vt, ldvt);
}
void mv2sgesvd_cpu_test()
{

	/* Locals */
	const int M = 6;//number of rows
	const int N = 5;//number of columns

	// Constants
	const magma_int_t ione = 1;
	const magma_int_t ineg_one = -1;
	const double d_neg_one = -1;
	const double nan = MAGMA_D_NAN;

	//U and V matrices
	mbv2vector jobU = mbv2vector::MagmaAllVec;
	mbv2vector jobV = mbv2vector::MagmaAllVec;

	const int LDA = M;//Leadning dimension for A
	const int LDU = M;//Leading dimensions for U
	const int LDVT = N;//Leading dimensions for N

	int n = N, m = M, lda = LDA, ldu = LDU, ldvt = LDVT, info;

	/* Local arrays (Col major matrix)*/
	float s[N], u[LDU * M], vt[LDVT * N];

	float a[LDA * M] = {
		8.79f,  6.11f, -9.15f, 9.57f, -3.49f, 9.84f,
		9.93f,  6.91f, -7.93f, 1.64f,  4.02f, 0.15f,
		9.83f,  5.04f,  4.86f, 8.83f,   9.8f, -8.99f,
		5.45f, -0.27f,  4.85f, 0.74f,  10.0f, -6.02f,
		3.16f,  7.98f,  3.01f,  5.8f,  4.27f, -5.31f
	};

	/* Executable statements */
	printf("mbv2sgesvd_cpu test\n");

	/* Compute SVD */
	info = mbv2sgesvd_cpu(jobU, jobV, m, n, a, lda, s, u, ldu, vt, ldvt);

	/* Check for convergence */
	if (info > 0) {
		printf("The algorithm computing SVD failed to converge.\n");

	}
	assert(0 == info);
	assert(27.47f == round_up(s[0], 2));
	assert(22.64f == round_up(s[1], 2));
	assert(8.56f == round_up(s[2], 2));
	assert(5.99f == round_up(s[3], 2));
	assert(2.01f == round_up(s[4], 2));

	/* Print singular values */
	print_matrix((char*)"Singular values", 1, n, s, 1);

	/* Print left singular vectors */
	print_matrix((char*)"Left singular vectors (stored columnwise)", m, n, u, ldu);

	/* Print right singular vectors */
	print_matrix((char*)"Right singular vectors (stored rowwise)", n, n, vt, ldvt);


}
void mv2sgesvds_test()
{
	/* Locals */
	const int M = 6;//number of rows
	const int N = 5;//number of columns

	// Constants
	const magma_int_t ione = 1;
	const magma_int_t ineg_one = -1;
	const double d_neg_one = -1;
	const double nan = MAGMA_D_NAN;

	const int LDA = M;//Leadning dimension for A
	const int LDU = M;//Leading dimensions for U
	const int LDVT = N;//Leading dimensions for N

	int n = N, m = M, lda = LDA, ldu = LDU, ldvt = LDVT, info;

	/* Local arrays (Col major matrix)*/
	float s[N], u[LDU * M], vt[LDVT * N];

	float a[LDA * M] = {
		8.79f,  6.11f, -9.15f, 9.57f, -3.49f, 9.84f,
		9.93f,  6.91f, -7.93f, 1.64f,  4.02f, 0.15f,
		9.83f,  5.04f,  4.86f, 8.83f,   9.8f, -8.99f,
		5.45f, -0.27f,  4.85f, 0.74f,  10.0f, -6.02f,
		3.16f,  7.98f,  3.01f,  5.8f,  4.27f, -5.31f
	};

	/* Executable statements */
	printf("mbv2sgesvds test\n");

	/* Compute SVD */
	info = mbv2sgesvds(m, n, a, s, u, vt);

	/* Check for convergence */
	if (info > 0) {
		printf("The algorithm computing SVD failed to converge.\n");

	}
	assert(0 == info);
	assert(27.47f == round_up(s[0], 2));
	assert(22.64f == round_up(s[1], 2));
	assert(8.56f == round_up(s[2], 2));
	assert(5.99f == round_up(s[3], 2));
	assert(2.01f == round_up(s[4], 2));

	/* Print singular values */
	print_matrix((char*)"Singular values", 1, n, s, 1);

	/* Print left singular vectors */
	print_matrix((char*)"Left singular vectors (stored columnwise)", m, n, u, ldu);

	/* Print right singular vectors */
	print_matrix((char*)"Right singular vectors (stored rowwise)", n, n, vt, ldvt);

}
void mv2sgesvd_test()
{
	/* Locals */
	const int M = 6;//number of rows
	const int N = 5;//number of columns

	// Constants
	const magma_int_t ione = 1;
	const magma_int_t ineg_one = -1;
	const double d_neg_one = -1;
	const double nan = MAGMA_D_NAN;

	//U and V matrices
	mbv2vector jobU = mbv2vector::MagmaAllVec;
	mbv2vector jobV = mbv2vector::MagmaAllVec;

	const int LDA = M;//Leadning dimension for A
	const int LDU = M;//Leading dimensions for U
	const int LDVT = N;//Leading dimensions for N

	int n = N, m = M, lda = LDA, ldu = LDU, ldvt = LDVT, info;

	/* Local arrays (Col major matrix)*/
	float s[N], u[LDU * M], vt[LDVT * N];

	float a[LDA * M] = {
		8.79f,  6.11f, -9.15f, 9.57f, -3.49f, 9.84f,
		9.93f,  6.91f, -7.93f, 1.64f,  4.02f, 0.15f,
		9.83f,  5.04f,  4.86f, 8.83f,   9.8f, -8.99f,
		5.45f, -0.27f,  4.85f, 0.74f,  10.0f, -6.02f,
		3.16f,  7.98f,  3.01f,  5.8f,  4.27f, -5.31f
	};

	/* Executable statements */
	printf("mbv2sgesvd test\n");

	/* Compute SVD */
	info = mbv2sgesvd(jobU, jobV, m, n, a, lda, s, u, ldu, vt, ldvt);

	/* Check for convergence */
	if (info > 0) {
		printf("The algorithm computing SVD failed to converge.\n");

	}
	assert(0 == info);
	assert(27.47f == round_up(s[0], 2));
	assert(22.64f == round_up(s[1], 2));
	assert(8.56f == round_up(s[2], 2));
	assert(5.99f == round_up(s[3], 2));
	assert(2.01f == round_up(s[4], 2));

	/* Print singular values */
	print_matrix((char*)"Singular values", 1, n, s, 1);

	/* Print left singular vectors */
	print_matrix((char*)"Left singular vectors (stored columnwise)", m, n, u, ldu);

	/* Print right singular vectors */
	print_matrix((char*)"Right singular vectors (stored rowwise)", n, n, vt, ldvt);

}





void testSVDQQ()
{
	magma_init();
	/* Locals */
	const int M = 6;//number of rows
	const int N = 5;//number of columns

	// Constants
	const magma_int_t ione = 1;
	const magma_int_t ineg_one = -1;
	const double d_neg_one = -1;
	const double nan = MAGMA_D_NAN;

	//U and V matrices
	magma_vec_t jobU = MagmaAllVec;
	magma_vec_t jobV = MagmaAllVec;

	const int LDA = M;//Leadning dimension for A
	const int LDU = M;//Leading dimensions for U
	const int LDVT = N;//Leading dimensions for N

	int n = N, m = M, lda = LDA, ldu = LDU, ldvt = LDVT, info;
	int min_mn = min(M, N);
	//query for workspace size
	magma_int_t query_magma;
	double dummy[1];
	double* h_work; // h_work - workspace
	magma_int_t* iwork, iunused[1];

	info = magma_dgesdd(jobU, m, n, NULL, lda, NULL, NULL, ldu, NULL, ldvt, dummy, ineg_one, iunused, &info);

	assert(info == 0);
	query_magma = (magma_int_t)MAGMA_D_REAL(dummy[0]);
	magma_dmalloc_pinned(&h_work, query_magma); // host mem . for h_work
	magma_imalloc_cpu(&iwork, 8 * min_mn);

	/* Local arrays (Col major matrix)*/
	double s[N], u[LDU * M], vt[LDVT * N];

	double a[LDA * M] = {
		8.79, 6.11, -9.15, 9.57, -3.49, 9.84,
		9.93, 6.91, -7.93, 1.64, 4.02, 0.15,
		9.83, 5.04, 4.86, 8.83, 9.8, -8.99,
		5.45, -0.27, 4.85, 0.74, 10, -6.02,
		3.16, 7.98, 3.01, 5.8, 4.27, -5.31
	};

	/* Executable statements */
	printf("magma_dgesvd (row-major, high-level) Example Program Results\n");

	/* Compute SVD */
	info = magma_dgesdd(jobU, m, n, a, lda, s, u, ldu, vt, ldvt, h_work, query_magma, iwork, &info);

	/* Check for convergence */
	if (info > 0) {
		printf("The algorithm computing SVD failed to converge.\n");
		exit(1);
	}

	/* Print singular values */
	print_matrix((char*)"Singular values", 1, n, s, 1);

	/* Print left singular vectors */
	print_matrix((char*)"Left singular vectors (stored columnwise)", m, n, u, ldu);

	/* Print right singular vectors */
	print_matrix((char*)"Right singular vectors (stored rowwise)", n, n, vt, ldvt);

	// Free memory
	magma_free_pinned(h_work); // free host memory
	magma_free_cpu(iwork);
	magma_finalize(); // finalize Magma
	magma_finalize();
}

void testSVD_Lapack()
{
	const int M = 6;//number of rows
	const int N = 5;//number of columns
	const int LDA = N;//Leadning dimension for A
	const int LDU = M;//Leading dimensions for U
	const int LDVT = N;//Leading dimensions for N

	/* Locals */
	MKL_INT m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT, info;
	double superb[min(M, N) - 1];
	/* Local arrays */
	double s[N], u[LDU * M], vt[LDVT * N];
	double a[LDA * M] = {//(ROW major matrix)
		8.79,  9.93,  9.83, 5.45,  3.16,
		6.11,  6.91,  5.04, -0.27,  7.98,
		-9.15, -7.93,  4.86, 4.85,  3.01,
		9.57,  1.64,  8.83, 0.74,  5.80,
		-3.49,  4.02,  9.80, 10.00,  4.27,
		9.84,  0.15, -8.99, -6.02, -5.31
	};
	/* Executable statements */
	printf("LAPACKE_dgesvd (row-major, high-level) Example Program Results\n");

	/* Compute SVD */
	info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, a, lda, s, u, ldu, vt, ldvt, superb);

	/* Check for convergence */
	if (info > 0) {
		printf("The algorithm computing SVD failed to converge.\n");
		return;
	}
	/* Print singular values */
	print_matrix((char*)"Singular values", 1, n, s, 1);
	/* Print left singular vectors */
	print_matrix((char*)"Left singular vectors (stored columnwise)", m, n, u, ldu);
	/* Print right singular vectors */
	print_matrix((char*)"Right singular vectors (stored rowwise)", n, n, vt, ldvt);

}