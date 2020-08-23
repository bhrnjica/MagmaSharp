#include <iostream>
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "mkl_types.h"
#include "mkl_cblas.h"
#include <stdio.h>
#include <stdarg.h>     /* va_list, va_start, va_arg, va_end */


#include "HelperTest.h"
#include <ctime>
#include <cuda.h>
#include "lblapack.h"
#include "mbmagma.h"
#include <cassert>
#include "MatrixTest.h"
using namespace LapackBinding;
using namespace MagmaBinding;

void mv2sgemm_test_magma_col()
{
    const float alpha = 1.0f, beta=1.0f;
    const int m=6, n=4, k = 5;
    const int lda = m, ldb = k, ldc = m;

	//(Row major matrix)
	const float A[m * k] = {
	 6.80f,	-6.05f,	-0.45f,	 8.32f,	 8.32f,	-9.67f,
	-2.11f,	-3.30f,	 2.58f,	 2.71f,  2.71f,	-5.14f,
	 5.66f,	 5.36f,	-2.70f,	 4.35f,	 4.35f,	-7.26f,
	 5.97f,	-4.44f,	 0.27f,	-7.17f,	-7.17f,	 6.08f,
	 8.23f,	 1.08f,	 9.04f,	 2.14f,	 2.14f,	-6.87f
	};
	const float B[k * n] = {
		 4.02f,	-1.56f,	 9.81f,	 9.81f,	 9.81f,
		 6.19f,	 4.0f,	-4.09f,	-4.09f,	-4.09f,
		-8.22f,	-8.67f,	-4.57f,	-4.57f,	-4.57f,
		-7.57f,	 1.75f,	-8.61f,	-8.61f,	-8.61f
	};

	 float C[m * n] = {
		0, 0, 0, 0,
	    0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
	};

	 mbv2sgemm(false, mbv2trans::NoTrans, mbv2trans::NoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
   
	print_matrix((char*)"Matrix mult result:",m,n,C, lda);

    printf("\nTest done!\n");

}

void mv2sgemm_test_magma_row()
{
	const float alpha = 1.0f, beta = 1.0f;
	const int m = 6, n = 4, k = 5;
	//with rowmajor layout leading dimension of the matrix is column number.
	//const int lda = k, ldb = n, ldc = n;
	const int lda = m, ldb = k, ldc = m;
	//(Row major matrix)
	const float A[m * k] = {
		6.80f, -2.11f,  5.66f,  5.97f,  8.23f,
	   -6.05f, -3.30f,  5.36f, -4.44f,  1.08f,
	   -0.45f,  2.58f, -2.70f,  0.27f,  9.04f,
		8.32f,  2.71f,  4.35f, -7.17f,  2.14f,
		8.32f,  2.71f,  4.35f, -7.17f,  2.14f,
	   -9.67f, -5.14f, -7.26f,  6.08f, -6.87f
	};
	const float B[k * n] = {
		4.02f,  6.19f, -8.22f, -7.57f,
	   -1.56f,  4.00f, -8.67f,  1.75f,
		9.81f, -4.09f, -4.57f, -8.61f,
		9.81f, -4.09f, -4.57f, -8.61f,
		9.81f, -4.09f, -4.57f, -8.61f
	};

	float C[m * n] = {
	   0, 0, 0, 0,
	   0, 0, 0, 0,
	   0, 0, 0, 0,
	   0, 0, 0, 0,
	   0, 0, 0, 0,
	   0, 0, 0, 0,
	};

	mbv2sgemm(true, mbv2trans::NoTrans, mbv2trans::NoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	
	print_matrix((char*)"Matrix mult result:", m, n, C, n, false);

	assert(225.45f == round_up(C[0], 2));
	assert(-47.58f == round_up(C[1], 2));

	assert(160.84f == round_up(C[22], 2));
	assert(133.52f == round_up(C[23], 2));


/* Result Matrix in RowMAjor ordr
 225.45 -47.58 -128.36 -226.16
   0.45 -58.83  69.20  22.80
  59.01 -19.50 -48.88 -48.99
  22.55  65.12 -88.78 -52.39
  22.55  65.12 -88.78 -52.39
 -109.83 -47.49 160.84 133.52
*/


	

	printf("\nTest done!\n");

}


void mv2sgemm_test_lapack_col()
{
	const float alpha = 1.0f, beta = 1.0f;
	const int m = 6, n = 4, k = 5;
	const int lda = m, ldb = k, ldc = m;

	//(Row major matrix)
	const float A[m * k] = {
	 6.80f,	-6.05f,	-0.45f,	 8.32f,	 8.32f,	-9.67f,
	-2.11f,	-3.30f,	 2.58f,	 2.71f,  2.71f,	-5.14f,
	 5.66f,	 5.36f,	-2.70f,	 4.35f,	 4.35f,	-7.26f,
	 5.97f,	-4.44f,	 0.27f,	-7.17f,	-7.17f,	 6.08f,
	 8.23f,	 1.08f,	 9.04f,	 2.14f,	 2.14f,	-6.87f
	};
	const float B[k * n] = {
		 4.02f,	-1.56f,	 9.81f,	 9.81f,	 9.81f,
		 6.19f,	 4.0f,	-4.09f,	-4.09f,	-4.09f,
		-8.22f,	-8.67f,	-4.57f,	-4.57f,	-4.57f,
		-7.57f,	 1.75f,	-8.61f,	-8.61f,	-8.61f
	};

	float C[m * n] = {
	   0, 0, 0, 0,
	   0, 0, 0, 0,
	   0, 0, 0, 0,
	   0, 0, 0, 0,
	   0, 0, 0, 0,
	   0, 0, 0, 0,
	};

	cblas_sgemm(CBLAS_LAYOUT::CblasColMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);



	print_matrix((char*)"Matrix mult result:", m, n, C, lda);

	printf("\nTest done!\n");

}

void mv2sgemm_test_lapack_row()
{
	const float alpha = 1.0f, beta = 1.0f;
	const int m = 6, n = 4, k = 5;
	//with rowmajor layout leading dimension of the matrix is column number.
	const int lda = k, ldb = n, ldc = n;

	//(Row major matrix)
	const float A[m * k] = {
		6.80f, -2.11f,  5.66f,  5.97f,  8.23f,
	   -6.05f, -3.30f,  5.36f, -4.44f,  1.08f,
	   -0.45f,  2.58f, -2.70f,  0.27f,  9.04f,
		8.32f,  2.71f,  4.35f, -7.17f,  2.14f,
		8.32f,  2.71f,  4.35f, -7.17f,  2.14f,
	   -9.67f, -5.14f, -7.26f,  6.08f, -6.87f
	};
	const float B[k * n] = {
		4.02f,  6.19f, -8.22f, -7.57f,
	   -1.56f,  4.00f, -8.67f,  1.75f,
		9.81f, -4.09f, -4.57f, -8.61f,
		9.81f, -4.09f, -4.57f, -8.61f,
		9.81f, -4.09f, -4.57f, -8.61f
	};

	float C[m * n] = {
	   0, 0, 0, 0,
	   0, 0, 0, 0,
	   0, 0, 0, 0,
	   0, 0, 0, 0,
	   0, 0, 0, 0,
	   0, 0, 0, 0,
	};

	cblas_sgemm(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);


	print_matrix((char*)"Matrix mult result:", m, n, C, ldc, false);

	assert(225.45f == round_up(C[0], 2));
	assert(-47.58f == round_up(C[1], 2));

	assert(160.84f == round_up(C[22], 2));
	assert(133.52f == round_up(C[23], 2));


	/* Result Matrix in RowMAjor ordr
	 225.45 -47.58 -128.36 -226.16
	   0.45 -58.83  69.20  22.80
	  59.01 -19.50 -48.88 -48.99
	  22.55  65.12 -88.78 -52.39
	  22.55  65.12 -88.78 -52.39
	 -109.83 -47.49 160.84 133.52
	*/

	printf("\nTest done!\n");

}