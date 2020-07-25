using MagmaSharp;
using System;
using Xunit;

namespace MagmaSharp.XUnit
{
    public class SolveTests
    {
        [Fact]
        public void TestFloat()
        {
            const int n = 5;
            const int nrhs = 3;
            const int lda = n;
            const int ldb = n;
            int[] ipiv = new int[5];

            //
            float[,] A = new float[lda, n]
            {
                { 6.80f, -6.05f, -0.45f,  8.32f,-9.67f },
                {-2.11f, -3.30f,  2.58f,  2.71f,-5.14f },
                { 5.66f,  5.36f, -2.70f,  4.35f,-7.26f },
                { 5.97f, -4.44f,  0.27f, -7.17f, 6.08f },
                { 8.23f,  1.08f,  9.04f,  2.14f,-6.87f },

            };
            float[,] B = new float[ldb, nrhs]
            {
                {4.02f, -1.56f,  9.81f},
                {6.19f,  4.00f, -4.09f},
                {-8.22f,-8.67f, -4.57f},
                {-7.57f, 1.75f, -8.61f},
                {-3.03f, 2.86f,  8.99f}
            };

            float[,] result = new float[ldb, nrhs]
            {
                 {-0.8f, -0.39f, 0.96f},
                 {-0.7f, -0.55f, 0.22f},
                 {0.59f,  0.84f, 1.90f},
                 {1.32f, -0.10f, 5.36f},
                 {0.57f,  0.11f, 4.04f},
            };

            var X = MagmaSharp.LinAlg.Solve(A, B, Device.CPU);
            for(int i=0; i<X.GetLength(0); i++)
            {
                for(int j=0; j<X.GetLength(1); j++)
                {
                    Assert.Equal(X[i,j], result[i,j], 2);
                }
            }
            //
            X = MagmaSharp.LinAlg.Solve(A, B, Device.GPU);
            for (int i = 0; i < X.GetLength(0); i++)
            {
                for (int j = 0; j < X.GetLength(1); j++)
                {
                    Assert.Equal(X[i, j], result[i, j], 2);
                }
            }
            //
            X = MagmaSharp.LinAlg.Solve(A, B, Device.CUSTOM);
            for (int i = 0; i < X.GetLength(0); i++)
            {
                for (int j = 0; j < X.GetLength(1); j++)
                {
                    Assert.Equal(X[i, j], result[i, j], 2);
                }
            }
        }

        [Fact]
        public void TestDouble()
        {
            const int n = 5;
            const int nrhs = 3;
            const int lda = n;
            const int ldb = n;
            int[] ipiv = new int[5];

            //
            double[,] A = new double[lda, n]
            {
                { 6.80, -6.05, -0.45,  8.32,-9.67 },
                {-2.11, -3.30,  2.58,  2.71,-5.14 },
                { 5.66,  5.36, -2.70,  4.35,-7.26 },
                { 5.97, -4.44,  0.27, -7.17, 6.08 },
                { 8.23,  1.08,  9.04,  2.14,-6.87 },

            };
            double[,] B = new double[ldb, nrhs]
            {
                {4.02f, -1.56f,  9.81f},
                {6.19f,  4.00f, -4.09f},
                {-8.22f,-8.67f, -4.57f},
                {-7.57f, 1.75f, -8.61f},
                {-3.03f, 2.86f,  8.99f}
            };

            double[,] result = new double[ldb, nrhs]
            {
                 {-0.8, -0.39,0.96},
                 {-0.7, -0.55,0.22},
                 {0.59, 0.84 ,1.9 },
                 {1.32, -0.1 ,5.36},
                 {0.57, 0.11 ,4.04},
            };

            var X = MagmaSharp.LinAlg.Solve(A, B, Device.CPU);
            for (int i = 0; i < X.GetLength(0); i++)
            {
                for (int j = 0; j < X.GetLength(1); j++)
                {
                    Assert.Equal(X[i, j], result[i, j], 2);
                }
            }
            //
            X = MagmaSharp.LinAlg.Solve(A, B, Device.GPU);
            for (int i = 0; i < X.GetLength(0); i++)
            {
                for (int j = 0; j < X.GetLength(1); j++)
                {
                    Assert.Equal(X[i, j], result[i, j], 2);
                }
            }
            //
            X = MagmaSharp.LinAlg.Solve(A, B, Device.CUSTOM);
            for (int i = 0; i < X.GetLength(0); i++)
            {
                for (int j = 0; j < X.GetLength(1); j++)
                {
                    Assert.Equal(X[i, j], result[i, j], 2);
                }
            }
        }
    }
}
