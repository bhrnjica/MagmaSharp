using System;
using Xunit;

namespace MagmaSharp.XUnit
{
    public class SvdTests
    {
        [Fact]
        public void TestFloat()
        {

            //result for s
            float[] result_s = new float[] {27.47f, 22.64f, 8.56f, 5.99f, 2.01f };

            //TODO
            float[,] resultU = new float[,]
            {
                { 8.79f,  9.93f,   9.83f,   5.45f,   3.16f, },
                {6.11f,   6.91f,   5.04f,  -0.27f,   7.98f, },
                {-9.15f,  -7.93f,   4.86f,   4.85f,   3.01f, },
                {9.57f,   1.64f,   8.83f,   0.74f,   5.80f, },
                {-3.49f,   4.02f,   9.80f,  10.00f,   4.27f, },
                { 9.84f,   0.15f,  -8.99f,  -6.02f,  -5.31f, },
            };

            //TODO
            float[,] resultvT = new float[,]
            {
                { 8.79f,  9.93f,   9.83f,   5.45f,   3.16f, },
                {6.11f,   6.91f,   5.04f,  -0.27f,   7.98f, },
                {-9.15f,  -7.93f,   4.86f,   4.85f,   3.01f, },
                {9.57f,   1.64f,   8.83f,   0.74f,   5.80f, },
                {-3.49f,   4.02f,   9.80f,  10.00f,   4.27f, },
                { 9.84f,   0.15f,  -8.99f,  -6.02f,  -5.31f, },
            };


            {
                //
                float[,] A = new float[,]
                {
                { 8.79f,  9.93f,   9.83f,   5.45f,   3.16f, },
                {6.11f,   6.91f,   5.04f,  -0.27f,   7.98f, },
                {-9.15f,  -7.93f,   4.86f,   4.85f,   3.01f, },
                {9.57f,   1.64f,   8.83f,   0.74f,   5.80f, },
                {-3.49f,   4.02f,   9.80f,  10.00f,   4.27f, },
                { 9.84f,   0.15f,  -8.99f,  -6.02f,  -5.31f, },
                };

                (float[] s, float[,] U, float[,] vT) = MagmaSharp.LinAlg.Svd(A, Device.GPU);

                for (int i = 0; i < result_s.Length; i++)
                    Assert.Equal(result_s[i], s[i], 2);

                //TODO uncomment and test
                //for (int i = 0; i < U.GetLength(0); i++)
                //{
                //    for (int j = 0; j < U.GetLength(1); j++)
                //    {
                //        Assert.Equal(U[i, j], resultU[i, j], 2);
                //    }
                //}

                //for (int i = 0; i < vT.GetLength(0); i++)
                //{
                //    for (int j = 0; j < vT.GetLength(1); j++)
                //    {
                //        Assert.Equal(vT[i, j], resultU[i, j], 2);
                //    }
                //}
            }

            {
                //
                float[,] A = new float[,]
                {
                { 8.79f,  9.93f,   9.83f,   5.45f,   3.16f, },
                {6.11f,   6.91f,   5.04f,  -0.27f,   7.98f, },
                {-9.15f,  -7.93f,   4.86f,   4.85f,   3.01f, },
                {9.57f,   1.64f,   8.83f,   0.74f,   5.80f, },
                {-3.49f,   4.02f,   9.80f,  10.00f,   4.27f, },
                { 9.84f,   0.15f,  -8.99f,  -6.02f,  -5.31f, },
                };

                (float[] s, float[,] U, float[,] vT) = MagmaSharp.LinAlg.Svd(A, Device.CPU);

                for (int i = 0; i < result_s.Length; i++)
                    Assert.Equal(result_s[i], s[i], 2);
            }

            {
                //
                float[,] A = new float[,]
                {
                { 8.79f,  9.93f,   9.83f,   5.45f,   3.16f, },
                {6.11f,   6.91f,   5.04f,  -0.27f,   7.98f, },
                {-9.15f,  -7.93f,   4.86f,   4.85f,   3.01f, },
                {9.57f,   1.64f,   8.83f,   0.74f,   5.80f, },
                {-3.49f,   4.02f,   9.80f,  10.00f,   4.27f, },
                { 9.84f,   0.15f,  -8.99f,  -6.02f,  -5.31f, },
                };

                (float[] s, float[,] U, float[,] vT) = MagmaSharp.LinAlg.Svd(A, Device.CUSTOM);

                for (int i = 0; i < result_s.Length; i++)
                    Assert.Equal(result_s[i], s[i], 2);
            }

        }
        [Fact]
        public void TestDouble()
        {
            //result for s
            double[] result_s = new double[] { 27.47f, 22.64f, 8.56f, 5.99f, 2.01f };

            //TODO
            double[,] resultU = new double[,]
            {
                { 8.79f,  9.93f,   9.83f,   5.45f,   3.16f, },
                {6.11f,   6.91f,   5.04f,  -0.27f,   7.98f, },
                {-9.15f,  -7.93f,   4.86f,   4.85f,   3.01f, },
                {9.57f,   1.64f,   8.83f,   0.74f,   5.80f, },
                {-3.49f,   4.02f,   9.80f,  10.00f,   4.27f, },
                { 9.84f,   0.15f,  -8.99f,  -6.02f,  -5.31f, },
            };

            //TODO
            double[,] resultvT = new double[,]
            {
                { 8.79f,  9.93f,   9.83f,   5.45f,   3.16f, },
                {6.11f,   6.91f,   5.04f,  -0.27f,   7.98f, },
                {-9.15f,  -7.93f,   4.86f,   4.85f,   3.01f, },
                {9.57f,   1.64f,   8.83f,   0.74f,   5.80f, },
                {-3.49f,   4.02f,   9.80f,  10.00f,   4.27f, },
                { 9.84f,   0.15f,  -8.99f,  -6.02f,  -5.31f, },
            };


            {
                //
                double[,] A = new double[,]
                {
                { 8.79f,  9.93f,   9.83f,   5.45f,   3.16f, },
                {6.11f,   6.91f,   5.04f,  -0.27f,   7.98f, },
                {-9.15f,  -7.93f,   4.86f,   4.85f,   3.01f, },
                {9.57f,   1.64f,   8.83f,   0.74f,   5.80f, },
                {-3.49f,   4.02f,   9.80f,  10.00f,   4.27f, },
                { 9.84f,   0.15f,  -8.99f,  -6.02f,  -5.31f, },
                };

                (double[] s, double[,] U, double[,] vT) = MagmaSharp.LinAlg.Svd(A, Device.GPU);

                for (int i = 0; i < result_s.Length; i++)
                    Assert.Equal(result_s[i], s[i], 2);

                //TODO uncomment and test
                //for (int i = 0; i < U.GetLength(0); i++)
                //{
                //    for (int j = 0; j < U.GetLength(1); j++)
                //    {
                //        Assert.Equal(U[i, j], resultU[i, j], 2);
                //    }
                //}

                //for (int i = 0; i < vT.GetLength(0); i++)
                //{
                //    for (int j = 0; j < vT.GetLength(1); j++)
                //    {
                //        Assert.Equal(vT[i, j], resultU[i, j], 2);
                //    }
                //}
            }

            {
                //
                double[,] A = new double[,]
                {
                { 8.79f,  9.93f,   9.83f,   5.45f,   3.16f, },
                {6.11f,   6.91f,   5.04f,  -0.27f,   7.98f, },
                {-9.15f,  -7.93f,   4.86f,   4.85f,   3.01f, },
                {9.57f,   1.64f,   8.83f,   0.74f,   5.80f, },
                {-3.49f,   4.02f,   9.80f,  10.00f,   4.27f, },
                { 9.84f,   0.15f,  -8.99f,  -6.02f,  -5.31f, },
                };

                (double[] s, double[,] U, double[,] vT) = MagmaSharp.LinAlg.Svd(A, Device.CPU);

                for (int i = 0; i < result_s.Length; i++)
                    Assert.Equal(result_s[i], s[i], 2);
            }

            {
                //
                double[,] A = new double[,]
                {
                { 8.79f,  9.93f,   9.83f,   5.45f,   3.16f, },
                {6.11f,   6.91f,   5.04f,  -0.27f,   7.98f, },
                {-9.15f,  -7.93f,   4.86f,   4.85f,   3.01f, },
                {9.57f,   1.64f,   8.83f,   0.74f,   5.80f, },
                {-3.49f,   4.02f,   9.80f,  10.00f,   4.27f, },
                { 9.84f,   0.15f,  -8.99f,  -6.02f,  -5.31f, },
                };

                (double[] s, double[,] U, double[,] vT) = MagmaSharp.LinAlg.Svd(A, Device.CUSTOM);

                for (int i = 0; i < result_s.Length; i++)
                    Assert.Equal(result_s[i], s[i], 2);
            }

        }
    }
}
