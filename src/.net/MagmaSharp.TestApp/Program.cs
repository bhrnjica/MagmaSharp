using System;
using System.Diagnostics;
using System.Linq;

namespace MagmaSharp.TestApp
{
    class Program
    {
        static void Main(string[] args)
        {

            //Matrix A
            var A = new double[4, 4]// { { 3, 3.2 }, { 3.5, 3.6 } };

                {
                { 1, 2, 3, 4},
                { 2, 2, 5, 4},
                { 3, 2, 6, 4},
                { 5, 5, 3, 4} };


            var C = LapackSharp.LinAlg.MInverse(A);


            PerformanceTest.SVD();
            return;

          //  TestSolveFloat();
           // TestDouble();


          //  TestSvdFloat();
            //return;
           // performanceTest();
          //  TestFloat();

           // foreach (var i in Enumerable.Range(0,100))
                //TestSvdFloat(); //TestFloat();

            return;
            
           
            return;
           // testEigen();
           // lssTest();
           //// dsvdTest();
           // return;
           // MagmaSharp.DeviceMng.GetDevices();
           // Console.WriteLine(MagmaSharp.DeviceMng.GetCUDAArch());

           // solve();
           // Console.WriteLine("Hello World!");
        }

        

      



        private static void printMatrix<T>(T[,] mat)
        {
            int n = mat.GetLength(0);//num of rows
            int m = mat.GetLength(1);//num of cols

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    Console.Write(Math.Round(Convert.ToDouble(mat[i, j]),2));
                    Console.Write("\t");
                }
                Console.Write("\n");
            }
        }



        public static void TestSolveFloat()
        {
            const int n = 5;
            const int nrhs = 3;
            int[] ipiv = new int[n];

            //
            float[,] A = new float[n, n]
            {
                { 6.80f, -6.05f, -0.45f,  8.32f,-9.67f },
                {-2.11f, -3.30f,  2.58f,  2.71f,-5.14f },
                { 5.66f,  5.36f, -2.70f,  4.35f,-7.26f },
                { 5.97f, -4.44f,  0.27f, -7.17f, 6.08f },
                { 8.23f,  1.08f,  9.04f,  2.14f,-6.87f },

            };
            float[,] B = new float[n, nrhs]
            {
                {4.02f, -1.56f,  9.81f},
                {6.19f,  4.00f, -4.09f},
                {-8.22f,-8.67f, -4.57f},
                {-7.57f, 1.75f, -8.61f},
                {-3.03f, 2.86f,  8.99f}
            };

            float[,] result = new float[n, nrhs]
            {
                 {-0.8f, -0.39f, 0.96f},
                 {-0.7f, -0.55f, 0.22f},
                 {0.59f,  0.84f, 1.90f},
                 {1.32f, -0.10f, 5.36f},
                 {0.57f,  0.11f, 4.04f},
            };

            var X = MagmaSharp.LinAlg.Solve(A, B, Device.GPU);

            for (int i = 0; i < X.GetLength(0); i++)
            {
                for (int j = 0; j < X.GetLength(1); j++)
                {
                    Debug.Assert((float)Math.Round(X[i, j],2)== result[i, j]);
                }
            }
            //
            //X = MagmaSharp.LinAlg.Solve(A, B, Device.GPU);
            //for (int i = 0; i < X.GetLength(0); i++)
            //{
            //    for (int j = 0; j < X.GetLength(1); j++)
            //    {
            //        Debug.Assert((float)Math.Round(X[i, j], 2) == result[i, j]);
            //    }
            //}
            ////
            //X = MagmaSharp.LinAlg.Solve(A, B, Device.CUSTOM);
            //for (int i = 0; i < X.GetLength(0); i++)
            //{
            //    for (int j = 0; j < X.GetLength(1); j++)
            //    {
            //        Debug.Assert((float)Math.Round(X[i, j], 2) == result[i, j]);
            //    }
            //}
        }


        public static void TestSvdFloat()
        {
            //result for s
            float[] result_s = new float[] { 27.47f, 22.64f, 8.56f, 5.99f, 2.01f };

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
                    {8.79f,  6.11f, -9.15f,  9.57f, -3.49f,  9.84f },
                    {9.93f,  6.91f, -7.93f,  1.64f,  4.02f,  0.15f },
                    {9.83f,  5.04f,  4.86f,  8.83f,  9.80f, -8.99f },
                    {5.45f, -0.27f,  4.85f,  0.74f, 10.00f, -6.02f },
                    { 3.16f,  7.98f,  3.01f,  5.80f,  4.27f, -5.31f }
                };

                (float[] s, float[,] U, float[,] vT) = MagmaSharp.LinAlg.Svd(A, true, true, device: Device.CUSTOM);

                for (int i = 0; i < result_s.Length; i++)
                    Debug.Assert(result_s[i] == (float)Math.Round(s[i],2));

               printMatrix<float>(U);

              
            }

           
        }


        public static void TestDouble()
        {
            //result for s
            double[] result_s = new double[] { 27.47f, 22.64f, 8.56f, 5.99f, 2.01f };

            //
            double[,] resultU = new double[,]
            {
               {-0.59f,  0.26f,   0.36f,   0.31f,   0.23f, 0.55F},
                {-0.40f,  0.24f,  -0.22f,  -0.75f,  -0.36f, 0.18F},
                {-0.03f, -0.60f,  -0.45f,   0.23f,  -0.31f, 0.54F},
                {-0.43f,  0.24f,  -0.69f,   0.33f,   0.16f, -0.39F},
                {-0.47f, -0.35f,   0.39f,   0.16f,  -0.52f, -0.46F},
                { 0.29f,  0.58f,  -0.02f,   0.38f,  -0.65f, 0.11F},
            };

            // 
            double[,] resultvT = new double[,]
            {
                 {-0.25f,  -0.40F,  -0.69F,  -0.37F,  -0.41F},
                { 0.81f,   0.36F,  -0.25F,  -0.37F,  -0.10F},
                {-0.26f,   0.70F,  -0.22F,   0.39F,  -0.49F},
                { 0.40f,  -0.45F,   0.25F,   0.43F,  -0.62F},
                {-0.22f,   0.14F,   0.59F,  -0.63F,  -0.44F},
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

                (double[] s, double[,] U, double[,] vT) = MagmaSharp.LinAlg.Svd(A, true, true, device: Device.GPU);

                for (int i = 0; i < result_s.Length; i++)
                    Debug.Assert(((double)Math.Round(result_s[i],2) == (double)Math.Round(s[i], 2)));


                for (int i = 0; i < U.GetLength(0); i++)
                {
                    for (int j = 0; j < U.GetLength(1); j++)
                    {
                       //
                        Debug.Assert(((double)Math.Round(resultU[i, j],2) == (double)Math.Round(U[i, j], 2)));
                    }
                }

                for (int i = 0; i < vT.GetLength(0); i++)
                {
                    for (int j = 0; j < vT.GetLength(1); j++)
                    {
                        //
                        Debug.Assert(((double)Math.Round(resultvT[i, j],2) == (double)Math.Round(vT[i, j], 2)));
                    }
                }
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

                (double[] s, double[,] U, double[,] vT) = MagmaSharp.LinAlg.Svd(A, true, true, device: Device.CPU);

                for (int i = 0; i < result_s.Length; i++)
                    Debug.Assert(((double)Math.Round(result_s[i], 2) == (double)Math.Round(s[i], 2)));


                for (int i = 0; i < U.GetLength(0); i++)
                {
                    for (int j = 0; j < U.GetLength(1); j++)
                    {
                        //
                        Debug.Assert(((double)Math.Round(resultU[i, j], 2) == (double)Math.Round(U[i, j], 2)));
                    }
                }

                for (int i = 0; i < vT.GetLength(0); i++)
                {
                    for (int j = 0; j < vT.GetLength(1); j++)
                    {
                        //
                        Debug.Assert(((double)Math.Round(resultvT[i, j], 2) == (double)Math.Round(vT[i, j], 2)));
                    }
                }
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

                (double[] s, double[,] U, double[,] vT) = MagmaSharp.LinAlg.Svd(A, true, true, device: Device.CUSTOM);

                for (int i = 0; i < result_s.Length; i++)
                    Debug.Assert(((double)Math.Round(result_s[i], 2) == (double)Math.Round(s[i], 2)));


                for (int i = 0; i < U.GetLength(0); i++)
                {
                    for (int j = 0; j < U.GetLength(1); j++)
                    {
                        //
                        Debug.Assert(((double)Math.Round(resultU[i, j], 2) == (double)Math.Round(U[i, j], 2)));
                    }
                }

                for (int i = 0; i < vT.GetLength(0); i++)
                {
                    for (int j = 0; j < vT.GetLength(1); j++)
                    {
                        //
                        Debug.Assert(((double)Math.Round(resultvT[i, j], 2) == (double)Math.Round(vT[i, j], 2)));
                    }
                }
            }

        }

    }
}
