using System;
using System.Diagnostics;
using System.Linq;

namespace MagmaSharp.TestApp
{
    class Program
    {
        static void Main(string[] args)
        {

           // foreach (var i in Enumerable.Range(0,100))
                TestSvdFloat(); //TestFloat();

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

        public static void TestFloat()
        {
            float[,] A = new float[,]
            {
             {-1.01f, 0.86f,-4.60f, 3.31f,-4.81f},
             { 3.98f, 0.53f,-7.04f, 5.29f, 3.55f},
             { 3.30f, 8.26f,-3.89f, 8.20f,-1.51f},
             { 4.43f, 4.96f,-7.66f,-7.33f, 6.18f},
             { 7.31f,-6.43f,-6.16f, 2.47f, 5.58f},

            };

            float[] result_wr = new float[] { 2.86f, 2.86f, -0.69f, -0.69f, -10.46f };
            float[] result_wi = new float[] { 10.76f, -10.76f, 4.70f, -4.70f, 0 };

            {
                (float[] wr, float[] wi) = MagmaSharp.LinAlg.Eigen(A, Device.CPU);

                for (int i = 0; i < result_wr.Length; i++)
                    Debug.Assert(result_wr[i] == (float)Math.Round(wr[i], 2));

              //  Console.WriteLine("Test Passed");

                for (int i = 0; i < result_wi.Length; i++)
                    Debug.Assert(result_wi[i] == (float)Math.Round(wi[i], 2));
               // Console.WriteLine("Test Passed");

            }


            {
                (float[] wr, float[] wi) = MagmaSharp.LinAlg.Eigen(A, Device.GPU);

                for (int i = 0; i < result_wr.Length; i++)
                    Debug.Assert(result_wr[i] == (float)Math.Round(wr[i], 2));

                for (int i = 0; i < result_wi.Length; i++)
                    Debug.Assert(result_wi[i] == (float)Math.Round(wi[i], 2));
               // Console.WriteLine("Test Passed");

            }

            {
                (float[] wr, float[] wi) = MagmaSharp.LinAlg.Eigen(A, Device.CUSTOM);

                for (int i = 0; i < result_wr.Length; i++)
                    Debug.Assert(result_wr[i]== (float)Math.Round(wr[i],2));

                for (int i = 0; i < result_wi.Length; i++)
                    Debug.Assert(result_wi[i]== (float)Math.Round(wi[i],2));
               // Console.WriteLine("Test Passed");

            }
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
                float[,] A = new float[,]
                {
                    { 8.79f,   9.93f,   9.83f,   5.45f,   3.16f, },
                    { 6.11f,   6.91f,   5.04f,  -0.27f,   7.98f, },
                    {-9.15f,  -7.93f,   4.86f,   4.85f,   3.01f, },
                    { 9.57f,   1.64f,   8.83f,   0.74f,   5.80f, },
                    {-3.49f,   4.02f,   9.80f,  10.00f,   4.27f, },
                    { 9.84f,   0.15f,  -8.99f,  -6.02f,  -5.31f, },
                };

                (float[] s, float[,] U, float[,] vT) = MagmaSharp.LinAlg.Svd(A, Device.GPU);

                for (int i = 0; i < result_s.Length; i++)
                    Debug.Assert(result_s[i] == (float)Math.Round(s[i],2));

               printMatrix<float>(U);

              
            }

           
        }
         
    }
}
