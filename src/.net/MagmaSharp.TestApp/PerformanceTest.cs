using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace MagmaSharp.TestApp
{
    public static class PerformanceTest
    {
        public static void SVD()
        {
            MagmaSharp.LinAlg.init();
            string strTxt = @"
******************************************
          MagmaSharp performance Test
******************************************";
            Console.WriteLine(strTxt);

            for(int i=5; i < 128;i++)
            {
                int m = 64*i;
                int n = 64*i;
                strTxt = $"Random generation of {m} x {n} matrix";
                Console.WriteLine(strTxt);
                double[,] matrix = generatedMatrix(m, n);
                Console.WriteLine($"Performance Test started......");


                //SVD using MAth.NET
                Stopwatch sw = Stopwatch.StartNew();
                if (n < 800)
                {
                    svdUsingMathNet(matrix);
                    Console.WriteLine($"Math.NET Execution time\t\t:{sw.Elapsed.TotalSeconds} s.");
                    Console.WriteLine($" ");
                }
                else
                {
                    Console.WriteLine($"Math.NET skipped test for m > 800");
                    Console.WriteLine($" ");
                }
                

                if(n < 500)
                {
                    sw.Restart();
                    svdUsingAccord(matrix);
                    Console.WriteLine($"Accord.NET Execution time\t:{sw.Elapsed.TotalSeconds} s.");
                    Console.WriteLine($" ");
                }
                else
                {
                    Console.WriteLine($"Accord.NET skipped test for m > 500");
                    Console.WriteLine($" ");
                }

                sw.Restart();
                var svs = MagmaSharp.LinAlg.Svd(matrix, true, true, device: Device.CUSTOM);
                Console.WriteLine($"LAPACK Execution time \t\t:{sw.Elapsed.TotalSeconds} s.");
                Console.WriteLine($" ");

                sw.Restart();
                var svs0 = MagmaSharp.LinAlg.Svd(matrix, true, true, device: Device.CPU);
                Console.WriteLine($"MAGMA Execution time\t\t:{sw.Elapsed.TotalSeconds} s.");
                Console.WriteLine($" ");
                Console.WriteLine($" ");
                Console.WriteLine($" ");
            }
            
            return;
            //var svdd = new Accord.Math.Decompositions.SingularValueDecomposition(matrix1);
            //Console.WriteLine($"Accord Execution time:{sw.Elapsed.TotalSeconds} s.");


            Console.WriteLine($" ");
            Console.WriteLine($"SVD calculation using Math.NET ......");
            Console.WriteLine($"SVD calculation using Math.NET ......");

        }

        private static void svdUsingMathNet(double[,] matrix)
        {
            var m = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.OfArray(matrix);
            var svd = m.Svd(true);
            //svd.U() * svd.W() * svd.VT()
        }
        private static void svdUsingAccord(double[,] matrix)
        {
            var svdd = new Accord.Math.Decompositions.SingularValueDecomposition(matrix);
            //svd.U() * svd.W() * svd.VT()
        }

        private static void svdUsingMagmaSharp(double[,] matrix)
        {
            var m = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.OfArray(matrix);
            var svd = m.Svd(true);
            //svd.U() * svd.W() * svd.VT()
        }

        private static float[,] generatesMatrix(int nRows, int nCols)
        {
            //The maximum matrix rank where managed method is faster than native is m=250
            int m = nRows;
            int n = nCols;

            var rndMatrix = Daany.nc.Rand(m, n).Select(x => Convert.ToSingle(x)).ToList();
            
            var matrix = new float[m, n];
            int k = 0;
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    var v = rndMatrix[k++];
                    matrix[i, j] = v;
                }
            }

            return matrix;
        }

        private static double[,] generatedMatrix(int nRows, int nCols)
        {
            //The maximum matrix rank where managed method is faster than native is m=250
            int m = nRows;
            int n = nCols;

            var rndMatrix = Daany.nc.Rand(m, n).Select(x => Convert.ToSingle(x)).ToList();

            var matrix = new double[m, n];
            int k = 0;
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    var v = rndMatrix[k++];
                    matrix[i, j] = v;
                }
            }

            return matrix;
        }
    }
}
