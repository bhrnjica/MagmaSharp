using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace MagmaSharp
{
    /// <summary>
    /// Utility class for data preparation before magmabinding method calls
    /// </summary>
    public static class Util
    {

        /// <summary>
        /// Takes 2D row major matrix to convert it to column major 1D array matrix
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="array"></param>
        /// <returns></returns>
        public static T[] To1DArray<T>(T[,] array)
        {
            int n = array.GetLength(0);//num of rows
            int m = array.GetLength(1);//num of cols
            T[] flatArray = new T[n * m];
            int k = 0;
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    flatArray[k++] = array[j, i];
                }
            }
            return flatArray;
        }

        /// <summary>
        /// Takes 1D column major matrix to convert it into 2D row major matrix
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="farray"></param>
        /// <param name="nRows"></param>
        /// <param name="nCols"></param>
        /// <param name="lda">leading dimension of the matrix</param>
        /// <returns></returns>
        public static T[,] To2dArray<T>(T[] farray, int nRows, int nCols, int lda)
        {
            T[,] array = new T[nRows, nCols];
            for (int j = 0; j < nCols; j++)
            {
                for (int i = 0; i < nRows; i++)
                {
                    array[i, j] = farray[i + j * lda];
                }
            }
            return array;
        }

        public static IntPtr ToNativeDArray(double[,] array)
        {
            double[] Af = Util.To1DArray<double>(array);
            IntPtr ptr = Marshal.AllocHGlobal(sizeof(double) * Af.Length);
            Marshal.Copy(Af, 0, ptr, Af.Length);
            return ptr;
        }

        public static IntPtr ToNativeSArray(float[,] array)
        {
            float[] Af = Util.To1DArray<float>(array);
            IntPtr ptr = Marshal.AllocHGlobal(sizeof(float) * Af.Length);
            Marshal.Copy(Af, 0, ptr, Af.Length);
            return ptr;
        }

        public static float[,] FromNativeToSArray(IntPtr narray, int nRows, int nCols, int lda)
        {
            var flatArray = new float[lda * nCols];
            Marshal.Copy(narray, flatArray, 0, flatArray.Length);
            return To2dArray<float>(flatArray, nRows, nCols, lda);
        }

        public static float[] FromNativeToSArray(IntPtr narray, int nCols)
        {
            var flatArray = new float[nCols];
            Marshal.Copy(narray, flatArray, 0, flatArray.Length);
            return flatArray;
        }


        public static double[,] FromNativeToDArray(IntPtr narray, int nRows, int nCols, int lda)
        {
            var flatArray = new double[lda * nCols];
            Marshal.Copy(narray, flatArray, 0, flatArray.Length);
            return To2dArray<double>(flatArray, nRows, nCols, lda);
        }

        public static double[] FromNativeToDArray(IntPtr narray, int nCols)
        {
            var flatArray = new double[nCols];
            Marshal.Copy(narray, flatArray, 0, flatArray.Length);
            return flatArray;
        }
    }
}
