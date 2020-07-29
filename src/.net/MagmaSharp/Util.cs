using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;

namespace MagmaSharp
{
    public static class GenericCopier<T>    //deep copy a list
    {
        public static T DeepCopy(object objectToCopy)
        {
            using (MemoryStream memoryStream = new MemoryStream())
            {
                BinaryFormatter binaryFormatter = new BinaryFormatter();
                binaryFormatter.Serialize(memoryStream, objectToCopy);
                memoryStream.Seek(0, SeekOrigin.Begin);
                return (T)binaryFormatter.Deserialize(memoryStream);
            }
        }
    }

    /// <summary>
    /// Utility class for data preparation before magmabinding method calls
    /// </summary>
    public static class Util
    {
        public static T DeepCopy<T>(T objectToCopy)
        {
            using (MemoryStream memoryStream = new MemoryStream())
            {
                BinaryFormatter binaryFormatter = new BinaryFormatter();
                binaryFormatter.Serialize(memoryStream, objectToCopy);
                memoryStream.Seek(0, SeekOrigin.Begin);
                return (T)binaryFormatter.Deserialize(memoryStream);
            }
        }

        public static T[] To1DRowArray<T>(T[,] array)
        {
            int n = array.GetLength(0);//num of rows
            int m = array.GetLength(1);//num of cols
            T[] flatArray = new T[n * m];
            int k = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    flatArray[k++] = array[i, j];
                }
            }
            return flatArray;
        }

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

        unsafe public static double[] To1DdArray<T>(double[,] array)
        {
            int n = array.GetLength(0);//num of rows
            int m = array.GetLength(1);//num of cols
            double[] flatArray = new double[n * m];
            if (flatArray.Length == array.Length)
            {
                fixed (double* pA = array, pB = flatArray)
                {
                    for(int i=0; i< flatArray.Length; i++)
                    {
                        pB[i] = pA[i];
                    }
                }
            }
            return flatArray;
        }
        unsafe public static float[] To1DsArray<T>(float[,] array)
        {
            int n = array.GetLength(0);//num of rows
            int m = array.GetLength(1);//num of cols
            float[] flatArray = new float[n * m];
            if (flatArray.Length == array.Length)
            {
                fixed (float* pA = array, pB = flatArray)
                {
                    for (int i = 0; i < flatArray.Length; i++)
                    {
                        pB[i] = pA[i];
                    }
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
