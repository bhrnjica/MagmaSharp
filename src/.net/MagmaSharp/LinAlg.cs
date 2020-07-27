using System;
using System.Runtime.InteropServices;
namespace MagmaSharp
{
    public enum Device
    {
        DEFAULT=0,
        CPU=1,
        GPU=2,
        CUSTOM=3,
    }
    public class LinAlg
    {
        static Device _device;
        static LinAlg()
        {
            try
            {
                var arch = DeviceMng.GetCUDAArch();
                if (arch == "No CUDA detected on this machine!")
                {
                    Console.WriteLine($"No CUDA detected on this machine!");
                    _device = Device.CPU;

                }
                else
                {
                    Console.WriteLine($"CUDA Arch ={arch} detected!");
                    _device = Device.GPU;
                }

            }
            catch
            {
                Console.WriteLine("No CUDA detected on this machine!");
                _device = Device.CPU;

            }

        }

        /// <summary>
        /// Dummy method to be called first 
        /// </summary>
        public static void init(){;}

        #region Solver- solver of system of linear equations
        // 
        [DllImport("LapackBinding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2sgesv_cpu(int n, int nrhs, IntPtr A, int lda, IntPtr ipiv, IntPtr B, int lbd);

        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2sgesv(int n, int nrhs, IntPtr A, int lda, IntPtr ipiv, IntPtr B,int lbd);

        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2sgesv_gpu(int n, int nrhs, IntPtr A, int ldda, IntPtr ipiv, IntPtr B, int lddb);

        //double
        [DllImport("LapackBinding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgesv_cpu(int n, int nrhs, IntPtr A, int lda, IntPtr ipiv, IntPtr B, int lbd);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgesv(int n, int nrhs, IntPtr A, int lda, IntPtr ipiv, IntPtr B, int lbd);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgesv_gpu(int n, int nrhs, IntPtr A, int lda, IntPtr ipiv, IntPtr B, int lbd);


        /// <summary>
        /// Solve linear system of equations A X = B.
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns>X - solution matrix</returns>
        public static float[,] Solve(float[,] A, float [,] B, Device device= Device.DEFAULT)
        {
            //define parameters
            int n = A.GetLength(0);
            int nhrs = B.GetLength(1);
            int lda = n;
            int ldb = B.GetLength(0);

            //define arrays
            int[] ipivv = new int[n];//permutation indices
            float[] Af = Util.To1DArray<float>(A);
            float[] Bf = Util.To1DArray<float>(B);


            // Initialize unmanaged memory to hold the array.
            int sizeA = Marshal.SizeOf(Af[0]) * Af.Length;
            int sizeB = Marshal.SizeOf(Bf[0]) * Bf.Length;
            int sizeIpiv = Marshal.SizeOf(ipivv[0]) * ipivv.Length;

            //copy managed arrays to native prior to pInvoke call
            IntPtr a = Marshal.AllocHGlobal(sizeA);
            IntPtr b = Marshal.AllocHGlobal(sizeB);
            IntPtr ipiv = Marshal.AllocHGlobal(sizeIpiv);

            //
            Marshal.Copy(Af, 0, a, Af.Length);
            Marshal.Copy(Bf, 0, b, Bf.Length);
            Marshal.Copy(ipivv, 0, ipiv, ipivv.Length);

            //pInvoke call
            int info = -1;
            if ((device == Device.DEFAULT || device==Device.GPU) && _device == Device.GPU)
                info = mbv2sgesv_gpu(n, nhrs, a, lda, ipiv, b, ldb);
            else if (device == Device.CPU && _device == Device.GPU)
                info = mbv2sgesv(n, nhrs, a, lda, ipiv, b, ldb);
            else
                info = mbv2sgesv_cpu(n, nhrs, a, lda, ipiv, b, ldb);
            //
            if (info != 0)
                throw new Exception($"magma_sgesv failed due to invalid parameter {-info}.");

            //back the array from native to managed
            //Marshal.Copy(a, Af, 0, Af.Length);
            Marshal.Copy(b, Bf, 0, Bf.Length);

            //matrix returned from Magma is column major, 
            //it should be transpose during creation as managed matrix array 
            var X = Util.To2dArray<float>(Bf,n, nhrs, ldb);

            //Free memory
            Marshal.FreeHGlobal(a);
            Marshal.FreeHGlobal(b);
            Marshal.FreeHGlobal(ipiv);
            //
            return X;
        }

        /// <summary>
        /// Solve linear system of equations A X = B.
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns>X - solution matrix</returns>
        public static double[,] Solve(double[,] A, double[,] B, Device device = Device.DEFAULT)
        {
            //define parameters
            int n = A.GetLength(0);
            int nhrs = B.GetLength(1);
            int lda = n;
            int ldb = B.GetLength(0);

            //define arrays
            int[] ipivv = new int[n];//permutation indices
            double[] Af = Util.To1DArray<double>(A);
            double[] Bf = Util.To1DArray<double>(B);


            // Initialize unmanaged memory to hold the array.
            int sizeA = Marshal.SizeOf(Af[0]) * Af.Length;
            int sizeB = Marshal.SizeOf(Bf[0]) * Bf.Length;
            int sizeIpiv = Marshal.SizeOf(ipivv[0]) * ipivv.Length;

            //copy managed arrays to native prior to pInvoke call
            IntPtr a = Marshal.AllocHGlobal(sizeA);
            IntPtr b = Marshal.AllocHGlobal(sizeB);
            IntPtr ipiv = Marshal.AllocHGlobal(sizeIpiv);

            //
            Marshal.Copy(Af, 0, a, Af.Length);
            Marshal.Copy(Bf, 0, b, Bf.Length);
            Marshal.Copy(ipivv, 0, ipiv, ipivv.Length);

            //pInvoke call
            int info = -1;
            if ((device == Device.DEFAULT || device == Device.GPU) && _device == Device.GPU)
                info = mbv2dgesv_gpu(n, nhrs, a, lda, ipiv, b, ldb);
            else if (device == Device.CPU && _device == Device.GPU)
                info = mbv2dgesv(n, nhrs, a, lda, ipiv, b, ldb);
            else
                info = mbv2dgesv_cpu(n, nhrs, a, lda, ipiv, b, ldb);
            //
            if (info != 0)
                throw new Exception($"magma_sgesv failed due to invalid parameter {-info}.");

            //back the array from native to managed
            //Marshal.Copy(a, Af, 0, Af.Length);
            Marshal.Copy(b, Bf, 0, Bf.Length);

            //matrix returned from Magma is column major, 
            //it should be transpose during creation as managed matrix array 
            var X = Util.To2dArray<double>(Bf, n, nhrs, ldb);

            //Free memory
            Marshal.FreeHGlobal(a);
            Marshal.FreeHGlobal(b);
            Marshal.FreeHGlobal(ipiv);
            //
            return X;
        }
        #endregion

        #region LSS - least square solver
        [DllImport("LapackBinding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2sgels_cpu(int m, int n, int nrhs, IntPtr A, int lda, IntPtr B, int lbd);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2sgels(int m, int n, int nrhs, IntPtr A, int lda, IntPtr B, int lbd);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2sgels_gpu(int m, int n, int nrhs, IntPtr A, int lda, IntPtr B, int lbd);

        [DllImport("LapackBinding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgels_cpu(int m, int n, int nrhs, IntPtr A, int lda, IntPtr B, int lbd);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgels(int m, int n, int nrhs, IntPtr A, int lda, IntPtr B, int lbd);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgels_gpu(int m, int n, int nrhs, IntPtr A, int lda, IntPtr B, int lbd);

        public static float[,] Lss(float[,] A, float[,] B, Device device = Device.DEFAULT)
        {
            //define parameters
            int m = A.GetLength(0);
            int n = A.GetLength(1);
            int ldb = B.GetLength(0);
            int nhrs = B.GetLength(1);
            int lda = m;

            //define arrays
            int[] ipivv = new int[n];//permutation indices
            float[] Af = Util.To1DArray<float>(A);
            float[] Bf = Util.To1DArray<float>(B);

            // Initialize unmanaged memory to hold the array.
            //copy managed arrays to native prior to pInvoke call
            IntPtr a = Marshal.AllocHGlobal(sizeof(float) * Af.Length);
            IntPtr b = Marshal.AllocHGlobal(sizeof(float) * Bf.Length);
            Marshal.Copy(Af, 0, a, Af.Length);
            Marshal.Copy(Bf, 0, b, Bf.Length);

            //pInvoke call
            int info = -1;
            if ((device == Device.DEFAULT || device == Device.GPU) && _device==Device.GPU)
                info = mbv2sgels_gpu(m, n, nhrs, a, lda, b, ldb);
            else if (device == Device.CPU && _device == Device.GPU)
                info = mbv2sgels(m, n, nhrs, a, lda, b, ldb);
            else
                info = mbv2sgels_cpu(m, n, nhrs, a, lda, b, ldb);
            //
            if (info != 0)
                throw new Exception($"magma_sgesv failed due to invalid parameter {-info}.");

            //back the array from native to managed
            //it should be transpose during creation as managed matrix array 
            var X = Util.FromNativeToSArray(b, n, nhrs, ldb);

            //Free memory
            Marshal.FreeHGlobal(a);
            Marshal.FreeHGlobal(b);
            //
            return X;
        }

        public static double[,] Lss(double[,] A, double[,] B, Device device = Device.DEFAULT)
        {
            //define parameters
            int m = A.GetLength(0);
            int n = A.GetLength(1);
            int ldb = B.GetLength(0);
            int nhrs = B.GetLength(1);
            int lda = m;


            //define arrays
            int[] ipivv = new int[n];//permutation indices
            double[] Af = Util.To1DArray<double>(A);
            double[] Bf = Util.To1DArray<double>(B);


            // Initialize unmanaged memory to hold the array.
            //copy managed arrays to native prior to pInvoke call
            IntPtr a = Marshal.AllocHGlobal(sizeof(double) * Af.Length);
            IntPtr b = Marshal.AllocHGlobal(sizeof(double) * Bf.Length);
            Marshal.Copy(Af, 0, a, Af.Length);
            Marshal.Copy(Bf, 0, b, Bf.Length);

            //pInvoke call
            int info = -1;
            if ((device == Device.DEFAULT || device == Device.GPU) && _device == Device.GPU)
                info = mbv2dgels_gpu(m, n, nhrs, a, lda, b, ldb);
            else if (device == Device.CPU && _device == Device.GPU)
                info = mbv2dgels(m, n, nhrs, a, lda, b, ldb);
            else
                info = mbv2dgels_cpu(m, n, nhrs, a, lda, b, ldb);
            //
            if (info != 0)
                throw new Exception($"magma_sgesv failed due to invalid parameter {-info}.");

            //back the array from native to managed
            var X = Util.FromNativeToDArray(b, n, nhrs, ldb);

            //Free memory
            Marshal.FreeHGlobal(a);
            Marshal.FreeHGlobal(b);
            //
            return X;
        }
        #endregion

        #region Eigenvalues 
        [DllImport("LapackBinding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2sgeevs_cpu(int n, IntPtr A, int lda, IntPtr wr, IntPtr wi, IntPtr VL, int ldvl, IntPtr VR, int ldvr);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2sgeevs(int n, IntPtr A, int lda, IntPtr wr, IntPtr wi, IntPtr VL, int ldvl, IntPtr VR, int ldvr);

        [DllImport("LapackBinding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgeevs_cpu(int n, IntPtr A, int lda, IntPtr wr, IntPtr wi, IntPtr VL, int ldvl, IntPtr VR, int ldvr);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgeevs(int n, IntPtr A, int lda, IntPtr wr, IntPtr wi, IntPtr VL, int ldvl, IntPtr VR, int ldvr);
       
        public static (float[] wr, float[] wi ) Eigen(float[,] A, Device device = Device.DEFAULT)
        {
            //define parameters
            int m = A.GetLength(0);
            if (m != A.GetLength(1))
                throw new Exception("Matrix A must be squared!");
            //
            int lda = m;//Leading dimension for A
            int ldvl = m;//Leading dimensions for VL
            int ldvr = m;//Leading dimensions for VR

            //create native arrays
            IntPtr Af = Util.ToNativeSArray(A);
            //
            IntPtr pwr = Marshal.AllocHGlobal(sizeof(float) * m);
            IntPtr pwi = Marshal.AllocHGlobal(sizeof(float) * m);
            //IntPtr pVl = Marshal.AllocHGlobal(sizeof(double) * ldvl * m);
            //IntPtr pVr = Marshal.AllocHGlobal(sizeof(double) * ldvr * m);

            //pInvoke call
            int info = -1;
            if ((device == Device.DEFAULT || device == Device.GPU) && _device == Device.GPU)
                info = mbv2sgeevs(m, Af, lda, pwr, pwi, IntPtr.Zero, ldvl, IntPtr.Zero, ldvr);
            else if (device == Device.CPU && _device == Device.GPU)
                info = mbv2sgeevs(m, Af, lda, pwr, pwi, IntPtr.Zero, ldvl, IntPtr.Zero, ldvr);
            else
                info = mbv2sgeevs_cpu(m, Af, lda, pwr, pwi, IntPtr.Zero, ldvl, IntPtr.Zero, ldvr);

            //
            if (info != 0)
                throw new Exception($"magma_sgesv failed due to invalid parameter {-info}.");

            //back the array from native to managed
           // var Am = Util.fromNativeToSArray(Af, m, m, lda);
            var wr = Util.FromNativeToSArray(pwr, m);
            var wi = Util.FromNativeToSArray(pwi, m);
            //var Vl = Util.fromNativeToSArray(pVl, m, m, ldvl);
            //var Vr = Util.fromNativeToSArray(pVr, m, m, ldvr);

            ////Free memory
            Marshal.FreeHGlobal(Af);
            Marshal.FreeHGlobal(pwr);
            Marshal.FreeHGlobal(pwi);
            //Marshal.FreeHGlobal(pVl);
            //Marshal.FreeHGlobal(pVr);
            //
            //return (Vl, Vr, wr, wi);
            return (wr, wi);
        }

        public static (double[] wr, double[] wi) Eigen(double[,] A, Device device = Device.DEFAULT)
        {
            //define parameters
            int m = A.GetLength(0);
            if (m != A.GetLength(1))
                throw new Exception("Matrix A must be squared!");
            //
            int lda = m;//Leading dimension for A
            int ldvl = m;//Leading dimensions for VL
            int ldvr = m;//Leading dimensions for VR

            //create native arrays
            IntPtr Af = Util.ToNativeDArray(A);
            //
            IntPtr pwr = Marshal.AllocHGlobal(sizeof(double) * m);
            IntPtr pwi = Marshal.AllocHGlobal(sizeof(double) * m);
            //IntPtr pVl = Marshal.AllocHGlobal(sizeof(double) * ldvl * m);
            //IntPtr pVr = Marshal.AllocHGlobal(sizeof(double) * ldvr * m);

            //pInvoke call
            int info = -1;
            if ((device == Device.DEFAULT || device == Device.GPU) && _device == Device.GPU)
                info = mbv2dgeevs(m, Af, lda, pwr, pwi, IntPtr.Zero, ldvl, IntPtr.Zero, ldvr);
            else if (device == Device.CPU && _device == Device.GPU)
                info = mbv2dgeevs(m, Af, lda, pwr, pwi, IntPtr.Zero, ldvl, IntPtr.Zero, ldvr);
            else
                info = mbv2dgeevs_cpu(m, Af, lda, pwr, pwi, IntPtr.Zero, ldvl, IntPtr.Zero, ldvr);
            //
            if (info != 0)
                throw new Exception($"magma_sgesv failed due to invalid parameter {-info}.");

            //back the array from native to managed
            var Am = Util.FromNativeToDArray(Af, m, m, lda);
            var wr = Util.FromNativeToDArray(pwr, m);
            var wi = Util.FromNativeToDArray(pwi, m);
            //var Vl = Util.fromNativeToDArray(pVl, m, m, ldvl);
            //var Vr = Util.fromNativeToDArray(pVr, m, m, ldvr);

            ////Free memory
            Marshal.FreeHGlobal(Af);
            Marshal.FreeHGlobal(pwr);
            Marshal.FreeHGlobal(pwi);
            //Marshal.FreeHGlobal(pVl);
            //Marshal.FreeHGlobal(pVr);
            //
            //return (Vl, Vr, wr, wi);
            return (wr, wi);
        }
        #endregion

        #region SVD singular value decomposition
        [DllImport("LapackBinding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2sgesvds_cpu(int m, int n, IntPtr A, IntPtr s, IntPtr U, IntPtr VT);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2sgesvds(int m, int n, IntPtr A, IntPtr s, IntPtr U, IntPtr VT);

        [DllImport("LapackBinding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgesvds_cpu(int m, int n, IntPtr A, IntPtr s, IntPtr U, IntPtr VT);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgesvds(int m, int n, IntPtr A, IntPtr s, IntPtr U, IntPtr VT);

        /// <summary>
        /// Solve linear system of equations A X = B.
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns>X - solution matrix</returns>
        public static (float[] s, float[,] U, float[,] vt ) Svd(float[,] A, Device device = Device.DEFAULT)
        {
            //define parameters
            int m = A.GetLength(0);//the number of rows
            int n = A.GetLength(1);//the number of columns

            //Initialize unmanaged memory to hold the array.
            float[] Af = Util.To1DArray<float>(A);
            IntPtr a = Marshal.AllocHGlobal(sizeof(float) * Af.Length);
            IntPtr s = Marshal.AllocHGlobal(sizeof(float) * n);
            IntPtr u = Marshal.AllocHGlobal(sizeof(float) * m * m);
            IntPtr vt= Marshal.AllocHGlobal(sizeof(float) * n * n);
            //
            Marshal.Copy(Af, 0, a, Af.Length);

            //pInvoke call
            int info = -1;
            if ((device == Device.DEFAULT || device == Device.GPU || device == Device.CPU ) && _device == Device.GPU)
                info = mbv2sgesvds(m, n, a, s, u, vt);
            else
                info = mbv2sgesvds_cpu(m, n, a, s, u, vt);

            //
            if (info != 0)
                throw new Exception($"magma_svd failed due to invalid parameter {-info}.");

            //back the array from native to managed
            //var aa = Util.FromNativeToSArray(a, Af.Length);
            var S = Util.FromNativeToSArray(s, n);
            var U = Util.FromNativeToSArray(u, m, m, m);
            var VT = Util.FromNativeToSArray(vt, n, n, n);

            ////Free memory
            Marshal.FreeHGlobal(a);
            Marshal.FreeHGlobal(s);
            Marshal.FreeHGlobal(u);
            Marshal.FreeHGlobal(vt);
            a = IntPtr.Zero;
            s = IntPtr.Zero;
            u = IntPtr.Zero;
            vt = IntPtr.Zero;
            //
            return (S, U, VT);
        }

        /// <summary>
        /// Solve linear system of equations A X = B.
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns>X - solution matrix</returns>
        public static (double[] s, double[,] U, double[,] vt) Svd(double[,] A, Device device = Device.DEFAULT)
        {
            //define parameters
            int m = A.GetLength(0);//the number of rows
            int n = A.GetLength(1);//the number of columns

            //Initialize unmanaged memory to hold the array.
            double[] Af = Util.To1DArray<double>(A);
            IntPtr a = Marshal.AllocHGlobal(sizeof(double) * Af.Length);
            IntPtr s = Marshal.AllocHGlobal(sizeof(double) * n);
            IntPtr u = Marshal.AllocHGlobal(sizeof(double) * m * m);
            IntPtr vt = Marshal.AllocHGlobal(sizeof(double) * n * n);
            Marshal.Copy(Af, 0, a, Af.Length);

            //pInvoke call
            int info = -1;
            if ((device == Device.DEFAULT || device == Device.GPU) && _device == Device.GPU)
                info = mbv2dgesvds(m, n, a, s, u, vt);
            else if (device == Device.CPU && _device == Device.GPU)
                info = mbv2dgesvds(m, n, a, s, u, vt);
            else
                info = mbv2dgesvds_cpu(m, n, a, s, u, vt);

            //
            if (info != 0)
                throw new Exception($"magma_svd failed due to invalid parameter {-info}.");

            //back the array from native to managed
            var S = Util.FromNativeToDArray(s, n);
            var U = Util.FromNativeToDArray(u, m, m, m);
            var VT = Util.FromNativeToDArray(vt, n, n, n);

            ////Free memory
            Marshal.FreeHGlobal(s);
            Marshal.FreeHGlobal(u);
            Marshal.FreeHGlobal(vt);
            //
            return (S, U, VT);
        }
        #endregion
    }
}
