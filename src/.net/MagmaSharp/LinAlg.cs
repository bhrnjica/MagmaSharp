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
    unsafe public class LinAlg
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
        private static extern int mbv2sgesv_cpu(bool rowmajor, int n, int nrhs, float* A, int lda, int* ipiv, float* B, int lbd);

        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2sgesv(bool rowmajor, int n, int nrhs, float* A, int lda, int* ipiv, float* B,int lbd);

        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2sgesv_gpu(bool rowmajor, int n, int nrhs, float* A, int ldda, int* ipiv, float* B, int lddb);

        //double
        [DllImport("LapackBinding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgesv_cpu(bool rowmajor, int n, int nrhs, double* A, int ldda, int* ipiv, double* B, int lddb);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgesv(bool rowmajor, int n, int nrhs, double* A, int ldda, int* ipiv, double* B, int lddb);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgesv_gpu(bool rowmajor, int n, int nrhs, double* A, int ldda, int* ipiv, double* B, int lddb);

        public static float[,] Solve(float[,] A, float[,] B, Device device = Device.DEFAULT)
        {
            //define parameters
            int info = -1;
            int n = A.GetLength(0);
            int nrhs = B.GetLength(1);
            var Ac = A.Clone() as float[,];
            var Bc = B.Clone() as float[,];

            //define arrays
            int[] ipiv = new int[n];//permutation indices
            fixed(float *pA = Ac, pB = Bc)
            {
                fixed(int* pipiv = ipiv)
                {
                    //pInvoke call
                    if ((device == Device.DEFAULT || device == Device.GPU) && _device == Device.GPU)
                        info = mbv2sgesv_gpu(true, n, nrhs, pA, n, pipiv, pB, n);
                    else if (device == Device.CPU && _device == Device.GPU)
                        info = mbv2sgesv(true, n, nrhs, pA, n, pipiv, pB, n);
                    else
                        info = mbv2sgesv_cpu(true, n, nrhs, pA, n, pipiv, pB, nrhs);
                }

            }
            //
            if (info != 0)
                throw new Exception($"magma_sgesv failed due to invalid parameter {-info}.");

            //
            return Bc;
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
            int info = -1;
            int n = A.GetLength(0);
            int nrhs = B.GetLength(1);
            var Ac = A.Clone() as double[,];
            var Bc = B.Clone() as double[,];
            //define arrays
            int[] ipiv = new int[n];//permutation indices
            fixed (double* pA = Ac, pB = Bc)
            {
                fixed (int* pipiv = ipiv)
                {
                    //pInvoke call
                    if ((device == Device.DEFAULT || device == Device.GPU) && _device == Device.GPU)
                        info = mbv2dgesv_gpu(true, n, nrhs, pA, n, pipiv, pB, n);
                    else if (device == Device.CPU && _device == Device.GPU)
                        info = mbv2dgesv(true, n, nrhs, pA, n, pipiv, pB, n);
                    else
                        info = mbv2dgesv_cpu(true, n, nrhs, pA, n, pipiv, pB, nrhs);
                }

            }
            //
            if (info != 0)
                throw new Exception($"magma_sgesv failed due to invalid parameter {-info}.");

            //
            return Bc;
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
        private static extern int mbv2sgesvds_cpu(bool rowmajor, int m, int n, float* A, float* s, float* U,bool calcU, float* VT, bool calcV);

        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2sgesvds(bool rowmajor, int m, int n, float* A, float* s, float* U, bool calcU, float* VT, bool calcV);

        [DllImport("LapackBinding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgesvds_cpu(bool rowmajor, int m, int n, double* A, double* s, double* U, bool calcU, double* VT, bool calcV);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgesvds(bool rowmajor, int m, int n, double* A, double* s, double* U, bool calcU, double* VT, bool calcV);


        /// <summary>
        ///  Decompose rectangular matrix A on A = U * s * Vt
        /// </summary>
        /// <param name="A"></param>
        /// <param name="device"></param>
        /// <returns></returns>
        public static (float[] s, float[,] U, float[,] vt) Svd(float[,] A, bool calcU, bool calcVt, Device device = Device.DEFAULT)
        {
            //define parameters
            int m = A.GetLength(0);//the number of rows
            int n = A.GetLength(1);//the number of columns
            var Ac = A.Clone() as float[,];
            //
            float[] s = new float[n];
            float[,] U = new float[m, m];
            float[,] VT = new float[n, n];
            
            //Initialize unmanaged memory to hold the array.
            fixed (float* pA = Ac, ps=s, pU=U, pVT=VT)
            {
                
                //pInvoke call
                int info = -1;
                if ((device == Device.DEFAULT || device == Device.GPU || device == Device.CPU) && _device == Device.GPU)
                    info = mbv2sgesvds(true, m, n, pA, ps, pU, calcU, pVT, calcVt);
                else
                    info = mbv2sgesvds_cpu(true, m, n, pA, ps, pU, calcU, pVT, calcVt);

                //
                if (info != 0)
                    throw new Exception($"magma_svd failed due to invalid parameter {-info}.");

                return (s, U, VT);
            }
        }

        /// <summary>
        /// Decompose rectangular matrix A on A = U * s * Vt
        /// </summary>
        /// <param name="A"></param>
        /// <param name="device"></param>
        /// <returns></returns>
        public static (double[] s, double[,] U, double[,] vt) Svd(double[,] A, bool calcU, bool calcVt, Device device = Device.DEFAULT)
        {
            //define parameters
            int m = A.GetLength(0);//the number of rows
            int n = A.GetLength(1);//the number of columns
            var Ac = A.Clone() as double[,];
            double[]   s = new double[n];
            double[,]  U = new double[m, m];
            double[,] VT = new double[n, n];

            //Initialize unmanaged memory to hold the array.
            fixed (double* pA = Ac, ps = s, pU = U, pVT = VT)
            {

                //pInvoke call
                int info = -1;
                if ((device == Device.DEFAULT || device == Device.GPU || device == Device.CPU) && _device == Device.GPU)
                    info = mbv2dgesvds(true, m, n, pA, ps, pU, calcU, pVT, calcVt);
                else
                    info = mbv2dgesvds_cpu(true, m, n, pA, ps, pU, calcU, pVT, calcVt);

                //
                if (info != 0)
                    throw new Exception($"magma_svd failed due to invalid parameter {-info}.");

                return (s, U, VT);
            }
        }
        #endregion
    }
}
