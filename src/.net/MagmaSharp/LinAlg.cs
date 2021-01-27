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
        private static extern int mbv2sgesv_cpu(bool rowmajor, int n, int nrhs, float* A, int lda, float* B, int lbd);

        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2sgesv(bool rowmajor, int n, int nrhs, float* A, int lda, float* B,int lbd);

        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2sgesv_gpu(bool rowmajor, int n, int nrhs, float* A, int ldda, float* B, int lddb);

        //double
        [DllImport("LapackBinding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgesv_cpu(bool rowmajor, int n, int nrhs, double* A, int ldda, double* B, int lddb);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgesv(bool rowmajor, int n, int nrhs, double* A, int ldda, double* B, int lddb);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgesv_gpu(bool rowmajor, int n, int nrhs, double* A, int ldda, double* B, int lddb);

        public static float[,] Solve(float[,] A, float[,] B, Device device = Device.DEFAULT)
        {
            //define parameters
            int info = -1;
            int n = A.GetLength(0);
            int nrhs = B.GetLength(1);
            var Ac = A.Clone() as float[,];
            var Bc = B.Clone() as float[,];

            //define arrays
            fixed(float *pA = Ac, pB = Bc)
            {
                //pInvoke call
                if ((device == Device.DEFAULT || device == Device.GPU) && _device == Device.GPU)
                    info = mbv2sgesv_gpu(true, n, nrhs, pA, n, pB, n);
                else if (device == Device.CPU && _device == Device.GPU)
                    info = mbv2sgesv(true, n, nrhs, pA, n, pB, n);
                else
                    info = mbv2sgesv_cpu(true, n, nrhs, pA, n, pB, nrhs);
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
            fixed (double* pA = Ac, pB = Bc)
            {
                //pInvoke call
                if ((device == Device.DEFAULT || device == Device.GPU) && _device == Device.GPU)
                    info = mbv2dgesv_gpu(true, n, nrhs, pA, n, pB, n);
                else if (device == Device.CPU && _device == Device.GPU)
                    info = mbv2dgesv(true, n, nrhs, pA, n, pB, n);
                else
                    info = mbv2dgesv_cpu(true, n, nrhs, pA, n, pB, nrhs);
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
        private static extern int mbv2sgels_cpu(bool rowmajor, int m, int n, int nrhs, float* A, int lda, float* B, int lbd);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2sgels(bool rowmajor, int m, int n, int nrhs, float* A, int lda, float* B, int lbd);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2sgels_gpu(bool rowmajor, int m, int n, int nrhs, float* A, int lda, float* B, int lbd);

        [DllImport("LapackBinding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgels_cpu(bool rowmajor, int m, int n, int nrhs, double* A, int lda, double* B, int lbd);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgels(bool rowmajor, int m, int n, int nrhs, double* A, int lda, double* B, int lbd);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgels_gpu(bool rowmajor, int m, int n, int nrhs, double* A, int lda, double* B, int lbd);

        public static float[,] Lss(float[,] A, float[,] B, Device device = Device.DEFAULT)
        {
            //define parameters
            int info = -1;
            int m = A.GetLength(0);
            int n = A.GetLength(1);
            int nrhs = B.GetLength(1);

            //define arrays
            var Ac = A.Clone() as float[,];
            var Bc = B.Clone() as float[,];

            fixed (float* pA = Ac, pB = Bc)
            {
                //pInvoke call
                if ((device == Device.DEFAULT || device == Device.GPU) && _device == Device.GPU)
                    info = mbv2sgels_gpu(true, m, n, nrhs, pA, m, pB, m);
                else if (device == Device.CPU && _device == Device.GPU)
                    info = mbv2sgels(true, m, n, nrhs, pA, m, pB, m);
                else
                    info = mbv2sgels_cpu(true, m, n, nrhs, pA, n, pB, nrhs);

                //
                if (info != 0)
                    throw new Exception($"magma_sgesv failed due to invalid parameter {-info}.");

                //X(n, nrhs) matrix is a submatrix of B(m, nrhs).
                var X = new float[n,nrhs];
                Array.Copy(Bc,X,n*nrhs);
                return X;
            }            
        }

        public static double[,] Lss(double[,] A, double[,] B, Device device = Device.DEFAULT)
        {
            //define parameters
            int info = -1;
            int m = A.GetLength(0);
            int n = A.GetLength(1);
            int nrhs = B.GetLength(1);

            //define arrays
            var Ac = A.Clone() as double[,];
            var Bc = B.Clone() as double[,];

            fixed (double* pA = Ac, pB = Bc)
            {
                //pInvoke call

                if ((device == Device.DEFAULT || device == Device.GPU) && _device == Device.GPU)
                    info = mbv2dgels_gpu(true, m, n, nrhs, pA, m, pB, m);
                else if (device == Device.CPU && _device == Device.GPU)
                    info = mbv2dgels(true, m, n, nrhs, pA, m, pB, m);
                else
                    info = mbv2dgels_cpu(true, m, n, nrhs, pA, n, pB, nrhs);

                //
                if (info != 0)
                    throw new Exception($"magma_dgesv failed due to invalid parameter {-info}.");

                //X(n, nrhs) matrix is a submatrix of B(m, nrhs).
                var X = new double[n, nrhs];
                Array.Copy(Bc, X, n * nrhs);
                return X;
            }

        }
        #endregion

        #region Eigenvalues 
        [DllImport("LapackBinding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2sgeevs_cpu(bool rowmajor, int n, float* A, int lda, float* wr, float* wi, float* VL, bool computeLeft, float* VR, bool computerRight);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2sgeevs(bool rowmajor, int n, float* A, int lda, float* wr, float* wi, float* VL, bool computeLeft, float* VR, bool computerRight);

        [DllImport("LapackBinding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgeevs_cpu(bool rowmajor, int n, double* A, int lda, double* wr, double* wi, double* VL, bool computeLeft, double* VR, bool computerRight);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2dgeevs(bool rowmajor, int n, double* A, int lda, double* wr, double* wi, double* VL, bool computeLeft, double* VR, bool computerRight);
       
        public static (float[] wr, float[] wi, float[,] VL, float[,] VR ) Eigen(float[,] A, bool computeLeft= false, bool computeRight= false, Device device = Device.DEFAULT)
        {
            //define parameters
            int info = -1;
            int m = A.GetLength(0);
            if (m != A.GetLength(1))
                throw new Exception("Matrix A must be squared!");

            //define arrays
            var Ac = A.Clone() as float[,];
            var wr = new float[m];
            var wi = new float[m];
            var VL = new float[m, m];
            var VR = new float[m, m];

            fixed (float* pA = Ac, pwr = wr, pwi = wi, pVL = VL, pVR = VR)
            {
                if ((device == Device.DEFAULT || device == Device.GPU) && _device == Device.GPU)
                    info = mbv2sgeevs(true, m, pA, m, pwr, pwi, pVL, computeLeft, pVR, computeRight);
                else if (device == Device.CPU && _device == Device.GPU)
                    info = mbv2sgeevs(true, m, pA, m, pwr, pwi, pVL, computeLeft, pVR, computeRight);
                else
                    info = mbv2sgeevs_cpu(true, m, pA, m, pwr, pwi, pVL, computeLeft, pVR, computeRight);
            }
            //
            return (wr, wi, VL, VR);
        }

        public static (double[] wr, double[] wi, double[,] VL, double[,] VR) Eigen(double[,] A, bool computeLeft, bool computeRight, Device device = Device.DEFAULT)
        {
            //define parameters
            int info = -1;
            int m = A.GetLength(0);
            if (m != A.GetLength(1))
                throw new Exception("Matrix A must be squared!");

            //define arrays
            var Ac = A.Clone() as double[,];
            var wr = new double[m];
            var wi = new double[m];
            var VL = new double[m, m];
            var VR = new double[m, m];

            fixed (double* pA = Ac, pwr = wr, pwi = wi, pVL = VL, pVR = VR)
            {
                if ((device == Device.DEFAULT || device == Device.GPU) && _device == Device.GPU)
                    info = mbv2dgeevs(true, m, pA, m, pwr, pwi, pVL, computeLeft, pVR, computeRight);
                else if (device == Device.CPU && _device == Device.GPU)
                    info = mbv2dgeevs(true, m, pA, m, pwr, pwi, pVL, computeLeft, pVR, computeRight);
                else
                    info = mbv2dgeevs_cpu(true, m, pA, m, pwr, pwi, pVL, computeLeft, pVR, computeRight);
            }
            //
            return (wr, wi, VL, VR);
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
