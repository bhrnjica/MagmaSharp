using System;
using System.Runtime.InteropServices;
namespace MagmaSharp
{

    //http://icl.cs.utk.edu/projectsfiles/magma/doxygen/group__magma__device.html
    public static class DeviceMng
    {
        #region Device Management
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern void printInfo();
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        public static extern int start();
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        public static extern int end();

        // Device Management
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int mbv2getdevice_arch();
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern void mbv2getdevices(IntPtr devices, int size, IntPtr num_dev);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern void mbv2getdevice(IntPtr device);
        [DllImport("Magmav2Binding.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern void mbv2setdevice(int device);

        public static string GetCUDAArch()
        {
            int ver = mbv2getdevice_arch();
            if(ver==0)
                return "No CUDA detected on this machine!";
            var strVer = ver.ToString();
            //
            return $"CUDA ARCH {strVer[0]}.{strVer[1]}.{strVer[2]}";
        }

        public static int[] GetDevices()
        {

            // Initialize unmanaged memory to hold the array.
            int size = 10;
            IntPtr devices = Marshal.AllocHGlobal(sizeof(Int32)* size);
            IntPtr devCount = Marshal.AllocHGlobal(sizeof(Int32));
            //
            mbv2getdevices(devices, size, devCount);
            //
            int[] dCount = new int[1];
            dCount[0] = -1;
            Marshal.Copy(devCount, dCount, 0, 1);
            if (dCount[0] < 0 || dCount[0] > 10)
                return dCount;

            int[] dvcs = new int[dCount[0]];
            Marshal.Copy(devices, dvcs, 0, dCount[0]);

            //Free memory
            Marshal.FreeHGlobal(devices);
            Marshal.FreeHGlobal(devCount);
            return dvcs;
        }
        public static int GetDevice()
        {
            //
            IntPtr devId = Marshal.AllocHGlobal(sizeof(Int32));
            int[] deviceId = new int[1];
            //
            mbv2getdevice(devId);
            //
            Marshal.Copy(devId, deviceId, 0, 1);
            //Free memory
            Marshal.FreeHGlobal(devId);

            return deviceId[0];
        }
        public static void SetDevice(int deviceId)
        {
            //
            mbv2setdevice(deviceId);
            return;
        }
        #endregion
    }
}
