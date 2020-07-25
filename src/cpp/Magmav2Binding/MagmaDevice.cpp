
#include "pch.h"
#include "MagmaDevice.h"
#include <stdio.h>
#include <cuda.h>
#include "magma_v2.h"
#include "magma_lapack.h"

namespace MagmaBinding
{
	void printInfo()
	{
		magma_print_environment();
	}

	int start()
	{
		return magma_init();
	}
	int end()
	{
		return magma_finalize();
	}

	int mbv2getdevice_arch()
	{
		magma_init();

			magma_int_t ff = magma_getdevice_arch();

		magma_finalize();

		return ff;
	}
	void mbv2getdevices(int* devices, int size, int* num_dev)
	{
		magma_init();
		magma_device_t dev[10];
		magma_int_t dsize = 10;
		magma_int_t numDev = 0;
		magma_getdevices(dev, dsize, &numDev);
		magma_finalize();
	}
	void mbv2getdevice(int* deviceId)
	{
		magma_init();
		magma_getdevice(deviceId);
		magma_finalize();
	}
	void mbv2setdevice(int deviceId)
	{
		magma_init();
		magma_setdevice(deviceId);
		magma_finalize();
	}
}

