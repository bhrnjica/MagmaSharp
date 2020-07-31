
MAGMA runtime libraries can be compiled and build from the source code located at https://bitbucket.org/icl/magma. 
However compiled windows static runtime libraries can be downloaded from several places on internet. One of the place is PyTorch repository.
For more info please visit:
 https://github.com/pytorch/pytorch/tree/5f4a01b2ea1dde566ff9ddd79d68fc3db2c2820c/.jenkins/pytorch/win-test-helpers/installation-helpers. 

The file structure for the runtime libraries must be in the form of:


MagmaSharp/MagmaLib/
	/Debug
		/Include
		/Lib
	/Release
		/Include
		/Lib
