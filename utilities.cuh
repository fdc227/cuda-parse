#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include "math.h"

cudaError_t arrayMalloc(void*** array, int length, size_t* size);
cudaError_t arraycpyHtoD(void*** array_d, void*** array_h, int length, size_t* size);
cudaError_t arraycpyHtoD_v2(void*** array_d, void** array_h, int length, size_t* size);
cudaError_t arraycpyDtoH(void*** array_h, void*** array_d, int length, size_t* size);
cudaError_t arraycpyDtoH_v2(void** array_h, void*** array_d, int length, size_t* size);
cudaError_t oneMalloc(void** a_d, size_t size);
cudaError_t onecpyHtoD(void* dev_a, void* a, size_t size);
cudaError_t onecpyDtoH(void* a, void* dev_a, size_t size);
cudaError_t oneSetdevice();
cudaError_t oneLastError();
cudaError_t oneCudaDeviceSync();
void write_array_to_file(double* A, std::string filename, int M, int N);
void write_array_to_file_simple(double* A, std::string filename, int size);
void write_array_to_file_simple_v(std::vector<double>& A, std::string filename, int size);

