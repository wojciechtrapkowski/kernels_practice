#define __CL_ENABLE_EXCEPTIONS
#define _SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING

#include "CL/cl.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <numeric>

#define TOL (0.001) // tolerance used in floating point comparisons

#ifndef VADD_KERNEL_PATH
#error "VADD_KERNEL_PATH not defined"
#endif

#ifndef MATMUL_KERNEL_PATH
#error "MATMUL_KERNEL_PATH not defined"
#endif

#ifndef REDUCE_KERNEL_PATH
#error "REDUCE_KERNEL_PATH not defined"
#endif

#include "host/matmul.hpp"

namespace {

cl::Device GetDevice() {

  std::optional<cl::Device> device;

  try {
    std::vector<cl::Platform> platforms;

    // Discover number of platforms
    cl::Platform::get(&platforms);
    std::cout << "\nNumber of OpenCL plaforms: " << platforms.size()
              << std::endl;

    // Investigate each platform
    std::cout << "\n-------------------------" << std::endl;
    for (std::vector<cl::Platform>::iterator plat = platforms.begin();
         plat != platforms.end(); plat++) {
      std::string s;
      plat->getInfo(CL_PLATFORM_NAME, &s);
      std::cout << "Platform: " << s << std::endl;

      plat->getInfo(CL_PLATFORM_VENDOR, &s);
      std::cout << "\tVendor:  " << s << std::endl;

      plat->getInfo(CL_PLATFORM_VERSION, &s);
      std::cout << "\tVersion: " << s << std::endl;

      // Discover number of devices
      std::vector<cl::Device> devices;
      plat->getDevices(CL_DEVICE_TYPE_ALL, &devices);
      std::cout << "\n\tNumber of devices: " << devices.size() << std::endl;

      if (!device) {
        device = devices.front();
      }

      // Investigate each device
      for (std::vector<cl::Device>::iterator dev = devices.begin();
           dev != devices.end(); dev++) {
        std::cout << "\t-------------------------" << std::endl;

        dev->getInfo(CL_DEVICE_NAME, &s);
        std::cout << "\t\tName: " << s << std::endl;

        dev->getInfo(CL_DEVICE_OPENCL_C_VERSION, &s);
        std::cout << "\t\tVersion: " << s << std::endl;

        int i;
        dev->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &i);
        std::cout << "\t\tMax. Compute Units: " << i << std::endl;

        size_t size;
        dev->getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
        std::cout << "\t\tLocal Memory Size: " << size / 1024 << " KB"
                  << std::endl;

        dev->getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &size);
        std::cout << "\t\tGlobal Memory Size: " << size / (1024 * 1024) << " MB"
                  << std::endl;

        dev->getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &size);
        std::cout << "\t\tMax Alloc Size: " << size / (1024 * 1024) << " MB"
                  << std::endl;

        dev->getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);
        std::cout << "\t\tMax Work-group Total Size: " << size << std::endl;

        std::vector<size_t> d;
        dev->getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &d);
        std::cout << "\t\tMax Work-group Dims: (";
        for (std::vector<size_t>::iterator st = d.begin(); st != d.end(); st++)
          std::cout << *st << " ";
        std::cout << "\x08)" << std::endl;

        std::cout << "\t-------------------------" << std::endl;
      }

      std::cout << "\n-------------------------\n";
    }
  } catch (cl::Error err) {
    std::cout << "OpenCL Error: " << err.what() << std::endl;
    std::cout << "Check cl.h for error codes." << std::endl;
    exit(-1);
  }

  if (!device) {
    throw std::runtime_error("No OpenCL device found.");
  }

  return device.value();
}

cl::Program CreateProgram(cl::Context &context, cl::CommandQueue &queue,
                          cl::Device &device, const char *path) {
  std::ifstream kernelFile(path);
  if (!kernelFile.is_open()) {
    throw std::runtime_error("Cannot open kernel file: " + std::string(path));
  }
  std::stringstream kernelStream;
  kernelStream << kernelFile.rdbuf();
  std::string kernelSource = kernelStream.str();

  cl::Program program = cl::Program(context, kernelSource.c_str());
  program.build({device});

  return program;
}

void RunVADDKernel(cl::Context &context, cl::CommandQueue &queue,
                   cl::Device &device) {
  constexpr auto LENGTH = 1024;

  auto program = CreateProgram(context, queue, device, VADD_KERNEL_PATH);

  auto vadd =
      cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int>(
          program, "vadd");

  std::vector<float> h_a(LENGTH);
  std::vector<float> h_b(LENGTH);
  std::vector<float> h_c(LENGTH);
  std::vector<float> h_d(LENGTH, 0xdeadbeef);

  // File the h_a & h_b randomly
  for (int i = 0; i < LENGTH; i++) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
    h_c[i] = rand() / (float)RAND_MAX;
  }

  cl::Buffer d_a = cl::Buffer(context, h_a.begin(), h_a.end(), true);

  cl::Buffer d_b = cl::Buffer(context, h_b.begin(), h_b.end(), true);

  cl::Buffer d_c = cl::Buffer(context, h_c.begin(), h_c.end(), true);

  cl::Buffer d_d =
      cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

  auto start = std::chrono::high_resolution_clock::now();

  vadd(cl::EnqueueArgs(queue, cl::NDRange(512, 1, 1), cl::NDRange(64, 1, 1)),
       d_a, d_b, d_c, d_d, LENGTH);

  queue.finish();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "Kernel execution time: " << duration.count() << " ms"
            << std::endl;

  cl::copy(queue, d_d, h_d.begin(), h_d.end());

  // Test results
  unsigned int correct = 0;
  for (int i = 0; i < LENGTH; i++) {
    float tmp = h_a[i] + h_b[i] + h_c[i] - h_d[i];
    if (tmp * tmp < TOL * TOL)
      correct++;
  }

  printf("C = A+B:  %d out of %d results were correct.\n", correct, LENGTH);
}
} // namespace

void RunMatmulKernel(cl::Context &context, cl::CommandQueue &queue,
                     cl::Device &device) {
  std::cout << MATMUL_KERNEL_PATH << std::endl;
  auto program = CreateProgram(context, queue, device, MATMUL_KERNEL_PATH);

  auto matmul =
      cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int>(
          program, "matmul");

  std::vector<float> h_a{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> h_b{5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> h_c(4);

  std::vector<float> result(4);
  HostKernels::matmul(h_a.data(), h_b.data(), result.data(), 2, 2, 2, 2);

  cl::Buffer d_a = cl::Buffer(context, h_a.begin(), h_a.end(), true);

  cl::Buffer d_b = cl::Buffer(context, h_b.begin(), h_b.end(), true);

  cl::Buffer d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * 4);

  auto start = std::chrono::high_resolution_clock::now();

  matmul(cl::EnqueueArgs(queue, cl::NDRange(4, 1, 1), cl::NDRange(2, 1, 1)), d_a, d_b, d_c, 2, 2, 2,
         2);

  queue.finish();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "Kernel execution time: " << duration.count() << " ms"
            << std::endl;

  cl::copy(queue, d_c, h_c.begin(), h_c.end());

  // Test results
  unsigned int correct = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if (result[i * 2 + j] == h_c[i * 2 + j]) {
        correct++;
      }
    }
  }

  printf("Matmul:  %d out of %d results were correct.\n", correct, 4);
}

void RunReduceKernel(cl::Context &context, cl::CommandQueue &queue,
                     cl::Device &device) {
  std::cout << REDUCE_KERNEL_PATH << std::endl;
  auto program = CreateProgram(context, queue, device, REDUCE_KERNEL_PATH);

  auto reduce =
      cl::make_kernel<cl::Buffer, cl::Buffer, int>(
          program, "reduce");

  std::vector<int> h_a{1, 2, 3, 4, 5, 6, 7, 8};
  int h_b = 0;

  cl::Buffer d_a = cl::Buffer(context, h_a.begin(), h_a.end(), true);
  
  cl::Buffer d_b = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(int));

  auto start = std::chrono::high_resolution_clock::now();

  reduce(cl::EnqueueArgs(queue, cl::NDRange(256), cl::NDRange(256)), d_a, d_b, (int)h_a.size());

  queue.finish();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "Kernel execution time: " << duration.count() << " ms"
            << std::endl;

  cl::copy(queue, d_b, &h_b, &h_b + 1);

  auto expected = std::accumulate(h_a.begin(), h_a.end(), 0); 
  if (expected != h_b) {
    std::cout << "Reduce: Incorrect result. Expected " << expected
              << " but got " << h_b << std::endl;
  } else {
    std::cout << "Reduce: Correct result." << std::endl;
  }
}

int main() {
  cl::Device device = GetDevice();
  cl::Context context = cl::Context(device);
  cl::CommandQueue queue = cl::CommandQueue(context, device);

  // RunVADDKernel(context, queue, device);
  // RunMatmulKernel(context, queue, device);
  RunReduceKernel(context, queue, device);
  return 0;
}
