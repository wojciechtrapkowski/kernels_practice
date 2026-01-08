#define __CL_ENABLE_EXCEPTIONS
#define _SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING

#include "CL/cl.hpp"

#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#define TOL (0.001) // tolerance used in floating point comparisons

#ifndef VADD_KERNEL_PATH
#error "VADD_KERNEL_PATH not defined"
#endif

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

      // Investigate each device
      for (std::vector<cl::Device>::iterator dev = devices.begin();
           dev != devices.end(); dev++) {
        if (!device) {
          device = *dev;
        }

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
      cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "vadd");

  std::vector<float> h_a(LENGTH);
  std::vector<float> h_b(LENGTH);
  std::vector<float> h_c(LENGTH, 0xdeadbeef);

  // File the h_a & h_b randomly
  for (int i = 0; i < LENGTH; i++) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
  }

  cl::Buffer d_a = cl::Buffer(context, h_a.begin(), h_a.end(), true);

  cl::Buffer d_b = cl::Buffer(context, h_b.begin(), h_b.end(), true);

  cl::Buffer d_c =
      cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

  vadd(cl::EnqueueArgs(queue, cl::NDRange(LENGTH)), d_a, d_b, d_c, LENGTH);

  queue.finish();

  cl::copy(queue, d_c, h_c.begin(), h_c.end());

  // Test results
  unsigned int correct = 0;
  for (int i = 0; i < LENGTH; i++) {
    float tmp = h_a[i] + h_b[i] - h_c[i];
    if (tmp * tmp < TOL * TOL)
      correct++;
  }

  printf("C = A+B:  %d out of %d results were correct.\n", correct, LENGTH);
}
} // namespace

int main() {
  cl::Device device = GetDevice();
  cl::Context context = cl::Context(device);
  cl::CommandQueue queue = cl::CommandQueue(context, device);

  RunVADDKernel(context, queue, device);

  return 0;
}
