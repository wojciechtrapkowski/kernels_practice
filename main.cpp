#define __CL_ENABLE_EXCEPTIONS

#include "tutorial/cl.hpp"

#include <optional>
#include <iostream>
#include <vector>

#define TOL (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024) // length of vectors a, b, and c

int main() {
  std::vector<cl::Platform> platforms;

  std::optional<cl::Device> device;

  try {
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

  const char *KernelSource =
      "\n"
      "__kernel void vadd(                                                 \n"
      "   __global float* a,                                                  "
      "\n"
      "   __global float* b,                                                  "
      "\n"
      "   __global float* c,                                                  "
      "\n"
      "   const unsigned int count)                                           "
      "\n"
      "{                                                                      "
      "\n"
      "   int i = get_global_id(0);                                           "
      "\n"
      "   if(i < count)                                                       "
      "\n"
      "       c[i] = a[i] + b[i];                                             "
      "\n"
      "}                                                                      "
      "\n"
      "\n";

  cl::Context context = cl::Context(device.value());
  cl::CommandQueue commands = cl::CommandQueue(context, device.value());

  // Create program
  cl::Program program = cl::Program(context, KernelSource);
  program.build({device.value()});

  cl::Kernel ko_vadd = cl::Kernel(program, "vadd");

  // Create the input (a, b) and output (c) arrays in device memory
  cl::Buffer d_a =
      cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * LENGTH);

  cl::Buffer d_b =
      cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * LENGTH);

  cl::Buffer d_c =
      cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

  std::vector<float> h_a(LENGTH);
  std::vector<float> h_b(LENGTH);

  // File the h_a & h_b randomly
  for (int i = 0; i < LENGTH; i++) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
  }

  std::vector<float> h_c(LENGTH);

  commands.enqueueWriteBuffer(d_a, CL_TRUE, 0, sizeof(float) * LENGTH, h_a.data());
  commands.enqueueWriteBuffer(d_b, CL_TRUE, 0, sizeof(float) * LENGTH, h_b.data());

  ko_vadd.setArg(0, d_a);
  ko_vadd.setArg(1, d_b);
  ko_vadd.setArg(2, d_c);
  ko_vadd.setArg(3, LENGTH);

  // Execute kernel
  cl::NDRange global(LENGTH);
  commands.enqueueNDRangeKernel(ko_vadd, cl::NullRange, global);
  commands.finish();

  // Read results back
  commands.enqueueReadBuffer(d_c, CL_TRUE, 0, sizeof(float) * LENGTH, h_c.data());

  // Test results
  unsigned int correct = 0;
  for (int i = 0; i < LENGTH; i++) {
    float tmp = h_a[i] + h_b[i] - h_c[i];
    if (tmp * tmp < TOL * TOL)
      correct++;
  }

  printf("C = A+B:  %d out of %d results were correct.\n", correct, LENGTH);

  return 0;
}
