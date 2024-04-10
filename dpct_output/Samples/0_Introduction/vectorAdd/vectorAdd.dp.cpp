/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <dpct/codepin/codepin.hpp>
#include "../../../generated_schema.hpp"
#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")

#include <helper_cuda.h>
#include <cmath>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
void vectorAdd(const float *A, const float *B, float *C,
                          int numElements, const sycl::nd_item<3> &item_ct1) {
  int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
          item_ct1.get_local_id(2);

  if (i < numElements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

/**
 * Host main routine
 */
int main(void) try {
  // Error code to check return values for CUDA calls
  dpct::err0 err = 0;

  // Print the vector length to be used, and compute its size
  int numElements = 50000;
  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);

  // Allocate the host input vector A
  float *h_A = (float *)malloc(size);

  // Allocate the host input vector B
  float *h_B = (float *)malloc(size);

  // Allocate the host output vector C
  float *h_C = (float *)malloc(size);

  // Verify that allocations succeeded
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // Allocate the device input vector A
  float *d_A = NULL;
  err = DPCT_CHECK_ERROR(
      d_A = (float *)sycl::malloc_device(size, dpct::get_in_order_queue()));
dpct::experimental::get_ptr_size_map()[*((void **)&d_A)] = size;

  // Allocate the device input vector B
  float *d_B = NULL;
  err = DPCT_CHECK_ERROR(
      d_B = (float *)sycl::malloc_device(size, dpct::get_in_order_queue()));
dpct::experimental::get_ptr_size_map()[*((void **)&d_B)] = size;

  // Allocate the device output vector C
  float *d_C = NULL;
  err = DPCT_CHECK_ERROR(
      d_C = (float *)sycl::malloc_device(size, dpct::get_in_order_queue()));
dpct::experimental::get_ptr_size_map()[*((void **)&d_C)] = size;

  // Copy the host input vectors A and B in host memory to the device input
  // vectors in
  // device memory
  printf("Copy input data from the host memory to the CUDA device\n");
  err = DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d_A, h_A, size).wait());

  err = DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d_B, h_B, size).wait());

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  /*
  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::experimental::gen_prolog_API_CP(
      "/home/tcs2/Avijit/cuda-samples/Samples/0_Introduction/vectorAdd/"
      "vectorAdd.cu:147:3",
      &dpct::get_in_order_queue(), "d_A", d_A, "d_B", d_B, "d_C", d_C,
      "numElements", numElements);
  dpct::get_in_order_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, blocksPerGrid) *
                            sycl::range<3>(1, 1, threadsPerBlock),
                        sycl::range<3>(1, 1, threadsPerBlock)),
      [=](sycl::nd_item<3> item_ct1) {
        vectorAdd(d_A, d_B, d_C, numElements, item_ct1);
      });
  /*
  DPCT1010:20: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::experimental::gen_epilog_API_CP(
      "/home/tcs2/Avijit/cuda-samples/Samples/0_Introduction/vectorAdd/"
      "vectorAdd.cu:147:3",
      &dpct::get_in_order_queue(), "d_A", d_A, "d_B", d_B, "d_C", d_C,
      "numElements", numElements);
err = 0;

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
  err = DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(h_C, d_C, size).wait());

  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  // Free device global memory
  err = DPCT_CHECK_ERROR(dpct::dpct_free(d_A, dpct::get_in_order_queue()));

  err = DPCT_CHECK_ERROR(dpct::dpct_free(d_B, dpct::get_in_order_queue()));

  err = DPCT_CHECK_ERROR(dpct::dpct_free(d_C, dpct::get_in_order_queue()));

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  printf("Done\n");
  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
