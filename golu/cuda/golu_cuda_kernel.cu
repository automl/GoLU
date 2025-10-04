// References - https://github.com/pytorch/pytorch
// Check https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/cuda for more CUDA Kernels of activations

#define TORCH_ASSERT_NO_OPERATORS
#define _USE_MATH_DEFINES

#include <cmath>

#include <thrust/tuple.h>

#include <ATen/native/Activation.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/Loops.cuh>

#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAMathCompat.h>


extern "C" {

// GoLU Forward CUDA Kernel Implementation
void GoLUForwardCUDAKernelImpl(at::TensorIteratorBase& iter, double alpha, double beta, double gamma) {

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "golu_forward_cuda", [&]() {
        
        using opmath_t = at::opmath_type<scalar_t>;
        
        const opmath_t alpha_ = static_cast<opmath_t>(alpha);
        const opmath_t beta_ = static_cast<opmath_t>(beta);
        const opmath_t gamma_ = static_cast<opmath_t>(gamma);
        
        at::native::gpu_kernel(
            iter, [alpha_, beta_, gamma_] GPU_LAMBDA(scalar_t x) -> scalar_t {

            x = static_cast<opmath_t>(x);

            return static_cast<opmath_t>(
                x * alpha_ * c10::cuda::compat::exp(-beta_ * c10::cuda::compat::exp(-gamma_ * x)));
        });
    });
}


// GoLU Backward CUDA Kernel Implementation
void GoLUBackwardCUDAKernelImpl(at::TensorIteratorBase& iter, double alpha, double beta, double gamma) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "golu_backward_cuda", [&]() {
        
        using opmath_t = at::opmath_type<scalar_t>;
        
        const opmath_t alpha_ = static_cast<opmath_t>(alpha);
        const opmath_t beta_ = static_cast<opmath_t>(beta);
        const opmath_t gamma_ = static_cast<opmath_t>(gamma);
        
        at::native::gpu_kernel(iter, [alpha_, beta_, gamma_] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
            
            dy = static_cast<opmath_t>(dy);
            x = static_cast<opmath_t>(x);
            
            opmath_t inner_exp = c10::cuda::compat::exp(-gamma_ * x);
            
            opmath_t grad_x = dy * alpha_ * c10::cuda::compat::exp(-beta_ * inner_exp) * (
                opmath_t(1) + beta_ * gamma_ * x * inner_exp);
            
            // NaN is the only value in IEEE Floating point which is not equal to itself
            // There's slight instability for extreme negative inputs in the backward pass
            // Otherwise the forward pass and extreme positive values in backward pass are stable
            // The problem we solve here is that 0 * +inf is invalid and leads to nans
            // Hence, we explicitly handle nan values here
            grad_x = (grad_x != grad_x) ? opmath_t(0) : grad_x;

            return static_cast<opmath_t>(grad_x);
        });
    });
}


} // extern "C"
