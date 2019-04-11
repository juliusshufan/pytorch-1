#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/Parallel.h>
#include <ATen/Dispatch.h>

namespace at { namespace native {


template<typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> fused_sgd_step_cpu_template(
    Tensor& param, Tensor& gradient, Tensor& momentum_buf, double learning_rate,
    double momentum, double dampening, double weight_decay, bool nesterov, bool momentum_buf_defined) {

  auto n = param.numel();

  auto lr = learning_rate;
  auto mom = momentum;
  auto dam = dampening;
  auto wd = weight_decay;

  auto p_ = param.data<scalar_t>();
  auto dp_ = gradient.data<scalar_t>();
  auto buf_ = momentum_buf.data<scalar_t>();

  parallel_for(0, n, 1, [=](int64_t begin, int64_t end){
    for (int64_t index = begin; index < end; index++) {
      if (wd != static_cast<scalar_t>(0)) {
        dp_[index] += wd * p_[index];
      }
      if (mom != static_cast<scalar_t>(0)) {
        if (momentum_buf_defined) {
          buf_[index] = buf_[index] * mom + (1 - dam) * dp_[index];
        } else {
          buf_[index] = dp_[index];
        }
        if (nesterov) {
          dp_[index] += mom * buf_[index];
        } else {
          dp_[index] = buf_[index];
        }
      }
      p_[index] -= lr * dp_[index];
    }
  });

  return std::tuple<Tensor, Tensor, Tensor>{param, gradient, momentum_buf};
}

std::tuple<Tensor, Tensor, Tensor> fused_sgd_step(
    Tensor& param, Tensor& gradient, Tensor& momentum_buf, double learning_rate,
    double momentum, double dampening, double weight_decay, bool nesterov, bool momentum_buf_defined) {
   
    return AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "fused_sgd_step", [&] {
        return fused_sgd_step_cpu_template<scalar_t>(param, gradient, momentum_buf, learning_rate,
            momentum, dampening, weight_decay, nesterov, momentum_buf_defined);
        });
}

}} // namespace at::native
