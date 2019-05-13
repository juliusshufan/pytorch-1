#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor& mkldnn_add_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  AT_ERROR("mkldnn_add_out: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_add(const Tensor& self, const Tensor& other, Scalar alpha) {
  AT_ERROR("mkldnn_add: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_add_(Tensor& self, const Tensor& other, Scalar alpha) {
  AT_ERROR("mkldnn_add_: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_mul_out(Tensor& result, const Tensor& self, const Tensor& other) {
  AT_ERROR("mkldnn_mul_out: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_mul(const Tensor& self, const Tensor& other) {
  AT_ERROR("mkldnn_mul: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_mul_(Tensor& self, const Tensor& other) {
   AT_ERROR("mkldnn_mul_: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

Tensor& mkldnn_add_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor& y = itensor_from_mkldnn(other);

  ideep::tensor& z = itensor_from_mkldnn(result);
  const std::vector<float> scales{1.0, alpha.to<float>()};
  ideep::sum::compute<AllocForMKLDNN>(scales, {x, y}, z);

  return result;
}

Tensor mkldnn_add(const Tensor& self, const Tensor& other, Scalar alpha) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor& y = itensor_from_mkldnn(other);

  ideep::tensor z;
  const std::vector<float> scales{1.0, alpha.to<float>()};
  ideep::sum::compute<AllocForMKLDNN>(scales, {x, y}, z);

  return new_with_itensor_mkldnn(std::move(z), self.options());
}

Tensor& mkldnn_add_(Tensor& self, const Tensor& other, Scalar alpha) {
  return native::mkldnn_add_out(self, self, other, alpha);
}

Tensor& mkldnn_mul_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto n = self.numel();

  ideep::tensor& z = itensor_from_mkldnn(result);
  ideep::tensor& x = itensor_from_mkldnn(self);

  auto* z_ = static_cast<float *>(z.get_data_handle());
  auto* x_ = static_cast<float *>(x.get_data_handle());

  float* y_;
  bool is_scalar = true;
  if (other.ndimension() != 0) { 
    is_scalar = false;
    ideep::tensor& y = itensor_from_mkldnn(other);
    y_ = static_cast<float *>(y.get_data_handle());
  }

  parallel_for(0, n, 1, [=](int64_t begin, int64_t end){
    for (int64_t index = begin; index < end; index++) {
      z_[index] = x_[index] * (is_scalar ?  other.item().to<float>(): y_[index]);
    }
  });

  return result;
}

Tensor mkldnn_mul(const Tensor& self, const Tensor& other) {
  Tensor result = new_with_sizes_mkldnn(self.sizes(), self.options());
  return native::mkldnn_mul_out(result, self, other);
}

Tensor& mkldnn_mul_(Tensor& self, const Tensor& other) {
  return native::mkldnn_mul_out(self, self, other);
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
