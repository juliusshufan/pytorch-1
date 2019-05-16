#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/cpu/vec256/vec256.h>

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
  AT_ASSERTM(result.sizes() == self.sizes(),
             "mkldnn_mul_out: currently mkldnn not support broadcasting");
  ideep::tensor& z = itensor_from_mkldnn(result);
  ideep::tensor& x = itensor_from_mkldnn(self);

  // For zero_dim densor
  if (other.ndimension() == 0) {
    ideep::eltwise_forward::compute<AllocForMKLDNN>(
      x, z, ideep::algorithm::eltwise_linear, ideep::prop_kind::forward_training, /*alpha*/ other.item().to<float>());

  return result;
  } else {
    AT_ASSERTM(self.sizes() == other.sizes(),
               "mkldnn_mul_out: currently mkldnn not support broadcasting");
    ideep::tensor y = itensor_from_mkldnn(other);
    auto* z_ = static_cast<float *>(z.get_data_handle());
    auto* x_ = static_cast<float *>(x.get_data_handle());
    auto* y_ = static_cast<float *>(y.get_data_handle());
    if (y.get_descriptor() != x.get_descriptor()){
       ideep::tensor y1(x.get_descriptor());
       y1.feed_from(y);
       y_ = static_cast<float *>(y1.get_data_handle());
    }

    auto n = self.numel();
    using Vec = vec256::Vec256<float>;

    parallel_for(0, n, 2048, [z_, x_, y_](int64_t begin, int64_t end){
      vec256::map2(
        [](Vec a, Vec b) {return a * b;},
        z_ + begin,
        x_ + begin,
        y_ + begin,
        end - begin);
    });

    return result;
  }
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
