#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor& mkldnn_zero_(Tensor& self) {
  AT_ERROR("mkldnn_zero_: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

Tensor& mkldnn_zero_(Tensor& self) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  memset(x.get_data_handle(), 0.0, x.get_size());
  return self;
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
