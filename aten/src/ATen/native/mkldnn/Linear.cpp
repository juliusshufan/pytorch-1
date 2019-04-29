#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor mkldnn_linear(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias) {
  AT_ERROR("mkldnn_linear: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_linear_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight) {
  AT_ERROR("mkldnn_linear_backward_input: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor> mkldnn_linear_backward_weights(
    const Tensor& grad_output, const Tensor& input, bool bias_defined) {
  AT_ERROR("mkldnn_linear_backward_weights: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_linear_backward(
    const Tensor& input, const Tensor& grad_output_t,
    const Tensor& weight, std::array<bool,3> output_mask) {
  AT_ERROR("mkldnn_linear_backward: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

Tensor mkldnn_linear(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor& w = itensor_from_mkldnn(weight);

  ideep::tensor y;
  if (bias.defined()) {
    ideep::tensor& b = itensor_from_mkldnn(bias);
    ideep::inner_product_forward::compute(x, w, b, y);
  } else {
    ideep::inner_product_forward::compute(x, w, y);
  }

  return new_with_itensor_mkldnn(std::move(y), self.options());
}

Tensor mkldnn_linear_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight){
  ideep::tensor& grady = itensor_from_mkldnn(grad_output);
  ideep::tensor& w = itensor_from_mkldnn(weight);

  ideep::tensor gradx;
  ideep::inner_product_backward_data::compute(grady, w, {input_size.begin(), input_size.end()}, gradx);

  return new_with_itensor_mkldnn(std::move(gradx), grad_output.options());
}

std::tuple<Tensor, Tensor> mkldnn_linear_backward_weights(
    const Tensor& grad_output, const Tensor& input, bool bias_defined) {
  ideep::tensor& grady = itensor_from_mkldnn(grad_output);
  ideep::tensor& x = itensor_from_mkldnn(input);
  ideep::tensor gradw, gradb;
  if (bias_defined) {
    ideep::inner_product_backward_weights::compute(x, grady, gradw, gradb);
  } else {
    ideep::inner_product_backward_weights::compute(x, grady, gradw);
  }

  return std::tuple<Tensor, Tensor>{new_with_itensor_mkldnn(std::move(gradw), grad_output.options()),
    new_with_itensor_mkldnn(std::move(gradb), grad_output.options())};
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_linear_backward(
    const Tensor& input, const Tensor& grad_output,
    const Tensor& weight, std::array<bool,3> output_mask) {
  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::mkldnn_linear_backward_input(input.sizes(), grad_output, weight);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = at::mkldnn_linear_backward_weights(grad_output, input, output_mask[2]);
  }
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
