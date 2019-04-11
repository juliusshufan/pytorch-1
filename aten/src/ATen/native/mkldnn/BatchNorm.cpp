#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& running_mean,
    const Tensor& running_var, bool training, double exponential_average_factor, double epsilon){
  AT_ERROR("mkldnn_batch_norm: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(
    const Tensor& input, const Tensor& grad_output, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean,
    const Tensor& save_var, double epsilon) {
  AT_ERROR("mkldnn_batch_norm_backward: ATen not compiled with MKLDNN support");
}

}} // namespace at::native

#else // AT_MKLDNN_EBABLED

#include <ATen/mkldnn/Types.h>
#include <ATen/mkldnn/TensorUtils.h>

namespace at { namespace native {

namespace {

constexpr int max_dim = 3;

struct BatchNormParams {
  int64_t dim;
  int64_t input_size[2 + max_dim];
  double epsilon;
  bool training;
  bool use_running_stat;
};

void setBatchNormParams(BatchNormParams* params, const Tensor& input,
    double epsilon, bool training, bool use_running_stat) {

  memset(params, 0, sizeof(BatchNormParams));

  params->dim = input.dim();
  for (int64_t i = 0; i < params->dim; ++i) {
    params->input_size[i] = input.size(i);
  }

  params->epsilon = epsilon;
  params->training = training;
  params->use_running_stat = use_running_stat;
}

struct BatchNormArgs {
  BatchNormParams params;
  tensor::dims input_tz;

  BatchNormArgs(const Tensor& input, const Tensor& running_mean,
      const Tensor& running_var, bool training, double epsilon) {

    bool use_running_stat = (running_mean.defined() && running_var.defined());

    setBatchNormParams(&params, input, epsilon, training, use_running_stat);

    for (int64_t i = 0; i < params.dim; ++i) {
      input_tz.push_back(params.input_size[i]);
    }
  }
};

}  // namespace

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& running_mean,
    const Tensor& running_var, bool training, double exponential_average_factor, double epsilon) {

  auto output = at::empty_like(input);

  int32_t ic = input.size(1);
  auto save_mean = at::empty({ic}, weight.options());
  auto save_var = at::empty({ic}, weight.options());

  BatchNormArgs args(input, running_mean, running_var, training, epsilon);

  auto input_ = GET_MKLDNN_TENSOR(input, args.input_tz);
  auto weight_ = GET_MKLDNN_TENSOR(weight, {ic});
  auto bias_ = GET_MKLDNN_TENSOR(bias, {ic});

  itensor output_, save_mean_, save_var_;

  if (training) {
    if (args.params.use_running_stat) {
      auto running_mean_ = GET_MKLDNN_TENSOR(running_mean, {ic});
      auto running_var_ = GET_MKLDNN_TENSOR(running_var, {ic});
      batch_normalization_forward_training::compute(
        input_, weight_, bias_, output_, save_mean_, save_var_,
        running_mean_, running_var_, exponential_average_factor, epsilon);
    } else {
      batch_normalization_forward_training::compute(input_, weight_, bias_, output_,
        save_mean_, save_var_, exponential_average_factor, epsilon);
    }
  } else {
    if (args.params.use_running_stat) {
      auto running_mean_ = GET_MKLDNN_TENSOR(running_mean, {ic});
      auto running_var_ = GET_MKLDNN_TENSOR(running_var, {ic});
      batch_normalization_forward_inference::compute(
        input_, running_mean_, running_var_, weight_, bias_, output_, epsilon);
    } else {
      batch_normalization_forward_inference::compute(input_, weight_, bias_, output_, epsilon);
    }
  }

  if (input.is_mkldnn()) {
    output = new_with_itensor_mkldnn(std::move(output_), input.options());
    save_mean = new_with_itensor_mkldnn(std::move(save_mean_), input.options());
    save_var = new_with_itensor_mkldnn(std::move(save_var_), input.options());
  } else {
    output_.reorder_to(output.data_ptr());
    save_mean_.reorder_to(save_mean.data_ptr());
    save_var_.reorder_to(save_var.data_ptr());
  }

  return std::tuple<Tensor, Tensor, Tensor>{output, save_mean, save_var};
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(const Tensor& input,
    const Tensor& grad_output, const Tensor& weight, const Tensor& running_mean,
    const Tensor& running_var, const Tensor& save_mean, const Tensor& save_var, double epsilon) {

  auto grad_input = at::empty_like(input);
  auto grad_weight = at::empty_like(weight);
  auto grad_bias = at::empty_like(weight);

  int32_t ic = input.size(1);
  
  BatchNormArgs args(input, running_mean, running_var, true, epsilon);

  auto type_ = get_mkldnn_dtype(grad_output);
  desc src_desc(args.input_tz, type_);
  desc statistic_desc_({ic}, type_);
  itensor src_, grady_, gradx, gradw, gradb, save_mean_, save_var_, scale_;

  src_.init(src_desc, input.data_ptr());
  grady_.init(src_desc, grad_output.data_ptr());
  gradx.init(src_desc, grad_input.data_ptr());

  gradw.init(statistic_desc_, grad_weight.data_ptr());
  gradb.init(statistic_desc_, grad_bias.data_ptr());

  save_mean_.init(statistic_desc_, save_mean.data_ptr());
  save_var_.init(statistic_desc_, save_var.data_ptr());
  scale_.init(statistic_desc_, weight.data_ptr());

  itensor gradx_, gradw_, gradb_;

  batch_normalization_backward::compute(src_, save_mean_, save_var_, grady_, scale_, gradx_, gradw_, gradb_, epsilon);

  reorder::compute(gradx_, gradx);
  reorder::compute(gradw_, gradw);
  reorder::compute(gradb_, gradb);

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

}}  // namespace at::native
#endif
