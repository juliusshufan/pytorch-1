#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(const Tensor& input_,
    const Tensor& weight_, const Tensor& bias_, const Tensor& running_mean_,
    const Tensor& running_var_, bool training, double exponential_average_factor, double epsilon){
  AT_ERROR("mkldnn_batch_norm: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(
    const Tensor& input_, const Tensor& grad_output, const Tensor& weight_,
    const Tensor& running_mean_, const Tensor& running_var_, const Tensor& save_mean,
    const Tensor& save_var, double epsilon) {
  AT_ERROR("mkldnn_batch_norm_backward: ATen not compiled with MKLDNN support");
}

}} // namespace at::native

#else // AT_MKLDNN_EBABLED

#include <ATen/mkldnn/Runtime.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>

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
  ideep::tensor::dims input_tz;

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

std::tuple<Tensor, Tensor, Tensor> _ideep_batch_norm(const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& running_mean,
    const Tensor& running_var, bool training, double exponential_average_factor, double epsilon) {

  bool use_running_stat = (running_mean.defined() && running_var.defined());
  const ideep::tensor& src_ = itensor_from_mkldnn(input);
  const ideep::tensor& weight_ = itensor_from_mkldnn(weight);
  const ideep::tensor& bias_ = itensor_from_mkldnn(bias);
  ideep::tensor dst_, save_mean_, save_var_;

  if (training) {
   if (use_running_stat) {
     ideep::tensor running_mean_ = itensor_from_mkldnn(running_mean);
     ideep::tensor running_var_ = itensor_from_mkldnn(running_var);
     ideep::batch_normalization_forward_training::compute(
       src_, weight_, bias_, dst_, save_mean_, save_var_,
       running_mean_, running_var_, exponential_average_factor, epsilon);
   } else {
     ideep::batch_normalization_forward_training::compute(src_, weight_, bias_, dst_,
       save_mean_, save_var_, exponential_average_factor, epsilon);
   }
  } else {
    if (use_running_stat) {
      ideep::tensor running_mean_ = itensor_from_mkldnn(running_mean);
      ideep::tensor running_var_ = itensor_from_mkldnn(running_var);
      ideep::batch_normalization_forward_inference::compute(
        src_, running_mean_, running_var_, weight_, bias_, dst_, epsilon);
    } else {
      ideep::batch_normalization_forward_inference::compute(src_, weight_, bias_, dst_, epsilon);
    }
  }

  auto output = new_with_itensor_mkldnn(std::move(dst_), input.options());
  auto mean_output = new_with_itensor_mkldnn(std::move(save_mean_), input.options());
  auto var_output = new_with_itensor_mkldnn(std::move(save_var_), input.options());
  return std::tuple<Tensor, Tensor, Tensor>{output, mean_output, var_output};
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(const Tensor& input_,
    const Tensor& weight_, const Tensor& bias_, const Tensor& running_mean_,
    const Tensor& running_var_, bool training, double exponential_average_factor, double epsilon) {

  if (input_.type_id() == MkldnnCPUTensorId()) {
    return _ideep_batch_norm(input_, weight_, bias_, running_mean_, running_var_, training, exponential_average_factor, epsilon);
  }

  auto input = input_.contiguous();
  auto weight = weight_.contiguous();
  auto bias =  bias_.contiguous();
  auto running_mean = running_mean_.defined() ? running_mean_.contiguous() : running_mean_;
  auto running_var = running_var_.defined() ? running_var_.contiguous() : running_var_;

  auto output = at::empty_like(input);

  int32_t ic = input.size(1);
  auto save_mean = at::empty({ic}, weight.options());
  auto save_var = at::empty({ic}, weight.options());

  BatchNormArgs args(input, running_mean, running_var, training, epsilon);

  auto type_ = ideep::tensor::data_type::f32;
  ideep::tensor::descriptor src_desc(args.input_tz, type_);
  ideep::tensor::descriptor statistic_desc_({ic}, type_);

  ideep::tensor src_, dst, scale_, shift_, mean, var, run_mean_, run_var_;
  src_.init(src_desc, input.data_ptr());
  dst.init(src_desc, output.data_ptr());

  scale_.init(statistic_desc_, weight.data_ptr());
  shift_.init(statistic_desc_, bias.data_ptr());

  mean.init(statistic_desc_, save_mean.data_ptr());
  var.init(statistic_desc_, save_var.data_ptr());

  ideep::tensor dst_, save_mean_, save_var_;

  if (training) {
    if (args.params.use_running_stat) {
      run_mean_.init(statistic_desc_, running_mean.data_ptr());
      run_var_.init(statistic_desc_, running_var.data_ptr());
      ideep::batch_normalization_forward_training::compute(
        src_, scale_, shift_, dst_, save_mean_, save_var_,
        run_mean_, run_var_, exponential_average_factor, epsilon);
    } else {
      ideep::batch_normalization_forward_training::compute(src_, scale_, shift_, dst_,
        save_mean_, save_var_, exponential_average_factor, epsilon);
    }
  } else {
    if (args.params.use_running_stat) {
      run_mean_.init(statistic_desc_, running_mean.data_ptr());
      run_var_.init(statistic_desc_, running_var.data_ptr());
      ideep::batch_normalization_forward_inference::compute(
        src_, run_mean_, run_var_, scale_, shift_, dst_, epsilon);
    } else {
      ideep::batch_normalization_forward_inference::compute(src_, scale_, shift_, dst_, epsilon);
    }
  }

  ideep::reorder::compute(dst_, dst);
  ideep::reorder::compute(save_mean_, mean);
  ideep::reorder::compute(save_var_, var);

  return std::tuple<Tensor, Tensor, Tensor>{output, save_mean, save_var};
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(const Tensor& input_,
    const Tensor& grad_output, const Tensor& weight_, const Tensor& running_mean_,
    const Tensor& running_var_, const Tensor& save_mean, const Tensor& save_var, double epsilon) {

  auto input = input_.contiguous();
  auto weight = weight_.contiguous();

  auto grad_input = at::empty_like(input);
  auto grad_weight = at::empty_like(weight);
  auto grad_bias = at::empty_like(weight);

  int32_t ic = input.size(1);

  auto type_ = ideep::tensor::data_type::f32;
  BatchNormArgs args(input, running_mean_, running_var_, true, epsilon);

  ideep::tensor::descriptor src_desc(args.input_tz, type_);
  ideep::tensor::descriptor statistic_desc_({ic}, type_);

  ideep::tensor src_, grady_, gradx, gradw, gradb, save_mean_, save_var_, scale_;

  src_.init(src_desc, input.data_ptr());
  grady_.init(src_desc, grad_output.data_ptr());
  gradx.init(src_desc, grad_input.data_ptr());

  gradw.init(statistic_desc_, grad_weight.data_ptr());
  gradb.init(statistic_desc_, grad_bias.data_ptr());

  save_mean_.init(statistic_desc_, save_mean.data_ptr());
  save_var_.init(statistic_desc_, save_var.data_ptr());
  scale_.init(statistic_desc_, weight.data_ptr());

  ideep::tensor gradx_, gradw_, gradb_;
  ideep::batch_normalization_backward::compute( src_, save_mean_, save_var_, grady_, scale_, gradx_, gradw_, gradb_, epsilon);

  ideep::reorder::compute(gradx_, gradx);
  ideep::reorder::compute(gradw_, gradw);
  ideep::reorder::compute(gradb_, gradb);

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

}}  // namespace at::native
#endif
