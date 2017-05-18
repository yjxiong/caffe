#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ColorConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  input_space_  = this->layer_param().color_conv_param().input();
  output_space_ = this->layer_param().color_conv_param().output();
}
template <typename Dtype>
void ColorConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  std::vector<int> bottom_shape = bottom[0]->shape();
  CHECK_EQ(bottom_shape.size(), 4) << "Images expected";
  CHECK_EQ(bottom_shape[1], input_space_ == ColorConvParameter::Gray ? 1 : 3) <<
    "Wrong number of channels";
  bottom_shape[1] = output_space_ == ColorConvParameter::Gray ? 1 : 3;
  top[0]->Reshape(bottom_shape);
}
typedef ColorConvParameter P;
namespace convert_color_cpu {
template <typename T, int, int> struct Converter {
};
template <typename T> struct Converter<T, P::RGB, P::RGB> {
  static void convert(const T * in, T * out) {
    out[0] = in[0];
    out[1] = in[1];
    out[2] = in[2];
  }
};
template <typename T>
T invS(T v) {
  return v <= T(0.04045) ? v/T(12.92) : pow((v + T(0.055))/T(1.055), T(2.4));
}
template <typename T>
T S(T v) {
  return v <= T(0.0031308) ? v*T(12.92) : T(1.055)*pow(v, T(1./2.4)) - T(0.055);
}
template <typename T> struct Converter<T, P::RGB, P::XYZ> {
  static void convert(const T * in, T * out) {
    T tmp[3] = {0};
    for (int i = 0; i < 3; i++)
      tmp[i] = invS(in[i]/T(255.));
    out[0] = T(0.4124564)*tmp[0] + T(0.3575761)*tmp[1] + T(0.1804375)*tmp[2];
    out[1] = T(0.2126729)*tmp[0] + T(0.7151522)*tmp[1] + T(0.0721750)*tmp[2];
    out[2] = T(0.0193339)*tmp[0] + T(0.1191920)*tmp[1] + T(0.9503041)*tmp[2];
  }
};
template <typename T> struct Converter<T, P::XYZ, P::RGB> {
  static void convert(const T * in, T * out) {
    T tmp[3] = {0};
    tmp[0] =  T(3.2404542)*in[0] - T(1.5371385)*in[1] - T(0.4985314)*in[2];
    tmp[1] = -T(0.9692660)*in[0] + T(1.8760108)*in[1] + T(0.0415560)*in[2];
    tmp[2] =  T(0.0556434)*in[0] - T(0.2040259)*in[1] + T(1.0572252)*in[2];
    for (int i = 0; i < 3; i++)
      out[i] = S(tmp[i])*T(255.);
  }
};
template <typename T> struct Converter<T, P::RGB, P::Gray> {
  static void convert(const T * in, T * out) {
    out[0] = T(0.2126729)*in[0] + T(0.7151522)*in[1] + T(0.0721750)*in[2];
  }
};
template <typename T> struct Converter<T, P::Gray, P::RGB> {
  static void convert(const T * in, T * out) {
    out[0] = out[1] = out[2] = in[0];
  }
};
template <typename T>
T invF(T v) {
  return v > T(0.206896) ? pow(v, T(3.)) : (T(116) * v - T(16.)) / T(903.3);
}
template <typename T>
T F(T v) {
  return v > T(0.008867) ? pow(v, T(1./3)) : (T(903.3) * v + T(16.)) / T(116.);
}
template <typename T> struct Converter<T, P::Lab, P::XYZ> {
  static void convert(const T * in, T * out) {
    const T R[3] = {0.95074, 1.0, 1.08883};  // Reference white D65
    T tmp[3] = {0};
    tmp[1] = (in[0]+16) / T(116.);
    tmp[0] = tmp[1] + in[1] / T(500.);
    tmp[2] = tmp[1] - in[2] / T(200.);
    for (int i = 0; i < 3; i++)
      out[i] = invF(tmp[i]) * R[i];
  }
};
template <typename T> struct Converter<T, P::XYZ, P::Lab> {
  static void convert(const T * in, T * out) {
    const T R[3] = {0.95074, 1.0, 1.08883};  // Reference white D65
    T tmp[3] = {0};
    for (int i = 0; i < 3; i++)
      tmp[i] = F(in[i] / R[i]);
    out[0] = T(116.)*tmp[1] - T(16.);
    out[1] = T(500.)*(tmp[0] - tmp[1]);
    out[2] = T(200.)*(tmp[1] - tmp[2]);
  }
};
template <typename T> struct Converter<T, P::RGB, P::Lab> {
  static void convert(const T * in, T * out) {
    T tmp[3] = {0};
    Converter<T, P::RGB, P::XYZ>::convert(in, tmp);
    Converter<T, P::XYZ, P::Lab>::convert(tmp, out);
  }
};
template <typename T> struct Converter<T, P::Lab, P::RGB> {
  static void convert(const T * in, T * out) {
    T tmp[3] = {0};
    Converter<T, P::Lab, P::XYZ>::convert(in, tmp);
    Converter<T, P::XYZ, P::RGB>::convert(tmp, out);
  }
};
template <typename T, int from, int to>
void convert(int N, const T * input, T * output,
    bool swap_in = 0, bool swap_out = 0) {
  const T *i1 = input, *i2 = input + N, *i3 = input + 2*N;
  if (swap_in) std::swap(i1, i3);
  T *o1 = output, *o2 = output + N, *o3 = output + 2*N;
  if (swap_out) std::swap(o1, o3);
  if (from == P::Gray) i2 = i3 = NULL;
  if (to   == P::Gray) o2 = o3 = NULL;
  typedef Converter<T, from, to> C;
  T in[3], out[3];
  for (int i = 0; i < N; i++) {
    in[0] = i1[i];
    if (i2) in[1] = i2[i];
    if (i3) in[2] = i3[i];

    C::convert(in, out);

    o1[i] = out[0];
    if (o2) o2[i] = out[1];
    if (o3) o3[i] = out[2];
  }
}
};  // namespace convert_color_cpu
template <typename T>
void ColorConvLayer<T>::Forward_cpu(const vector<Blob<T>*>& bottom,
    const vector<Blob<T>*>& top) {
  using convert_color_cpu::convert;
  const int N = bottom[0]->shape()[0], Cb = bottom[0]->shape()[1];
  const int Ct = top[0]->shape()[1], count = bottom[0]->count() / (N*Cb);
  ColorConvParameter::ColorSpace in = input_space_, out = output_space_;
  bool swap_in = (in == P::BGR), swap_out = (out == P::BGR);
  if (swap_in) in = P::RGB;
  if (swap_out) out = P::RGB;
  for (int n = 0; n < N; n++) {
    const T * input = bottom[0]->cpu_data() + n * count * Cb;
    T * output = top[0]->mutable_cpu_data() + n * count * Ct;
#define CONVERT(A, B) if (in == A && out == B)\
    return convert<T, A, B>(count, input, output, swap_in, swap_out);
    CONVERT(P::Gray, P::RGB);
    CONVERT(P::Lab, P::RGB);
    CONVERT(P::Lab, P::XYZ);
    CONVERT(P::RGB, P::Gray);
    CONVERT(P::RGB, P::Lab);
    CONVERT(P::RGB, P::RGB);
    CONVERT(P::RGB, P::XYZ);
    CONVERT(P::XYZ, P::Lab);
    CONVERT(P::XYZ, P::RGB);
  }
}

template <typename Dtype>
void ColorConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(ColorConvLayer);
#endif

INSTANTIATE_CLASS(ColorConvLayer);
REGISTER_LAYER_CLASS(ColorConv);

}  // namespace caffe
