#ifdef USE_MPI
#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/mpi_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_test_forward(
    const int num, const int channels, const int spatial_dim,
    const Dtype* scale, const Dtype* bias, const Dtype* mean, const Dtype* var,
    const Dtype eps, const Dtype* bottom_data, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    int c = (index / spatial_dim) % channels;
    top_data[index] = (bottom_data[index] - mean[c]) / sqrt(var[c] + eps)
        * scale[c] + bias[c];
  }
}

template <typename Dtype>
__global__ void kernel_local_stats(int num, int channels, int spatial_dim,
    const Dtype norm_factor,
    const Dtype* bottom_data, Dtype* mean, Dtype* var) {
  // store local E[x] to mean, E[x^2] to var temporarily
  __shared__ Dtype buffer1[CAFFE_CUDA_NUM_THREADS];
  __shared__ Dtype buffer2[CAFFE_CUDA_NUM_THREADS];
  const int tid = threadIdx.x;
  const int c = blockIdx.x;

  // load and accumulate data on each thread
  buffer1[tid] = buffer2[tid] = 0;
  for (int i = tid; i < num * spatial_dim; i += blockDim.x) {
    const int index = i / spatial_dim * channels * spatial_dim
        + c * spatial_dim + i % spatial_dim;
    buffer1[tid] += bottom_data[index];
    buffer2[tid] += bottom_data[index] * bottom_data[index];
  }
  __syncthreads();

  // do tree reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      buffer1[tid] += buffer1[tid + s];
      buffer2[tid] += buffer2[tid + s];
    }
    __syncthreads();
  }

  // save the result back
  if (tid == 0) {
    mean[c] = buffer1[0] / norm_factor;
    var[c] = buffer2[0] / norm_factor;
  }
}

template <typename Dtype>
__global__ void kernel_backward_scale_bias(
    const int num, const int channels, const int spatial_dim,
    const Dtype* mean, const Dtype* var, const Dtype eps,
    const Dtype* top_diff, const Dtype* bottom_data,
    Dtype* scale_diff, Dtype* bias_diff) {
  __shared__ Dtype buffer1[CAFFE_CUDA_NUM_THREADS];
  __shared__ Dtype buffer2[CAFFE_CUDA_NUM_THREADS];
  const int tid = threadIdx.x;
  const int c = blockIdx.x;

  // load and accumulate data on each thread
  buffer1[tid] = buffer2[tid] = 0;
  for (int i = tid; i < num * spatial_dim; i += blockDim.x) {
    const int index = i / spatial_dim * channels * spatial_dim
        + c * spatial_dim + i % spatial_dim;
    buffer1[tid] += top_diff[index] * (bottom_data[index] - mean[c])
                                    / sqrt(var[c] + eps);
    buffer2[tid] += top_diff[index];
  }
  __syncthreads();

  // do tree reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      buffer1[tid] += buffer1[tid + s];
      buffer2[tid] += buffer2[tid + s];
    }
    __syncthreads();
  }

  // save the result back
  if (tid == 0) {
    scale_diff[c] = buffer1[0];
    bias_diff[c] = buffer2[0];
  }
}

template <typename Dtype>
__global__ void kernel_backward_bottom(
    const int num, const int channels, const int spatial_dim,
    const Dtype* scale, const Dtype* bias,
    const Dtype* mean, const Dtype* var, const Dtype eps,
    const Dtype norm_factor,
    const Dtype* top_diff, const Dtype* scale_diff, const Dtype* bias_diff,
    const Dtype* bottom_data, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    int c = (index / spatial_dim) % channels;
    const Dtype inv_std = Dtype(1) / sqrt(var[c] + eps);
    const Dtype x_norm = (bottom_data[index] - mean[c]) * inv_std;
    bottom_diff[index] = scale[c] * inv_std *
        (top_diff[index] - (x_norm * scale_diff[c] + bias_diff[c]) / norm_factor);
  }
}


template <typename Dtype>
void SyncBNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (this->phase_ == TEST) {
    kernel_test_forward<<<CAFFE_GET_BLOCKS(bottom[0]->count()),
        CAFFE_CUDA_NUM_THREADS>>>(
      num_, channels_, height_ * width_,
      this->blobs_[0]->gpu_data(),
      this->blobs_[1]->gpu_data(),
      this->blobs_[2]->gpu_data(),
      this->blobs_[3]->gpu_data(),
      bn_eps_,
      bottom[0]->gpu_data(),
      top[0]->mutable_gpu_data()
    );
    CUDA_POST_KERNEL_CHECK;
  } else {
    const int m = num_ * height_ * width_ * Caffe::MPI_all_rank();
    // compute local E[x] and E[x^2]
    kernel_local_stats<<<channels_, CAFFE_CUDA_NUM_THREADS>>>(
      num_, channels_, height_ * width_,
      static_cast<Dtype>(m),
      bottom[0]->gpu_data(),
      mean_buffer_.mutable_gpu_data(),
      var_buffer_.mutable_gpu_data()
    );
    CUDA_POST_KERNEL_CHECK;
    // sync E[x] and E[x^2]
    mpi_force_synchronize();
    caffe_iallreduce(mean_buffer_.mutable_cpu_data(), channels_);
    caffe_iallreduce(var_buffer_.mutable_cpu_data(), channels_);
    mpi_force_synchronize();
    // var = (E[x^2] - E[x]^2) * bias_correction_factor
    caffe_gpu_mul(channels_, mean_buffer_.gpu_data(), mean_buffer_.gpu_data(),
                  top[0]->mutable_gpu_data());  // reuse the top buffer
    caffe_gpu_sub(channels_, var_buffer_.gpu_data(), top[0]->gpu_data(),
                  var_buffer_.mutable_gpu_data());
    if (m > 1) {
      caffe_gpu_scal(channels_, Dtype(m) / (m-1),
                     var_buffer_.mutable_gpu_data());
    }
    // update running mean and var
    caffe_gpu_axpby(mean_buffer_.count(),
        Dtype(1) - bn_momentum_, mean_buffer_.gpu_data(),
        bn_momentum_, this->blobs_[2]->mutable_gpu_data());
    caffe_gpu_axpby(var_buffer_.count(),
        Dtype(1) - bn_momentum_, var_buffer_.gpu_data(),
        bn_momentum_, this->blobs_[3]->mutable_gpu_data());
    // compute output
    kernel_test_forward<<<CAFFE_GET_BLOCKS(bottom[0]->count()),
        CAFFE_CUDA_NUM_THREADS>>>(
      num_, channels_, height_ * width_,
      this->blobs_[0]->gpu_data(),
      this->blobs_[1]->gpu_data(),
      mean_buffer_.gpu_data(),
      var_buffer_.gpu_data(),
      bn_eps_,
      bottom[0]->gpu_data(),
      top[0]->mutable_gpu_data()
    );
    CUDA_POST_KERNEL_CHECK;
  }
}

template <typename Dtype>
void SyncBNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    CHECK(this->param_propagate_down_[0] && this->param_propagate_down_[1])
        << "SyncBN layer params should backprop when the layer backprops";
    // compute local scale and bias diff
    kernel_backward_scale_bias<<<channels_, CAFFE_CUDA_NUM_THREADS>>>(
      num_, channels_, height_ * width_,
      mean_buffer_.gpu_data(),
      var_buffer_.gpu_data(),
      bn_eps_,
      top[0]->gpu_diff(),
      bottom[0]->gpu_data(),
      mean_buffer_.mutable_gpu_diff(),  // temp use for local scale diff
      var_buffer_.mutable_gpu_diff() // temp use for local bias diff
    );
    CUDA_POST_KERNEL_CHECK;
    // sync scale and bias diff
    mpi_force_synchronize();
    caffe_iallreduce(mean_buffer_.mutable_cpu_diff(), channels_);
    caffe_iallreduce(var_buffer_.mutable_cpu_diff(), channels_);
    mpi_force_synchronize();
    // add to param blobs diff
    caffe_gpu_axpy(channels_, Dtype(1) / Caffe::MPI_all_rank(),
                   mean_buffer_.gpu_diff(),
                   this->blobs_[0]->mutable_gpu_diff());
    caffe_gpu_axpy(channels_, Dtype(1) / Caffe::MPI_all_rank(),
                   var_buffer_.gpu_diff(),
                   this->blobs_[1]->mutable_gpu_diff());
    // compute bottom diff
    kernel_backward_bottom<<<CAFFE_GET_BLOCKS(bottom[0]->count()),
        CAFFE_CUDA_NUM_THREADS>>>(
      num_, channels_, height_ * width_,
      this->blobs_[0]->gpu_data(),
      this->blobs_[1]->gpu_data(),
      mean_buffer_.gpu_data(),
      var_buffer_.gpu_data(),
      bn_eps_,
      static_cast<Dtype>(num_ * height_ * width_ * Caffe::MPI_all_rank()),
      top[0]->gpu_diff(),
      mean_buffer_.gpu_diff(),
      var_buffer_.gpu_diff(),
      bottom[0]->gpu_data(),
      bottom[0]->mutable_gpu_diff()
    );
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SyncBNLayer);

}  // namespace caffe
#endif