#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <boost/unordered_map.hpp>
#include <cudnn.h>

using boost::unordered_map;

namespace caffe {

// Set to three for the benefit of the backward pass, which
// can use separate streams for calculating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently
#define CUDNN_FWD_STREAMS_PER_GROUP 1
#define CUDNN_BWD_STREAMS_PER_GROUP 2

template <typename Dtype>
shared_ptr<SyncedMemory> CuDNNConvolutionLayer<Dtype>::workspaceData_fwd;
template <typename Dtype>
shared_ptr<SyncedMemory> CuDNNConvolutionLayer<Dtype>::workspaceData_bwd_filter;
template <typename Dtype>
shared_ptr<SyncedMemory> CuDNNConvolutionLayer<Dtype>::workspaceData_bwd_data;

template <typename Dtype>
size_t CuDNNConvolutionLayer<Dtype>::conv_layer_count = 0;


/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

  if (conv_layer_count == 0){
    workspaceData_fwd = shared_ptr<SyncedMemory>(new SyncedMemory());
    workspaceData_bwd_filter = shared_ptr<SyncedMemory>(new SyncedMemory());
    workspaceData_bwd_data = shared_ptr<SyncedMemory>(new SyncedMemory());
  }
  conv_layer_count++;

  // Initialize CUDA streams and cuDNN.
  int total_streams_per_group = CUDNN_FWD_STREAMS_PER_GROUP + CUDNN_BWD_STREAMS_PER_GROUP;
  stream_         = new cudaStream_t[this->group_ * total_streams_per_group];
  handle_         = new cudnnHandle_t[this->group_ * total_streams_per_group];


  // initialize size arrays
  workspace_fwd_offsets_ = new size_t[bottom.size()];
  workspace_bwd_filter_offsets_ = new size_t[bottom.size()];
  workspace_bwd_data_offsets_ = new size_t[bottom.size()];


  for (size_t i = 0; i < bottom.size(); ++i) {
    // initialize all to default algorithms
    fwd_algo_.push_back((cudnnConvolutionFwdAlgo_t)0);
    bwd_filter_algo_.push_back((cudnnConvolutionBwdFilterAlgo_t)0);
    bwd_data_algo_.push_back((cudnnConvolutionBwdDataAlgo_t)0);
    // default algorithms don't require workspace
    workspace_fwd_offsets_[i] = 0;
    workspace_bwd_filter_offsets_[i] = 0;
    workspace_bwd_data_offsets_[i] = 0;
  }

  for (int g = 0; g < this->group_ * total_streams_per_group; g++) {
    CUDA_CHECK(cudaStreamCreate(&stream_[g]));
    CUDNN_CHECK(cudnnCreate(&handle_[g]));
    CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
  }

  // Set the indexing parameters.
  weight_offset_ = (this->num_output_ / this->group_)
      * (this->channels_ / this->group_) * this->kernel_h_ * this->kernel_w_;
  bias_offset_ = (this->num_output_ / this->group_);

  // Create filter descriptor.
  cudnn::createFilterDesc<Dtype>(&filter_desc_,
      this->num_output_ / this->group_, this->channels_ / this->group_,
      this->kernel_h_, this->kernel_w_);

  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    cudnnTensorDescriptor_t bottom_desc;
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);
    cudnnTensorDescriptor_t top_desc;
    cudnn::createTensor4dDesc<Dtype>(&top_desc);
    top_descs_.push_back(top_desc);
    cudnnConvolutionDescriptor_t conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&conv_desc);
    conv_descs_.push_back(conv_desc);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
  }

  handles_setup_ = true;
  need_benchmark_ = true;

}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::Reshape(bottom, top);
  bottom_offset_ = (this->channels_ / this->group_)
      * this->height_ * this->width_;
  top_offset_ = (this->num_output_ / this->group_)
      * this->height_out_ * this->width_out_;

  // Specify workspace limit for kernels directly until we have a
  // planning strategy and a rewrite of Caffe's GPU memory mangagement.
  //
  // However this can be tuned by the "richness" parameter in the solver protobuf
  // By setting richness, you can increase the memory available to cuDNN and thus
  // let it choose fast but space consuming algorithms.
  if (need_benchmark_){
    for (int i = 0; i < bottom.size(); i++) {

      cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
                                    this->num_,
                                    this->channels_ / this->group_,
                                    this->height_, this->width_,
                                    this->channels_ * this->height_ * this->width_,
                                    this->height_ * this->width_,
                                    this->width_, 1);
      cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
                                    this->num_,
                                    this->num_output_ / this->group_,
                                    this->height_out_, this->width_out_,
                                    this->num_output_ * this->height_out_ * this->width_out_,
                                    this->height_out_ * this->width_out_,
                                    this->width_out_, 1);
      cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i],
                                       filter_desc_, this->pad_h_, this->pad_w_,
  #if CUDNN_VERSION_MIN(6, 0, 0)
                                       this->stride_h_, this->stride_w_, this->dilation_h_, this->dilation_w_);
  #else
                                       this->stride_h_, this->stride_w_);
  #endif

      // choose forward and backward algorithms + workspace(s)
      const int kRequestedForwardAlgoCount = 6;
      vector<cudnnConvolutionFwdAlgoPerf_t> fwd_perf;
      fwd_perf.resize(kRequestedForwardAlgoCount);
      int returnedAlgoCount;
      CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(handle_[0],
                                                       bottom_descs_[i],
                                                       filter_desc_,
                                                       conv_descs_[i],
                                                       top_descs_[i],
                                                       kRequestedForwardAlgoCount,
                                                       &returnedAlgoCount,
                                                       &fwd_perf[0]));

      // choose the fastest within limit
      // if all algorithms exceed memory limit, we will use the 0 algorithm with no workspace
      for (int a = 0; a < kRequestedForwardAlgoCount; ++a){
        if (fwd_perf[a].memory * this->group_ < (Caffe::cudnn_mem_richness() * 1024 * 1024)
            || Caffe::cudnn_mem_richness() == 0){
          fwd_algo_[i] = fwd_perf[a].algo;
          break;
        }
      }

      // choose backward algorithm for filter
      const int kRequestedBackwardFilterAlgoCount = 4;
      vector<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filter_perf;
      bwd_filter_perf.resize(kRequestedBackwardFilterAlgoCount);
      CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithm(handle_[0],
                                                              bottom_descs_[i],
                                                              top_descs_[i],
                                                              conv_descs_[i],
                                                              filter_desc_,
                                                              kRequestedBackwardFilterAlgoCount,
                                                              &returnedAlgoCount,
                                                              &bwd_filter_perf[0]));

      // choose the fastest within limit
      // if all algorithms exceed memory limit, we will use the 0 algorithm with no workspace
      for (int a = 0; a < kRequestedBackwardFilterAlgoCount; ++a){
        if (bwd_filter_perf[a].memory * this->group_ < (Caffe::cudnn_mem_richness() * 1024 * 1024)
            || Caffe::cudnn_mem_richness() == 0){
          bwd_filter_algo_[i] = bwd_filter_perf[a].algo;
          break;
        }
      }

      // choose backward algo for data
      const int kRequestedBackwardDataAlgoCount = 4;
      vector<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_perf;
      bwd_data_perf.resize(kRequestedBackwardDataAlgoCount);
      CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(handle_[0],
                                                            filter_desc_,
                                                            top_descs_[i],
                                                            conv_descs_[i],
                                                            bottom_descs_[i],
                                                            kRequestedBackwardDataAlgoCount,
                                                            &returnedAlgoCount,
                                                            &bwd_data_perf[0]));

      // choose the fastest within limit
      // if all algorithms exceed memory limit, we will use the 0 algorithm with no workspace
      for (int a = 0; a < kRequestedBackwardDataAlgoCount; ++a){
        if (bwd_data_perf[a].memory * this->group_ <(Caffe::cudnn_mem_richness() * 1024 * 1024)
            || Caffe::cudnn_mem_richness() == 0){
          bwd_data_algo_[i] = bwd_data_perf[a].algo;
          break;
        }
      }

      need_benchmark_ = false;
    }
  }


  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_ / this->group_, 1, 1);
  }

  AdjustWorkSpaces();
}

template<typename Dtype>
void CuDNNConvolutionLayer<Dtype>::AdjustWorkSpaces() {

  size_t workspace_size_fwd = 0;
  size_t workspace_size_bwd_data = 0;
  size_t workspace_size_bwd_filter = 0;

  for (int i = 0; i < fwd_algo_.size(); ++i){
    size_t workspace_size;
    cudnnGetConvolutionForwardWorkspaceSize(handle_[0],
                                            bottom_descs_[i], filter_desc_,
                                            conv_descs_[i],
                                            top_descs_[i],
                                            fwd_algo_[i], &workspace_size);
    workspace_fwd_offsets_[i] = workspace_size;
    workspace_size_fwd = std::max(workspace_size * this->group_, workspace_size_fwd);

    cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_[1],
                                                   bottom_descs_[i], top_descs_[i],
                                                   conv_descs_[i],
                                                   filter_desc_,
                                                   bwd_filter_algo_[i], &workspace_size);
    workspace_bwd_filter_offsets_[i] = workspace_size;
    workspace_size_bwd_filter = std::max(workspace_size * this->group_, workspace_size_bwd_filter);

    cudnnGetConvolutionBackwardDataWorkspaceSize(handle_[2],
                                                 filter_desc_,
                                                 top_descs_[i],
                                                 conv_descs_[i],
                                                 bottom_descs_[i],
                                                 bwd_data_algo_[i], &workspace_size);
    workspace_bwd_data_offsets_[i] = workspace_size;
    workspace_size_bwd_data = std::max(workspace_size * this->group_, workspace_size_bwd_data);
  }

  workspaceData_fwd->Resize(workspace_size_fwd);
  workspaceData_bwd_filter->Resize(workspace_size_bwd_filter);
  workspaceData_bwd_data->Resize(workspace_size_bwd_data);
}

template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  for (int i = 0; i < bottom_descs_.size(); i++) {
    cudnnDestroyTensorDescriptor(bottom_descs_[i]);
    cudnnDestroyTensorDescriptor(top_descs_[i]);
    cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
  }
  if (this->bias_term_) {
    cudnnDestroyTensorDescriptor(bias_desc_);
  }
  cudnnDestroyFilterDescriptor(filter_desc_);

  int total_stream_per_group = CUDNN_FWD_STREAMS_PER_GROUP + CUDNN_BWD_STREAMS_PER_GROUP;
  for (int g = 0; g < this->group_ * total_stream_per_group; g++) {
    cudaStreamDestroy(stream_[g]);
    cudnnDestroy(handle_[g]);
  }

  --conv_layer_count;
  if (conv_layer_count == 0){
    workspaceData_fwd.reset();
    workspaceData_bwd_filter.reset();
    workspaceData_bwd_data.reset();
  }

  delete [] stream_;
  delete [] handle_;
  delete [] workspace_fwd_offsets_;
  delete [] workspace_bwd_data_offsets_;
  delete [] workspace_bwd_filter_offsets_;

}

INSTANTIATE_CLASS(CuDNNConvolutionLayer);

}   // namespace caffe
#endif
