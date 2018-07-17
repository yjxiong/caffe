#ifdef USE_CUDNN
#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#if CUDNN_VERSION_MIN(5, 0, 0)

namespace caffe {

template <typename Dtype>
void CuDNNBNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BNLayer<Dtype>::LayerSetUp(bottom, top);
  save_mean_.ReshapeLike(*(this->blobs_[2]));
  save_inv_variance_.ReshapeLike(*(this->blobs_[3]));

  // Initialize CUDNN.
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  cudnn::createTensor4dDesc<Dtype>(&bn_param_desc_);
  handles_setup_ = true;
  
  LOG(INFO)<<"using cuDNN BN engine";
}

template <typename Dtype>
void CuDNNBNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Do not call BNLayer::Reshape function as some members are unnecessary
  top[0]->ReshapeLike(*(bottom[0]));

  // CUDNN tensors
  cudnn::setTensorNdDesc<Dtype>(&bottom_desc_, bottom[0]->shape());
  cudnn::setTensorNdDesc<Dtype>(&top_desc_, bottom[0]->shape());
  // Fix to the spatial mode
  CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_param_desc_,
      bottom_desc_, CUDNN_BATCHNORM_SPATIAL));

  if (this->frozen_){
	if (bottom[0]->num_axes() < 5) {
	    this->num_ = bottom[0]->shape(0);
	    this->channels_ = bottom[0]->shape(1);
	    this->height_ = bottom[0]->shape(2);
	    this->width_ = bottom[0]->shape(3);
	    this->spatial_statistic_.Reshape(this->num_, this->channels_, 1, 1);
	    this->batch_statistic_.Reshape(1, this->channels_, 1, 1);
	    this->spatial_sum_multiplier_.Reshape(1, 1, this->height_, this->width_);
	    this->batch_sum_multiplier_.Reshape(this->num_, 1, 1, 1);
	}
	if (bottom[0]->num_axes() >=5) {
		this->num_ = bottom[0]->shape(0);
	    	this->channels_ = bottom[0]->shape(1);
	    	this->height_ = bottom[0]->shape(2) * bottom[0]->shape(3);
	    	this->width_ = bottom[0]->shape(4);
		vector<int> blob_shape_(5,0);
		blob_shape_[0] = bottom[0]->shape(0);
		blob_shape_[1] = bottom[0]->shape(1);
		blob_shape_[2] = 1;
		blob_shape_[3] = 1;
		blob_shape_[4] = 1;
	    this->spatial_statistic_.Reshape(blob_shape_);
		blob_shape_[0] = 1;
		blob_shape_[1] = bottom[0]->shape(1);
		blob_shape_[2] = 1;
		blob_shape_[3] = 1;
		blob_shape_[4] = 1;
	    this->batch_statistic_.Reshape(blob_shape_);
		blob_shape_[0] = 1;
		blob_shape_[1] = 1;
		blob_shape_[2] = bottom[0]->shape(2);
		blob_shape_[3] = bottom[0]->shape(3);
		blob_shape_[4] = bottom[0]->shape(4);
	    this->spatial_sum_multiplier_.Reshape(blob_shape_);
		blob_shape_[0] = bottom[0]->shape(0);
		blob_shape_[1] = 1;
		blob_shape_[2] = 1;
		blob_shape_[3] = 1;
		blob_shape_[4] = 1;
	    this->batch_sum_multiplier_.Reshape(blob_shape_);
	
	}
    this->broadcast_buffer_.ReshapeLike(*(bottom[0]));

    caffe_set(this->spatial_sum_multiplier_.count(), Dtype(1),
      this->spatial_sum_multiplier_.mutable_cpu_data());
    caffe_set(this->batch_sum_multiplier_.count(), Dtype(1),
      this->batch_sum_multiplier_.mutable_cpu_data());

  }
}

template <typename Dtype>
CuDNNBNLayer<Dtype>::~CuDNNBNLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyTensorDescriptor(bn_param_desc_);
  cudnnDestroy(handle_);
}

INSTANTIATE_CLASS(CuDNNBNLayer);

}  // namespace caffe
#endif
#endif
