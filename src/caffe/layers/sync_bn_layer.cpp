#ifdef USE_MPI
#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SyncBNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  bn_momentum_ = this->layer_param_.bn_param().momentum();
  bn_eps_ = this->layer_param_.bn_param().eps();
  // Initialize parameters
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(4);
    vector<int> shape;
    shape.push_back(1);
    shape.push_back(bottom[0]->channels());
    // slope
    this->blobs_[0].reset(new Blob<Dtype>(shape));
    shared_ptr<Filler<Dtype> > slope_filler(GetFiller<Dtype>(
        this->layer_param_.bn_param().slope_filler()));
    slope_filler->Fill(this->blobs_[0].get());
    // bias
    this->blobs_[1].reset(new Blob<Dtype>(shape));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
        this->layer_param_.bn_param().bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());
    // moving average mean
    this->blobs_[2].reset(new Blob<Dtype>(shape));
    caffe_set(this->blobs_[2]->count(), Dtype(0),
        this->blobs_[2]->mutable_cpu_data());
    // moving average variance
    this->blobs_[3].reset(new Blob<Dtype>(shape));
    caffe_set(this->blobs_[3]->count(), Dtype(0),
        this->blobs_[3]->mutable_cpu_data());
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  // runing average stats does not use weight decay and learning rate
  while (this->layer_param_.param_size() < 4){
    this->layer_param_.mutable_param()->Add();
  }
  this->layer_param_.mutable_param(2)->set_lr_mult(Dtype(0));
  this->layer_param_.mutable_param(2)->set_decay_mult(Dtype(0));
  this->layer_param_.mutable_param(3)->set_lr_mult(Dtype(0));
  this->layer_param_.mutable_param(3)->set_decay_mult(Dtype(0));
}

template <typename Dtype>
void SyncBNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  top[0]->ReshapeLike(*(bottom[0]));

  mean_buffer_.Reshape(1, channels_, 1, 1);
  var_buffer_.Reshape(1, channels_, 1, 1);
}

template <typename Dtype>
void SyncBNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void SyncBNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}


#ifdef CPU_ONLY
STUB_GPU(SyncBNLayer);
#endif

INSTANTIATE_CLASS(SyncBNLayer);
REGISTER_LAYER_CLASS(SyncBN);

}  // namespace caffe

#endif