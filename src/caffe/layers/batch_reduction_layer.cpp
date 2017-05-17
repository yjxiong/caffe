#include <algorithm>
#include <cfloat>
#include <vector>
#include <caffe/caffe.hpp>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void BatchReductionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  op_ = this->layer_param_.batch_reduction_param().reduction_param().operation();
  axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.batch_reduction_param().reduction_param().axis());

  // load levels
  int n_level = this->layer_param_.batch_reduction_param().level_size();

  pos_ = this->layer_param_.batch_reduction_param().pos();

  if (n_level == 0) {
    this->layer_param_.mutable_batch_reduction_param()->add_level(1);
    n_level = 1;
  }
  levels_.reserve(this->layer_param_.batch_reduction_param().level_size());

  if (n_level > 1){
    CHECK(!pos_)<<"Cannot do pos-sensitive reduction when level is more than 1";
  }

  for (int i = 0; i < n_level; ++i){
    levels_.push_back(this->layer_param_.batch_reduction_param().level(i));
    ticks_.push_back(levels_.back() * levels_.back());
    max_tick_ = std::max(ticks_.back(), max_tick_);
  }

  // top-k reduction currently only works with single level
  if (op_ == ReductionParameter_ReductionOp_TOPK){
    CHECK(n_level <= 1)<<"For now top-k reduction only works with 1 level";
  }

}

template <typename Dtype>
void BatchReductionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  vector<int> top_shape(bottom[0]->shape().begin(),
                        bottom[0]->shape().begin() + axis_);

  // if levels = [1], we do global reduction instead
  if ((levels_.size() != 1) || (levels_[0] != 1)){
    top_shape.push_back(levels_.size());
    int red_dim = 0;
    for (int i = 0; i < ticks_.size(); ++i) red_dim += ticks_[i];
    CHECK_EQ(red_dim, bottom[0]->shape(axis_));
  }else{
    ticks_[0] = bottom[0]->shape(axis_); // levels=[1] means we reduce along the whole axis
  }

  if (pos_){
    CHECK_GE(bottom[0]->shape().size() - 2, axis_)<<"In pos mode, there are two axis reduced";
    for (int i = axis_ + 2; i < bottom[0]->shape().size(); ++i) {
      top_shape.push_back(bottom[0]->shape()[i]);
    }
    step_ = bottom[0]->count(axis_+2);
    num_ = bottom[0]->count(0, axis_);
  }else {
    for (int i = axis_ + 1; i < bottom[0]->shape().size(); ++i) {
      top_shape.push_back(bottom[0]->shape()[i]);
    }
    step_ = bottom[0]->count(axis_+1);
    num_ = bottom[0]->count(0, axis_);
  }
  top[0]->Reshape(top_shape);

  //LOG_INFO<<num_<<" "<<step_;
  CHECK_EQ(step_ * num_ * levels_.size(), top[0]->count());

  // will add these later
  if (op_ == ReductionParameter_ReductionOp_SUMSQ || op_ == ReductionParameter_ReductionOp_ASUM){
    NOT_IMPLEMENTED;
  }

  ticks_blob_.Reshape(ticks_.size(), 1, 1, 1);
  Dtype* tick_data = ticks_blob_.mutable_cpu_data();
  for (int i = 0; i < levels_.size(); ++i){
    tick_data[i] = ticks_[i];
    max_tick_ = std::max(ticks_[i], max_tick_);
  }

  // reshape idx blob in top-k case
  if (op_ == ReductionParameter_ReductionOp_TOPK){
    argsort_idx_.Reshape(bottom[0]->shape());
  }else{
    argsort_idx_.Reshape(1,1,1,1);
  }
}

template <typename Dtype>
bool comparator( const std::pair<Dtype, int>& left, const std::pair<Dtype, int>& right){
  return left.first >= right.first; // use descending order here
}

template <typename Dtype>
void BatchReductionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* idx_data = argsort_idx_.mutable_cpu_data();


  caffe_set(top[0]->count(), Dtype(0), top_data);

  if (op_ != ReductionParameter_ReductionOp_TOPK) {
    for (int n = 0; n < num_; ++n) {
      //printf(" levels: %d\n", levels_.size());
      for (int l = 0; l < levels_.size(); ++l) {
        int tick = ticks_[l];
        Dtype coeff = (op_ == ReductionParameter_ReductionOp_MEAN) ? Dtype(1) / Dtype(tick) : Dtype(1);
        for (int t = 0; t < tick; ++t) {
          caffe_cpu_axpby(step_, coeff, bottom_data, Dtype(1), top_data);
          int stride = (t == (tick -1))?1:tick+1;
          bottom_data += (pos_)?step_*stride:step_;
        }
        top_data += step_;
      }
    }
  }else {
    int k = this->layer_param_.batch_reduction_param().reduction_param().k();
    int tick = ticks_[0];

    vector<std::pair<Dtype, int> > buffer;
    buffer.resize(tick);


    // num_ outer loops
    caffe_set(top[0]->count(), Dtype(0), top_data);
    caffe_set(bottom[0]->count(), Dtype(0), idx_data);
    for (int n = 0; n < num_; ++n) {
      // step_ inner loops
      for (int i = 0; i < step_; ++i) {
        //fill data
        for (int t = 0; t < tick; ++t){
          buffer[t] = std::make_pair(bottom_data[t * step_ + i], t);
        }
        // perform sort
        std::sort(buffer.begin(), buffer.end(), comparator<Dtype>);

        // obtain output and index
        Dtype accum = 0;
        for (int k_out = 0; k_out < k; ++k_out){
          std::pair<Dtype, int>& p = buffer[k_out];
          accum += p.first;
          idx_data[p.second*step_ + i] = k_out+1;
        }
        // set top data
        top_data[i] = accum / Dtype(k);
      }
      top_data += step_;
      bottom_data += tick*step_;
      idx_data += tick*step_;

    }


  }


}

template <typename Dtype>
void BatchReductionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  // Get bottom_data, if needed.
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* idx_data = argsort_idx_.mutable_cpu_data();

  if (pos_){
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  }

  if (op_ != ReductionParameter_ReductionOp_TOPK) {
    for (int i = 0; i < num_; ++i) {
      for (int l = 0; l < levels_.size(); ++l) {
        int tick = ticks_[l];
        Dtype coeff = (op_ == ReductionParameter_ReductionOp_MEAN) ? Dtype(1) / Dtype(tick) : Dtype(1);
        for (int t = 0; t < tick; ++t) {
          caffe_cpu_axpby(step_, coeff, top_diff, Dtype(0), bottom_diff);
          //offset bottom_data each input step
          int stride = (t == (tick -1))?1:tick+1;
          bottom_diff += (pos_)?step_*stride:step_;
        }
        //offset bottom_data each output step
        top_diff += step_;
      }
    }
  }else {
    int tick = ticks_[0];
    int k = this->layer_param_.batch_reduction_param().reduction_param().k();

    // num_ outer loops
    for (int n = 0; n < num_; ++n) {
      // step_ inner loops
      for (int i = 0; i < step_; ++i) {
        //fill data
        Dtype diff = top_diff[i] / Dtype(k);
        for (int t = 0; t < tick; ++t){
          bottom_diff[t * step_ + i] = (idx_data[t * step_ + i] >= 1)?diff:0;
        }

      }
      top_diff += step_;
      bottom_diff += tick*step_;
      idx_data += tick*step_;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BatchReductionLayer);
#endif

INSTANTIATE_CLASS(BatchReductionLayer);
REGISTER_LAYER_CLASS(BatchReduction);

}  // namespace caffe
