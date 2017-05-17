
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/common_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class BatchReductionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BatchReductionLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(1, 1, 5, 2)),
        blob_top_data_(new Blob<Dtype>()) {
    Dtype* bottom_data = blob_bottom_data_->mutable_cpu_data();
    int i = 0;
    bottom_data[0 * 2 + i] = 1;
    bottom_data[1 * 2 + i] = 2;
    bottom_data[2 * 2 + i] = 3;
    bottom_data[3 * 2 + i] = 2;
    bottom_data[4 * 2 + i] = 1;
    i = 1;
    bottom_data[0 * 2 + i] = 2;
    bottom_data[1 * 2 + i] = 3;
    bottom_data[2 * 2 + i] = 4;
    bottom_data[3 * 2 + i] = 3;
    bottom_data[4 * 2 + i] = 2;

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_top_vec_.push_back(blob_top_data_);
  }
  virtual ~BatchReductionLayerTest() {
    delete blob_bottom_data_;
    delete blob_top_data_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BatchReductionLayerTest, TestDtypesAndDevices);

TYPED_TEST(BatchReductionLayerTest, TestSetUp) {
typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BatchReductionParameter* batch_reduction_param = layer_param.mutable_batch_reduction_param();
  batch_reduction_param->mutable_reduction_param()->set_operation(ReductionParameter_ReductionOp_MEAN);
  BatchReductionLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  batch_reduction_param->mutable_reduction_param()->set_operation(ReductionParameter_ReductionOp_TOPK);
  batch_reduction_param->mutable_reduction_param()->set_axis(2);
  batch_reduction_param->mutable_reduction_param()->set_k(3);
  BatchReductionLayer<Dtype> topk_layer(layer_param);
  topk_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
}


TYPED_TEST(BatchReductionLayerTest, TestMeanForwardBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BatchReductionParameter* batch_reduction_param = layer_param.mutable_batch_reduction_param();
  batch_reduction_param->mutable_reduction_param()->set_operation(ReductionParameter_ReductionOp_MEAN);
  batch_reduction_param->mutable_reduction_param()->set_axis(2);
  BatchReductionLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(3, this->blob_top_data_->num_axes());
  EXPECT_EQ(2, this->blob_top_data_->shape(2));

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->blob_top_data_->cpu_data();

  EXPECT_NEAR(1.8, top_data[0], 0.001);
  EXPECT_NEAR(2.8, top_data[1], 0.001);

  Dtype* top_diff = this->blob_top_data_->mutable_cpu_diff();

  top_diff[0] = 0.5;
  top_diff[1] = 1.5;

  vector<bool> propagate_down(1, true);
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

  const Dtype* bottom_diff = this->blob_bottom_data_->cpu_diff();

  for (int r = 0; r < 2; ++r){
    for (int i = 0; i < 5; ++i){
      EXPECT_NEAR(bottom_diff[i * 2 + r], 0.1 + 0.2 * r, 0.0001);
    }
  }
}

TYPED_TEST(BatchReductionLayerTest, TestTopKForwardBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BatchReductionParameter* batch_reduction_param = layer_param.mutable_batch_reduction_param();
  batch_reduction_param->mutable_reduction_param()->set_operation(ReductionParameter_ReductionOp_TOPK);
  batch_reduction_param->mutable_reduction_param()->set_axis(2);
  batch_reduction_param->mutable_reduction_param()->set_k(3);
  BatchReductionLayer<Dtype> layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(3, this->blob_top_data_->num_axes());
  EXPECT_EQ(2, this->blob_top_data_->shape(2));

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->blob_top_data_->cpu_data();

  EXPECT_NEAR(7.0/3.0, top_data[0], 0.001);
  EXPECT_NEAR(10.0/3.0, top_data[1], 0.001);

  Dtype* top_diff = this->blob_top_data_->mutable_cpu_diff();

  top_diff[0] = 0.6;
  top_diff[1] = 1.8;

  vector<bool> propagate_down(1, true);
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

  const Dtype* bottom_diff = this->blob_bottom_data_->cpu_diff();

  for (int r = 0; r < 2; ++r){
    for (int i = 1; i < 4; ++i){
      EXPECT_NEAR(bottom_diff[i * 2 + r], 0.2 + 0.4 * r, 0.0001);
    }
    EXPECT_NEAR(bottom_diff[0 * 2 + r], 0.0, 0.0001);
    EXPECT_NEAR(bottom_diff[5 * 2 + r], 0.0, 0.0001);
  }
}

//TYPED_TEST(BatchReductionLayerTest, TestGradient) {
//  typedef typename TypeParam::Dtype Dtype;
//  LayerParameter layer_param;
//  ROIPoolingParameter* roi_pooling_param =
//      layer_param.mutable_roi_pooling_param();
//  roi_pooling_param->set_pooled_h(6);
//  roi_pooling_param->set_pooled_w(6);
//  ROIPoolingLayer<Dtype> layer(layer_param);
//  GradientChecker<Dtype> checker(1e-4, 1e-2);
//  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//      this->blob_top_vec_, 0);
//}

}  // namespace caffe
