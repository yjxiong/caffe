#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef USE_MPI
#include "mpi.h"
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#endif

namespace caffe{
template <typename Dtype>
SegDataLayer<Dtype>:: ~SegDataLayer<Dtype>(){
	this->JoinPrefetchThread();
}

template <typename Dtype>
void SegDataLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){

	const int stride = this->layer_param_.transform_param().stride();
	const string& source = this->layer_param_.seg_data_param().source();
	const string& root_dir = this->layer_param_.seg_data_param().root_dir();


	LOG(INFO) << "Opening file: " << source;
	std:: ifstream infile(source.c_str());
	string img_filename;
	string label_filename;
	while (infile >> img_filename >> label_filename){
		lines_.push_back(std::make_pair(root_dir + img_filename, root_dir + label_filename));
	}

	if (this->layer_param_.seg_data_param().shuffle()){
		const unsigned int prefectch_rng_seed = 17;//caffe_rng_rand(); // magic number
		prefetch_rng_.reset(new Caffe::RNG(prefectch_rng_seed));
		ShuffleImages();
	}

	LOG(INFO) << "A total of " << lines_.size() << " images.";
	lines_id_ = 0;

	Datum datum_data, datum_label;
	CHECK(ReadSegDataToDatum(lines_[lines_id_].first, lines_[lines_id_].second, &datum_data, &datum_label, true));


	int crop_height = datum_data.height() / stride * stride;
	int crop_width = datum_data.width() / stride * stride;

  	if (this->layer_param_.transform_param().has_upper_size())
	{
		crop_height = std::min(crop_height, this->layer_param_.transform_param().upper_size());
		crop_width = std::min(crop_width, this->layer_param_.transform_param().upper_size());
	}
	else if (this->layer_param_.transform_param().has_upper_height() && this->layer_param_.transform_param().has_upper_width())
	{
		crop_height = std::min(crop_height, this->layer_param_.transform_param().upper_height());
		crop_width = std::min(crop_width, this->layer_param_.transform_param().upper_width());
	}


	top[0]->Reshape(1, datum_data.channels(), crop_height, crop_width);
	this->prefetch_data_.Reshape(1, datum_data.channels(), crop_height, crop_width);

	top[1]->Reshape(1, datum_label.channels(), crop_height, crop_width);
	this->prefetch_label_.Reshape(1, datum_label.channels(), crop_height, crop_width);

	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();
	LOG(INFO) << "output label size: " << top[1]->num() << "," << top[1]->channels() << "," << top[1]->height() << "," << top[1]->width();
}

template <typename Dtype>
void SegDataLayer<Dtype>::ShuffleImages(){
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void SegDataLayer<Dtype>::InternalThreadEntry(){

	Datum datum_data, datum_label;
	CHECK(this->prefetch_data_.count());
	
	const int lines_size = lines_.size();

	CHECK_GT(lines_size, lines_id_);
	CHECK(ReadSegDataToDatum(lines_[lines_id_].first, lines_[lines_id_].second, &datum_data, &datum_label, true));

	this->data_transformer_->Transform(datum_data, datum_label, &this->prefetch_data_, &this->prefetch_label_);

	if (this->layer_param_.seg_data_param().balance())
	{
		for (int t = 0; t < 10; t++)
		{
			std::vector<int> cnt(256, 0); int max_label_cnt = 0;
			for (int p1 = 0; p1 < this->prefetch_label_.height(); p1 ++)
	  	  		for (int p2 = 0; p2 < this->prefetch_label_.width(); p2 ++)
	  	  		{
	  	  			int label_value = (int)this->prefetch_label_.data_at(0, 0, p1, p2);
	  	  			cnt[label_value]++;
	  	  		}
	  	  	for (int i = 0; i<cnt.size(); i++)
	  			max_label_cnt = std::max(max_label_cnt, cnt[i]);

	  		if (max_label_cnt > 0.8 * this->prefetch_label_.count())
	  			this->data_transformer_->Transform(datum_data, datum_label, &this->prefetch_data_, &this->prefetch_label_);
  			else
  				break;
		}
	}

	
	if (false)
	{
	  	cv::Mat im_data(this->prefetch_data_.height(), this->prefetch_data_.width(), CV_8UC3);
	  	cv::Mat im_label(this->prefetch_label_.height(), this->prefetch_label_.width(), CV_8UC1);

	  	for (int p1 = 0; p1 < this->prefetch_data_.height(); p1 ++)
	  		for (int p2 = 0; p2 < this->prefetch_data_.width(); p2 ++)
	  		{
	  			im_data.at<uchar>(p1, p2*3+0) = (uchar)(this->prefetch_data_.data_at(0, 0, p1, p2)+104);
	  			im_data.at<uchar>(p1, p2*3+1) = (uchar)(this->prefetch_data_.data_at(0, 1, p1, p2)+117);
	  			im_data.at<uchar>(p1, p2*3+2) = (uchar)(this->prefetch_data_.data_at(0, 2, p1, p2)+123);	
	  		}
  		for (int p1 = 0; p1 < this->prefetch_label_.height(); p1 ++)
  	  		for (int p2 = 0; p2 < this->prefetch_label_.width(); p2 ++)
  	  			im_label.at<uchar>(p1, p2) = this->prefetch_label_.data_at(0, 0, p1, p2);
	  	int tot = rand() * 10000 + rand() + lines_id_;
	  	char temp_path[200];
	  	sprintf(temp_path, "temp/%d_0_%s.jpg", tot, lines_[lines_id_].first.substr(18+16,3).c_str());
	  	imwrite(temp_path, im_data);
	  	sprintf(temp_path, "temp/%d_1_%s.jpg", tot, lines_[lines_id_].first.substr(18+16,3).c_str());
	  	imwrite(temp_path, im_label);
	}



	//next iteration
	lines_id_++;
	if (lines_id_ >= lines_.size()) {
		// We have reached the end. Restart from the first.
		DLOG(INFO) << "Restarting data prefetching from start.";
		lines_id_ = 0;
		if (this->layer_param_.seg_data_param().shuffle()) {
			ShuffleImages();
		}
	}
}

INSTANTIATE_CLASS(SegDataLayer);
REGISTER_LAYER_CLASS(SegData);
}
