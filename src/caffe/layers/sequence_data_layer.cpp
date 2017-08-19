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

#ifdef USE_MPI
#include "mpi.h"
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#endif

namespace caffe{
template <typename Dtype>
SequenceDataLayer<Dtype>:: ~SequenceDataLayer<Dtype>(){
	this->JoinPrefetchThread();
}

template <typename Dtype>
void SequenceDataLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const int new_height  = this->layer_param_.sequence_data_param().new_height();
	const int new_width  = this->layer_param_.sequence_data_param().new_width();
	const int num_frames  = this->layer_param_.sequence_data_param().num_frames();
	const int num_segments = this->layer_param_.sequence_data_param().num_segments();
        const int num_shots = this->layer_param_.sequence_data_param().num_shots();
	const string& video_source = this->layer_param_.sequence_data_param().video_source();
        const string& shot_source = this->layer_param_.sequence_data_param().shot_source();

	// loading video files (path, duration, label)
        LOG(INFO) << "Opening video source file: " << video_source;
	std:: ifstream infile(video_source.c_str());
	string filename;
	int label;
	int length;
	while (infile >> filename >> length >> label) {
		lines_.push_back(std::make_pair(filename,label));
		lines_duration_.push_back(length);
	}

        // loading shot files (filename, duration)
        LOG(INFO) << "Opening shot source file: " << shot_source;
        std:: ifstream infile_shot(shot_source.c_str());
        while (infile_shot >> filename >> length) {
		std:: ifstream shot_file(filename.c_str());
                int start = 0;
                int end = 0;
                int tmp;
                vector<std::pair<int, int> > tmp_vec;
                while (shot_file >> tmp) {
                	if (start==0) {
				start = tmp;
				continue;			
			}
                        if (this->layer_param_.sequence_data_param().modality() == SequenceDataParameter_Modality_RGB)
				end = tmp - 1;
                        if (this->layer_param_.sequence_data_param().modality() == SequenceDataParameter_Modality_FLOW)
				end = tmp - 2;
			if (end-start+1 > num_frames)
				tmp_vec.push_back(std::make_pair(start, end));
			start = tmp;
                }
               	if (this->layer_param_.sequence_data_param().modality() == SequenceDataParameter_Modality_RGB)
			end = length;
		if (this->layer_param_.sequence_data_param().modality() == SequenceDataParameter_Modality_FLOW)
			end = length - 1;
		if (end-start+1 > num_frames)
			tmp_vec.push_back(std::make_pair(start, end));
		lines_shot_.push_back(tmp_vec);
	}
 
	// shuffling the file_list, duration_list, shot_list	
	if (this->layer_param_.sequence_data_param().shuffle()){
		const unsigned int prefectch_rng_seed = caffe_rng_rand();
		prefetch_rng_1_.reset(new Caffe::RNG(prefectch_rng_seed));
		prefetch_rng_2_.reset(new Caffe::RNG(prefectch_rng_seed));
		prefetch_rng_3_.reset(new Caffe::RNG(prefectch_rng_seed));
		ShuffleSequences();
	}
        
	LOG(INFO) << "A total of " << lines_.size() << " videos.";
	LOG(INFO) << "A total of " << lines_shot_.size() << " shots.";
        for (int i = 0; i < lines_shot_.size(); ++i)
		LOG(INFO) << lines_[i].first << " " << lines_shot_[i].size();

	//check name patter
	if (this->layer_param_.sequence_data_param().name_pattern() == ""){
		if (this->layer_param_.sequence_data_param().modality() == SequenceDataParameter_Modality_RGB){
			name_pattern_ = "image_%04d.jpg";
		}else if (this->layer_param_.sequence_data_param().modality() == SequenceDataParameter_Modality_FLOW){
			name_pattern_ = "flow_%c_%04d.jpg";
		}
	}else{
		name_pattern_ = this->layer_param_.sequence_data_param().name_pattern();
	}

        const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
	frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));

	Datum datum;
        lines_id_ = 0;
        vector<std::pair<int, int> > cur_shot_list = lines_shot_[lines_id_];
        vector<int> offsets;

        for (int i = 0; i < num_shots; ++i) {
		int shot_idx = i;
		if (i >= cur_shot_list.size()) {
			caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
			shot_idx = (*frame_rng)() % (cur_shot_list.size());		
		}
		int start_idx = cur_shot_list[shot_idx].first;
		int end_idx = cur_shot_list[shot_idx].second;
		int average_duration = (int) (end_idx-start_idx+1)/num_segments;
		for (int j = 0; j < num_segments; ++j) {
			if (average_duration < num_frames) {
				offsets.push_back(start_idx-1);
				continue;
			}
			caffe::rng_t* frame_rng1 = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
			int offset = (*frame_rng1)() % (average_duration - num_frames + 1);
			offsets.push_back(start_idx-1+offset+j*average_duration);
		}
	}
	if (this->layer_param_.sequence_data_param().modality() == SequenceDataParameter_Modality_FLOW)
		CHECK(ReadSegmentFlowToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
									 offsets, new_height, new_width, num_frames, &datum, name_pattern_.c_str()));
	else
		CHECK(ReadSegmentRGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
									offsets, new_height, new_width, num_frames, &datum, true, name_pattern_.c_str()));
	const int crop_size = this->layer_param_.transform_param().crop_size();
	const int batch_size = this->layer_param_.sequence_data_param().batch_size();
	if (crop_size > 0){
		top[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
		this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
	} else {
		top[0]->Reshape(batch_size, datum.channels(), datum.height(), datum.width());
		this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());
	}
	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

	top[1]->Reshape(batch_size, 1, 1, 1);
	this->prefetch_label_.Reshape(batch_size, 1, 1, 1);

	vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
	this->transformed_data_.Reshape(top_shape);
}

template <typename Dtype>
void SequenceDataLayer<Dtype>::ShuffleSequences(){
	caffe::rng_t* prefetch_rng1 = static_cast<caffe::rng_t*>(prefetch_rng_1_->generator());
	caffe::rng_t* prefetch_rng2 = static_cast<caffe::rng_t*>(prefetch_rng_2_->generator());
        caffe::rng_t* prefetch_rng3 = static_cast<caffe::rng_t*>(prefetch_rng_3_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng1);
	shuffle(lines_duration_.begin(), lines_duration_.end(), prefetch_rng2);
	shuffle(lines_shot_.begin(), lines_shot_.end(), prefetch_rng3);
}

template <typename Dtype>
void SequenceDataLayer<Dtype>::InternalThreadEntry(){

	Datum datum;
	CHECK(this->prefetch_data_.count());
	Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
	Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
	SequenceDataParameter sequence_data_param = this->layer_param_.sequence_data_param();
	const int batch_size = sequence_data_param.batch_size();
	const int new_height = sequence_data_param.new_height();
	const int new_width = sequence_data_param.new_width();
	const int num_frames = sequence_data_param.num_frames();
	const int num_segments = sequence_data_param.num_segments();
	const int num_shots = sequence_data_param.num_shots();
	const int lines_size = lines_.size();

	for (int item_id = 0; item_id < batch_size; ++item_id){
		CHECK_GT(lines_size, lines_id_);
		
		vector<std::pair<int, int> > cur_shot_list = lines_shot_[lines_id_];
		caffe::rng_t* frame_rng3 = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
		shuffle(cur_shot_list.begin(), cur_shot_list.end(), frame_rng3);
        	vector<int> offsets;

        	for (int i = 0; i < num_shots; ++i) {
			int shot_idx = i;
			if (i >= cur_shot_list.size()) {
				caffe::rng_t* frame_rng1 = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
				shot_idx = (*frame_rng1)() % (cur_shot_list.size());		
			}
			int start_idx = cur_shot_list[shot_idx].first;
			int end_idx = cur_shot_list[shot_idx].second;
			int average_duration = (int) (end_idx-start_idx+1)/num_segments;
			for (int j = 0; j < num_segments; ++j) {
				if (average_duration < num_frames) {
					offsets.push_back(start_idx-1);
					continue;
				}
				caffe::rng_t* frame_rng2 = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
				int offset = (*frame_rng2)() % (average_duration - num_frames + 1);
				offsets.push_back(start_idx-1+offset+j*average_duration);
			}
		}

		if (this->layer_param_.sequence_data_param().modality() == SequenceDataParameter_Modality_FLOW) {
			if(!ReadSegmentFlowToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
									   offsets, new_height, new_width, num_frames, &datum, name_pattern_.c_str())) {
				continue;
			}
		} else { 
			if(!ReadSegmentRGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
									  offsets, new_height, new_width, num_frames, &datum, true, name_pattern_.c_str())) {
				continue;
			}
		}

		int offset1 = this->prefetch_data_.offset(item_id);
    	        this->transformed_data_.set_cpu_data(top_data + offset1);
		this->data_transformer_->Transform(datum, &(this->transformed_data_));
		top_label[item_id] = lines_[lines_id_].second;
		//LOG()

		//next iteration
		lines_id_++;
		if (lines_id_ >= lines_size) {
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if(this->layer_param_.sequence_data_param().shuffle()){
				ShuffleSequences();
			}
		}
	}
}

INSTANTIATE_CLASS(SequenceDataLayer);
REGISTER_LAYER_CLASS(SequenceData);
}
