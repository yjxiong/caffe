#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

// The CUDA kernel actually runs the reduction
template <typename Dtype>
__global__ void BatchReductionForwardKer(const int step, const int num,
                                         const int n_level, const Dtype* ticks,
                                         const bool mean, const bool forward, const bool pos,
                                         Dtype* bottom, Dtype* top) {
    Dtype* bottom_ptr = bottom;
    Dtype* top_ptr = top;
    CUDA_KERNEL_LOOP(index, step){
        for (int n = 0; n < num; ++n){
            for (int l = 0; l < n_level; ++l){
                int tick = ticks[l];
                Dtype coeff = (mean)? Dtype(1)/Dtype(tick) : Dtype(1);
                for (int t = 0; t < tick; ++t){
                    if (forward){
                        top_ptr[index] += bottom_ptr[index] * coeff;
                    }else{
                        bottom_ptr[index] = top_ptr[index] * coeff;
                    }
                    int stride = (t == (tick -1))?1:tick+1;
                    bottom_ptr += (pos)?step*stride:step;
                }
                top_ptr += step;
            }
        }
    }
}

template <typename Dtype>
__device__ void swap(Dtype* data, int i0, int i1){
    Dtype tmp = data[i0];
    data[i0] = data[i1];
    data[i1] = tmp;
}

template <typename Dtype>
__device__ void partitionRank(Dtype* data, int* idx, int n, int k){

    if (k >= n){
        // trivial case
        for (int i = 0; i < n; ++i){
            idx[i] = 256;
        }
    }

    int loc = -1;

    int start = 0;
    int end = n;

    while (loc != k) {
        int left = start;
        int right = end - 1;
        Dtype val = data[start];
        while (left < right) {
            if (data[right--] > val) {
                swap(data, ++right, ++left);
                swap(idx, right, left);
            }
        }
        swap(data, start, right);
        swap(idx, start, right);
        loc = right + 1;
        start = (loc > k)?start:loc;
        end = (loc > k)?(loc - 1):end;
    }

    for (int i = 0; i < k; ++i){
        idx[idx[i] % 256] += 1<<8; // store the flag on the 8-th bit
    }
}

// The CUDA kernel for Top-k reduction
template <typename Dtype>
__global__ void BatchReductionTopKForwardKernel(const int step, const int num,
                                                const int n_level, const int k, const Dtype* ticks,
                                                Dtype* idx, const bool pos,
                                                const Dtype* bottom, Dtype* top){
    const Dtype* bottom_ptr = bottom;
    Dtype* top_ptr = top;
    Dtype* idx_ptr = idx;

    Dtype buffer[256];
    int idx_buffer[256];

    Dtype coeff = Dtype(1)/Dtype(k);
    CUDA_KERNEL_LOOP(index, step){
        for (int n = 0; n < num; ++n){
            for (int l = 0; l < n_level; ++l){
                int tick = ticks[l];
                for (int t = 0; t < tick; ++t){
                    buffer[t] = bottom_ptr[index];
                    idx_buffer[t] = t;
                    int stride = (t == (tick -1))?1:tick+1;
                    bottom_ptr += (pos)?step*stride:step;
                }

                //selection rank algorithm
                partitionRank(buffer, idx_buffer, tick, k);

                for (int t = 0; t < tick; ++t){
                    top_ptr[index] += buffer[t] * ((t < k)?coeff:0);
                    idx_ptr[index] = idx_buffer[t] / 256;

                    int stride = (t == (tick -1))?1:tick+1;
                    idx_ptr += (pos)?step*stride:step;
                }
                top_ptr += step;
            }
        }
    }
}

template <typename Dtype>
__global__ void BatchReductionTopKBackwardKernel(const int step, const int num,
                                                const int n_level, const Dtype* ticks, int k,
                                                const Dtype* idx, const bool pos,
                                                Dtype* bottom, const Dtype* top){
    Dtype* bottom_ptr = bottom;
    const Dtype* top_ptr = top;
    const Dtype* idx_ptr = idx;
    CUDA_KERNEL_LOOP(index, step){
        for (int n = 0; n < num; ++n){
            for (int l = 0; l < n_level; ++l){
                int tick = ticks[l];
                Dtype coeff = Dtype(1)/Dtype(k);
                for (int t = 0; t < tick; ++t){

                    bottom_ptr[index] = top_ptr[index] * coeff * (idx_ptr[index]);

                    int stride = (t == (tick -1))?1:tick+1;
                    bottom_ptr += (pos)?step*stride:step;
                    idx_ptr += (pos)?step*stride:step;
                }
                top_ptr += step;
            }
        }
    }
}

template <typename Dtype>
void BatchReductionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    if ((op_ != ReductionParameter_ReductionOp_TOPK)){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        const Dtype* tick_data = this->ticks_blob_.gpu_data();
        const bool kMean = (this->op_ == ReductionParameter_ReductionOp_MEAN);
        const int n_level = this->levels_.size();

        const bool kForward = true; // forward

        caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
        //invoke kernel
        BatchReductionForwardKer<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(step_), CAFFE_CUDA_NUM_THREADS>>>(
            step_, num_, n_level, tick_data,
            kMean, kForward, pos_,
            (Dtype*)bottom_data, top_data);
    }else{
        int k = this->layer_param_.batch_reduction_param().reduction_param().k();
        if (max_tick_ >= 256){
            // for reduction on sequences longer than 256 elements,
            // maybe std:sort() will be faster than my poor SelectionRank CUDA implementation
            Forward_cpu(bottom, top);
        }else{
            const Dtype* bottom_data = bottom[0]->gpu_data();
            Dtype* top_data = top[0]->mutable_gpu_data();
            Dtype* idx_data = argsort_idx_.mutable_gpu_data();
            const Dtype* tick_data = this->ticks_blob_.gpu_data();
            const int n_level = this->levels_.size();

            caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
            BatchReductionTopKForwardKernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(step_), CAFFE_CUDA_NUM_THREADS>>>(
                step_, num_, n_level, k, tick_data,
                idx_data, pos_,
                bottom_data, top_data);

        }
    }

}

template <typename Dtype>
void BatchReductionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if ((op_ != ReductionParameter_ReductionOp_TOPK)){
        const Dtype *top_diff = top[0]->gpu_diff();
        Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
        const Dtype *tick_data = this->ticks_blob_.gpu_data();
        const bool kMean = (this->op_ == ReductionParameter_ReductionOp_MEAN);
        const int n_level = this->levels_.size();

        const bool kForward = false; // backward

        //invoke kernel
        BatchReductionForwardKer<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(step_), CAFFE_CUDA_NUM_THREADS>>>(
            step_, num_, n_level, tick_data,
            kMean, kForward, pos_,
            bottom_diff, (Dtype*)top_diff);
    }else{
        // Backward does not need n to be less 256
        int k = this->layer_param_.batch_reduction_param().reduction_param().k();
        const Dtype *top_diff = top[0]->gpu_diff();
        Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
        const Dtype* idx_data = argsort_idx_.gpu_data();
        const Dtype* tick_data = this->ticks_blob_.gpu_data();
        const int n_level = this->levels_.size();

        BatchReductionTopKBackwardKernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(step_), CAFFE_CUDA_NUM_THREADS>>>(
            step_, num_, n_level, tick_data, k,
            idx_data, pos_,
            bottom_diff, top_diff);
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(BatchReductionLayer);

}  // namespace caffe
