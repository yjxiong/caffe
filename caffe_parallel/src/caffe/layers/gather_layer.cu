#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GatherLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {

  #ifdef USE_MPI
  if (Caffe::parallel_mode() == Caffe::MPI){
    for (int i = 0; i < bottom.size(); ++i) {
      //Gather the bottom to the top
      MPI_Allgather((void*)bottom[i]->gpu_data(), bottom[i]->count(),
                    (sizeof(Dtype) == 4) ? MPI_FLOAT : MPI_DOUBLE,
                    (void*)top[i]->mutable_gpu_data(), bottom[i]->count(),
                    (sizeof(Dtype) == 4) ? MPI_FLOAT : MPI_DOUBLE,
                    MPI_COMM_WORLD);
    }
  }
  #endif
  //Do nothing if not if MPI mode
}

template <typename Dtype>
void GatherLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  #ifdef USE_MPI
    if (Caffe::parallel_mode() == Caffe::MPI){
      for (int i = 0; i < bottom.size(); ++i) {
          //Scatter the top diff to buttom
          if (propagate_down[i]) {
          MPI_Scatter((void*)top[i]->gpu_diff(), bottom[i]->count(),
                      (sizeof(Dtype) == 4) ? MPI_FLOAT : MPI_DOUBLE,
                      (void*)bottom[i]->mutable_gpu_diff(), bottom[i]->count(),
                      (sizeof(Dtype) == 4) ? MPI_FLOAT : MPI_DOUBLE,
                      0,
                      MPI_COMM_WORLD);
          //compensate the scale on diff IMPORTANT
          caffe_gpu_scal(bottom[i]->count(), Dtype(Caffe::MPI_all_rank()),
                         bottom[i]->mutable_gpu_diff());
        }
      }
    }
  #endif
}

INSTANTIATE_LAYER_GPU_FUNCS(GatherLayer);

}  // namespace caffe
