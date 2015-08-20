% test video and its optical flow field
video_name = 'test.avi';
video_flow = 'test/';

% spatial prediction
model_def_file = 'cuhk_action_spatial_vgg_16_deploy.prototxt';
model_file = 'cuhk_action_spatial_vgg_16_split1.caffemodel';
mean_file = 'rgb_mean.mat';
gpu_id = 0;

caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_def_file, model_file, 'test');

spatial_prediction = VideoSpatialPrediction(video_name, mean_file, net);

caffe.reset_all();


% temporal prediction
model_def_file = 'cuhk_action_temporal_vgg_16_deploy.prototxt';
model_file = 'cuhk_action_temporal_vgg_16_split1.caffemodel';
mean_file = 'flow_mean.mat';
gpu_id = 0;

caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_def_file, model_file, 'test');

temporal_prediction = VideoTemporalPrediction(video_flow, mean_file, net);

caffe.reset_all();