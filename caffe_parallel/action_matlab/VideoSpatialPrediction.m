function prediction = VideoSpatialPrediction(vid_name, mean_file, net)
num=25;
vidObj = VideoReader(vid_name);
duration = vidObj.NumberOfFrames;
step = floor((duration-1)/(num-1));
rgb = zeros(256,340,3,num,'single');
rgb_flip = zeros(256,340,3,num,'single');


for i = 1:num
    img = single(imresize(read(vidObj,(i-1)*step+1),[256,340]));
    rgb(:,:,:,i) = img;
    rgb_flip(:,:,:,i) = img(:,end:-1:1,:);
end

rgb_1 = rgb(1:224,1:224,:,:);
rgb_2 = rgb(1:224,end-223:end,:,:);
rgb_3 = rgb(16:16+223,60:60+223,:,:);
rgb_4 = rgb(end-223:end,1:224,:,:);
rgb_5 = rgb(end-223:end,end-223:end,:,:);
rgb_f_1 = rgb_flip(1:224,1:224,:,:);
rgb_f_2 = rgb_flip(1:224,end-223:end,:,:);
rgb_f_3 = rgb_flip(16:16+223,60:60+223,:,:);
rgb_f_4 = rgb_flip(end-223:end,1:224,:,:);
rgb_f_5 = rgb_flip(end-223:end,end-223:end,:,:);

rgb = cat(4,rgb_1,rgb_2,rgb_3,rgb_4,rgb_5,rgb_f_1,rgb_f_2,rgb_f_3,rgb_f_4,rgb_f_5);
d = load(mean_file);
IMAGE_MEAN = single(d.image_mean);
rgb = bsxfun(@minus,rgb(:,:,[3,2,1],:),IMAGE_MEAN);
rgb = permute(rgb,[2,1,3,4]);



prediction = zeros(101,size(rgb,4));
batch_size = 50;
num_batches = ceil(size(rgb,4)/batch_size);
rgbs = zeros(224,224,3,batch_size,'single');

for bb = 1:num_batches
	range = 1 + batch_size*(bb-1): min(size(rgb,4),batch_size*bb);
	rgbs(:,:,:,mod(range-1,batch_size)+1) = single(rgb(:,:,:,range));
	out_put = net.forward({rgbs});
	out_put = squeeze(out_put{1});
	prediction(:,range) = out_put(:,mod(range-1,batch_size)+1);
end

end