from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import time

#config_file = '../configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py'
#checkpoint_file = '../checkpoints/DECOUPLED_SOLO_R50_3x.pth'

config_file = 'configs/solov2/solov2_light_448_r34_fpn_8gpu_3x.py'
checkpoint_file = 'checkpoints/SOLOv2_LIGHT_448_R34_3x.pth'

#config_file = '../configs/solov2/solov2_light_448_r18_fpn_8gpu_3x.py'
#checkpoint_file = '../checkpoints/SOLOv2_LIGHT_448_R18_3x.pth'


# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
#model = init_detector(config_file, checkpoint_file, device='cpu')

# test a single image
img = 'demo/kitti.png'

'''
result是一个list，表示每张图片的分割结果
result[i]是一个tuple，分别是(seg_masks, cate_labels, cate_scores)
'''
result = inference_detector(model, img)

print(result[0][0].shape)
print(result[0][1].shape)
print(result[0][2].shape)

show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file="kitti_out.jpg")


