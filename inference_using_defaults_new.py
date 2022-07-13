import cv2
import os
import torch
from torch.distributions.weibull import Weibull
from torch.distributions.transforms import AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from detectron2.utils.logger import setup_logger
setup_logger()


import src.config.defaults
import src.engine
import src.config
import src.modeling
import src.data

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from importlib_resources import path



import numpy as np

def getFileName1(path,suffix):
    
    input_template_All=[]
    f_list = os.listdir(path)
    for i in f_list:
        # os.path.splitext():
        if os.path.splitext(i)[1] ==suffix:
            input_template_All.append(i)
            #print(i)
    return input_template_All



ourimg='/data/CODE/xcr/OWOD-master/ourimg'
vocimg='/data/CODE/xcr/OWOD-master/image'
cocoimg='/data/CODE/xcr/stud-main/image_for_inference/coco2017'
bddimg='/data/CODE/xcr/stud-main/image_for_inference/bdd100k/b1c9c847-3bda4659'
cocoval='/data/CODE/xcr/stud-main/datasets/coco2017/val2017'
list_jpg=getFileName1(vocimg,".jpg")
list_png=getFileName1(ourimg,".png")#文件名列表
list_jpg_coco=getFileName1(cocoimg,".jpg")
list_jpg_bdd100k=getFileName1(bddimg,".jpg")
list_jpg_coco_val=getFileName1(cocoval,'.jpg')

# Get image

#im = cv2.imread("/home/fk1/workspace/OWOD/datasets/VOC2007/JPEGImages/" + file_name + ".jpg")
#im = cv2.imread("/data/CODE/xcr/OWOD-master/image/2007_000033.jpg")
# model = '/home/fk1/workspace/OWOD/output/old/t1_20_class/model_0009999.pth'
# model = '/home/fk1/workspace/OWOD/output/t1_THRESHOLD_AUTOLABEL_UNK/model_final.pth'
# model = '/home/fk1/workspace/OWOD/output/t1_clustering_with_save/model_final.pth'
model = '/data/CODE/xcr/stud-main/resnet_bdd/model_final_resnet_bdd.pth'
# model = '/home/fk1/workspace/OWOD/output/t2_ft/model_final.pth'
#model = '/data/CODE/xcr/OWOD-master/results/t2_final/model_final.pth'
# model = '/home/fk1/workspace/OWOD/output/t3_ft/model_final.pth'
#model = '/data/CODE/xcr/OWOD-master/results/t3_final/model_final.pth'
#model = '/home/fk1/workspace/OWOD/output/t4_ft/model_final.pth'
#cfg_file = '/home/fk1/workspace/OWOD/configs/OWOD/t1/t1_test.yaml'
cfg_file = '/data/CODE/xcr/stud-main/configs/BDD100k/stud_resnet_ood_coco.yaml'
#cfg_file = '/data/CODE/xcr/OWOD-master/configs/OWOD/t2/t2_test.yaml'
#cfg_file = '/data/CODE/xcr/OWOD-master/configs/OWOD/t3/t3_test.yaml'



# Get the configuration ready
cfg = get_cfg()

cfg.merge_from_file(cfg_file)
#print(cfg)
cfg.MODEL.WEIGHTS = model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.61
# cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.8
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4

# POSITIVE_FRACTION: 0.25
# NMS_THRESH_TEST: 0.5
# SCORE_THRESH_TEST: 0.05
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 21

predictor = DefaultPredictor(cfg)
save_path='/data/CODE/xcr/stud-main/outputimg/ourimg/'
save_path_voc='/data/CODE/xcr/stud-main/outputimg/voc/'
save_path_coco2017='/data/CODE/xcr/stud-main/outputimg/coco/'
save_path_bdd100k='/data/CODE/xcr/stud-main/outputimg/bdd100k/'
save_path_bdd100k_ood='/data/CODE/xcr/stud-main/outputimg/ood_bdd100k/'
save_path_coco2017_ood='/data/CODE/xcr/stud-main/outputimg/ood_coco/'
save_path_coco2017_val_ood='/data/CODE/xcr/stud-main/outputimg/coco2017val_ood'



def visualize_inference(model, inputs, results, savedir, name, cfg, energy_threshold=None):
    """
    A function used to visualize final network predictions.
    It shows the original image and up to 20
    predicted object bounding boxes on the original image.

    Valuable for debugging inference methods.

    Args:
        inputs (list): a list that contains input to the model.
        results (List[Instances]): a list of #images elements.
    """
    import cv2
    from detectron2.utils.visualizer import ColorMode, _SMALL_OBJECT_AREA_THRESH
    from detectron2.data import MetadataCatalog
    from src.engine.myvisualizer import MyVisualizer
    max_boxes = 20

    # required_width = inputs[0]['width']
    # required_height = inputs[0]['height']

    # img = inputs[0]["image"].cpu().numpy()
    # assert img.shape[0] == 3, "Images should have 3 channels."
    # if model.input_format == "RGB":
    #     img = img[::-1, :, :]
    # img = img.transpose(1, 2, 0)
    # img = cv2.resize(img, (required_width, required_height))
    # breakpoint()
    results = results['instances']
    predicted_boxes = results.pred_boxes.tensor.cpu().numpy()


    v_pred = MyVisualizer(inputs, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    # print(len(predicted_boxes))
    # breakpoint()
    labels = results.det_labels[0:max_boxes]
    scores = results.scores[0:max_boxes]
    print(labels)
    print(scores)
    # breakpoint()

    inter_feat = results.inter_feat[0:max_boxes]
    print(inter_feat)
    print(torch.logsumexp(inter_feat[:, :-1], dim=1).cpu().data.numpy())
    print((np.argwhere(
            torch.logsumexp(inter_feat[:, :-1], dim=1).cpu().data.numpy() < energy_threshold)).reshape(-1))
    if energy_threshold:
        labels[(np.argwhere(
            torch.logsumexp(inter_feat[:, :-1], dim=1).cpu().data.numpy() < energy_threshold)).reshape(-1)] = 8
    print(labels)
    # # if name == '133631':
    #     # breakpoint()
    # # breakpoint()
    if len(scores) == 0 or max(scores) <= 0.0:
        return

    v_pred = v_pred.overlay_covariance_instances(
        labels=labels,
        scores=scores,
        boxes=predicted_boxes[0:max_boxes], covariance_matrices=None,
        score_threshold = 0.0)
        # covariance_matrices=predicted_covar_mats[0:max_boxes])

    prop_img = v_pred.get_image()
    vis_name = f"{max_boxes} Highest Scoring Results"
    # cv2.imshow(vis_name, prop_img)
    # cv2.savefig
    cv2.imwrite(savedir + '/' + name + '.jpg', prop_img)
    cv2.waitKey()



# im=cv2.imread('/data/CODE/xcr/stud-main/image_for_inference/coco2017/000000343821.jpg')
# outputs=predictor(im)
# print(outputs)
# print(outputs['instances'].pred_cls_probs)
# print(outputs['instances'].det_labels)
# print(outputs['instances'].scores)
# print(outputs['instances'].inter_feat.cpu().data.numpy())


# save_path_bdd100k_threshold6p5='/data/CODE/xcr/stud-main/outputimg/bdd100k_threshold6p5'
# for i in list_jpg_bdd100k:
#     im=cv2.imread(bddimg+'/'+i)
#     outputs=predictor(im)
#     #print(outputs)
#     # print(im.shape)
#     # print(im.shape[0])
#     # print(im.shape[1])
#     visualize_inference(model, im,
#                                   outputs,
#                                   savedir=save_path_bdd100k_threshold6p5,
#                                   name=i,
#                                   cfg=cfg,
#                                   energy_threshold=6.5)
    
save_path_coco_threshold6p5='/data/CODE/xcr/stud-main/outputimg/coco_threshold6p5'
for i in list_jpg_coco_val:
    im=cv2.imread(cocoval+'/'+i)
    outputs=predictor(im)
    #print(outputs)
    # print(im.shape)
    # print(im.shape[0])
    # print(im.shape[1])
    visualize_inference(model, im,
                                  outputs,
                                  savedir=save_path_coco_threshold6p5,
                                  name=i,
                                  cfg=cfg,
                                  energy_threshold=6.5)
    
# new_save_path='/data/CODE/xcr/stud-main/outputimg/newpath'    
# for i in list_jpg_coco:
#     im=cv2.imread(cocoimg+'/'+i)
#     outputs=predictor(im)
#     # print(outputs)
#     # print(im.shape)
#     # print(im.shape[0])
#     # print(im.shape[1])
#     visualize_inference(model, im,
#                                   outputs,
#                                   savedir=new_save_path,
#                                   name=i,
#                                   cfg=cfg,
#                                   energy_threshold=8.3)
                                 
    
# for i in list_jpg_coco_val:
#     im=cv2.imread(cocoval+'/'+i)
#     outputs=predictor(im)
#     print(outputs)
#     # print(im.shape)
#     # print(im.shape[0])
#     # print(im.shape[1])
#     visualize_inference(model, im,
#                                   outputs,
#                                   savedir=save_path_coco2017_ood,
#                                   name=i,
#                                   cfg=cfg)
#                                  # energy_threshold=8.868)