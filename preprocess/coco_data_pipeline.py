import logging
import os

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.dataloader import DataLoader, DataLoaderIter

from preprocess.InputProcess import load_image_gt, mold_image
from tasks.bbox.AnchorProcess import (generate_pyramid_anchors,
                                      generate_random_rois)
from tasks.bbox.BboxProcess import compute_iou, compute_overlaps
from tasks.merge_task import build_detection_targets, build_rpn_targets
from tasks.utils import log

class sDataLoader(DataLoader):
	def get_stream(self):
		while True:
			for data in DataLoaderIter(self):
				yield data


def CocoLoader(dataset, config, shuffle=True, augment=True, batch_size=16, num_workers=16):
    dataset = CocoData(dataset, config, augment=augment)
    data_loader = sDataLoader(dataset, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers, drop_last=True)
    return data_loader


class CocoData(Dataset):
    # class for loading and processing coco images
    # input:
    # inp_size: resolution for model input
    # feat_stride: stride from model inpt to rpn features
    # preprocess: image normalization
    # training: boolean value for mode
    # note that data directories are all defined in __init__()
    def __init__(self, dataset, config, augment=True):
        self.dataset = dataset
        self.augment = augment
        self.image_ids = np.copy(dataset.image_ids)
        self.config = config
        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        self.anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                config.RPN_ANCHOR_RATIOS,
                                                config.BACKBONE_SHAPES,
                                                config.BACKBONE_STRIDES,
                                                config.RPN_ANCHOR_STRIDE)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):

        config = self.config
        dataset = self.dataset
        # Get GT bounding boxes and masks for image.
        image_id = self.image_ids[index]
        image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
            load_image_gt(dataset, config, image_id, augment=self.augment,
            use_mini_mask=config.USE_MINI_MASK)
                           
                    
        # RPN Targets
        rpn_match, rpn_bbox = build_rpn_targets(image.shape, self.anchors,
                                            gt_class_ids, gt_boxes, config)
                                                                
        # If more instances than fits in the array, sub-sample from them.

        if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]


        batch_gt_boxes = np.zeros(
            (config.MAX_GT_INSTANCES, 4), dtype=gt_boxes.dtype)
            
        batch_gt_masks = np.zeros((gt_masks.shape[0], gt_masks.shape[1],
            config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)                                   
                                               
        batch_gt_class_ids = np.zeros(
                    (config.MAX_GT_INSTANCES),dtype=gt_class_ids.dtype)      
                                                             
        batch_gt_boxes[:gt_boxes.shape[0],:] = gt_boxes
        batch_gt_masks[:,:,:gt_masks.shape[2]] = gt_masks  
        batch_gt_class_ids[:gt_class_ids.shape[0]] = gt_class_ids                                        
        # Add to batch
        rpn_match = rpn_match[:, np.newaxis]
        image = mold_image(image.astype(np.float32), config)
        batch_gt_class_ids = batch_gt_class_ids.astype(np.float32) 
        batch_gt_masks = batch_gt_masks.astype(np.float32)  
        image = image.transpose(2, 0, 1)
        return image, image_meta, rpn_match, rpn_bbox, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks                                                      
