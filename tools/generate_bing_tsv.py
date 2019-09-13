#!/usr/bin/env python


"""Generate bottom-up attention features as a tsv file. Can use multiple gpus, each produces a 
   separate tsv file that can be merged later (e.g. by using merge_tsv function). 
   Modify the load_image_ids script as necessary for your data location. """


# Example:
# ./tools/generate_bing_tsv.py --gpu 0 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --out output_bing_test_100k.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final_iter_320000.caffemodel --split bing_img --input /data/users/kragga/visual_qna/agiPT/agiPT/projects/visual_qna/lxmert/data/bing/test_split_40M_0_00_100k.tsv --lines 100000
import caffe
import sys
# To remove the _cafe not found error
sys.path.insert(0,'/opt/caffe/python/caffe')
import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect, _get_blobs, vis_detections, vis_multiple, vis_relations
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import io
import wand.image

import argparse
import pprint
import time, os, sys
import numpy as np
import cv2
import csv
from multiprocessing import Process
import random
import json
import base64
import matplotlib.pyplot as plt
from StringIO import StringIO
from PIL import Image
import random as rand

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36. 
MIN_BOXES = 36
MAX_BOXES = 36

def load_image_ids(split_name):
    ''' Load a list of (path,image_id tuples). Modify this to suit your data locations. '''
    split = []
    if split_name == 'coco_test2014':
        with open('data/coco/annotations/image_info_test2014.json') as f:
            data = json.load(f)
            for item in data['images']:
                image_id = int(item['id'])
                filepath = os.path.join('data/test2014/', item['file_name'])
                split.append((filepath,image_id))
    elif split_name == 'coco_test2015':
        with open('data/coco/annotations/image_info_test2015.json') as f:
            data = json.load(f)
            for item in data['images']:
                image_id = int(item['id'])
                filepath = os.path.join('data/test2015/', item['file_name'])
                split.append((filepath,image_id))
    elif split_name == 'genome':
        with open('data/visualgenome/image_data.json') as f:
            for item in json.load(f):
                image_id = int(item['image_id'])
                filepath = os.path.join('data/visualgenome/', item['url'].split('rak248/')[-1])
                split.append((filepath, image_id))
    elif split_name == 'bing_img':
       with open('./data/bing/test.tsv') as f:
            for item in f:
                item_arr = item.split('\t')
                if len(item_arr) > 13:
                    image_id = (item_arr[13])
                    img = (item_arr[0])
                    split.append((img, image_id))
                    
    else:
        print("unknown split")
    return split



def readb64(base64_string):
    sbuf = StringIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

def vis_detection_helper(im, class_name, dets, thresh=0.0):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    new_file_name = str(rand.random()) + "_test.jpg"
    print(new_file_name)
    plt.savefig(new_file_name)

def vis_detections(scores, boxes, attr_boxes, im_file, im):
    # Visualize detections for each class
    CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind : 4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detection_helper(im, cls, dets, thresh=CONF_THRESH)
    
    new_file_name = im_file+str(rand.random())+"_test.jpg"
    plt.savefig(new_file_name)


def get_detections_from_im_bing(net, im, image_id, conf_thresh=0.2):
    """
    :param net:
    :param im_file: full path to an image
    :param image_id:
    :param conf_thresh:
    :return: all information from detection and attr prediction
    """
    
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    attr_prob = net.blobs['attr_prob'].data
    pool5 = net.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1, cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

    objects = np.argmax(cls_prob[keep_boxes][:, 1:], axis=1)
    objects_conf = np.max(cls_prob[keep_boxes][:, 1:], axis=1)
    attrs = np.argmax(attr_prob[keep_boxes][:, 1:], axis=1)
    attrs_conf = np.max(attr_prob[keep_boxes][:, 1:], axis=1)

    return {
        "img_id": image_id,
        "img_h": np.size(im, 0),
        "img_w": np.size(im, 1),
        "objects_id": base64.b64encode(objects),  # int64
        "objects_conf": base64.b64encode(objects_conf),  # float32
        "attrs_id": base64.b64encode(attrs),  # int64
        "attrs_conf": base64.b64encode(attrs_conf),  # float32
        "num_boxes": len(keep_boxes),
        "boxes": base64.b64encode(cls_boxes[keep_boxes]),  # float32
        "features": base64.b64encode(pool5[keep_boxes])  # float32
    }


def generate_tsv_bing(gpu_id, prototxt, weights, input_file, outfile, total_num_records):
    caffe.set_mode_cpu()
    # caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffe.TEST, weights=weights)
    total = int(total_num_records)
    count = 0
    with open(input_file) as f:
        print("input_file:", input_file)
        
        for item in f:
            try:
                item_arr = item.split('\t')
                if len(item_arr) > 13:
                    image_id = (item_arr[13]).rstrip()
                    im_file = (item_arr[0])
                    
                    # First check if file exists, and if it is complete
                    # wanted_ids=set([image_id[1] for image_id in image_ids])
                    # found_ids=set()
                    if os.path.exists(outfile):
                        with open(outfile) as tsvfile:
                            reader=csv.DictReader(tsvfile, delimiter='\t',
                                                fieldnames=FIELDNAMES)
                      
                with open(outfile, 'ab') as tsvfile:
                    writer=csv.DictWriter(tsvfile, delimiter='\t',
                                        fieldnames=FIELDNAMES)
                    _t={'misc': Timer()}
                    
                    # for im_file, image_id in image_ids:
                    #     if image_id in missing:
                    
                    try :
                        _t['misc'].tic()
                        im = readb64(im_file)
                        # Check if the image is more than 200, 200 as otherwise it leads to "Check failed: error == cudaSuccess (9 vs. 0) invalid configuration argument"
                        if (im.shape[0] >= 100 and im.shape[1] >= 100):
                            writer.writerow(get_detections_from_im_bing(
                                net, im, image_id))
                            _t['misc'].toc()
                            if (count % 100) == 0:
                                print 'GPU {:d}: {:d} {:.3f}s (projected finish: {:.2f} hours)' \
                                    .format(gpu_id, count+1, _t['misc'].average_time,
                                            _t['misc'].average_time*(total-count)/3600)
                            count += 1
                    except Exception as e:
                        print(e)
                        print("error_encountered for:", count)

            except Exception as e:
                print("exception in reading a")
                print(e)  

def merge_tsvs():
    test = ['/work/data/tsv/test2015/resnet101_faster_rcnn_final_test.tsv.%d' % i for i in range(8)]

    outfile = '/work/data/tsv/merged.tsv'
    with open(outfile, 'ab') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)   
        
        for infile in test:
            with open(infile) as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
                for item in reader:
                    try:
                      writer.writerow(item)
                    except Exception as e:
                      print e                           


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to use',
                        default=None, type=str)
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--split', dest='data_split',
                        help='dataset to use',
                        default='karpathy_train', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--input', dest = 'input_file', 
                        help = 'Input file', default = None)
    parser.add_argument('--lines', dest = 'total_lines', 
                        help = 'Total Lines', default = None)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

    

                      
     
if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    gpu_id = args.gpu_id
    gpu_list = gpu_id.split(',')
    gpus = [int(i) for i in gpu_list]

    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    # image_ids = load_image_ids(args.data_split)
    # random.seed(10)
    # random.shuffle(image_ids)
    # Split image ids between gpus
    # image_ids = [image_ids[i::len(gpus)] for i in range(len(gpus))]
    
    caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))
    procs = []    
    
    for i,gpu_id in enumerate(gpus):
        outfile = '%s.%d' % (args.outfile, gpu_id)
        p = Process(target=generate_tsv_bing,
                    args=(gpu_id, args.prototxt, args.caffemodel, args.input_file, outfile, args.total_lines))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()            
                  
