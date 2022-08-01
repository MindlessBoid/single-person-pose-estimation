import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import itertools

def eval_OKS(predictions, gt_path):
  # prediciton can be a list or path of json file
  oks_preds = []
  image_ids = []

  if type(predictions) == str:
    predictions = _load_predictions(predictions)
  
  for p in predictions:
    ann_id = p['ann_id']
    image_id = p['image_id']
    
    xs = p['xs/pred']
    ys = p['ys/pred']
    coco_kpts = []
    for x, y in zip(xs, ys):
      coco_kpts.append(int(x))
      coco_kpts.append(int(y))
      coco_kpts.append(1) # doesnt matter but always set to 1

    confs = p['confs']
    score = np.mean(confs)

    oks = _create_oks_obj(ann_id, image_id, coco_kpts, score)

    image_ids.append(image_id)
    oks_preds.append(oks)


  # Run coco eval
  cocoGt= COCO(gt_path)
  cocoDt = cocoGt.loadRes(oks_preds)

  annType = "keypoints"
  cocoEval = COCOeval(cocoGt,cocoDt,annType)
  cocoEval.params.imgIds = image_ids
  cocoEval.params.catIds = [1] # Person category
  cocoEval.evaluate()
  cocoEval.accumulate()
  print('\nSummary: ')
  cocoEval.summarize()
  
  return cocoEval.stats

def eval_PCK(predictions, keypoint_labels, pck_threshold = 0.05):
  '''
    Use bbox instead of torso or head box
  '''
  if type(predictions) == str:
    predictions = _load_predictions(predictions)

  correct_kps = dict(zip(keypoint_labels, itertools.repeat(0)))
  vis_kps = dict(zip(keypoint_labels, itertools.repeat(0)))
  for p in predictions:
    bbox = p['original_bbox']
    xs_pred = p['xs/pred']
    ys_pred = p['ys/pred']
    xs_gt = p['xs/gt']
    ys_gt = p['ys/gt']
    vs = p['vs']
    
    diameter = np.sqrt(bbox[2]**2 + bbox[3]**2)
    threshold = pck_threshold*diameter
    '''threshold = 0.2 * 25 # if both diameters aer not avail

    # torso diameter: left_hip index 11  to right_hip index 12
    if vs[11] > 0 and vs[12] > 0:
      d = np.sqrt((xs_gt[11]-xs_gt[12])**2 + (ys_gt[11]-ys_gt[12])**2)
      threshold = 0.5*d
    # head bone: left ear index 1, right ear index 2
    elif vs[1] > 0 and vs[2] > 0:
      d = np.sqrt((xs_gt[1]-xs_gt[2])**2 + (ys_gt[1]-ys_gt[2])**2)
      threshold = 0.2*d
    '''
    for x0, y0, x1, y1, v, label in zip(xs_gt, ys_gt, xs_pred, ys_pred, vs, keypoint_labels):
      if v > 0:
        dist = np.sqrt((x0-x1)**2 + (y0-y1)**2)
        vis_kps[label] += 1
        if dist <= threshold:
          correct_kps[label] += 1

  stats = []
  for label in keypoint_labels:
    percent = correct_kps[label]/vis_kps[label]
    stats.append(percent)
    print(f'{label}: {percent:.2f}%')

  return stats
  

def predict_ds(compiled_model, ds, ds_length, batch_size, heatmaps_to_keypoints_func, save_path = 'result.json', conf_threshold = 1e-6):
  num_iters = int(np.ceil(ds_length/batch_size))
  it = iter(ds)
  predictions = []

  for i in range(num_iters):
    images_batch, meta = next(it)
    pred_heatmaps_batch = compiled_model.predict(images_batch)

    for j, hms in enumerate(pred_heatmaps_batch[-1]): # only last output of the model
      
      prediction = {}

      kpts = heatmaps_to_keypoints_func(hms, conf_threshold = conf_threshold)
      # xs, ys
      xs_pred = kpts[:, 0]/hms.shape[1]# normalize
      ys_pred = kpts[:, 1]/hms.shape[0] # normalize
      vs = meta['keypoints/vis'][j].numpy()
      bbox_w = int(meta['bbox_w'][j])
      bbox_h = int(meta['bbox_h'][j])
      bbox_x = float(meta['bbox_x'][j])
      bbox_y = float(meta['bbox_y'][j])
      original_bbox = meta['original_bbox'][j].numpy()
      xs_gt = (meta['keypoints/x'][j] / bbox_w).numpy()
      ys_gt = (meta['keypoints/y'][j] / bbox_h).numpy()

      adjusted_xs_pred, adjusted_ys_pred = _undo_bbox(bbox_x, bbox_y, bbox_w, bbox_h, xs_pred, ys_pred)
      adjusted_xs_gt, adjusted_ys_gt = _undo_bbox(bbox_x, bbox_y, bbox_w, bbox_h, xs_gt, ys_gt)
      # conf
      confs = kpts[:, 2]

      # dont use np types otherwise its hard to write to json
      prediction['xs/pred'] = adjusted_xs_pred.astype(dtype=float).tolist()
      prediction['ys/pred'] = adjusted_ys_pred.astype(dtype=float).tolist()
      prediction['xs/gt'] = adjusted_xs_gt.astype(dtype=float).tolist()
      prediction['ys/gt'] = adjusted_ys_gt.astype(dtype=float).tolist()
      prediction['vs'] = vs.astype(dtype=int).tolist()
      prediction['confs'] = confs.astype(dtype=float).tolist()
      prediction['image_id'] = int(meta['image_id'][j])
      prediction['ann_id'] = int(meta['ann_id'][j])
      prediction['original_bbox'] = original_bbox.astype(dtype=float).tolist()

      predictions.append(prediction)

  # save
  _save_predictions(predictions, save_path) 

  return predictions


######################### private ####################



def _undo_bbox(x, y, width, height, normalized_xs, normalized_ys):
 
  undo_xs = normalized_xs * width + x
  undo_ys = normalized_ys * height + y

  return undo_xs, undo_ys  

def _create_oks_obj(ann_id, image_id, pred_kpts, score):
  oks_obj = {}
  oks_obj['image_id'] = image_id
  oks_obj['ann_id'] = ann_id
  oks_obj['category_id'] = 1
  oks_obj['keypoints'] = pred_kpts
  oks_obj['score'] = score
  return oks_obj

def _save_predictions(predictions, path):
  with open(path, 'w') as outfile:
    json.dump(predictions, outfile)

def _load_predictions(path):
  with open(path, 'r') as openfile:
    predictions = json.load(openfile)
  return predictions


