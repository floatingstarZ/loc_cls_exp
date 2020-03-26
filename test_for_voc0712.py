from argparse import ArgumentParser

import mmcv
import numpy as np

from mmdet import datasets
from mmdet.core import eval_map
def parse_args(args=''):
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    if args == '':
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    return args

def voc_eval(result_file, dataset, iou_thr=0.5):
    det_results = mmcv.load(result_file)
    gt_bboxes = []
    gt_labels = []
    gt_ignore = []
    for i in range(len(dataset)):
        ann = dataset.get_ann_info(i)
        bboxes = ann['bboxes']
        labels = ann['labels']
        if 'bboxes_ignore' in ann:
            ignore = np.concatenate([
                np.zeros(bboxes.shape[0], dtype=np.bool),
                np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
            ])
            gt_ignore.append(ignore)
            bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
            labels = np.concatenate([labels, ann['labels_ignore']])
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
    if not gt_ignore:
        gt_ignore = None
    if hasattr(dataset, 'year') and dataset.year == 2007:
        dataset_name = 'voc07'
    else:
        dataset_name = dataset.CLASSES
    eval_map(
        det_results,
        gt_bboxes,
        gt_labels,
        gt_ignore=gt_ignore,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        print_summary=True)

def voc_coco_eval(result_file, dataset, iou_thr=0.5, pass_difficult=True):
    det_results = mmcv.load(result_file)
    gt_bboxes = []
    gt_labels = []
    gt_ignore = []
    for i in range(len(dataset)):

        ann = dataset.get_ann_info(i)
        org_bboxes = ann['bboxes']
        org_labels = ann['labels']
        if 'others' in ann.keys():
            bboxes = []
            labels = []
            bboxes_ignore = []
            labels_ignore = []
            for i, d in enumerate(ann['others']):
                # if pass_difficult:
                #     continue
                d = d['difficult']
                if not d:
                    bboxes.append(org_bboxes[i])
                    labels.append(org_labels[i])
                else:
                    bboxes_ignore.append(org_bboxes[i])
                    labels_ignore.append(org_labels[i])
            bboxes = np.array(bboxes)
            labels = np.array(labels)
            bboxes_ignore = np.array(bboxes_ignore)
            labels_ignore = np.array(labels_ignore)

            ignore = np.concatenate([
                np.zeros(bboxes.shape[0], dtype=np.bool),
                np.ones(bboxes_ignore.shape[0], dtype=np.bool)
            ])
            gt_ignore.append(ignore)
            if len(bboxes_ignore) > 0:
                bboxes = np.vstack([bboxes, bboxes_ignore])
                labels = np.concatenate([labels, labels_ignore])
        else:
            bboxes = org_bboxes
            labels = org_labels
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
    if not gt_ignore:
        gt_ignore = None
    dataset_name = 'voc07'
    eval_map(
        det_results,
        gt_bboxes,
        gt_labels,
        gt_ignore=gt_ignore,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        print_summary=True)


def main():
    args = ['./results/Retina_r50_voc.pkl',
            './configs/voc/retinanet_r50_fpn_1x_voc_coco.py']
    args = parse_args(args)
    print(args)
    cfg = mmcv.Config.fromfile(args.config)
    test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
    voc_coco_eval(args.result, test_dataset, args.iou_thr)


if __name__ == '__main__':
    main()
