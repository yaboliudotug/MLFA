# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os
class DatasetCatalog(object):
    DATA_DIR = "/disk/liuyabo/research/daod_sigma/datasets/"
    DATASETS = {
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2017_test_dev": {
            "img_dir": "coco/test2017",
            "ann_file": "coco/annotations/image_info_test-dev2017.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
        "keypoints_coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/person_keypoints_train2014.json",
        },
        "keypoints_coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_val2014.json"
        },
        "keypoints_coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_minival2014.json",
        },
        "keypoints_coco_2014_valminusminival": {
            "img_dir":
            "coco/val2014",
            "ann_file":
            "coco/annotations/person_keypoints_valminusminival2014.json",
        },
        "voc_2007_train": {
            "data_dir": "voc/VOC2007",
            "split": "train"
        },
        "voc_2007_train_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_train2007.json"
        },
        "voc_2007_val": {
            "data_dir": "voc/VOC2007",
            "split": "val"
        },
        "voc_2007_val_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_val2007.json"
        },
        "voc_2007_test": {
            "data_dir": "VOCdevkit/VOC2007",
            "split": "test"
        },
        "voc_2007_test_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_test2007.json"
        },
        "voc_2012_train": {
            "data_dir": "voc/VOC2012",
            "split": "train"
        },
        "voc_2012_train_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_train2012.json"
        },
        "voc_2012_val": {
            "data_dir": "voc/VOC2012",
            "split": "val"
        },
        "voc_2012_val_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_val2012.json"
        },
        "voc_2012_test": {
            "data_dir": "voc/VOC2012",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        },
        "cityscapes_train_cocostyle": {
            "img_dir": "Cityscapes/leftImg8bit/train",
            "ann_file": "Cityscapes/cocoAnnotations/cityscapes_train_cocostyle.json"
        },
        "cityscapes_train_caronly_cocostyle": {
            "img_dir": "Cityscapes/leftImg8bit/train",
            "ann_file": "Cityscapes/cocoAnnotations/cityscapes_train_caronly_cocostyle.json"
        },
        "cityscapes_val_caronly_cocostyle": {
            "img_dir": "Cityscapes/leftImg8bit/val",
            "ann_file": "Cityscapes/cocoAnnotations/cityscapes_val_caronly_cocostyle.json"
        },
        "cityscapes_foggy_train_cocostyle": {
            "img_dir": "Cityscapes/leftImg8bit_foggy/train",
            "ann_file": "Cityscapes/cocoAnnotations/cityscapes_foggy_train_cocostyle.json"
        },
        "cityscapes_val_cocostyle": {
            "img_dir": "Cityscapes/leftImg8bit/val",
            "ann_file": "Cityscapes/cocoAnnotations/cityscapes_val_cocostyle.json"
        },
        "cityscapes_foggy_val_cocostyle": {
            "img_dir": "Cityscapes/leftImg8bit_foggy/val",
            "ann_file": "Cityscapes/cocoAnnotations/cityscapes_foggy_val_cocostyle.json"
        },
        "sim10k_trainval_caronly": {
            "data_dir": "Sim10k",
            "split": "trainval10k_caronly"
        },
        "sim10k_splited_train_caronly": {
            "data_dir": "Sim10k",
            "split": "splited_train10k_caronly"
        },
        "sim10k_splited_val_caronly": {
            "data_dir": "Sim10k",
            "split": "splited_val10k_caronly"
        },
        "watercolor_train": {
            "data_dir": "watercolor",
            "split": "train"
        },
        "watercolor_test": {
            "data_dir": "watercolor",
            "split": "test"
        },
        "comic_train": {
            "data_dir": "comic",
            "split": "train"
        },
        "comic_test": {
            "data_dir": "comic",
            "split": "test"
        },
        "clipart_train": {
            "data_dir": "clipart/",
            "split": "train"
        },
        "clipart_test": {
            "data_dir": "clipart/",
            "split": "test"
        },
        "voc_2007_trainval_cocostyle": {
            "img_dir": "VOCdevkit/VOC2007/JPEGImages",
            "ann_file": "VOCdevkit/trainval_2007.json"
        },
        "voc_2012_trainval_cocostyle": {
            "img_dir": "VOCdevkit/VOC2012/JPEGImages",
            "ann_file": "VOCdevkit/trainval_2012.json"
        },
        "clipart_train_cocostyle": {
            "img_dir": "clipart/JPEGImages",
            "ann_file": "clipart/train_clipart.json"
        },
        "clipart_test_cocostyle": {
            "img_dir": "clipart/JPEGImages",
            "ann_file": "clipart/test_clipart.json"
        },
        "train_test_clipart_cocostyle": {
            "img_dir": "clipart/JPEGImages",
            "ann_file": "clipart/train_test_clipart.json"
        },
        "clipart_voc": {
            "data_dir": "clipart/",
            "split": "all"
        },
        "clipart_cocostyle": {
            "img_dir": "clipart/JPEGImages",
            "ann_file": "clipart/train_test_clipart.json"
        },
        "water_pascal_2007": {
            "data_dir": "VOCdevkit/VOC2007/",
            "split": "voc2007_water"
        },
        "water_pascal_2012": {
            "data_dir": "VOCdevkit/VOC2012/",
            "split": "voc2012_water"
        },
        "water_train": {
            "data_dir": "watercolor/watercolor",
            "split": "train"
        },
        "water_test": {
            "data_dir": "watercolor/watercolor",
            "split": "test"
        },
        "water_train_cocostyle": {
            "img_dir": "watercolor/JPEGImages",
            "ann_file": "watercolor/train.json"
        },
        "water_test_cocostyle": {
            "img_dir": "watercolor/JPEGImages",
            "ann_file": "watercolor/test.json"
        },
        "voc_2007_trainval": {
            "data_dir": "VOCdevkit/VOC2007/",
            "split": "trainval"
        },
        "voc_2012_trainval": {
            "data_dir": "VOCdevkit/VOC2012/",
            "split": "trainval"
        },
        "bdd100k_train_cocostyle": {
            "img_dir": "BDD100k/images/train/",
            "ann_file": "BDD100k/cocoAnnotations/bdd100k_train_da.json"
        },
        "bdd100k_val_cocostyle": {
            "img_dir": "BDD100k/images/val",
            "ann_file": "BDD100k/cocoAnnotations/bdd100k_val_da.json"
        },
        "voc_2007_cyclegan": {
            "data_dir": "style-transferred/style-transferred/VOC2007_to_clipart/",
            # "split": "da_transferred_train"
            "split": "train"
        },
        "voc_2012_cyclegan": {
            "data_dir": "style-transferred/style-transferred/VOC2012_to_clipart/",
            # "split": "da_transferred_train"
            "split": "train"
        },
        "voc_2007_cyclegan_water_color": {
            "data_dir": "style-transferred/style-transferred/VOC2007_to_watercolor/",
            # "split": "transferrd_train",
            "split": "train"
        },
        "voc_2012_cyclegan_water_color": {
            "data_dir": "style-transferred/style-transferred/VOC2012_to_watercolor/",
            # "split": "transferrd_train"
            "split": "train"
        },
        "voc_2007_cyclegan_water_color_test": {
            "data_dir": "style-transferred/style-transferred/VOC2007_to_watercolor/",
            "split": "val"
        },
        "voc_2012_cyclegan_water_color_test": {
            "data_dir": "style-transferred/style-transferred/VOC2012_to_watercolor/",
            "split": "val"
        },
        "voc_2007_cyclegan_comic": {
            "data_dir": "style-transferred/style-transferred/VOC2007_to_comic/",
            "split": "train"
            # "split": "transferred_train"
        },
        "voc_2012_cyclegan_comic": {
            "data_dir": "style-transferred/style-transferred/VOC2012_to_comic/",
            "split": "train"
            # "split": "transferred_train"
        },
        "kitti_train_caronly": {
            "data_dir": "KITTI",
            "split": "train_caronly"
        },
        "kitti_splited_train_caronly": {
            "data_dir": "KITTI",
            "split": "splited_train_caronly"
        },
        "kitti_splited_val_caronly": {
            "data_dir": "KITTI",
            "split": "splited_val_caronly"
        },

    }

    @staticmethod
    def get(name):
        if "kitti" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="KittiDataset",
                args=args,
            )
        elif "sim10k" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="Sim10kDataset",
                args=args,
            )
        elif "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "water" in name or 'comic' in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="WaterColorDataset",
                args=args,
            )
        elif "voc" in name or 'clipart' in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )

        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
        "FAIR/20171220/X-101-64x4d": "ImageNetPretrained/20171220/X-101-64x4d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
