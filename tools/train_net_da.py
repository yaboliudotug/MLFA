# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import random
import argparse
import os
import numpy as np
import torch
from fcos_core.config import cfg
from fcos_core.data import make_data_loader, make_data_loader_4tta
from fcos_core.solver import make_lr_scheduler
from fcos_core.solver import make_optimizer
from fcos_core.engine.inference import inference
from fcos_core.engine.trainer import do_train
from fcos_core.modeling.backbone import build_backbone
from fcos_core.modeling.rpn.rpn import build_rpn, build_middle_head
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, get_rank
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir


def train(cfg, local_rank, distributed=False, test_only=False):
    ##########################################################################
    ############################# Initial MODEL ##############################
    ##########################################################################
    MODEL = {}
    device = torch.device(cfg.MODEL.DEVICE)
    backbone = build_backbone(cfg).to(device)
    fcos = build_rpn(cfg, backbone.out_channels, device).to(device)
    ##########################################################################
    #################### Initial Optimizer and Scheduler #####################
    ##########################################################################
    optimizer = {}
    scheduler = {}
    optimizer["backbone"] = make_optimizer(cfg, backbone, name='backbone')
    optimizer["fcos"] = make_optimizer(cfg, fcos, name='fcos')
    scheduler["backbone"] = make_lr_scheduler(cfg, optimizer["backbone"], name='backbone')
    scheduler["fcos"] = make_lr_scheduler(cfg, optimizer["fcos"], name='fcos')
    ##########################################################################
    ########################### Save MODEL to Dict ###########################
    ##########################################################################
    MODEL["backbone"] = backbone
    MODEL["fcos"] = fcos
    ##########################################################################
    ################################ Training ################################
    ##########################################################################
    arguments = {}
    arguments["iteration"] = 0
    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, MODEL, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(f=cfg.MODEL.WEIGHT)

    data_loader = {}
    data_set = {}
    
    data_loader["source_4tta"], data_set["source_4tta"] = make_data_loader_4tta(cfg, shuffle=True, drop_last=False, dataset_name='source', is_distributed=distributed)
    data_loader["val_4tta"], data_set["val_4tta"] = make_data_loader_4tta(cfg, shuffle=True, drop_last=False, dataset_name='val', is_distributed=distributed)

    if not test_only:
        do_train(
            MODEL,
            data_loader,
            optimizer,
            checkpointer,
            device,
            cfg,
        )
        # do_train(
        #     MODEL,
        #     data_loader,
        #     data_set,
        #     optimizer,
        #     scheduler,
        #     checkpointer,
        #     device,
        #     None,
        #     arguments,
        #     cfg,
        #     distributed,
        #     None,
        # )
    return MODEL


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run_test(cfg, MODEL, distributed):
    if distributed:
        MODEL["backbone"] = MODEL["backbone"].module
        MODEL["fcos"] = MODEL["fcos"].module

    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            cfg,
            MODEL,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.ATSS_ON or cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="./configs/debug_vis_c2f.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--test_only",
        dest="test_only",
        help="Test the input MODEL directly, without training",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--use_tensorboard",
        dest="use_tensorboard",
        help="Use tensorboardX logger (Requires tensorboardX installed)",
        action="store_true",
        default=True,
    )

    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("fcos_core", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    setup_seed(1234)
    MODEL = train(cfg, args.local_rank, args.distributed, args.test_only)

    if args.test_only:
        run_test(cfg, MODEL, args.distributed)

if __name__ == "__main__":
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    main()