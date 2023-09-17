import argparse
import torch
import os
import os.path as osp
import cv2
from loguru import logger
from tracker.utils.parser import get_config
from tracker.utils.timer import Timer
from yolox.utils import get_model_info, fuse_model, postprocess
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils.visualize import plot_tracking
from tracker.LG_Track import LG_Track

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
trackerTimer = Timer()
timer = Timer()

def parse_args():
    parser = argparse.ArgumentParser(description='LG_Track')
    parser.add_argument('--datasets', type=str, default='MOT17', help='MOT17, MOT20')
    parser.add_argument('--split', type=str, default='train', help='train, test')
    parser.add_argument("--default-parameters", type=str, default=False, help="use the default parameters as in the paper")
    args = parser.parse_args()
    return args


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            device=torch.device("cuda"),
            fp16=False
    ):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        if img is None:
            raise ValueError("Empty image: ", img_info["file_name"])

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)

        return outputs, img_info

def image_track(predictor, cfg, args):

    img_path = os.path.join(cfg.data_path, args.split, args.seq_name, 'img1')
    files = get_image_list(img_path)
    files.sort()
    num_frames = len(files)

    tracker = LG_Track(cfg, args, frame_rate=30)
    results = []
    out_dets = []
    for frame_id, img_path in enumerate(files, 1):

        outputs, img_info = predictor.inference(img_path, timer)
        scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))
        img = img_info["raw_img"]
        if outputs[0] is not None:
            outputs = outputs[0].cpu().numpy()
            detections = outputs[:, :7]
            detections[:, :4] /= scale
        if cfg.save_det_results:
            for i, det in enumerate(detections):
                out_dets.append(
                    f"{frame_id},-1,{det[0]:.2f},{det[1]:.2f},{det[2]:.2f},{det[3]:.2f},{det[4]*det[5]:.2f},{det[4]:.2f},{det[5]:.2f}\n"
                )

        trackerTimer.tic()
        online_targets = tracker.update(detections, img)
        trackerTimer.toc()

        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > cfg.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > cfg.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                # save results
                results.append(
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                )
        timer.toc()
        if cfg.save_img:
            online_im = plot_tracking(
                img, online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
            img_folder = osp.join(cfg.output_root, 'trk_img', args.seq_name)
            os.makedirs(img_folder, exist_ok=True)
            cv2.imwrite(osp.join(img_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info(
                'Processing frame {}/{} ({:.2f} fps)'.format(frame_id, num_frames, 1. / max(1e-5, timer.average_time)))

    if cfg.save_det_results:
        det_folder = osp.join(cfg.output_root, 'detections')
        os.makedirs(det_folder, exist_ok=True)
        det_file = osp.join(det_folder, args.seq_name + '.txt')
        with open(det_file, 'w') as f:
            f.writelines(out_dets)

    folder = osp.join(cfg.output_root, 'results')
    os.makedirs(folder, exist_ok=True)
    file = osp.join(folder, args.seq_name + '.txt')
    with open(file, 'w') as f:
        f.writelines(results)
    logger.info(f"save results to {file}")

def main(exp, cfg, args):

    if cfg.conf is not None:
        exp.test_conf = cfg.conf
    if cfg.nms is not None:
        exp.nmsthre = cfg.nms
    model = exp.get_model().to(torch.device(cfg.device))
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    ckpt_file = cfg.det_ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")

    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    if cfg.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)
    if cfg.fp16:
        model = model.half()

    predictor = Predictor(model, exp, torch.device(cfg.device), cfg.fp16)
    image_track(predictor, cfg, args)


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config()
    args.config_path = './configs/%s.yaml' % args.datasets
    cfg.merge_from_file(args.config_path)
    mainTimer = Timer()
    mainTimer.tic()
    if args.split == 'train':
        for seq in cfg.train_seqs:
            for name in cfg.det_name:
                if name == '':
                    args.seq = seq
                    args.seq_name = args.datasets + '-' + seq
                    exp = get_exp(cfg.exp_file, args.seq_name)
                    main(exp, cfg, args)
                else:
                    args.seq = seq
                    args.seq_name = args.datasets + '-' + seq + '-' + name
                    exp = get_exp(cfg.exp_file, args.seq_name)
                    main(exp, cfg, args)
    else:
        for seq in cfg.test_seqs:
            for name in cfg.det_name:
                if name == '':
                    args.seq = seq
                    args.seq_name = args.datasets + '-' + seq
                    exp = get_exp(cfg.exp_file, args.seq_name)
                    if args.default_parameters:
                        if args.seq_name == 'MOT20-06' or 'MOT20-08':
                            exp.test_size = (736, 1920)
                    main(exp, cfg, args)
                else:
                    args.seq = seq
                    args.seq_name = args.datasets + '-' + seq + '-' + name
                    exp = get_exp(cfg.exp_file, args.seq_name)
                    main(exp, cfg, args)
    mainTimer.toc()
    print("TOTAL TIME END-to-END (with loading networks and images): ", mainTimer.total_time)
    print("TOTAL TIME (Detector + Tracker): " + str(timer.total_time) + ", FPS: " + str(1.0 / timer.average_time))
