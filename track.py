import argparse
import os
import os.path as osp
import cv2
import numpy as np
from loguru import logger
from tracker.utils.parser import get_config
from tracker.utils.timer import Timer
from yolox.utils.visualize import plot_tracking
from tracker.LG_Track import LG_Track

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
trackerTimer = Timer()
timer = Timer()


def parse_args():
    parser = argparse.ArgumentParser(description='LG_Track')
    parser.add_argument('--datasets', type=str, default='MOT17', help='MOT17, MOT20')
    parser.add_argument('--split', type=str, default='test', help='train, test')
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


def image_track(dets, cfg, args):

    img_path = os.path.join(cfg.data_path, args.split, args.seq_name, 'img1')
    files = get_image_list(img_path)
    files.sort()
    num_frames = len(files)

    tracker = LG_Track(cfg, args, frame_rate=30)
    results = []
    for frame_id, img_path in enumerate(files, 1):

        detections = dets[dets[:, 0] == frame_id]
        img = cv2.imread(img_path)

        timer.tic()
        online_targets = tracker.update(detections, img)

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

    folder = osp.join(cfg.output_root, 'results')
    os.makedirs(folder, exist_ok=True)
    file = osp.join(folder, args.seq_name + '.txt')
    with open(file, 'w') as f:
        f.writelines(results)
    logger.info(f"save results to {file}")


def main(cfg, args):

    input_path = os.path.join(cfg.det_file, f"{args.seq_name}.txt")
    dets = np.loadtxt(input_path, delimiter=',')
    image_track(dets, cfg, args)


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
                    main(cfg, args)
                else:
                    args.seq = seq
                    args.seq_name = args.datasets + '-' + seq + '-' + name
                    main(cfg, args)
    else:
        for seq in cfg.test_seqs:
            for name in cfg.det_name:
                if name == '':
                    args.seq = seq
                    args.seq_name = args.datasets + '-' + seq
                    main(cfg, args)
                else:
                    args.seq = seq
                    args.seq_name = args.datasets + '-' + seq + '-' + name
                    main(cfg, args)
    mainTimer.toc()
    print("TOTAL TIME END-to-END (with loading networks and images): ", mainTimer.total_time)
    print("TOTAL TIME (Detector + Tracker): " + str(timer.total_time) + ", FPS: " + str(1.0 / timer.average_time))
