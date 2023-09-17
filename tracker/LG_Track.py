import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import deque
from scipy.spatial.distance import cdist
from tracker import matching
from tracker.gmc import GMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter

from fast_reid.fast_reid_interfece import FastReIDInterface


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, pos, cls, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.pos = pos
        self.cls = cls
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9
        self.time = 0

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

        self.score = new_track.score
        self.pos = new_track.pos
        self.cls = new_track.cls

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.pos = new_track.pos
        self.cls = new_track.cls

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class LG_Track(object):
    def __init__(self, cfg, args, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.cfg = cfg
        self.args = args
        self.cls = cfg.cls
        self.pos = cfg.pos
        self.match_thresh_a = self.cfg.match_thresh_a
        self.match_thresh_b = self.cfg.match_thresh_b
        self.match_thresh_c = self.cfg.match_thresh_c
        self.match_thresh_d = self.cfg.match_thresh_d
        self.new_track_thresh = cfg.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * cfg.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        self.encoder = FastReIDInterface(cfg.reid_config, cfg.reid_weights, torch.device(cfg.device))

        self.gmc = GMC(method=cfg.cmc_method, verbose=[args.seq_name, False])

    def update(self, output_results, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 9:
                detections = []
                for i, det in enumerate(output_results):
                    if det[7] > 0.1:
                        detections.append(det)
                if detections != []:
                    detections = np.asarray(detections)
                    bboxs = detections[:, 2:6]
                else:
                    bboxs = []
            else:
                detections = []
                for i, det in enumerate(output_results):
                    if det[4] > 0.1:
                        detections.append(det)
                if detections != []:
                    detections = np.asarray(detections)
                    bboxs = detections[:, :4]
                else:
                    bboxs = []
        else:
            bboxs = []
            detections = []

        '''Extract embeddings '''
        features_keep = self.encoder.inference(img, bboxs)

        if len(detections) > 0:
            if output_results.shape[1] == 9:
                dets = [STrack(STrack.tlbr_to_tlwh(tlbr), s, p, c, f) for
                       (tlbr, s, p, c, f) in zip(bboxs, detections[:, 6], detections[:, 7], detections[:, 8], features_keep)]
            else:
                dets = [STrack(STrack.tlbr_to_tlwh(tlbr), s, p, c, f) for
                        (tlbr, s, p, c, f) in zip(bboxs, detections[:, 4] * detections[:, 5], detections[:, 4], detections[:, 5], features_keep)]
        else:
            dets = []

        if self.args.default_parameters:
            if self.args.datasets == 'MOT20':
                if self.args.seq == '06' or self.args.seq == '08':
                    self.new_track_thresh = 0.4
                    self.pos = 0.3
                    self.cls = 0.65
                    self.match_thresh_a = self.match_thresh_b = 0.75
            if self.args.datasets == 'MOT17':
                if self.args.seq == '05' or self.args.seq == '06':
                    self.max_time_lost = 14
                if self.args.seq == '13':
                    self.max_time_lost = 25
                if self.args.seq == '01' or self.args.seq == '06':
                    self.new_track_thresh = 0.75
                if self.args.seq == '12':
                    self.new_track_thresh = 0.8

        '''selected detections'''
        high_dets = []
        high_pos = []
        high_cls = []
        low_dets = []
        for i, det in enumerate(dets):
            if det.pos >= self.pos and det.cls >= self.cls:
                high_dets.append(det)
            if det.pos >= self.pos and det.cls < self.cls:
                high_pos.append(det)
            if det.pos < self.pos and det.cls >= self.cls:
                high_cls.append(det)
            if det.pos < self.pos and det.cls < self.cls:
                low_dets.append(det)

        tracked_stracks = [track for track in self.tracked_stracks if track.is_activated]
        unconfirmed = [track for track in self.tracked_stracks if not track.is_activated]
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        # Fix camera motion
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        ''' the 1st association '''
        trks_a = joint_stracks(strack_pool, unconfirmed)
        ious_dists = matching.iou_distance(trks_a, high_dets, pos=True)
        match_a, u_trk_a, u_det_a = matching.linear_assignment(ious_dists, thresh=self.match_thresh_a)
        for itrk, idet in match_a:
            track = trks_a[itrk]
            det = high_dets[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # supplement
        trks_sup = [trks_a[i] for i in u_trk_a]
        dets_sup = [high_dets[i] for i in u_det_a]
        emb_dists_sup = matching.embedding_distance(trks_sup, dets_sup, score=True)
        dists_sup = matching.gate_cost_matrix(self.kalman_filter, emb_dists_sup, trks_sup, dets_sup)
        match_sup, u_trk_sup, u_det_sup = matching.linear_assignment(dists_sup, thresh=self.match_thresh_c)
        for itrk, idet in match_sup:
            track = trks_sup[itrk]
            det = dets_sup[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Delete the unconfirmed which is un_matched
        for j in u_trk_sup:
            track = trks_sup[j]
            if not track.is_activated:
                track.mark_removed()
                removed_stracks.append(track)

        ''' the 2nd association '''
        dets_b = [dets_sup[i] for i in u_det_sup] + high_pos
        trks_b = [trks_sup[i] for i in u_trk_sup if trks_sup[i].is_activated]
        dists_b = matching.iou_distance(trks_b, dets_b, pos=True)
        match_b, u_trk_b, u_det_b = matching.linear_assignment(dists_b, thresh=self.match_thresh_b)
        for itrk, idet in match_b:
            track = trks_b[itrk]
            det = dets_b[idet]
            det_features = np.asarray([det.curr_feat], dtype=np.float)
            track_features = np.asarray([track.smooth_feat], dtype=np.float)
            cost_matrix = cdist(track_features, det_features, metric='cosine')
            # Appearance restriction
            if cost_matrix[0][0] < 0.5:
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            else:
                u_det_b = u_det_b.tolist()
                u_det_b.append(idet)
                u_det_b = np.asarray(u_det_b)
                u_trk_b = u_trk_b.tolist()
                u_trk_b.append(itrk)
                u_trk_b = np.asarray(u_trk_b)

        ''' the 3rd association '''
        trks_c = [trks_b[i] for i in u_trk_b]
        dets_c = high_cls
        emb_dists_c = matching.embedding_distance(trks_c, dets_c, score=True)
        dists_c = matching.gate_cost_matrix(self.kalman_filter, emb_dists_c, trks_c, dets_c)
        match_c, u_trk_c, u_det_c = matching.linear_assignment(dists_c, thresh=self.match_thresh_c)
        for itrk, idet in match_c:
            track = trks_c[itrk]
            det = dets_c[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' the 4th association '''
        dets_d = [dets_c[i] for i in u_det_c] + [dets_b[j] for j in u_det_b] + low_dets
        trks_d = [trks_c[i] for i in u_trk_c if trks_c[i].state == TrackState.Tracked]
        emb_d = matching.embedding_distance(trks_d, dets_d, score=True)
        iou = matching.iou_distance(trks_d, dets_d, pos=True)
        cost = 0.5 * emb_d + 0.5 * iou
        match_d, u_trk_d, u_det_d = matching.linear_assignment(cost, thresh=self.match_thresh_d)
        for itrk, idet in match_d:
            track = trks_d[itrk]
            det = dets_d[idet]
            track.update(det, self.frame_id)
            activated_starcks.append(track)

        for i in u_trk_d:
            track = trks_d[i]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        ''' Initialization trajectory '''
        for i in u_det_d:
            det = dets_d[i]
            if det.score >= self.new_track_thresh:
                det.activate(self.kalman_filter, self.frame_id)
                activated_starcks.append(det)
        ''' Delete Lost tracks that exceed the threshold '''
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        output_stracks = [track for track in self.tracked_stracks]
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb, pos=True)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
