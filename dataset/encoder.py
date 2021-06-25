import torch
import numpy as np
from itertools import chain
from typing import List, Tuple
from .packet import Slice, SimplePacket
from .flow import Flow


class FeatureExtractor:
    def __init__(self):
        pass

    def _encode(self, flow: Flow, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, flow: Flow, *args, **kwargs):
        ret = self._encode(flow, *args, **kwargs)
        return ret


class BasicExtractor(FeatureExtractor):
    def _encode(self, flow: Flow, *args, **kwargs):
        return {
            'src_port': flow.id.src_port,
            'dst_port': flow.id.dst_port,
            'proto': flow.id.proto,
            'duration': flow.length
        }


class PacketNumExtractor(FeatureExtractor):
    def _encode(self, flow: Flow, *args, **kwargs):
        # fn = float(len(flow.forward_packets) / (flow.length + 1))
        # bn = float(len(flow.backward_packets) / (flow.length + 1))
        fn = len(flow.forward_packets)
        bn = len(flow.backward_packets)
        return {
            'f_pack_num': fn, 'b_pack_num': bn, 'pack_num_ratio': fn / (bn + 1.)
        }


class PacketLengthExtractor(FeatureExtractor):
    def _encode(self, flow: Flow, *args, **kwargs):
        def compute(ps: List[SimplePacket]):
            if len(ps) == 0:
                return 0.0, 0.0
            ls = [p.length for p in ps]
            return np.mean(ls), np.std(ls)

        f_mean, f_std = compute(flow.forward_packets)
        b_mean, b_std = compute(flow.backward_packets)
        # fb = float(sum(x.length for x in flow.forward_packets) / (flow.length + 1))
        # bb = float(sum(x.length for x in flow.backward_packets) / (flow.length + 1))
        return {
            'f_len_mean': f_mean, 'f_len_std': f_std,
            'b_len_mean': b_mean, 'b_len_std': b_std,
            'f_b_ratio': f_mean / (b_mean + 1.)
        }


class IntervalExtractor(FeatureExtractor):
    def _encode(self, flow: Flow, *args, **kwargs):
        def mean_interval(slice: Slice):
            if len(slice) <= 1:
                return 0.0, 0.0
            intervals = [y.timestamp - x.timestamp for x, y in zip(slice[:-1], slice[1:])]
            return np.mean(intervals), np.std(intervals)

        f_mean, f_std = mean_interval(flow.forward_packets)
        b_mean, b_std = mean_interval(flow.backward_packets)

        return {
            'f_itv_mean': f_mean, 'b_itv_mean': b_mean,
            'f_itv_std': f_std, 'b_itv_std': b_std,
        }


class FlowEncoder:
    def __init__(self, extractors: List[FeatureExtractor]):
        self.extractors = extractors

    def __call__(self, flow: Flow):
        feats = dict()
        for extractor in self.extractors:
            feats.update(extractor(flow))

        return feats
