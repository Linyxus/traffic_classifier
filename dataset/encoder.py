import torch
from itertools import chain
from typing import List
from .packet import Slice
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
        fn = float(len(flow.forward_packets) / (flow.length + 1))
        bn = float(len(flow.backward_packets) / (flow.length + 1))
        return {
            'f_pack_num': fn, 'b_pack_num': bn, 'pack_num_ratio': fn / (bn + 1.)
        }


class PacketLengthExtractor(FeatureExtractor):
    def _encode(self, flow: Flow, *args, **kwargs):
        fb = float(sum(len(x) for x in flow.forward_packets) / (flow.length + 1))
        bb = float(sum(len(x) for x in flow.backward_packets) / (flow.length + 1))
        return {
            'f_byte_num': fb, 'b_byte_num': bb, 'byte_num_ratio': fb / (bb + 1.)
        }


class IntervalExtractor(FeatureExtractor):
    def _encode(self, flow: Flow, *args, **kwargs):
        def mean_interval(slice: Slice) -> float:
            if len(slice) <= 1:
                return 0.0
            intervals = [y.time - x.time for x, y in zip(slice[:-1], slice[1:])]
            return sum(intervals) / len(intervals)

        fi = float(mean_interval(flow.forward_packets))
        bi = float(mean_interval(flow.backward_packets))

        return {
            'f_itv': fi, 'b_itv': bi
        }


class FlowEncoder:
    def __init__(self, extractors: List[FeatureExtractor]):
        self.extractors = extractors

    def __call__(self, flow: Flow):
        feats = dict()
        for extractor in self.extractors:
            feats.update(extractor(flow))

        return feats
