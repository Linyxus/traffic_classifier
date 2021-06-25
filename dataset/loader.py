from time import perf_counter
import os
import os.path as osp

import ray
from scapy.all import rdpcap
from typing import List

import pandas as pd

from .encoder import *
from .flow import construct_flows
from .packet import load_traffic


DEFAULT_CLASS_LIST = 'Audio_Streaming Browsing Chat Email File_Transfer P2P Video_Streaming VoIP'.split(' ')


ALL_EXTRACTORS = {
    'basic': BasicExtractor,
    'packet_num': PacketNumExtractor,
    'packet_length': PacketLengthExtractor,
    'interval': IntervalExtractor
}


DEFAULT_EXTRACTOR_LIST = ['basic', 'packet_num', 'packet_length', 'interval']


class TrafficDataset:
    def __init__(self, root: str, classes: List[str] = None, extractors: List[str] = None, slice_length: float = 15.0):
        if classes is None:
            classes = DEFAULT_CLASS_LIST
        if extractors is None:
            extractors = DEFAULT_EXTRACTOR_LIST

        self.classes = classes
        self.c2i = {k: v for v, k in enumerate(self.classes)}
        self.extractors = [ALL_EXTRACTORS[f]() for f in extractors]
        self.encoder = FlowEncoder(self.extractors)

        self.slice_length = slice_length
        self.root_dir = root

    @property
    def raw_dir(self):
        return osp.join(self.root_dir, 'raw')

    @property
    def processed_dir(self):
        res = osp.join(self.root_dir, 'processed')
        os.makedirs(res, exist_ok=True)
        return res

    @property
    def processed_file(self):
        filename = f'data-{self.slice_length}.csv'
        return osp.join(self.processed_dir, filename)

    def encode_flow(self, flow: Flow, label: str):
        res = self.encoder(flow)
        res['label'] = label
        return res

    def encode_flows(self, flows: List[Flow], label: str) -> List[dict]:
        @ray.remote
        def f(flow: Flow):
            return self.encode_flow(flow, label)

        print(f'encoding {len(flows)} flows')
        res = [f.remote(x) for x in flows]
        res = ray.get(res)
        return res

    def encode_traffic(self, traffic: List[SimplePacket], label: str):
        flows = construct_flows(traffic, slice_length=self.slice_length)
        res = self.encode_flows(flows, label=label)
        return res

    def scan_dir(self):
        res = []
        for c in self.classes:
            i = self.c2i[c]
            base_dir = osp.join(self.raw_dir, c)
            for root, _, files in os.walk(base_dir):
                for name in files:
                    path = osp.join(root, name)
                    res.append((path, i))
        return res

    def load_dataset(self):
        data_paths = self.scan_dir()

        datas = []

        @ray.remote
        def f(info):
            path, label = info
            print(f'[TODO] start loading {path} ...')
            tic = perf_counter()
            traffic = load_traffic(path)
            toc = perf_counter()
            print(f'[LOAD] file {path}, {len(traffic)} packets in total, {toc - tic:.4f} secs elapsed')
            tic = perf_counter()
            x = self.encode_traffic(traffic, label)
            toc = perf_counter()
            print(f'[DONE] processing {path} in {toc - tic:.4f} secs')
            return x

        datas = [f.remote(info) for info in data_paths]
        datas = ray.get(datas)

        data = list(chain.from_iterable(datas))
        df = pd.DataFrame(data)

        df.to_csv(self.processed_file)

        return df
