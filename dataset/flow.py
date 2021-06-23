from itertools import chain
from typing import List
from dataclasses import dataclass
from .packet import Slice, Packet, slice_traffic


@dataclass
class FlowID:
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    proto: int

    @property
    def backward(self):
        return FlowID(
            src_ip=self.dst_ip, src_port=self.dst_port, dst_ip=self.src_ip, dst_port=self.dst_port, proto=self.proto)

    def __hash__(self):
        key = f'{self.src_ip}:{self.src_port} ->> {self.dst_ip}:{self.dst_port} : {self.proto}'
        return hash(key)


@dataclass
class Flow:
    id: FlowID
    forward_packets: Slice
    backward_packets: Slice

    timespan: float

    @property
    def length(self) -> float:
        all_p = self.forward_packets + self.backward_packets
        epoch_start = min([p.time for p in all_p])
        epoch_end = max([p.time for p in all_p])
        return float(epoch_end - epoch_start)


def get_descriptor(packet: Packet) -> FlowID:
    if 'IP' not in packet:
        return None
    ip = packet['IP']
    src_ip = ip.src
    dst_ip = ip.dst
    src_port = ip.sport
    dst_port = ip.dport
    proto = ip.proto
    return FlowID(src_ip=src_ip, src_port=src_port, dst_ip=dst_ip, dst_port=dst_port, proto=proto)


def extract_flows(slice: Slice, timespan: float) -> List[Flow]:
    def push_packet(d: dict, k: FlowID, v: Packet):
        if k in d:
            prev = d[k]
            prev.append(v)
        else:
            d[k] = [v]

    forward_packets = dict()
    backward_packets = dict()
    for p in slice:
        fid = get_descriptor(p)
        if fid is None:
            continue
        push_packet(forward_packets, fid, p)
        push_packet(backward_packets, fid.backward, p)

    keys = set(forward_packets.keys()).union(set(backward_packets.keys()))

    def construct_flow(key: FlowID) -> Flow:
        fp = forward_packets[key] if key in forward_packets else []
        bp = backward_packets[key] if key in backward_packets else []

        return Flow(id=key, forward_packets=fp, backward_packets=bp, timespan=timespan)

    return [construct_flow(k) for k in keys]


def construct_flows(traffic: List[Packet], slice_length: float = 15.0) -> List[Flow]:
    slices = slice_traffic(traffic, slice_length=slice_length)

    def f(slice: Slice):
        return extract_flows(slice, timespan=slice_length)

    fss = [f(x) for x in slices]

    return list(chain.from_iterable(fss))
