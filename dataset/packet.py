from time import perf_counter
from typing import Optional
from dataclasses import dataclass
from typing import List, Tuple
from scapy.all import Packet, rdpcap


@dataclass
class SimplePacket:
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    proto: int
    timestamp: float
    length: int

    @staticmethod
    def from_packet(pkt: Packet) -> Optional['SimplePacket']:
        if 'IP' not in pkt:
            return None
        ip = pkt['IP']
        for k in ['src', 'dst', 'sport', 'dport', 'proto']:
            if not hasattr(ip, k):
                return None
        src_ip = ip.src
        dst_ip = ip.dst
        src_port = ip.sport
        dst_port = ip.dport
        proto = ip.proto
        timestamp = float(pkt.time)
        length = len(pkt)

        return SimplePacket(
            src_ip=src_ip,
            src_port=src_port,
            dst_ip=dst_ip,
            dst_port=dst_port,
            proto=proto,
            timestamp=timestamp,
            length=length
        )


Slice = List[SimplePacket]


def load_traffic(path: str) -> List[SimplePacket]:
    tic = perf_counter()
    log_interval = 2000
    i = 0
    traffic = []
    pcap = rdpcap(path)

    for p in pcap:
        i += 1
        p = SimplePacket.from_packet(p)
        if p is not None:
            traffic.append(p)

        if i % log_interval == 0:
            toc = perf_counter()
            print(f'loading {path}: {i} packets already loaded, {toc - tic:.4f} secs elapsed')

    return traffic


def slice_traffic(traffic: List[SimplePacket], slice_length: float = 10.0) -> List[Slice]:
    def divide_traffic(flow: List[SimplePacket]) -> Tuple[List[SimplePacket], List[SimplePacket]]:
        if len(flow) == 0:
            return [], []
        epoch_start = flow[0].timestamp
        epoch_end = epoch_start + slice_length
        i = 0
        while i < len(flow) and flow[i].timestamp <= epoch_end:
            i += 1
        return flow[:i], flow[i:]

    def recur(flow: List[SimplePacket], acc: List[Slice]) -> List[Slice]:
        if len(flow) == 0:
            return acc
        slice, rem = divide_traffic(flow)
        return recur(rem, acc + [slice] if len(slice) > 0 else acc)

    return recur(traffic, [])
