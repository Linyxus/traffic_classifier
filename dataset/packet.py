from typing import List, Tuple
from scapy.all import Packet


Slice = List[Packet]


def slice_traffic(flow: List[Packet], slice_length: float = 10.0) -> List[Slice]:
    def divide_traffic(flow: List[Packet]) -> Tuple[List[Packet], List[Packet]]:
        if len(flow) == 0:
            return [], []
        epoch_start = flow[0].time
        epoch_end = epoch_start + slice_length
        i = 0
        while i < len(flow) and flow[i].time <= epoch_end:
            i += 1
        return flow[:i], flow[i:]

    def recur(flow: List[Packet], acc: List[Slice]) -> List[Slice]:
        if len(flow) == 0:
            return acc
        slice, rem = divide_traffic(flow)
        return recur(rem, acc + [slice] if len(slice) > 0 else acc)

    return recur(flow, [])
