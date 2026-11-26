"""Service catalogue: forwarding latency l_n, download time M_n, normalized demands a_{n,r}.

Units: time in ms, bandwidth in Mbps, request sizes in kbit (kbit / Mbps = ms),
disk in MB (MB*8000 kbit / Mbps = ms), CPU in cores, RAM in GB.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Catalog:
    l: np.ndarray          # (N,) forwarding latency to cloud
    M: np.ndarray          # (N,) download time (request + transfer)
    a: np.ndarray          # (N,3) normalized demands in (0,1]: cpu, ram, disk
    raw: np.ndarray | None = None
    caps: np.ndarray | None = None

    @property
    def N(self) -> int:
        return len(self.l)


def from_physical(raw, caps=None, cap_mult=8.0, U=100.0, D=1000.0, rtt=20.0, req_kbit=100.0):
    """Build a catalogue from raw (N,3) demands [cpu, ram, disk_MB].

    l_n = rtt + req/U            (round trip + time to push the request over the uplink)
    M_n = rtt/2 + req/U + S_n/D  (M^req_n + S_n/D, eq. (26))
    caps default to max(cap_mult * mean demand, max demand) per resource.
    """
    raw = np.asarray(raw, float)
    N = raw.shape[0]
    if caps is None:
        caps = np.maximum(cap_mult * raw.mean(0), raw.max(0))
    caps = np.asarray(caps, float)
    a = raw / caps
    req = np.broadcast_to(np.asarray(req_kbit, float), (N,))
    l = rtt + req / U
    M = rtt / 2 + req / U + raw[:, 2] * 8000.0 / D
    return Catalog(l=l, M=M, a=a, raw=raw, caps=caps)
