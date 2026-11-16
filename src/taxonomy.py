"""Fault categories used by the symptom profiler, the retriever and the evaluators.

SOFI only labels samples as fault / no-fault, so the category is inferred from the
symptom signature (see ``symptoms.py``). Each category lists:

* ``proposal_faults`` - the wired-network faults from the project proposal it covers,
* ``query_terms``     - expansion text used when retrieval confidence is low,
* ``relevance``       - regexes defining which KB passages count as relevant for the
  retrieval evaluation (a passage must match at least ``min_matches`` of them).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaultCategory:
    key: str
    title: str
    description: str
    proposal_faults: tuple[str, ...]
    query_terms: str
    relevance: tuple[str, ...]
    min_matches: int = 2


CATEGORIES: dict[str, FaultCategory] = {c.key: c for c in [
    FaultCategory(
        "link_down", "Link / interface down",
        "One or more monitored interfaces report an operational status other than up.",
        ("link down", "cable cut", "line card failure"),
        "interface operational status down ifOperStatus link down lowerLayerDown carrier loss "
        "cable unplugged transceiver failure line card",
        (r"ifOperStatus", r"\boper(ational)?\s*status", r"link\s*(is\s*)?down", r"lowerLayerDown",
         r"\bcarrier\b", r"linkDown", r"\bnotPresent\b", r"interface\s+(is\s+)?down"),
    ),
    FaultCategory(
        "device_unreachable", "Device unreachable / management plane lost",
        "The switch stops answering ICMP echo or SNMP polls.",
        ("line card failure", "protocol issues", "misconfiguration"),
        "device unreachable ping fails ICMP echo no reply destination unreachable SNMP timeout "
        "management plane loss node failure",
        (r"\bping\b", r"\becho\b", r"unreachable", r"\bICMP\b", r"time(d)?[- ]?out", r"\bSNMP\b",
         r"no (reply|response)"),
    ),
    FaultCategory(
        "device_restart", "Device restart",
        "The switch uptime counter reset, indicating a reload or crash.",
        ("line card failure",),
        "device reboot reload crash sysUpTime reset coldStart warmStart power failure",
        (r"sysUpTime", r"coldStart", r"warmStart", r"re(boot|load|start|initiali[sz])",
         r"discontinuit", r"\bcrash"),
    ),
    FaultCategory(
        "traffic_loss", "Traffic loss with links up (black-holing)",
        "Traffic through the switch collapses while interfaces stay operationally up, "
        "e.g. a downstream cable cut, forwarding or configuration fault.",
        ("cable cut", "misconfiguration", "protocol issues"),
        "traffic drop throughput collapse black hole packets not forwarded interface up but no traffic "
        "packet loss forwarding failure routing loop spanning tree blocking misconfiguration",
        (r"packet loss", r"\bloss\b", r"black ?hol", r"forward(ing)?", r"throughput", r"\bdrop",
         r"spanning tree|\bSTP\b|blocking", r"routing loop|\bloop\b", r"misconfigur"),
    ),
    FaultCategory(
        "interface_errors", "Interface errors (faulty cabling / duplex mismatch)",
        "Inbound or outbound error counters rise above their baseline.",
        ("faulty cabling", "misconfiguration"),
        "input errors CRC FCS errors alignment errors runts late collisions duplex mismatch "
        "autonegotiation faulty cable bad transceiver ifInErrors",
        (r"\bCRC\b", r"\bFCS\b", r"ifInErrors|ifOutErrors|in errors|input errors", r"alignment",
         r"collision", r"duplex", r"auto-?negotiat", r"\bcabl(e|ing)\b", r"\brunts?\b", r"rx_crc|rx_errors"),
    ),
    FaultCategory(
        "packet_discards", "Packet discards / congestion (high utilization)",
        "Discard counters increase or utilisation approaches line rate.",
        ("high link utilization",),
        "packet discards output drops buffer overflow congestion high utilization queue full "
        "ifOutDiscards bandwidth saturation",
        (r"discard", r"congest", r"\bbuffer", r"\bqueue", r"utili[sz]ation", r"bandwidth",
         r"overflow", r"\bdrops?\b"),
    ),
    FaultCategory(
        "high_latency", "Latency degradation",
        "ICMP round-trip time rises well above its baseline.",
        ("high link utilization", "protocol issues"),
        "high latency round trip time increased delay jitter congestion queueing delay slow response",
        (r"latenc", r"round[- ]trip|\bRTT\b", r"\bdelay", r"jitter", r"queu(e|ing)", r"response time"),
    ),
    FaultCategory(
        "unclassified", "Unclassified anomaly",
        "The detector flagged the sample but no single symptom pattern dominates.",
        (),
        "network fault troubleshooting methodology isolate the problem check interface counters "
        "logs and recent changes",
        (r"troubleshoot", r"isolat", r"diagnos", r"counter", r"\blogs?\b"),
    ),
]}
