"""Openly available troubleshooting documentation for the RAG knowledge base.

Only sources that permit automated retrieval are fetched. Vendor guides that block
scripted downloads (e.g. Cisco TAC notes) can be saved manually into
``data/kb/local/`` as PDF, HTML or TXT and are picked up by ``netdiag build-kb``.
"""

RFCS = {
    "rfc2863": "The Interfaces Group MIB (ifOperStatus, error and discard counters)",
    "rfc3635": "Definitions of Managed Objects for the Ethernet-like Interface Types (FCS, alignment, collisions, duplex)",
    "rfc1213": "MIB-II",
    "rfc3418": "SNMPv2 MIB (sysUpTime, coldStart/warmStart)",
    "rfc1157": "Simple Network Management Protocol (SNMP)",
    "rfc792": "Internet Control Message Protocol",
    "rfc2151": "A Primer on Internet and TCP/IP Tools and Utilities",
    "rfc7276": "An Overview of Operations, Administration, and Maintenance (OAM) Tools",
    "rfc6632": "An Overview of the IETF Network Management Standards",
    "rfc5880": "Bidirectional Forwarding Detection (BFD)",
    "rfc2544": "Benchmarking Methodology for Network Interconnect Devices",
    "rfc7680": "A One-Way Loss Metric for IP Performance Metrics (IPPM)",
    "rfc7799": "Active and Passive Metrics and Methods",
    "rfc5424": "The Syslog Protocol",
}

WIKIPEDIA = [
    "Duplex mismatch",
    "Autonegotiation",
    "Ethernet frame",
    "Ethernet physical layer",
    "Twisted pair",
    "Network congestion",
    "Packet loss",
    "Bufferbloat",
    "Broadcast storm",
    "Link aggregation",
    "Spanning Tree Protocol",
    "Routing loop",
    "Network switch",
    "Simple Network Management Protocol",
    "Ping (networking utility)",
    "Traceroute",
    "Bit error rate",
    "Unidirectional Link Detection",
    "Link Layer Discovery Protocol",
    "Network monitoring",
]

WEB_PAGES = {
    "linux-interface-statistics": "https://www.kernel.org/doc/html/latest/networking/statistics.html",
    "vyos-troubleshooting": "https://docs.vyos.io/en/latest/troubleshooting/index.html",
}
