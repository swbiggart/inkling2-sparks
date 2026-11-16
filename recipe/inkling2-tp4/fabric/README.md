# Fabric

The four hosts exchange activations over 200G RoCE. Two layouts:

| Folder | Layout | Status |
| --- | --- | --- |
| `switch/` and `host/` | One 200G link per host to a switch (both PCIe halves used), one L2 segment | **Default. The release was measured on it.** |
| `ring-experimental/` | Switchless ring, two cables to each neighbour | Experimental, not validated for this release |

For the switch: configure the switch (`switch/`), install the host setup on each host (`host/`), keep the
default `NCCL_ENV` in `config.example.sh`, and check the ports with `../status.sh CONFIG` (and `--bw`).

**After any cable change, reboot the hosts whose cables moved.** A runtime NIC re-probe caps transmit at about
13 Gb/s per PCIe half until the next boot. `status.sh` warns about it.
