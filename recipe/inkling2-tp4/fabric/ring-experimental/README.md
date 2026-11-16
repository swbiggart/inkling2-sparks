# Switchless ring (EXPERIMENTAL, not validated for this release)

> **Unvalidated.** The release was measured on the switch (`../switch/`, `../host/`). This ring setup ran
> earlier versions of the server, but it has not passed this release's gates. Use the switch if you can.

Four hosts, each with two 200G cables to each neighbour (one a PCIe half), in a ring: rank 0 - 1 - 2 - 3 - 0. Every
host uses all four of its ports.

## Addressing

`ring-fabric.sh RANK [PREFIX]` (root, runtime only) gives each port an address on its link's /24:

- Rank r's port 0 is cabled to rank r+1's port 1 (mod 4). Link L joins rank L's port 0 (host octet 10) and rank
  L+1's port 1 (host octet 11).
- Half 0 of link L is `PREFIX.L.0/24`; half 1 is `PREFIX.(L+4).0/24`.

| Rank | `enp1s0f0np0` | `enp1s0f1np1` | `enP2p1s0f0np0` | `enP2p1s0f1np1` |
| --- | --- | --- | --- | --- |
| 0 | `P.0.10` | `P.3.11` | `P.4.10` | `P.7.11` |
| 1 | `P.1.10` | `P.0.11` | `P.5.10` | `P.4.11` |
| 2 | `P.2.10` | `P.1.11` | `P.6.10` | `P.5.11` |
| 3 | `P.3.10` | `P.2.11` | `P.7.10` | `P.6.11` |

The scheme (one /24 per cable, `.10` and `.11`, MTU 9000) follows the switchless-ring guide in MiaAI-Lab's
DeepSeek-v4.1-Flash-DGX-Sparks; the second half's subnets are added here. See `THIRD_PARTY_NOTICES.md`.

MTU 9000, `arp_ignore=1`, `arp_announce=2` on every port. Nothing persists: run it again after a reboot, or adapt
`../host/` to the four ports. The same re-probe trap applies: reboot a host after any cable change.

## NCCL on the ring

In the config, replace the switch's `NCCL_ENV` with the commented ring block in `config.example.sh`:

- `NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1` (all four ports)
- `NCCL_CROSS_NIC=1 NCCL_IB_SUBNET_AWARE_ROUTING=1 NCCL_IB_SUBNET_PREFIX_LEN=24`: subnet-aware routing sends each
  connection out of the port whose /24 holds the peer's address.
- `NCCL_ALGO=Ring` with 4 channels, and the hosts in `HOSTS` in ring order.
- `GRAPH=$ROOT/nccl/graph-rank{R}.xml`: a per-rank NCCL graph file (`NCCL_GRAPH_FILE`). The files used were
  generated for these hosts by `graph_xml` in MiaAI-Lab's four-Spark DeepSeek recipe (`scripts/nodes.sh`), whose NCCL
  settings this block follows; they gave four one-way channels 0 -> 1 -> 2 -> 3, two on each half, and are not
  included here. Generate yours with that recipe, or let NCCL write its own search result with
  `NCCL_GRAPH_DUMP_FILE=FILE` on one run; pin that file per rank. The ring was only run with a graph file.

Measured on the ring before the switch, with every link healthy: NCCL reaches about 18 GB/s bus bandwidth at 8-64
MiB a rank; raw RDMA reaches about 98 Gb/s per half with both halves of a link sending. A ring all-gather runs at
the pace of its slowest link, so one host with a re-probed NIC (about 13 Gb/s a half) caps the whole ring at about
3 GB/s.

## Opt-in transports on the ring

- `TF_INK2_ROCE_MULTI=1` with `TF_TP_ROCE_RELAY=1` and `TF_TP_ROCE_HCA=rocep1s0f0:3,rocep1s0f1:3` (one HCA to each
  neighbour; each neighbour relays half of the opposite rank's data). See the config's opt-in block for the other
  settings it needs.
- `TF_ROCE=1` does not work on the ring: its connect probe needs every rank to reach every other rank on the same
  HCA, so every rank keeps NCCL.
- `TF_INK2_XROWS=1` (prompt row sums) needs every pair of ranks connected and does not work on the ring.
