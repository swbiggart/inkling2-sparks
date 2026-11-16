# Switch configuration

The four hosts each have one 200G link to the switch. `crs812-fabric.rsc` configures a MikroTik CRS812-class switch
for them; the notes below cover other switches too.

This is the configuration the release was measured on: four GB10 hosts on a CRS812-8DS-2DQ-2DDQ (RouterOS 7.20),
through two QSFP56-DD 400G-to-2x200G breakout cables.

## What the script does

1. Sets each breakout leg to `auto-negotiation=no speed=200G-baseCR4`.
2. Then sets `mtu=9000 l2mtu=9216` on the legs. **Order matters: changing a port's speed resets its MTU**, so a
   speed change made after the MTU silently puts the port back to 1500.
3. Puts the legs on the bridge with `pvid=86`. RouterOS adds them to VLAN 86 as untagged members.
4. Turns on `vlan-filtering=yes` on the bridge. ether1, ether2 and the bridge itself stay on VLAN 1, so management
   keeps working, the fabric is cut off from the LAN, and the switch chip still forwards in hardware.
5. Adds a DHCP client on the bridge for management, with no default route, DNS or NTP.

The script holds no MAC or IP addresses. Edit the leg list at the top if your cabling differs, then:

```
# from a machine that reaches the switch's management address
scp crs812-fabric.rsc admin@<switch>:
ssh admin@<switch> '/import file-name=crs812-fabric.rsc'
```

Check the result on the switch:

```
/interface ethernet print where name~"qsfp56-dd"        # mtu 9000, l2mtu 9216 on the legs
/interface ethernet monitor qsfp56-dd-1-1 once           # rate: 200Gbps, auto-negotiation: disabled
/interface bridge vlan print                             # VLAN 86 untagged on the four legs
/interface bridge port print                             # H (hardware offload) flag on every port
```

On each host, `status.sh CONFIG` (in the recipe folder) reads both fabric ports' speed, MTU and carrier and looks
for a NIC re-probe; `status.sh CONFIG --bw` runs a short `ib_write_bw` (from perftest, installed on each host)
between neighbouring ranks.

## Cables

- **Use a 400G-to-2x200G breakout (QSFP56-DD to 2x QSFP56), one leg per host.** A GB10 host's ConnectX-7 carries
  both of its PCIe halves on that one 200G port (about 98 Gb/s each with every rank sending, about 196 Gb/s a host).
- **A 2x100G splitter caps each host at 100G.** A cable like the MCP7H50 splits a port into two 100G legs, and the
  switch can only bring them up at `100G-baseCR2`. It works, but at half the bandwidth.

## Re-plugging a cable: reboot the host

Unplugging and re-plugging a host's cable, or moving it to another port, makes the mlx5 driver re-probe the NIC.
After a runtime re-probe the NIC transmits at only about 13 Gb/s per PCIe half until the host reboots. Its kernel log
shows `PCIe slot power capability was not advertised` (or `Detected insufficient power on the PCIe slot`). A clean
boot with the cable in place restores the full rate. This matches a report on the NVIDIA Developer Forums
([ConnectX-7 Inter-Spark Link Capped at ~13 Gbps](https://forums.developer.nvidia.com/t/connectx-7-inter-spark-link-capped-at-13-gbps-expected-200-gbps-pcie-power-throttling-27w/363461/4)),
and we measured the same on these hosts.

So, **after any cable change, reboot the hosts whose cables moved**. `status.sh` warns when it finds the message in
a host's kernel log since boot. The fabric unit in `../host/` brings the ports back up after a re-probe, but it
cannot restore the transmit rate.

## Other switches

Any switch that forwards 200GbE between the four hosts works. It needs:

- **One L2 segment** that holds the fabric ports: an access VLAN, or a plain bridge. The two PCIe halves of each host
  use two different /24 subnets on that one segment (see `../host/`), and NCCL routes each half to the same half on
  the other hosts.
- **Jumbo frames:** host MTU 9000, so the switch's MTU or L2 MTU at 9216 or more on those ports.
- **Fixed speed** when auto-negotiation does not settle on 200G with your cables. Set the speed before the MTU if the
  switch resets one with the other.
- **No PFC or ECN.** The release numbers were measured without them. Plain RoCE v2 (GID index 3) over an ordinary lossy
  Ethernet segment was enough at these loads. If you see `rx_out_of_sequence` or `packet_seq_err` counters climb under long
  prompts (`ethtool -S <port>`), that is the first place to look.
- Keep the fabric off the management LAN, so that NCCL's bootstrap and the HTTP endpoint stay on the LAN interface
  (`SOCKET_IFNAME` in the config) and the data stays on the fabric.
