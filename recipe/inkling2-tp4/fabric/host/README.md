# Host fabric setup

Each host's switch-facing 200G port appears as two netdevs, one for each PCIe half of the ConnectX-7:
`enp1s0f0np0` (HCA `rocep1s0f0`) and `enP2p1s0f0np0` (HCA `roceP2p1s0f0`) with the cable in cage 0. Both carry
traffic, about 98 Gb/s each with every rank sending.

The setup gives each half an address on its own /24, the same two /24s on every host, with the host number
(rank + 1) as the last octet:

| Rank | Half 0 (`enp1s0f0np0`) | Half 1 (`enP2p1s0f0np0`) |
| --- | --- | --- |
| 0 | `NET0.1/24` | `NET1.1/24` |
| 1 | `NET0.2/24` | `NET1.2/24` |
| 2 | `NET0.3/24` | `NET1.3/24` |
| 3 | `NET0.4/24` | `NET1.4/24` |

Both halves sit on one L2 segment, so each port answers ARP only for its own address (`arp_ignore=1`,
`arp_announce=2`). MTU 9000 on both. NCCL uses GID index 3 (RoCE v2 over IPv4) and routes half 0 to half 0 and half
1 to half 1 by subnet.

## Files

- `inkling2-fabric.sh`: sets MTU, ARP settings and the address on both netdevs, and brings them up. It reads
  `/etc/default/inkling2-fabric`.
- `inkling2-fabric.conf.example`: that file's settings, explained.
- `inkling2-fabric.service`: a oneshot unit that runs the script at boot.
- `99-inkling2-fabric.rules.in`: a udev rule template. It runs the unit again when the driver recreates a port,
  matched by the port's MAC address. `install.sh` fills in the MACs.
- `install.sh`, `uninstall.sh`.

## Install (root, on each host)

```
cd recipe/inkling2-tp4/fabric/host
sudo ./install.sh RANK --dry-run                      # shows the config and udev rule it would write
sudo ./install.sh RANK                                # RANK 0..3, in the same order as HOSTS in the config
sudo ./install.sh RANK --nets "A.B.C A.B.D"           # your own two /24s (default 198.18.0 and 198.18.1)
```

`install.sh` writes `/etc/default/inkling2-fabric`, installs the script to `/usr/local/sbin/`, the unit to
`/etc/systemd/system/` (enabled and started), and the rule to `/etc/udev/rules.d/`. Check it:

```
ip -br address show dev enp1s0f0np0; ip -br address show dev enP2p1s0f0np0
cat /sys/class/net/enp1s0f0np0/speed /sys/class/net/enp1s0f0np0/mtu     # 200000, 9000
ping -c 2 -M do -s 8972 -I enp1s0f0np0 NET0.<other host>                 # jumbo frames pass the switch
```

If NetworkManager manages these ports it may undo the settings. Mark them unmanaged first, for example with
`/etc/NetworkManager/conf.d/99-inkling2-fabric-unmanaged.conf`:

```
[keyfile]
unmanaged-devices=interface-name:enp1s0f0np0;interface-name:enP2p1s0f0np0
```

## Uninstall

```
sudo ./uninstall.sh            # disables the unit and removes the four files; addresses stay until reboot
sudo ./uninstall.sh --flush    # also removes the addresses now
```

## After a cable change

A replugged cable makes the driver re-probe the NIC. The udev rule brings the ports back up with their MTU and
addresses, but the NIC transmits at only about 13 Gb/s per half until the host reboots. Reboot it. `status.sh` in the
recipe folder warns about this state.
