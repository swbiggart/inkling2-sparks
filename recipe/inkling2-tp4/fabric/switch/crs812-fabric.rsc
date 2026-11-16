# Inkling-2 TP4 fabric on a MikroTik CRS812-class switch (written for a CRS812-8DS-2DQ-2DDQ on RouterOS 7.20).
#
# Four hosts, one 200G link each, through two QSFP56-DD 400G-to-2x200G breakout cables. One L2 segment (VLAN 86)
# for the fabric ports; management stays on VLAN 1 (ether1 and the bridge itself), so the fabric is isolated from
# the LAN and the bridge keeps hardware offload. No PFC or ECN.
#
# Start from the factory configuration (its "bridge" holds every port). Edit the leg list below to match your
# cabling, upload the file and run `/import file-name=crs812-fabric.rsc`, or paste it into a terminal.
# Read README.md in this folder first: the order of the steps matters. The steps sit in one { } block so the
# local variables hold when the text is pasted into a terminal as well as when it is imported.

{
  # The breakout legs the four hosts use. A 2x200G breakout puts its legs on lanes 1 and 5 of a QSFP56-DD cage.
  :local legs {"qsfp56-dd-1-1";"qsfp56-dd-1-5";"qsfp56-dd-2-1";"qsfp56-dd-2-5"}
  :local fabricVlan 86

  # 1. Speed first, with auto-negotiation off. A speed change resets the port's MTU, so it must come before step 2.
  :foreach p in=$legs do={
    /interface ethernet set [/interface ethernet find default-name=$p] auto-negotiation=no speed=200G-baseCR4
  }

  # 2. Jumbo frames on the legs (RoCE at MTU 9000 needs the L2 MTU above it).
  :foreach p in=$legs do={
    /interface ethernet set [/interface ethernet find default-name=$p] mtu=9000 l2mtu=9216
  }

  # 3. The legs untagged on the fabric VLAN. RouterOS adds the untagged VLAN 86 entries to the bridge VLAN table
  #    from the ports' pvid. ether1, ether2 and the bridge stay on VLAN 1 (pvid 1, the default).
  :foreach p in=$legs do={
    :if ([:len [/interface bridge port find interface=$p]] = 0) do={
      /interface bridge port add bridge=bridge interface=$p pvid=$fabricVlan comment="inkling2 fabric"
    } else={
      /interface bridge port set [/interface bridge port find interface=$p] pvid=$fabricVlan
    }
  }

  # 4. VLAN filtering on, last, once every port has its pvid (management on VLAN 1 is not interrupted).
  /interface bridge set [/interface bridge find name=bridge] vlan-filtering=yes

  # 5. Management: a DHCP client on the bridge (VLAN 1). No default route, DNS or NTP from it.
  :if ([:len [/ip dhcp-client find interface=bridge]] = 0) do={
    /ip dhcp-client add interface=bridge add-default-route=no use-peer-dns=no use-peer-ntp=no comment="management"
  }
}

# Optional: a name for the switch, and drop the factory static address once DHCP has bound.
# /system identity set name=<switch-name>
# /ip address remove [/ip address find comment=defconf]
