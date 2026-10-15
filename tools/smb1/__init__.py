"""Super Mario Bros.-specific calibration tools.

These are the one-game half of the tool set: RAM-map calibration and palette
probing, which only make sense for a ROM whose memory layout a profile claims to
know. Everything generic lives one level up in ``tools/``.

Run them as scripts, e.g.::

    python tools/smb1/find_ram.py --rom roms/smb1.nes
"""
