Put your ROM here as *.nes -- e.g. roms/smb1.nes or roms/<any game>.nes

Every commercial ROM is copyrighted, so this project never ships one and never
downloads one. Supply your own legally obtained dump, then pass it explicitly:

    python run.py --rom roms/<your game>.nes --profile generic

`--profile` selects a JSON file in src/profiles/ that describes the game (HUD
height, what "advance" means in buttons, optionally a RAM map). `generic` needs
no tuning for a typical NES side-scroller; copy src/profiles/generic.json and
edit it for anything else.

Everything else in the project works without a commercial ROM: run

    python tools/make_test_rom.py

to generate roms/inputtest.nes, a homebrew ROM written by this repo whose only
purpose is to prove the emulator wrapper, the controller bitmap and the video
path all work.

inputtest.nes is generated, not copyrighted, and is deliberately gitignored along
with every other *.nes -- an exception in .gitignore is exactly how a real dump
sneaks into a repo six months later.
