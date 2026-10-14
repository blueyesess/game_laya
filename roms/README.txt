Put your ROM here as *.nes -- e.g. roms/smb1.nes

Super Mario Bros. is copyrighted, so this project never ships one and never
downloads one. Supply your own legally obtained dump.

Everything else in the project works without it: run

    python tools/make_test_rom.py

to generate roms/inputtest.nes, a homebrew ROM written by this repo whose only
purpose is to prove the emulator wrapper, the controller bitmap and the video
path all work.

inputtest.nes is generated, not copyrighted, and is tracked by git.
