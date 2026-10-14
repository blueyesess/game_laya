// Extract the frame-blit code from web/index.html and exercise it.
//
// The bug this guards against is subtle and was invisible to every other test in the
// repo: ImageData is RGBA (4 bytes/pixel) while nes_py serves RGB (3 bytes/pixel), and
// writing one stream into the other shifts every pixel by a byte. The screen still
// showed the HUD, the blocks and the ground in the right places, so it read like an
// emulator fault.
//
// So this does not re-implement the blit. It lifts the real `rgba32`, the real
// endianness probe and the real `blit` out of index.html and runs them, which means
// the shipped code is what gets tested:
//
//   1. fixture  a 4x2 image with known pixels expands to the canonical RGBA layout
//   2. parity   the little-endian Uint32 fast path agrees with the portable byte loop
//   3. real     (optional) a captured emulator frame, expanded for comparison in Python
//
// Usage: node blit_check.js <index.html> [in.rgb] [out.rgba] [width] [height]

const fs = require('fs');

const [, , htmlPath, rgbPath, outPath, widthArg, heightArg] = process.argv;
const W = parseInt(widthArg || '256', 10);
const H = parseInt(heightArg || '240', 10);

function fail(msg) {
  process.stderr.write('  [FAIL] ' + msg + '\n');
  process.exitCode = 1;
}
function pass(msg) {
  process.stdout.write('  [PASS] ' + msg + '\n');
}
function bail(msg) {
  fail(msg);
  process.exit(1);
}

// ---- lift the code under test out of the dashboard -------------------------
const html = fs.readFileSync(htmlPath, 'utf8');
const script = (html.match(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/) || [])[1] || '';

function grab(pattern, what) {
  const m = script.match(pattern);
  if (!m) bail('could not find ' + what + ' in index.html -- has it been renamed?');
  return m[0];
}

const rgba32Src = grab(/const rgba32 = new Uint32Array\([^;]*\);/, 'rgba32 view');
const endianSrc = grab(/const LITTLE_ENDIAN = \(\(\) => \{[\s\S]*?\}\)\(\);/, 'LITTLE_ENDIAN probe');
const blitSrc = grab(/function blit\(rgb\) \{[\s\S]*?\n\}/, 'blit()');

// Every isolated instantiation needs the same three pieces: blit() closes over
// rgba32 and LITTLE_ENDIAN, and rgba32 is derived from the image passed in.
const fullBuild = new Function('image', rgba32Src + '\n' + endianSrc + '\n' + blitSrc + '\nreturn { blit, LITTLE_ENDIAN };');
// Forcing the portable branch: pass the flag as a parameter and drop the const probe,
// since re-declaring LITTLE_ENDIAN in the same scope would be a syntax error.
const portableBuild = new Function(
  'image', 'LITTLE_ENDIAN',
  rgba32Src + '\n' + blitSrc.replace('if (LITTLE_ENDIAN)', 'if (false)') + '\nreturn blit;'
);

const hostImage = { data: new Uint8ClampedArray(W * H * 4) };
const host = fullBuild(hostImage);
const { blit, LITTLE_ENDIAN } = host;

// ---- 1. fixture -------------------------------------------------------------
{
  const pixels = [
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [18, 52, 86],
    [1, 2, 3], [250, 251, 252], [0, 0, 0], [255, 255, 255],
  ];
  const rgb = new Uint8Array(pixels.flat());
  const want = rgb.length;
  if (want === pixels.length * 4) bail('fixture assumption wrong: RGB and RGBA lengths collided');

  const fixtureImage = { data: new Uint8ClampedArray(pixels.length * 4) };
  fullBuild(fixtureImage).blit(rgb);

  const d = fixtureImage.data;
  let ok = true;
  let firstBad = null;
  for (let i = 0; i < pixels.length; i++) {
    const q = i * 4;
    const expect = [pixels[i][0], pixels[i][1], pixels[i][2], 255];
    for (let c = 0; c < 4; c++) {
      if (d[q + c] !== expect[c] && ok) {
        ok = false;
        firstBad = 'pixel ' + i + ' channel ' + c + ': got ' + d[q + c] + ', want ' + expect[c];
      }
    }
  }
  if (ok) pass('blit expands 8 known pixels to canonical RGBA');
  else fail('blit produced wrong RGBA -- ' + firstBad);
}

// ---- 2. fast path vs portable path -----------------------------------------
{
  const probe = new Uint8Array([11, 22, 33, 44, 55, 66]);
  const a = { data: new Uint8ClampedArray(12) };
  const b = { data: new Uint8ClampedArray(12) };
  fullBuild(a).blit(probe);
  portableBuild(b, false)(probe);
  if (Buffer.compare(Buffer.from(a.data), Buffer.from(b.data)) === 0) {
    pass(LITTLE_ENDIAN ? 'Uint32 fast path matches the portable byte loop'
                       : 'portable byte loop matches (host is big-endian)');
  } else {
    fail('fast path and byte loop disagree: [' + Array.from(a.data) + '] vs [' + Array.from(b.data) + ']');
  }
}

// ---- 3. the real frame ------------------------------------------------------
if (rgbPath && outPath) {
  const raw = fs.readFileSync(rgbPath);
  if (raw.length !== W * H * 3) {
    fail('input rgb is ' + raw.length + ' bytes, expected ' + W * H * 3);
    process.exit(1);
  }
  blit(new Uint8Array(raw));
  fs.writeFileSync(outPath, Buffer.from(hostImage.data.buffer));
  pass('expanded ' + raw.length + ' RGB bytes into ' + hostImage.data.length +
       ' RGBA bytes -> ' + outPath);
}
