# Real-game compatibility survey

Survey of the Glide-capable titles in a real Lutris library (the games that had
`dgvoodoo2: true`), to ground the roadmap in what actual games require.

## Key finding: glide2x vs glide3x

GlideVK currently ships **`glide3x.dll`** only. But most classic Glide titles
link **`glide2x.dll`** (Glide 2.x, the 1996–1998 Voodoo1/2 era). Confirmed by
dumping each renderer's import table (`objdump -x ... | grep 'DLL Name: glide'`):

| Game | Renderer binary | Links |
|------|-----------------|-------|
| POD Gold | `PODX3Dfx.exe`, `donut.exe` | **glide2x** |
| Deus Ex (Unreal) | `GlideDrv.dll` | **glide2x** |
| Clive Barker's Undying (Unreal) | `GlideDrv.dll` | **glide2x** |
| Metal Fatigue | `GlideRendEng.dll` | **glide2x** |
| **Diablo II** | `D2Glide.dll` | **glide3x** ✅ |

So a **`glide2x.dll` shell** (reusing `glidevk-core`) is the single biggest
unlock for real-game testing — it covers the Unreal-engine games, POD, and most
of the 1997–98 catalogue. Glide2 is the same pipeline with a fixed `GrVertex`
struct (no `grVertexLayout`) and a slightly different entry-point set.

The GOG builds already run these in Glide mode via **nGlide** (bundled
`glide.dll`/`glide2x.dll`/`glide3x.dll` + `nglide_config.exe`), so a GlideVK DLL
dropped into the game dir is a clean drop-in swap.

## Diablo II — the glide3x test target

`D2Glide.dll` imports **35** `glide3x` functions. GlideVK now exports all of
them (verified: `comm -23 d2_needs our_exports` is empty), so D2 can load our
DLL. The functions it needs:

```
grGlideInit grGlideShutdown grSstSelect grSstWinOpen grSstWinClose
grBufferClear grBufferSwap grFinish grGet
grColorCombine grAlphaCombine grAlphaBlendFunction grConstantColorValue
grColorMask grDepthMask grChromakeyMode grChromakeyValue grDitherMode
grVertexLayout grDrawVertexArray grDrawVertexArrayContiguous grDrawLine grDrawPoint
grTexSource grTexDownloadMipMap grTexDownloadTable grTexCombine grTexFilterMode
grTexMinAddress grTexMaxAddress
grLfbLock grLfbUnlock grLoadGammaTable guGammaCorrectionRGB
```

Added for D2 (P3a tail): `grGet` (reports a Voodoo-ish single-TMU / 16 MB / 256px
capability set), `grTexMin/MaxAddress`, a non-crashing `grLfbLock` (hands back a
real writable 16-bit staging buffer), and no-op stubs for the rest.

### What D2 still needs to render correctly
- **LFB upload** ✅ — done (P3b). `grLfbLock`/`grLfbUnlock` writes now composite
  into the presented image (verified via `examples/wine_lfb.c`).
- Real **ST scaling** (texel-space → normalized) and **Z range** — currently
  passed through; fine for the synthetic harness, not for a real game. Best
  driven by an actual D2 frame once the win32→win64 prefix migration is done.
- Read-locks and partial-region LFB (unwritten pixels currently keep the staging
  buffer's prior contents rather than the live framebuffer).

## Deus Ex — real-game render status (glide2x)

Deus Ex runs on GlideVK and renders its intro (Eidos logo, the 3D flying-logo
sequence). Getting there fixed a chain of real Glide-vs-Vulkan mismatches:

- **Decorated exports** — Glide2 games import `_grXxx@N` (stdcall-decorated);
  Glide3 (D2) imports undecorated. `glide2x.def` + `build.rs` add the decorated
  aliases.
- **Snap bias** — Unreal's GlideDrv adds `(3<<18)=786432` to vertex X/Y (the
  Glide fixed-point snapping bias); subtract it (`unsnap`).
- **Perspective `oow`** — Deus Ex routes W through the *vertex* `oow` (via
  `grHints`); the shader does `s = sow/oow` then normalizes by texture size.
  **Correction (was a real bug):** GlideDrv does not zero `tmuvtx[0].oow` — it
  leaves it **uninitialized**. That slot holds stack garbage that *varies run to
  run*: sometimes the denormal `1.345e-42` (the int `960` = screen width
  reinterpreted as f32), sometimes normal-looking junk like `5.9e3` or `-1.7e38`.
  `read_vertex` used to *prefer* the per-TMU slot (skipping only an exact `0.0`),
  so it divided `sow` by that garbage — exploding texture coordinates to ±inf and
  rendering the intro logos/menu as garbage that changed on every launch (the
  "invisible Eidos logo / badly-rendered Ion Storm & menu"). Fix: trust the
  vertex `oow` (= 1/z, the field GlideDrv actually populates); fall back to the
  per-TMU slot only if the vertex one is 0/denormal, else affine `1.0`. Verified:
  `inf`/`NaN` texture coordinates dropped 200 → 0 across an intro capture, and the
  result is now deterministic. Regression-tested in
  `uninitialized_tmu_oow_is_ignored_for_vertex_oow`.
- **ABGR color format** — `grSstWinOpen` declares `cformat=1` (ABGR); unpack
  `grConstantColorValue`/`grBufferClear`/`grChromakeyValue` accordingly (R/B
  were swapped).
- **Base-mip only** — `grTexDownloadMipMapLevel` sends one level per call; keep
  only `this_lod == large_lod` (base) to avoid over-reading sub-mip data.
- **Constant-color byte order** — Deus Ex declares an ABGR framebuffer
  (`grSstWinOpen` cformat=1) but supplies `grConstantColorValue` packed as
  `0xRRGGBBAA` (RGBA) regardless — a GlideDrv quirk real 3dfx/nGlide honour.
  Reading it as the declared ABGR swaps R↔B, rendering the green Ion Storm /
  Eidos accent sprites as dim blue. `unpack_constant_color` special-cases this
  (confirmed: the constant color ramps `0x004e2818`, whose dominant `0x4e` byte
  only lands in green under RGBA; forcing it green produced a grid of green
  accent dots, matching the intro). The framebuffer/clear/chroma paths stay
  ABGR. Note the **glow halo itself is genuinely white** — its per-vertex colors
  are a gray `R=G=B` ramp (0→255) in the game's own geometry, so a white glow is
  faithful; the green lives only in the constant-color accent sprites.
- **Logical resolution** — the game renders in its requested space (960×720);
  map geometry to that (push-constant viewport) and let the swapchain scale to
  the window, so 2D overlays center correctly. `grSstScreenWidth/Height` report
  it too.

**Focus-loss crash (alt-tab / minimize) — two causes, both fixed.** The
second, found later: `deusex_glidevk.sh install` used to point
`WindowedRenderDevice` at GlideDrv too. Unreal's GlideDrv is fullscreen-only
(its `SetRes` returns 0 for a windowed request), and on `WM_KILLFOCUS` the engine
re-initialises with the *windowed* device — so any focus loss produced
`3d hardware initialization failed` → `Assertion failed: RenDev`, with real Glide
drivers as well. The script now leaves the windowed device on SoftDrv; the game
drops to software while minimised and re-enters Glide fullscreen on restore
(verified: minimise + restore mid-intro, clean exit). Also fixed alongside: the
`GR_RESOLUTION_*` table had code 7 as 960x720; per `glide.h` it is 640x480 (9 is
960x720), so every viewport was mis-sized — now a `ScreenResolution` enum in
core with the header values. The first cause, the device thrash: On
`WM_KILLFOCUS` in fullscreen, Unreal *resets* the render device with a
`grSstWinClose` → `grGlideShutdown` → `grSstWinOpen` → `grGlideShutdown` dance.
GlideVK used to destroy and recreate the entire Vulkan device on each of those,
and that thrash left Unreal's `RenDev` null so its `EndFullscreen` assert
(`WinViewport.cpp:2187`, seen in Wine as C++ exception `e06d7363`) fired. Real
Glide drivers (nGlide/dgVoodoo2) *reset* a persistent device, which is why they
never crashed. Fix: keep the Vulkan backend alive across that dance —
`grSstWinClose`/`grGlideShutdown` no longer tear it down, and `grSstWinOpen`
reuses the live backend (the swapchain re-creates itself on the next present when
the window size changed). Verified by reproducing the crash (launch on `:0`,
`xdotool windowminimize`) then confirming the game survives both minimize and
restore. Windowed mode (`examples/deusex_glidevk.sh install windowed`) remains a
belt-and-braces option but is no longer required.

**Intro logos rendered as grids of dark dots — was a texture-coordinate space
bug, now fixed (and verified against nGlide).** Glide's s/t coordinates are not
in texels: they span `[0,256)` on the texture's *larger* axis whatever its pixel
size, and `[0,256/aspect)` on the smaller one. GlideVK normalised by the pixel
size instead, so a 32x32 logo texture addressed with s,t≈(218,131) — a point
inside the texture — tiled 8x and every tile seam showed as a dark dot. The
shader now maps `uv = st * major / (256 * texSize)`. The per-draw "clamp affine
primitives" heuristic that hid the symptom is gone; `grTexClampMode` is honoured
directly (a `tex_clamp` flag in the shared core state, both shells). Effect on
Deus Ex: the Ion Storm logo now renders as a clean shape, and the emblem shows
its green glow instead of flat dark blue. The convention was pinned down by
`examples/wine_texcoord.c` run beside nGlide's `glide2x.dll` — see *Differential
testing* below.

**Eidos logo (the first intro segment) rendered black — fixed; it was
palette timing.** The 3dfx logo that nGlide shows before it is nGlide's own
splash (drawn by the DLL, `NGLIDE_SPLASH`), not the game. The Eidos mark is a
mesh with a 64x64 `GR_TEXFMT_P_8` skin, and the `GLIDEVK_TRACE_FRAMES` call
trace shows Unreal's order per texture: `grTexDownloadMipMapLevel` →
`grTexSource` → **`grTexDownloadTable` (its palette)** → draws. Glide's TMU
applies the palette at draw time, but GlideVK decoded palettised textures at
download/bind time — against the *previous* texture's palette (the all-black
one of the 2x2 base texture that Unreal binds just before). Every P8 texture
therefore decoded black. Fix: P8 downloads keep only their indices; at each
draw the bound P8 texture is resolved against the palette live at that moment
(`TextureCache::p8_draw`), cached per palette *content* (`variant_key`, up to
`MAX_P8_VARIANTS` per address) so Unreal's alternating per-texture palettes
map back to resident variants instead of re-uploading. This replaced the
"re-decode at bind, capped once per frame" scheme, and the intro runs at ~70
fps instead of ~10. The logic lives in the shared core (`Context::bind_texture`
/ `download_p8` / `bind_for_draw`), so glide3x has it too; nGlide's glide3x
confirms the same draw-time semantics (`examples/wine3_palette.c`).

**GPU crash (GPUVM/TCP fault) during the intro — texture re-upload race.** The P8
palette re-decode replaces a texture mid-frame; destroying the old `GpuTexture`
immediately faults the GPU because draws already recorded that frame still
reference it. Fixed by retiring replaced textures and destroying them only after
the frame's fence signals (`retired_textures`, drained in `present_inner`).

**Output gamma (`grGammaCorrectionValue`) — was stubbed, is the reason dim
content read as black.** Deus Ex sets `UseGammaCorrection=True` and requests
gamma **1.4**; real 3dfx applies this at the RAMDAC, lifting dim geometry into
the visible range. We now forward it to the backend and apply `pow(rgb, 1/gamma)`
in the combiner's final output (per-fragment; exact for single-layer geometry
over black, which is where it matters). This alone takes the flying Deus Ex
emblem from pure black to a visible dark-blue shape.

**Palettised (`GR_TEXFMT_P_8`) textures — superseded.** The earlier
"re-decode lazily at bind, capped once per texture per frame" approach is gone;
see the Eidos entry above for the draw-time, per-palette-content variant cache
that replaced it. The long-term option remains a palette LUT in the shader.

**Flat intro logos (Eidos mark, Deus Ex arcs, Ion Storm mark) render but are
nearly invisible without gamma.** They draw with
`grColorCombine(func=Local, local=Iterated)` (output = the iterated vertex
color) and blend `(One, OneMinusSrcColor)`. Forcing the vertex color bright
reveals the exact logo geometry, so position/combine/blend are all correct — the
emblem is simply drawn in the game's own *very dim* per-vertex colors (peak
≈`[0.04, 0.07, 0.15]`, most vertices `[0,0,0]`), which vanish on black. Same
blend as the glow; the glow is bright only because its texture is white. Since
this is the game's own Glide output. Untextured 3D geometry (the Deus Ex emblem
arcs, the pure-3D Eidos logo) is shaded the same way: its per-vertex colours are
read correctly at the standard `r@12/g@16/b@20` offsets, but the game lights them
dimly (`[0,0,0]` unlit → `~[0.03,0.06,0.11]` lit), so they're near-invisible on
black. **`GLIDEVK_GAMMA=<v>` overrides the output gamma as a brightness knob** —
e.g. `GLIDEVK_GAMMA=3.0` makes the emblem arcs and orb clearly visible (verified
against a normal-vs-boosted capture). It's the practical way to match a reference
renderer's brightness without hard-coding a value.

**Ion Storm text hidden by its glow — W-buffer depth, now fixed.** With
blending right, the "ION STORM" letters vanished behind the green glow. The
call trace (`GLIDEVK_TRACE_FRAMES=1100..1400`) shows the game draws the letter
meshes first and the glow *after* them, opaque, with depth writes on and
`GR_CMP_LEQUAL` — on hardware the depth test rejects the glow where the letters
are nearer. Every draw of ours had depth 1.0: depth came from the vertex `ooz`
field, which Unreal never sets, while the game runs
`grDepthBufferMode(GR_DEPTHBUFFER_WBUFFER)`, where depth is `w = 1/oow`.
`DepthMode::vertex_depth` (core) now derives depth from `oow` in W-buffer
mode (`w / 65535`, nearer = smaller) and from `ooz` in Z-buffer mode;
`examples/wine_wbuffer.c` confirms against nGlide that the nearer quad wins
regardless of draw order, and that a later draw wins with depth off. W is
interpolated per vertex for now (exact for small primitives); moving it to
the fragment shader is the follow-up if large walls show sorting errors.

**In-game objects untextured, and 14-40 fps — the descriptor pool ran dry.**
Loading a save showed the world textured but many surfaces flat-shaded (they
were the 1x1 white fallback tinted by the level light). Instrumenting the
binds showed up to 275 texture binds per frame finding nothing at their
address, across 1094 distinct addresses — every one of which had been
downloaded moments earlier. Cause: each `upload_texture` created a fresh
image plus **two descriptor sets** (REPEAT + CLAMP), the pool was created
without `FREE_DESCRIPTOR_SET`, and `destroy_texture` never returned them, so
after a few thousand uploads every allocation failed. The error was discarded
with `let _ =`, so the upload silently left nothing at that address and the
next bind fell back to white. Games re-download constantly (Unreal refreshes
lightmaps every frame), so in-game this happened within seconds. Fixes: the
pool allows freeing and `destroy_texture` frees both sets; a re-upload of the
same address at the same size now overwrites the image in place instead of
building a new one (guarded by a `used_frame` stamp so it never touches a
texture already bound in the frame being recorded — earlier frames are
complete, their fence is waited on at present); and a failed upload is logged
instead of swallowed. Measured in the same save: texture misses per frame 275
-> 0, uploads per frame 29-119 -> 3-18, upload time 4-55 ms -> 0.7-3.8 ms per
frame, and 14-96 fps -> 102-145 fps. The `perf` line now also reports
`uploads/frame`, `upload=<ms>/frame` and `texmiss/frame`.

**Frame rate collapsed with resolution (~2 fps at 1600x1200, ~7 at
800x600) — the frame-capture diagnostic was on for every frame.** The capture
setting defaulted to `-1`, and negative values mean "every |n| frames", so with
no `GLIDEVK_CAPTURE` at all each present copied the whole image to host memory
and converted every pixel on the CPU — reading from write-combined, uncached
memory, which is what made it so slow and resolution-dependent. It went
unnoticed because every scripted test run set `GLIDEVK_CAPTURE`, which
replaced the default with a sparse interval: those runs showed ~140 fps while
real sessions crawled. Fixes: a `CaptureMode` enum (`Off` by default, pinned by
a test); readback buffers allocated from `HOST_CACHED` memory; the per-frame
copy is now GPU-only, and the CPU converts just the pixels a
`grLfbReadRegion` asks for, or the full frame on a requested capture frame.
The capture buffer also now follows swapchain growth — it was sized once, so
a larger mode chosen in-game overran it on the GPU and the host read past its
mapping. Measured without any diagnostic flags: 144 fps (the monitor's
refresh cap) at 1600x1200 and 800x600, fullscreen or `GLIDEVK_WINDOWED`, with
the per-frame copy costing ~0.3 ms at 1600x1200. Testing note: launching the
game again while the previous instance is still exiting makes it quit before
loading the renderer (empty `DeusEx.log`, crash marker left behind); wait for
`wineserver -w` between scripted runs.

**Resolution changes and fullscreen did nothing — Glide is a fullscreen API,
now honoured.** On 3dfx hardware `grSstWinOpen` switched the display to the
requested mode; Unreal then sizes its window from `grSstScreenWidth/Height`,
except that Deus Ex never actually resizes it, so with GlideVK a request for
800x600 rendered into the untouched 640x480 window, and a 1600x1200 pick from
the Display Settings menu just relaid the UI for a viewport four times the
window. nGlide gets all of this from Direct3D fullscreen. GlideVK now does the
equivalent itself (`glidevk-core::win32`): `grSstWinOpen` calls
`ChangeDisplaySettingsExW(CDS_FULLSCREEN)` for the Glide resolution and makes
the game window a borderless popup of that size at the origin; `grSstWinClose`
restores the desktop mode. Wine emulates the mode switch by scaling the
fullscreen window and its mouse input to the monitor, which is what makes
"fullscreen" and in-game resolution changes work. If Wine rejects a legacy mode
(macOS commonly returns `BADMODE` for 1600x1200), GlideVK still makes the game
window a borderless window covering its current monitor; the rejected mode
remains the logical Glide coordinate space. The swapchain now follows the
window's surface extent every frame instead of waiting for an out-of-date error
Wine does not always raise. `GLIDEVK_WINDOWED=1` keeps the desktop mode and
adds a draggable caption to the game window while preserving its client size;
`GLIDEVK_WINDOWED=0` or leaving it unset selects the fullscreen path. The
logical Glide resolution can still differ from that client size; the renderer
scales to the client area.
Note that toggling *off* fullscreen in the game still means Unreal's software
renderer: GlideDrv is fullscreen-only by design, as on real hardware.

**Main menu without any text — `guColorCombineFunction` table, now from the
SDK.** The menu's frames, bars and cursor drew but every label was missing.
The menu-frame trace shows the glyph tiles (font textures decode fine, white
glyphs with alpha) drawn with `TEXTURE_TIMES_CCRGB` (code 5) — Unreal's tile
path modulates the texture by the *constant* colour, which holds the UI green.
Our `guColorCombineFunction` handled four codes and let the rest fall through
to modulate-by-iterated, and the iterated colour on tiles is black, so the text
was drawn black. The helper is now the SDK's own table (`gu.c`: every
`GrColorCombineFnc_t` as its `grColorCombine` expansion), same for
`guAlphaSource`, verified against nGlide by `examples/wine_gucombine.c`. Found
alongside and fixed per `glide.h`: `GR_COMBINE_FUNCTION` 0x9 / 0x10 are
`SCALE_MINUS_OTHER_ADD_LOCAL[_ALPHA]` (`f*(-other)+local`), not minus-local;
`GR_COMBINE_LOCAL_DEPTH` is 2; and `invert` inverts the combine unit's output
(`GR_COLORCOMBINE_ONE` is an inverted ZERO), not the factor. The code→enum
maps now live in core (`CombineFunction::from_glide` etc.) for both shells.
nGlide renders the two DELTA0 codes black, contradicting the SDK's flat-shade
semantics, so the harness skips them.

**Blend factor codes — were permuted, now per `glide.h` and verified against
nGlide.** `map_blend_factor` had 3/4/5/6 shuffled (ONE was 6, ONE_MINUS_SRC_ALPHA
was 3) and its "verified by wine_blend" claim was circular: that harness used
the same wrong defines. `examples/wine_blendmode.c` walks every code on both
sides through the reference; the resulting table lives in
`BlendFactor::from_glide` (core, shared by both shells): 0 ZERO, 1 SRC_ALPHA,
2 SRC/DST_COLOR, 3 DST_ALPHA, 4 ONE, 5 ONE_MINUS_SRC_ALPHA, 6 ONE_MINUS_SRC/
DST_COLOR, 7 ONE_MINUS_DST_ALPHA, 0xf ALPHA_SATURATE (src) / PREFOG_COLOR
(dst, = source colour with fog off). The `_COLOR` codes name the other
operand, so the side matters. With the real table Unreal's translucent
`(ONE, ONE_MINUS_SRC_COLOR)` glow and modulated `(DST_COLOR, SRC_COLOR)`
lightmaps composite as intended; the emblem is now a lit metallic mesh rather
than a flat dark shape. Iterated vertex colours are also clamped to [0,255] in
the vertex shader, as the hardware does (Unreal leaves unused channels as
garbage such as 885 / -497).

**Safe-mode dialog after a crash.** Unreal drops a 0-byte `System/Running.ini`
marker at boot and removes it only on a *clean* exit. Any crash (e.g. the
focus-loss bug above, or a level-transition re-init) leaves it behind, so the
next launch detects "we died last time" and shows the safe-mode/renderer picker
instead of booting. `examples/deusex_glidevk.sh run` now `rm -f`s it before every
launch, so the game always starts straight into the intro.

Remaining fidelity TODOs: mipmapping (the Eidos logo aliases into a glow +
dots), the flickering "Deus" letter, and the intermittent level-transition
re-init. The `GLIDEVK_CAPTURE=<file.ppm>` frame dump makes these debuggable
offline.

## Metal Fatigue — glide2x texture-memory accounting

Metal Fatigue's `GlideRendEng.dll` uses `grTexMinAddress`,
`grTexMaxAddress`, and `grTexCalcMemRequired` to lay out its own TMU heap.
The latter therefore has to report the full inclusive LOD chain, including the
SST-1's four-texel minimum/fetch granularity and final eight-byte alignment—not
just the base mip. Returning only the base level makes consecutive textures
overlap in game-managed TMU address space. A 1600×1200 run reached roughly
53 FPS with no Vulkan presentation or texture-miss errors before an abrupt
crash in the game renderer; the corrected accounting is the next isolated
change to test against that path.

## Launch notes
- Deus Ex: `examples/deusex_glidevk.sh install` then `run` (direct launch in
  the prefix; the Wine virtual desktop it used to need is gone). Set
  `GLIDEVK_LOG`, `GLIDEVK_CAPTURE`, `GLIDEVK_TEXDUMP`, `GLIDEVK_TRACE_FRAMES`
  / `GLIDEVK_TRACE_TIME`, `GLIDEVK_WINDOWED` in the environment as needed.
- D2's prefix is a **win32** prefix — needs a 32-bit-capable wine
  (`wine-10.8-staging-i386` in the Lutris runners), not a WoW64-only build.
- Force Glide with `-3dfx` (Lutris's default launch does not).
- Drop `glide3x.dll` beside `Diablo II.exe` (the game dir has none; the loader
  finds it before the system32 nGlide copy) — non-destructive.

## Differential testing — the reference decides

The `examples/wine_*.c` harnesses used to check GlideVK against GlideVK's own
assumptions (the comment in `wine_texmap.c` literally described our divide and
then wrote coordinates to match it, so it could only pass). The texture-space
bug above survived that. The fix is methodological:

- **`examples/run_diff.sh [harness...]`** builds `glide2x.dll`, runs each harness
  beside a *reference* `glide2x.dll` (default: the nGlide copy backed up by
  `deusex_glidevk.sh`; override with `GLIDE2X_REF`) and beside ours, and diffs
  their stdout. `PASS` means we match the reference, not our expectation.
- Harnesses in this style read pixels back with **`grLfbReadRegion`**, which every
  Glide 2 implementation exports (the older ones use GlideVK-only debug exports
  and cannot run against a reference). GlideVK's `grLfbReadRegion` is now real:
  it serves the last presented frame as RGB565 (the first call turns on a
  per-frame host copy, so a read before any subsequent present returns zeros).
- `wine_texcoord.c` is the first such harness: nGlide answers `red green blue
  white` / `red green` for the 0..256 mapping, and so do we now.
- Harness `GR_RESOLUTION` codes were also wrong (5 = 640x350, not 640x480); they
  now use 7.
- `wine_blendmode.c`: every `GrAlphaBlendFnc_t` code on both sides, quantised
  to 8 levels per channel so 565 rounding cannot flip a bucket.
- `wine3_palette.c` (glide3x): downloads a P8 texture, binds it, uploads its
  palette *after* the bind, draws; then swaps palettes without rebinding. The
  reference answers green / red / blue — the palette live at each draw — and
  so do we. Run with `GLIDE_API=glide3x`; the runner then builds `glide3x.dll`
  and defaults the reference to nGlide's `glide3x.dll` in the Deus Ex install.
  Both shells' `grLfbReadRegion` are real and share
  `Backend::read_region_rgb565`.
- `wine_gucombine.c`: every `guColorCombineFunction` and `guAlphaSource` code
  with texture, iterated and constant inputs chosen to be distinguishable.
- `wine_wbuffer.c`: near-then-far and far-then-near quads under
  `GR_DEPTHBUFFER_WBUFFER` + `LEQUAL` with `ooz` left at 0, plus depth off.
- Performance: set `GLIDEVK_LOG=1` and `GLIDEVK_PROFILE_ONLY=1` for a
  lightweight `perf` line every 120 frames and slow texture-bind, download and
  present calls (100 ms or more). Profile mode suppresses detailed vertex,
  texture and call traces. Each line also reports the longest individual draw,
  upload and present call in the sample, useful for short intermittent stalls.
  Present timings include any sleep required by the `GLIDEVK_FPS` limit.
  The upload time is included in draw submission time;
  the first sample excludes work before the first swap. `GLIDEVK_TRACE_TIME=<a>..<b>`
  (seconds) traces by wall time when detailed logging is enabled.
- Diagnostics that made the palette bug visible: `GLIDEVK_TRACE_FRAMES=<n>` or
  `<start>..<end>` logs every traced call (downloads, palette uploads, binds,
  draws with blend/depth state, clears, swaps) in order for those frames; `GLIDEVK_TEXDUMP=<dir>` writes each
  distinct uploaded texture, and each P8 variant, as a PPM.

When something in a game looks wrong, the order is: reproduce the exact Glide
call pattern in a harness, run it through the reference, then fix — not the
other way round.

