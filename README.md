# GlideVK

A native Linux **3dfx Glide → Vulkan** translation layer. The Glide equivalent
of [DXVK](https://github.com/doitsujin/dxvk) / [D7VK](https://github.com/WinterSnowfall/d7vk):
drop-in `glide3x.dll` / `glide2x.dll` that translate Glide calls **directly** to
Vulkan under Wine — no Direct3D or OpenGL hop.

> Status: **P3 in progress — real games.** `glide2x.dll` + `glide3x.dll`
> shells over a shared core. Deus Ex (Unreal, glide2x) boots, plays its
> whole intro and shows its main menu with GlideVK: the Eidos and Ion Storm
> logos, the flying title, the lit metallic emblem and the menu text all render,
> and in-game resolution changes switch the (Wine-emulated) display mode like
> a 3dfx driver did (see
> [`docs/game-compatibility.md`](docs/game-compatibility.md)). Behaviour is
> checked **against a reference Glide DLL** (nGlide) with
> `examples/run_diff.sh`, which found five root causes in a row: Glide's s/t
> span `[0,256)` on the major axis, not texels; the blend-factor table was
> permuted; palettes apply at draw time, not bind time; W-buffer depth
> comes from `oow`, not `ooz`; and the `gu` combine helpers must expand
> exactly as the SDK's `gu.c` does. The Wine smoke
> tests (`examples/run_wine_probe.sh [32|64]`) still pass on the 32-bit ship
> target.

Origin: proposed in [lutris/dgvoodoo2#5](https://github.com/lutris/dgvoodoo2/issues/5)
as the missing FOSS piece for legacy-API coverage on Linux, alongside D7VK,
nGlide, and cncdraw.

## Layout

```
crates/
  glidevk-core/   Backend-agnostic core: Glide state machine, texture format
                  conversion, LFB emulation, state-hash -> pipeline/shader cache.
                  Host-buildable and unit-tested (no Vulkan loader required).
  glide3x/        The glide3x.dll shell — Glide 3.x entry points -> core.
docs/
  feasibility.md  Feasibility study, architecture, and roadmap.
```

  glide2x/        The glide2x.dll shell — Glide 2.x entry points -> core
                  (fixed GrVertex, gu* helpers, inverted texture LOD).

## Building

### Core (host — for development & tests)

```sh
cargo test                       # runs core + shell unit tests natively
cargo check -p glidevk-core --features vulkan   # type-check the ash backend
```

The core builds on any host so the state machine, texture decode, and LFB
packing can be developed and tested without a Vulkan loader.

### Differential tests (against a reference Glide DLL)

```sh
examples/run_diff.sh                    # wine_texcoord beside nGlide and beside ours, diffed
GLIDE_REF=/path/glide2x.dll examples/run_diff.sh wine_texcoord wine_blendmode
GLIDE_API=glide3x examples/run_diff.sh wine3_palette   # glide3x shell, wine3_* harnesses
```

Harnesses that read back through `grLfbReadRegion` run unchanged against any
Glide 2 implementation, so the reference defines correct behaviour. See
`docs/game-compatibility.md` § *Differential testing*.

### glide3x.dll (Wine target — the shipping artifact)

Glide games are 32-bit, so the primary target is `i686-pc-windows-gnu`
(`x86_64-pc-windows-gnu` also builds, for the rare 64-bit consumer):

```sh
rustup target add i686-pc-windows-gnu
cargo build -p glide3x --release --target i686-pc-windows-gnu
```

On macOS, some i686 MinGW packages (including Homebrew's current one) use the
SJLJ exception runtime. Rust's prebuilt `i686-pc-windows-gnu` standard library
expects the DWARF-2 `_Unwind_Resume` symbol, so those runtimes fail at link
time with `undefined reference to '_Unwind_Resume'`. The workspace's
`panic = "abort"` release setting does not change the already-built Rust std.
This matches [Rust issue #158933](https://github.com/rust-lang/rust/issues/158933);
the `build-std` workaround is tracked in [Rust issue #79609](https://github.com/rust-lang/rust/issues/79609).

Use an i686 MinGW runtime that provides DWARF-2 unwinding, or rebuild Rust std
with the project's aborting panic strategy using nightly:

```sh
rustup toolchain install nightly
rustup component add rust-src --toolchain nightly
rustup target add --toolchain nightly i686-pc-windows-gnu
cargo +nightly build -Z build-std=std,panic_abort -p glide3x --release --target i686-pc-windows-gnu
```

Replace `glide3x` with `glide2x` to build that DLL. The target must remain
installed because `build-std` does not provide Rust's Windows startup objects.

### GitHub Actions DLL artifacts and releases

The [Windows DLLs workflow](.github/workflows/windows-dlls.yml) runs the native
tests and Clippy, cross-builds both shipping PE32 DLLs with the pinned nightly
`build-std` workaround, verifies that they are i686 DLLs with the expected
exports, and uploads:

- `glide2x.dll`
- `glide3x.dll`
- `SHA256SUMS`

Every push, pull request, and manual workflow run gets a `glidevk-i686-windows`
workflow artifact. Pushing a tag beginning with `v` also attaches those same
three files to the corresponding GitHub Release.

Exports are forced undecorated via `#[export_name]` and mirrored in
[`crates/glide3x/glide3x.def`](crates/glide3x/glide3x.def) for the MinGW linker.
Vulkan is reached through the in-prefix `vulkan-1.dll` → `winevulkan`, exactly as
DXVK does. (`x86_64-w64-mingw32-gcc` and `wine` are the runtime prerequisites.)

### Frame-rate limit

GlideVK limits presentation to 60 frames per second by default in both Glide 2
and Glide 3. Set the environment variable `GLIDEVK_FPS` to a whole number from
1 to 1000 to change the limit, or set `GLIDEVK_FPS=0` to disable the limit.
Invalid values fall back to 60. Set it before starting the game. The limit is a
ceiling; rendering speed and display synchronization may produce a lower rate.

### Install into a Wine prefix (once the backend lands)

```sh
cp target/i686-pc-windows-gnu/release/glide3x.dll "$WINEPREFIX/drive_c/windows/system32/"
# then set glide3x to native,builtin as a Wine DLL override
```

## Roadmap

- **P0** ✅ — DLL loads under Wine; Vulkan reached via winevulkan; `grSstWinOpen`
  builds a Win32 surface + swapchain; `grBufferClear`/`grBufferSwap` clear +
  present. Verified end-to-end (headless probe + 60-frame windowed present).
- **P1** ✅ — first triangle: `grDrawTriangle` / `grDrawVertexArray` /
  `grDrawVertexArrayContiguous` → Vulkan draw (render pass, passthrough Gouraud
  pipeline, persistently-mapped vertex buffer, `grVertexLayout` parsing,
  window→clip transform, primitive assembly for list/strip/fan). Verified via
  center-pixel readback on 32- and 64-bit.
- **P2** ✅ — textures + combine units. `grTexDownloadMipMap`/`grTexSource`
  decode (RGB565/1555/4444/P8/A8/I8) and upload to GPU images with descriptor
  sets; a combiner **ubershader** implements the Glide color/alpha combine units
  driven by push constants (`grColorCombine`/`grAlphaCombine`/`grTexCombine`,
  `GR_PARAM_ST0`). Verified via center-pixel readback on 32- and 64-bit.
  (Chose an ubershader over per-key GLSL→SPIR-V codegen: identical output, and
  it dodges a naga 30.0 frontend bug — per-state shader specialization is a
  deferred P4 perf optimization.)
- **P3a** ✅ — alpha blend (`grAlphaBlendFunction`), depth buffer (`grDepthBufferMode`/
  `Function`/`grDepthMask`, `GR_PARAM_Z`), alpha test + chroma-key discard. The
  pipeline cache keys `VkPipeline`s on baked blend/depth state; combine/chroma/
  alpha-test are push-constant driven. Verified via center-pixel readback.
- **P3b (LFB)** ✅ — `grLfbLock`/`grLfbUnlock` write path: the game's 16/32-bit
  framebuffer writes are unpacked (RGB565/555/1555/8888) and composited into the
  presented image on swap. Verified via `examples/wine_lfb.c` (LFB-written yellow
  overwrites a blue clear). This is the path Diablo II's 2D/menus use. Remaining
  P3b: **fog** (per-pixel fog blend + table), read-locks / partial-region LFB
  preservation, and the offscreen-backbuffer model for finely interleaved
  LFB↔triangle ordering.
- **glide2x shell** ✅ — `crates/glide2x` reusing `glidevk-core`: fixed
  `GrVertex`, `grSstQueryHardware`, inverted texture LOD, and the `gu*` combine
  helpers. Exports cover **Deus Ex / POD / Undying**'s `glide2x` imports, and it
  renders a textured/combined triangle correctly under Wine (verified via
  `examples/wine_glide2.c`). See [`docs/game-compatibility.md`](docs/game-compatibility.md).

### Real-game status
- **`glide2x.dll`** — verified drop-in for the Unreal-engine games, POD, and the
  1997–98 catalogue (covers their import tables; renders through the fixed
  `GrVertex` path). Best first real-render target once 64-bit prefixes are in place.
- **`glide3x.dll`** — now exports the **complete documented Glide 3.x API
  surface** (every entry in [`glide3x.def`](crates/glide3x/glide3x.def) has a
  body — the P0–P3 render path wired to `glidevk-core`, plus state save/restore,
  screen/origin queries, single-level texture download, tex-mem accounting,
  clamp-mode select, and the `guFogGenerate*` table builders; remaining
  future-milestone calls — fog render, NCC, region LFB — are accepted no-ops).
  Verified loadable drop-in for **Diablo II** (all 35 `D2Glide.dll` imports);
  full-frame LFB (P3b) is done, region-LFB is the last piece for its 2D/menus.
- **P4** — fidelity (16-bit dither / 22-bit post-filter, NCC textures),
  performance, HUD/config, Lutris install path.

## Licensing

GlideVK code: `MIT OR Apache-2.0`. The reference material (open 3dfx Glide SDK,
SPDX `Glide`) is a permissive, source-available license permitting derivative
works; API semantics and export signatures are taken from it. This is a clean,
from-scratch FOSS implementation — the entire point of the originating thread.
