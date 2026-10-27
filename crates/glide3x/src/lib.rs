//! `glide3x` — the 3dfx Glide 3.x API entry points.
//!
//! This is a thin shell: each `grXxx` export validates/translates its arguments
//! and drives [`glidevk_core`]. It carries no rendering logic of its own.
//!
//! ## Calling convention & exports
//!
//! Glide uses `FX_CALL` == `__stdcall` on Win32. On the 32-bit target
//! (`i686-pc-windows-gnu` — where nearly all Glide games live) `extern "system"`
//! *is* stdcall; on x86_64 there is a single ABI and it is a no-op. We force the
//! **undecorated** export names 3dfx games import (`grDrawTriangle`, not
//! `_grDrawTriangle@N`) with `#[export_name]`, backed by `glide3x.def` for the
//! MinGW linker.
//!
//! The full documented API surface (see [`glide3x.def`]) is now exported: the
//! P0–P3 render path has real bodies, and the remaining entry points are either
//! wired to `glidevk-core` or accepted-and-validated no-ops for state we don't
//! yet act on (fog render, NCC, region LFB — future milestones) via [`stub!`],
//! so no game fails to load on a missing import. Signatures follow `glide.h` /
//! `g3ext.h` from the open SDK; on the 32-bit stdcall target the argument counts
//! must match the prototypes exactly to keep the call stack balanced.
//!
//! [`glide3x.def`]: ../../../crates/glide3x/glide3x.def
#![allow(non_snake_case, clippy::missing_safety_doc)]

use std::sync::Mutex;

use glidevk_core::state::DepthMode;
use glidevk_core::types::*;
use glidevk_core::vk::{BlendConfig, CombineParams, DepthConfig, DrawVertex};
use glidevk_core::Context;

/// The single active Glide context. Glide is effectively single-context in
/// practice; a `Mutex<Option<_>>` keeps the shell `Send`/`Sync` and null-safe
/// before `grSstWinOpen`.
static CONTEXT: Mutex<Option<Context>> = Mutex::new(None);

fn with_ctx<R>(f: impl FnOnce(&mut Context) -> R) -> Option<R> {
    let mut guard = CONTEXT.lock().ok()?;
    guard.as_mut().map(f)
}

// ---------------------------------------------------------------------------
// Library lifecycle
// ---------------------------------------------------------------------------

#[export_name = "grGlideInit"]
pub extern "system" fn grGlideInit() {
    // No global device yet; per-window setup happens in grSstWinOpen.
}

#[export_name = "grGlideShutdown"]
pub extern "system" fn grGlideShutdown() {
    if let Ok(mut g) = CONTEXT.lock() {
        *g = None;
    }
}

#[export_name = "grGlideGetVersion"]
pub unsafe extern "system" fn grGlideGetVersion(version: *mut u8) {
    // "Glide3x" — 8 bytes incl. NUL, matching what apps expect.
    if !version.is_null() {
        let s = b"Glide3x\0";
        std::ptr::copy_nonoverlapping(s.as_ptr(), version, s.len());
    }
}

/// Saved render states keyed by a token written into the caller's state blob.
/// `grGlideGetState` snapshots the live [`State`] and hands the game an opaque
/// id; `grGlideSetState` restores by that id. Using the caller's own buffer as
/// the id carrier keeps multiple saved states independent and alias-free.
static SAVED_STATES: Mutex<Option<std::collections::HashMap<u32, glidevk_core::State>>> =
    Mutex::new(None);
static NEXT_STATE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

/// `grGlideGetState(state)` — snapshot the current render state into `state`
/// (opaque to the game; sized per `grGet(GLIDE_STATE_SIZE)`).
#[export_name = "grGlideGetState"]
pub unsafe extern "system" fn grGlideGetState(state: *mut core::ffi::c_void) {
    if state.is_null() {
        return;
    }
    with_ctx(|ctx| {
        let id = NEXT_STATE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Ok(mut g) = SAVED_STATES.lock() {
            g.get_or_insert_with(Default::default)
                .insert(id, ctx.state.clone());
        }
        (state as *mut u32).write_unaligned(id);
    });
}

/// `grGlideSetState(state)` — restore a state previously captured by
/// `grGlideGetState`. Unknown tokens are ignored.
#[export_name = "grGlideSetState"]
pub unsafe extern "system" fn grGlideSetState(state: *const core::ffi::c_void) {
    if state.is_null() {
        return;
    }
    let id = (state as *const u32).read_unaligned();
    let restored = SAVED_STATES
        .lock()
        .ok()
        .and_then(|g| g.as_ref().and_then(|m| m.get(&id).cloned()));
    if let Some(s) = restored {
        with_ctx(|ctx| ctx.state = s);
    }
}

// ---------------------------------------------------------------------------
// Context / window
// ---------------------------------------------------------------------------

/// `grSstWinOpen(hwnd, res, refresh, cformat, origin, nColBuffers, nAuxBuffers)`
/// -> context handle (0 on failure).
///
/// On the Windows/Wine build this binds a winevulkan swapchain to `hwnd`; on a
/// native build it creates the context without a surface (the DLL logic and its
/// tests still run).
#[export_name = "grSstWinOpen"]
pub extern "system" fn grSstWinOpen(
    hwnd: usize,
    screen_resolution: u32,
    _refresh_rate: i32,
    _color_format: i32,
    origin_location: i32,
    _num_color_buffers: i32,
    _num_aux_buffers: i32,
) -> GrContext {
    let mut guard = match CONTEXT.lock() {
        Ok(g) => g,
        Err(_) => return 0,
    };
    // `mut` is used only on the Windows path (backend assignment).
    #[allow(unused_mut)]
    let mut ctx = Context::new();

    #[cfg(windows)]
    {
        let (w, h) = resolution_to_size(screen_resolution);
        // Glide is fullscreen-only: switch the display like a 3dfx driver did
        // (Wine emulates it by scaling the window); GR_RESOLUTION_NONE (0xff)
        // means "use the window" and GLIDEVK_WINDOWED=1 opts out.
        if screen_resolution != ScreenResolution::None as u32
            && std::env::var_os("GLIDEVK_WINDOWED").is_none()
            && glidevk_core::win32::set_display_mode(w, h).is_ok()
        {
            glidevk_core::win32::make_fullscreen_window(hwnd as isize, w, h);
        }
        // SAFETY: GetModuleHandleW(NULL) returns this process's base HINSTANCE.
        let hinstance = unsafe { GetModuleHandleW(core::ptr::null()) };
        match glidevk_core::vk::Backend::new_windowed(hwnd as isize, hinstance, w, h) {
            Ok(mut backend) => {
                // GR_ORIGIN_UPPER_LEFT == 0 (Vulkan-native); LOWER_LEFT == 1.
                backend.set_origin_upper_left(origin_location == 0);
                ctx.backend = Some(backend);
            }
            Err(_e) => return 0, // TODO: route to grErrorSetCallback
        }
    }
    #[cfg(not(windows))]
    let _ = (hwnd, screen_resolution, origin_location);

    *guard = Some(ctx);
    1 // non-zero context handle
}

#[cfg(windows)]
extern "system" {
    fn GetModuleHandleW(module_name: *const u16) -> isize;
}

#[export_name = "grSstWinClose"]
pub extern "system" fn grSstWinClose(_context: GrContext) -> FxBool {
    if let Ok(mut g) = CONTEXT.lock() {
        *g = None;
    }
    #[cfg(windows)]
    if std::env::var_os("GLIDEVK_WINDOWED").is_none() {
        glidevk_core::win32::restore_display_mode();
    }
    1
}

#[export_name = "grSstScreenWidth"]
pub extern "system" fn grSstScreenWidth() -> u32 {
    with_ctx(|ctx| ctx.backend.as_ref().map(|b| b.logical_size().0))
        .flatten()
        .unwrap_or(640)
}

#[export_name = "grSstScreenHeight"]
pub extern "system" fn grSstScreenHeight() -> u32 {
    with_ctx(|ctx| ctx.backend.as_ref().map(|b| b.logical_size().1))
        .flatten()
        .unwrap_or(480)
}

/// `grSstOrigin(origin)` — GR_ORIGIN_UPPER_LEFT (0, Vulkan-native) or
/// GR_ORIGIN_LOWER_LEFT (1). Flips the window→clip Y transform in the backend.
#[export_name = "grSstOrigin"]
pub extern "system" fn grSstOrigin(origin: i32) {
    with_ctx(|ctx| {
        if let Some(backend) = ctx.backend.as_mut() {
            backend.set_origin_upper_left(origin == 0);
        }
    });
}

// ---------------------------------------------------------------------------
// Frame lifecycle
// ---------------------------------------------------------------------------

#[export_name = "grBufferSwap"]
pub extern "system" fn grBufferSwap(_swap_interval: i32) {
    with_ctx(|ctx| {
        if let Some(backend) = ctx.backend.as_mut() {
            // End the frame (render pass begun at grBufferClear / first draw)
            // and present it.
            if backend.present().is_err() {
                ctx.present_errors += 1;
            }
        }
    });
}

#[export_name = "grBufferClear"]
pub extern "system" fn grBufferClear(color: u32, alpha: u8, depth: u32) {
    with_ctx(|ctx| {
        // Glide clear color is packed per the context color format; assume RGBA.
        let r = ((color >> 24) & 0xff) as f32 / 255.0;
        let g = ((color >> 16) & 0xff) as f32 / 255.0;
        let b = ((color >> 8) & 0xff) as f32 / 255.0;
        let a = alpha as f32 / 255.0;
        ctx.clear_color = [r, g, b, a];
        if let Some(backend) = ctx.backend.as_mut() {
            if backend
                .clear_masked(
                    [r, g, b, a],
                    depth,
                    ctx.state.color_mask_rgb,
                    ctx.state.color_mask_alpha,
                    ctx.state.depth_write,
                )
                .is_err()
            {
                ctx.present_errors += 1;
            }
        }
    });
}

#[export_name = "grRenderBuffer"]
pub extern "system" fn grRenderBuffer(_buffer: i32) {}

// ---------------------------------------------------------------------------
// Combine / blend state  (feeds glidevk_core::State -> PipelineKey)
// ---------------------------------------------------------------------------

#[export_name = "grColorCombine"]
pub extern "system" fn grColorCombine(
    function: i32,
    factor: i32,
    local: i32,
    other: i32,
    invert: FxBool,
) {
    with_ctx(|ctx| {
        ctx.state.color_combine.function = CombineFunction::from_glide(function);
        ctx.state.color_combine.factor = CombineFactor::from_glide(factor);
        ctx.state.color_combine.local = CombineLocal::from_glide(local);
        ctx.state.color_combine.other = CombineOther::from_glide(other);
        ctx.state.color_combine.invert = invert != 0;
    });
}

#[export_name = "grAlphaCombine"]
pub extern "system" fn grAlphaCombine(
    function: i32,
    factor: i32,
    local: i32,
    other: i32,
    invert: FxBool,
) {
    with_ctx(|ctx| {
        ctx.state.alpha_combine.function = CombineFunction::from_glide(function);
        ctx.state.alpha_combine.factor = CombineFactor::from_glide(factor);
        ctx.state.alpha_combine.local = CombineLocal::from_glide(local);
        ctx.state.alpha_combine.other = CombineOther::from_glide(other);
        ctx.state.alpha_combine.invert = invert != 0;
    });
}

#[export_name = "grAlphaBlendFunction"]
pub extern "system" fn grAlphaBlendFunction(
    rgb_src: i32,
    rgb_dst: i32,
    alpha_src: i32,
    alpha_dst: i32,
) {
    with_ctx(|ctx| {
        ctx.state.blend_src_rgb = BlendFactor::from_glide(rgb_src, true);
        ctx.state.blend_dst_rgb = BlendFactor::from_glide(rgb_dst, false);
        ctx.state.blend_src_alpha = BlendFactor::from_glide(alpha_src, true);
        ctx.state.blend_dst_alpha = BlendFactor::from_glide(alpha_dst, false);
    });
}

#[export_name = "grConstantColorValue"]
pub extern "system" fn grConstantColorValue(color: u32) {
    with_ctx(|ctx| {
        // Glide constant color is packed per the color format; assume ARGB here.
        let a = ((color >> 24) & 0xff) as f32 / 255.0;
        let r = ((color >> 16) & 0xff) as f32 / 255.0;
        let g = ((color >> 8) & 0xff) as f32 / 255.0;
        let b = (color & 0xff) as f32 / 255.0;
        ctx.state.constant_color = [r, g, b, a];
    });
}

// ---------------------------------------------------------------------------
// Depth / fog / chroma
// ---------------------------------------------------------------------------

#[export_name = "grDepthBufferMode"]
pub extern "system" fn grDepthBufferMode(mode: i32) {
    with_ctx(|ctx| {
        ctx.state.depth_mode = match mode {
            0 => DepthMode::Disable,
            1 => DepthMode::ZBuffer,
            2 => DepthMode::WBuffer,
            _ => DepthMode::Disable,
        };
    });
}

#[export_name = "grDepthBufferFunction"]
pub extern "system" fn grDepthBufferFunction(function: i32) {
    // GR_CMP_* code, consumed directly by the pipeline cache.
    with_ctx(|ctx| ctx.state.depth_compare = function);
}

#[export_name = "grDepthMask"]
pub extern "system" fn grDepthMask(enable: FxBool) {
    with_ctx(|ctx| ctx.state.depth_write = enable != 0);
}

#[export_name = "grFogMode"]
pub extern "system" fn grFogMode(mode: i32) {
    with_ctx(|ctx| ctx.state.fog_enabled = mode != 0);
}

#[export_name = "grChromakeyMode"]
pub extern "system" fn grChromakeyMode(mode: i32) {
    with_ctx(|ctx| ctx.state.chromakey_enabled = mode != 0);
}

#[export_name = "grChromakeyValue"]
pub extern "system" fn grChromakeyValue(value: u32) {
    // Unpacked as RGBA8 (as clear/constant color). TODO: real chroma-key is
    // compared in the buffer's native format (e.g. RGB565).
    with_ctx(|ctx| {
        ctx.state.chroma_key = [
            ((value >> 24) & 0xff) as f32 / 255.0,
            ((value >> 16) & 0xff) as f32 / 255.0,
            ((value >> 8) & 0xff) as f32 / 255.0,
            (value & 0xff) as f32 / 255.0,
        ];
    });
}

#[export_name = "grAlphaTestFunction"]
pub extern "system" fn grAlphaTestFunction(function: i32) {
    with_ctx(|ctx| ctx.state.alpha_test_func = function);
}

#[export_name = "grAlphaTestReferenceValue"]
pub extern "system" fn grAlphaTestReferenceValue(value: u8) {
    with_ctx(|ctx| ctx.state.alpha_test_ref = value as f32 / 255.0);
}

// ---------------------------------------------------------------------------
// Vertex layout + drawing
// ---------------------------------------------------------------------------

/// Byte offsets, within the game's vertex struct, of the parameters we read.
/// Populated by `grVertexLayout`. XY is effectively always present; color is
/// either separate RGB(+A) floats (0..255) or a packed PARGB word.
#[derive(Clone, Copy)]
struct VertexLayout {
    xy: u32,
    z: Option<u32>,
    rgb: Option<u32>,
    a: Option<u32>,
    pargb: Option<u32>,
    st0: Option<u32>,
}

static LAYOUT: Mutex<VertexLayout> = Mutex::new(VertexLayout {
    xy: 0,
    z: None,
    rgb: None,
    a: None,
    pargb: None,
    st0: None,
});

// GR_PARAM_* (glide.h) — the subset P1/P2/P3 consume.
const GR_PARAM_XY: i32 = 0x01;
const GR_PARAM_Z: i32 = 0x02;
const GR_PARAM_A: i32 = 0x06;
const GR_PARAM_RGB: i32 = 0x07;
const GR_PARAM_PARGB: i32 = 0x08;
const GR_PARAM_ST0: i32 = 0x10;

// GR_* primitive types for grDrawVertexArray.
const GR_POLYGON: i32 = 3;
const GR_TRIANGLE_STRIP: i32 = 4;
const GR_TRIANGLE_FAN: i32 = 5;
const GR_TRIANGLES: i32 = 6;

#[export_name = "grVertexLayout"]
pub extern "system" fn grVertexLayout(param: i32, offset: i32, mode: i32) {
    let enabled = mode != 0; // GR_PARAM_ENABLE == 1
    if let Ok(mut l) = LAYOUT.lock() {
        let off = offset as u32;
        match param {
            GR_PARAM_XY => l.xy = off,
            GR_PARAM_Z => l.z = enabled.then_some(off),
            GR_PARAM_RGB => l.rgb = enabled.then_some(off),
            GR_PARAM_A => l.a = enabled.then_some(off),
            GR_PARAM_PARGB => l.pargb = enabled.then_some(off),
            GR_PARAM_ST0 => l.st0 = enabled.then_some(off),
            _ => {} // W/Q and ST1/ST2 (multitexture) land later.
        }
    }
}

#[inline]
unsafe fn read_f32(base: *const u8, off: u32) -> f32 {
    (base.add(off as usize) as *const f32).read_unaligned()
}

/// Read one game vertex into the backend's normalized form (window-space
/// position + RGBA in 0..1).
unsafe fn read_vertex(base: *const u8, l: &VertexLayout) -> DrawVertex {
    let pos = [read_f32(base, l.xy), read_f32(base, l.xy + 4)];
    let color = if let Some(o) = l.pargb {
        let p = (base.add(o as usize) as *const u32).read_unaligned(); // 0xAARRGGBB
        [
            ((p >> 16) & 0xff) as f32 / 255.0,
            ((p >> 8) & 0xff) as f32 / 255.0,
            (p & 0xff) as f32 / 255.0,
            ((p >> 24) & 0xff) as f32 / 255.0,
        ]
    } else {
        // Glide iterated color components are floats in [0, 255].
        let (r, g, b) = match l.rgb {
            Some(o) => (
                read_f32(base, o),
                read_f32(base, o + 4),
                read_f32(base, o + 8),
            ),
            None => (255.0, 255.0, 255.0),
        };
        let a = l.a.map(|o| read_f32(base, o)).unwrap_or(255.0);
        [r / 255.0, g / 255.0, b / 255.0, a / 255.0]
    };
    // TODO(P3): real Glide ST are texel-space (s/w, t/w) needing divide-by-tex-
    // dimension + perspective correction. Passed through as-is for now.
    let tex = l
        .st0
        .map(|o| [read_f32(base, o), read_f32(base, o + 4)])
        .unwrap_or([0.0, 0.0]);
    // TODO: real Glide Z is 1/w or a device-range value; passed through in 0..1.
    let depth = l.z.map(|o| read_f32(base, o)).unwrap_or(0.0);
    DrawVertex {
        pos,
        color,
        tex,
        depth,
        // TODO(glide3): perspective via GR_PARAM_Q. Affine (oow=1) for now; the
        // backend still normalizes `tex` by the bound texture size.
        oow: 1.0,
    }
}

/// Expand a primitive of `mode` into a triangle list.
fn to_triangle_list(mode: i32, v: &[DrawVertex]) -> Vec<DrawVertex> {
    let mut out = Vec::new();
    match mode {
        GR_TRIANGLES => {
            let (triangles, remainder) = v.as_chunks::<3>();
            debug_assert!(remainder.is_empty());
            for triangle in triangles {
                out.extend_from_slice(triangle);
            }
        }
        GR_TRIANGLE_STRIP => {
            for i in 0..v.len().saturating_sub(2) {
                if i % 2 == 0 {
                    out.extend_from_slice(&[v[i], v[i + 1], v[i + 2]]);
                } else {
                    out.extend_from_slice(&[v[i + 1], v[i], v[i + 2]]);
                }
            }
        }
        GR_TRIANGLE_FAN | GR_POLYGON => {
            for i in 1..v.len().saturating_sub(1) {
                out.extend_from_slice(&[v[0], v[i], v[i + 1]]);
            }
        }
        _ => {} // points/lines: not P1.
    }
    out
}

fn submit_triangles(tris: &[DrawVertex]) {
    with_ctx(|ctx| {
        let clamp = ctx.state.tex_clamp;
        let combine = CombineParams::from_state(&ctx.state);
        let blend = BlendConfig::from_state(&ctx.state);
        let depth = DepthConfig::from_state(&ctx.state);
        let chroma_enable = ctx.state.chromakey_enabled;
        let chroma = [
            ctx.state.chroma_key[0],
            ctx.state.chroma_key[1],
            ctx.state.chroma_key[2],
        ];
        let (at_func, at_ref) = (ctx.state.alpha_test_func, ctx.state.alpha_test_ref);
        ctx.bind_for_draw();
        if let Some(backend) = ctx.backend.as_mut() {
            backend.set_combine(combine);
            backend.set_blend(blend);
            backend.set_depth(depth);
            backend.set_chroma(chroma_enable, chroma);
            backend.set_alpha_test(at_func, at_ref);
            backend.set_clamp(clamp);
            let _ = backend.draw_triangles(tris);
        }
    });
}

#[export_name = "grDrawTriangle"]
pub unsafe extern "system" fn grDrawTriangle(
    a: *const core::ffi::c_void,
    b: *const core::ffi::c_void,
    c: *const core::ffi::c_void,
) {
    let l = match LAYOUT.lock() {
        Ok(l) => *l,
        Err(_) => return,
    };
    let tris = [
        read_vertex(a as *const u8, &l),
        read_vertex(b as *const u8, &l),
        read_vertex(c as *const u8, &l),
    ];
    submit_triangles(&tris);
}

#[export_name = "grDrawVertexArray"]
pub unsafe extern "system" fn grDrawVertexArray(
    mode: i32,
    count: i32,
    pointers: *const *const core::ffi::c_void,
) {
    if count <= 0 || pointers.is_null() {
        return;
    }
    let l = match LAYOUT.lock() {
        Ok(l) => *l,
        Err(_) => return,
    };
    let ptrs = std::slice::from_raw_parts(pointers, count as usize);
    let verts: Vec<DrawVertex> = ptrs
        .iter()
        .map(|&p| read_vertex(p as *const u8, &l))
        .collect();
    submit_triangles(&to_triangle_list(mode, &verts));
}

#[export_name = "grDrawVertexArrayContiguous"]
pub unsafe extern "system" fn grDrawVertexArrayContiguous(
    mode: i32,
    count: i32,
    vertices: *const core::ffi::c_void,
    stride: i32,
) {
    if count <= 0 || stride <= 0 || vertices.is_null() {
        return;
    }
    let l = match LAYOUT.lock() {
        Ok(l) => *l,
        Err(_) => return,
    };
    let base = vertices as *const u8;
    let verts: Vec<DrawVertex> = (0..count as usize)
        .map(|i| read_vertex(base.add(i * stride as usize), &l))
        .collect();
    submit_triangles(&to_triangle_list(mode, &verts));
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

/// Glide3 `GrTexInfo` (glide.h). LOD/aspect are log2 encodings.
#[repr(C)]
pub struct GrTexInfo {
    small_lod_log2: u32,
    large_lod_log2: u32,
    aspect_ratio_log2: u32,
    format: u32,
    data: *const core::ffi::c_void,
}

/// Map `GR_TEXFMT_*` to a core [`TexFormat`]. `None` = not yet decodable.
fn map_texfmt(v: u32) -> Option<TexFormat> {
    match v {
        0x02 => Some(TexFormat::Alpha8),
        0x03 => Some(TexFormat::Intensity8),
        0x05 => Some(TexFormat::P8),
        0x0a => Some(TexFormat::Rgb565),
        0x0b => Some(TexFormat::Argb1555),
        0x0c => Some(TexFormat::Argb4444),
        _ => None, // 332 / NCC / 8332 / 88: P4 fidelity work.
    }
}

fn texfmt_bpp(f: TexFormat) -> usize {
    match f {
        TexFormat::Rgb565
        | TexFormat::Argb1555
        | TexFormat::Argb4444
        | TexFormat::Argb8332
        | TexFormat::AlphaIntensity88 => 2,
        _ => 1,
    }
}

/// Texture dimensions from the LOD + aspect log2 fields. `large_lod_log2` is
/// log2 of the larger dimension; aspect narrows the other (wide ratios only for
/// now — tall ratios are a TODO).
fn tex_dims(info: &GrTexInfo) -> (u32, u32) {
    let dim = 1u32 << (info.large_lod_log2 & 0xff).min(11);
    match info.aspect_ratio_log2 {
        1 => (dim, (dim >> 1).max(1)),
        2 => (dim, (dim >> 2).max(1)),
        3 => (dim, (dim >> 3).max(1)),
        _ => (dim, dim),
    }
}

#[export_name = "grTexDownloadMipMap"]
pub unsafe extern "system" fn grTexDownloadMipMap(
    _tmu: u32,
    start_address: u32,
    _even_odd: u32,
    info: *const GrTexInfo,
) {
    if info.is_null() {
        return;
    }
    download_texture(start_address, &*info);
}

/// `grTexDownloadMipMapLevel` — download a single LOD level. We do not mipmap,
/// so only the base (largest) level is uploaded; smaller levels are ignored.
/// In Glide3 the LOD is a log2 code, so the base level is `this_lod == large_lod`.
#[export_name = "grTexDownloadMipMapLevel"]
pub unsafe extern "system" fn grTexDownloadMipMapLevel(
    _tmu: u32,
    start_address: u32,
    this_lod: u32,
    large_lod: u32,
    aspect_ratio_log2: u32,
    format: u32,
    _even_odd: u32,
    data: *const core::ffi::c_void,
) {
    if this_lod != large_lod {
        return;
    }
    let info = GrTexInfo {
        small_lod_log2: large_lod,
        large_lod_log2: large_lod,
        aspect_ratio_log2,
        format,
        data,
    };
    download_texture(start_address, &info);
}

/// Decode a texture from a `GrTexInfo` and upload it to the GPU at `start_address`.
/// Shared by `grTexDownloadMipMap` / `grTexDownloadMipMapLevel`.
unsafe fn download_texture(start_address: u32, info: &GrTexInfo) {
    if info.data.is_null() {
        return;
    }
    let fmt = match map_texfmt(info.format) {
        Some(f) => f,
        None => return,
    };
    let (w, h) = tex_dims(info);
    let src_len = (w as usize) * (h as usize) * texfmt_bpp(fmt);
    let src = std::slice::from_raw_parts(info.data as *const u8, src_len);
    with_ctx(|ctx| {
        // Palettised textures are decoded per draw against the palette live
        // at that moment (Glide applies its LUT when drawing; games upload a
        // texture's palette after binding it) — see `Context::bind_for_draw`.
        if matches!(fmt, TexFormat::P8) {
            ctx.download_p8(start_address, w, h, src);
            return;
        }
        let decoded = match ctx.textures.decode(fmt, w, h, src) {
            Ok(t) => t,
            Err(_) => return,
        };
        ctx.clear_p8(start_address);
        if let Some(backend) = ctx.backend.as_mut() {
            if backend
                .upload_texture(start_address as u64, w, h, &decoded.rgba)
                .is_ok()
                && ctx.bound_texture == start_address
            {
                backend.set_texture(start_address as u64);
            }
        }
    });
}

#[export_name = "grTexSource"]
pub extern "system" fn grTexSource(
    _tmu: u32,
    start_address: u32,
    _even_odd: u32,
    _info: *const GrTexInfo,
) {
    with_ctx(|ctx| {
        ctx.bind_texture(start_address);
    });
}

#[export_name = "grTexDownloadTable"]
pub unsafe extern "system" fn grTexDownloadTable(_table_type: i32, data: *const core::ffi::c_void) {
    if data.is_null() {
        return;
    }
    // Palette download for GR_TEXFMT_P_8: 256 ARGB entries.
    let src = std::slice::from_raw_parts(data as *const u32, 256);
    let mut palette = [0u32; 256];
    palette.copy_from_slice(src);
    with_ctx(|ctx| ctx.textures.set_palette(palette));
}

#[export_name = "grTexCombine"]
pub extern "system" fn grTexCombine(
    tmu: u32,
    rgb_function: i32,
    rgb_factor: i32,
    alpha_function: i32,
    alpha_factor: i32,
    rgb_invert: FxBool,
    alpha_invert: FxBool,
) {
    with_ctx(|ctx| {
        let tc = &mut ctx.state.tex_combine[(tmu as usize).min(2)];
        tc.rgb.function = CombineFunction::from_glide(rgb_function);
        tc.rgb.factor = CombineFactor::from_glide(rgb_factor);
        tc.rgb.invert = rgb_invert != 0;
        tc.alpha.function = CombineFunction::from_glide(alpha_function);
        tc.alpha.factor = CombineFactor::from_glide(alpha_factor);
        tc.alpha.invert = alpha_invert != 0;
        tc.enabled = true;
        // Consumed by the shader once multitexture (TMU1/2) lands; single-TMU0
        // draws sample tex0 directly and select it via grColorCombine other.
    });
}

/// `grTexClampMode(tmu, s_mode, t_mode)` — GR_TEXTURECLAMP_WRAP (0) /
/// _CLAMP (1) / _MIRROR_EXTEND (2). We map any non-wrap mode to a clamped
/// sampler (mirror is approximated as clamp for now).
#[export_name = "grTexClampMode"]
pub extern "system" fn grTexClampMode(_tmu: u32, s_mode: i32, t_mode: i32) {
    with_ctx(|ctx| ctx.state.tex_clamp = s_mode != 0 || t_mode != 0);
}

/// `grTexCalcMemRequired(smallLod, largeLod, aspect, format)` -> bytes needed in
/// TMU memory for a mipmap chain. We report the base (largest) level only, which
/// is what we actually upload; games use this to plan their texture heap.
#[export_name = "grTexCalcMemRequired"]
pub extern "system" fn grTexCalcMemRequired(
    _small_lod: u32,
    large_lod: u32,
    aspect: u32,
    format: u32,
) -> u32 {
    let info = GrTexInfo {
        small_lod_log2: large_lod,
        large_lod_log2: large_lod,
        aspect_ratio_log2: aspect,
        format,
        data: core::ptr::null(),
    };
    let (w, h) = tex_dims(&info);
    let bpp = map_texfmt(format).map(texfmt_bpp).unwrap_or(2);
    w * h * bpp as u32
}

/// `grTexTextureMemRequired(evenOdd, info)` -> bytes needed for the texture
/// described by `info`. Same accounting as `grTexCalcMemRequired`.
#[export_name = "grTexTextureMemRequired"]
pub unsafe extern "system" fn grTexTextureMemRequired(
    _even_odd: u32,
    info: *const GrTexInfo,
) -> u32 {
    if info.is_null() {
        return 0;
    }
    let info = &*info;
    let (w, h) = tex_dims(info);
    let bpp = map_texfmt(info.format).map(texfmt_bpp).unwrap_or(2);
    w * h * bpp as u32
}

// ---------------------------------------------------------------------------
// Capabilities, LFB, and misc state (needed so real games load + init)
// ---------------------------------------------------------------------------

/// `grGet` parameter selectors (`glide.h`). Only the ones real games query are
/// enumerated; the rest fall through to a benign default.
mod grget {
    pub const BITS_DEPTH: u32 = 0x01;
    pub const BITS_RGBA: u32 = 0x03;
    pub const FOG_TABLE_ENTRIES: u32 = 0x04;
    pub const GAMMA_TABLE_ENTRIES: u32 = 0x05;
    pub const GLIDE_STATE_SIZE: u32 = 0x06;
    pub const GLIDE_VERTEXLAYOUT_SIZE: u32 = 0x07;
    pub const IS_BUSY: u32 = 0x08;
    pub const LFB_PIXEL_PIPE: u32 = 0x09;
    pub const MAX_TEXTURE_SIZE: u32 = 0x0a;
    pub const MAX_TEXTURE_ASPECT_RATIO: u32 = 0x0b;
    pub const MEMORY_FB: u32 = 0x0c;
    pub const MEMORY_TMU: u32 = 0x0d;
    pub const MEMORY_UMA: u32 = 0x0e;
    pub const NUM_BOARDS: u32 = 0x0f;
    pub const NON_POWER_OF_TWO_TEXTURES: u32 = 0x10;
    pub const NUM_FB: u32 = 0x11;
    pub const NUM_TMU: u32 = 0x13;
    pub const REVISION_FB: u32 = 0x15;
    pub const REVISION_TMU: u32 = 0x16;
    pub const TEXTURE_ALIGN: u32 = 0x23;
}

const TMU_MEMORY_BYTES: u32 = 16 * 1024 * 1024;

/// `grGet(pname, plength, params)` — report capabilities. Writes up to `plength`
/// bytes into `params`, returns the number of bytes written. We advertise a
/// single Voodoo-ish TMU with 16 MB and 256px textures.
#[export_name = "grGet"]
pub unsafe extern "system" fn grGet(pname: u32, plength: u32, params: *mut i32) -> u32 {
    if params.is_null() || plength < 4 {
        return 0;
    }
    use grget::*;
    // Most queries return a single int; BITS_RGBA returns four.
    let write = |vals: &[i32]| {
        let n = (plength as usize / 4).min(vals.len());
        std::ptr::copy_nonoverlapping(vals.as_ptr(), params, n);
        (n * 4) as u32
    };
    match pname {
        BITS_DEPTH => write(&[16]),
        BITS_RGBA => write(&[8, 8, 8, 8]),
        FOG_TABLE_ENTRIES => write(&[64]),
        GAMMA_TABLE_ENTRIES => write(&[256]),
        GLIDE_STATE_SIZE => write(&[8192]),
        GLIDE_VERTEXLAYOUT_SIZE => write(&[256]),
        IS_BUSY => write(&[0]),
        LFB_PIXEL_PIPE => write(&[0]),
        MAX_TEXTURE_SIZE => write(&[256]),
        MAX_TEXTURE_ASPECT_RATIO => write(&[3]),
        MEMORY_FB => write(&[4 * 1024 * 1024]),
        MEMORY_TMU => write(&[TMU_MEMORY_BYTES as i32]),
        MEMORY_UMA => write(&[0]),
        NUM_BOARDS => write(&[1]),
        NON_POWER_OF_TWO_TEXTURES => write(&[0]),
        NUM_FB => write(&[1]),
        NUM_TMU => write(&[1]),
        REVISION_FB | REVISION_TMU => write(&[1]),
        TEXTURE_ALIGN => write(&[8]),
        _ => write(&[1]), // benign default for unhandled queries
    }
}

#[export_name = "grTexMinAddress"]
pub extern "system" fn grTexMinAddress(_tmu: u32) -> u32 {
    0
}

#[export_name = "grTexMaxAddress"]
pub extern "system" fn grTexMaxAddress(_tmu: u32) -> u32 {
    TMU_MEMORY_BYTES - 1
}

/// Glide3 `GrLfbInfo_t` (glide.h). `size` is set by the caller.
#[repr(C)]
pub struct GrLfbInfo {
    pub size: i32,
    pub lfb_ptr: *mut core::ffi::c_void,
    pub stride_in_bytes: u32,
    pub write_mode: i32,
    pub origin: i32,
}

#[export_name = "grLfbLock"]
pub unsafe extern "system" fn grLfbLock(
    _lock_type: u32,
    _buffer: u32,
    write_mode: i32,
    origin: i32,
    _pixel_pipeline: FxBool,
    info: *mut GrLfbInfo,
) -> FxBool {
    if info.is_null() {
        return 0;
    }
    with_ctx(|ctx| {
        if let Some(backend) = ctx.backend.as_mut() {
            let (ptr, stride) = backend.lfb_lock(write_mode);
            (*info).lfb_ptr = ptr as *mut core::ffi::c_void;
            (*info).stride_in_bytes = stride;
            (*info).write_mode = write_mode;
            (*info).origin = origin;
            1
        } else {
            0
        }
    })
    .unwrap_or(0)
}

#[export_name = "grLfbUnlock"]
pub extern "system" fn grLfbUnlock(_lock_type: u32, _buffer: u32) -> FxBool {
    with_ctx(|ctx| {
        if let Some(backend) = ctx.backend.as_mut() {
            backend.lfb_unlock();
        }
    });
    1
}

// No-op state setters — accepted so games load and run; wired to real behavior
// as needed. (grDrawLine/Point render nothing yet.)
/// `grColorMask(rgb, alpha)` — enable/disable colour / alpha writes.
#[export_name = "grColorMask"]
pub extern "system" fn grColorMask(rgb: FxBool, alpha: FxBool) {
    with_ctx(|ctx| {
        ctx.state.color_mask_rgb = rgb != 0;
        ctx.state.color_mask_alpha = alpha != 0;
    });
}
/// `GR_LFB_SRC_FMT_*` region writes and reads. Region LFB (partial-rectangle
/// upload/readback with source-format conversion) is deferred P3b work; the
/// full-frame `grLfbLock`/`Unlock` path above is what the target games use.
/// These accept the call and report success so a game's blit path doesn't error
/// out — they do not yet move pixels.
#[export_name = "grLfbWriteRegion"]
pub extern "system" fn grLfbWriteRegion(
    _dst_buffer: i32,
    _dst_x: u32,
    _dst_y: u32,
    _src_format: i32,
    _src_width: u32,
    _src_height: u32,
    _pixel_pipeline: FxBool,
    _src_stride: i32,
    _src_data: *const core::ffi::c_void,
) -> FxBool {
    1
}

/// `grLfbReadRegion`: a rectangle of the last presented frame as RGB565 (see
/// `Backend::read_region_rgb565` for the readback semantics).
///
/// # Safety
/// `dst_data` must point to at least `src_height * dst_stride` writable bytes.
#[export_name = "grLfbReadRegion"]
pub unsafe extern "system" fn grLfbReadRegion(
    _src_buffer: i32,
    src_x: u32,
    src_y: u32,
    src_width: u32,
    src_height: u32,
    dst_stride: u32,
    dst_data: *mut core::ffi::c_void,
) -> FxBool {
    if dst_data.is_null() || src_width == 0 || src_height == 0 {
        return 0;
    }
    let dst = std::slice::from_raw_parts_mut(
        dst_data as *mut u8,
        (src_height as usize) * (dst_stride as usize),
    );
    with_ctx(|ctx| match ctx.backend.as_mut() {
        Some(b) => {
            b.read_region_rgb565(src_x, src_y, src_width, src_height, dst_stride, dst);
            1
        }
        None => 0,
    })
    .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Fog table helpers (gu*) — pure computation the games rely on even before we
// render fog. `GR_FOG_TABLE_SIZE` = 64 (matches grGet(FOG_TABLE_ENTRIES)).
// ---------------------------------------------------------------------------

const GR_FOG_TABLE_SIZE: usize = 64;

/// Fill `fogtable` (64 `GrFog_t` bytes) with fog blend factors sampled from
/// `blend(w)`, where `w` is the eye-space W at each table index. Entry i=63 is
/// forced fully-fogged (255), per the Glide convention.
unsafe fn fill_fog_table(fogtable: *mut u8, blend: impl Fn(f32) -> f32) {
    if fogtable.is_null() {
        return;
    }
    let table = std::slice::from_raw_parts_mut(fogtable, GR_FOG_TABLE_SIZE);
    for (i, slot) in table.iter_mut().enumerate() {
        let w = gu_fog_index_to_w(i as i32);
        *slot = (blend(w).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
    table[GR_FOG_TABLE_SIZE - 1] = 255;
}

/// Canonical `guFogTableIndexToW`: `2^(i/4) / 8`.
fn gu_fog_index_to_w(i: i32) -> f32 {
    2.0_f32.powf(i.max(0) as f32 / 4.0) / 8.0
}

#[export_name = "guFogTableIndexToW"]
pub extern "system" fn guFogTableIndexToW(i: i32) -> f32 {
    gu_fog_index_to_w(i)
}

/// `guFogGenerateExp(fogtable, density)` — exponential fog `1 - e^(-density·w)`.
#[export_name = "guFogGenerateExp"]
pub unsafe extern "system" fn guFogGenerateExp(fogtable: *mut u8, density: f32) {
    fill_fog_table(fogtable, |w| 1.0 - (-density * w).exp());
}

/// `guFogGenerateExp2(fogtable, density)` — squared-exponential fog.
#[export_name = "guFogGenerateExp2"]
pub unsafe extern "system" fn guFogGenerateExp2(fogtable: *mut u8, density: f32) {
    fill_fog_table(fogtable, |w| {
        let d = density * w;
        1.0 - (-(d * d)).exp()
    });
}

/// `guFogGenerateLinear(fogtable, nearZ, farZ)` — linear ramp from `nearZ`
/// (no fog) to `farZ` (full fog).
#[export_name = "guFogGenerateLinear"]
pub unsafe extern "system" fn guFogGenerateLinear(fogtable: *mut u8, near_z: f32, far_z: f32) {
    let span = far_z - near_z;
    fill_fog_table(fogtable, |w| {
        if span.abs() < f32::EPSILON {
            0.0
        } else {
            (w - near_z) / span
        }
    });
}

// ---------------------------------------------------------------------------
// Accepted-but-minimal entry points. Argument counts match the Glide 3.x
// prototypes so the stdcall stack stays balanced on the 32-bit target.
// ---------------------------------------------------------------------------
stub!("grCoordinateSpace", gr_coordinate_space(mode: i32));
stub!("grDitherMode", gr_dither_mode(mode: i32));
stub!("grSstSelect", gr_sst_select(which: i32));
stub!("grTexFilterMode", gr_tex_filter_mode(tmu: u32, minf: i32, magf: i32));
stub!("grTexMipMapMode", gr_tex_mipmap_mode(tmu: u32, mode: i32, lod_blend: FxBool));
stub!("grTexLodBiasValue", gr_tex_lod_bias(tmu: u32, bias: f32));
stub!("grTexNCCTable", gr_tex_ncc_table(table: i32));
stub!("grFinish", gr_finish());
stub!("grFlush", gr_flush());
stub!("grReset", gr_reset(what: u32));
stub!("grEnable", gr_enable(mode: u32));
stub!("grDisable", gr_disable(mode: u32));
stub!("grErrorSetCallback", gr_error_set_callback(cb: *const core::ffi::c_void));
stub!("grCullMode", gr_cull_mode(mode: i32));
stub!("grDepthBiasLevel", gr_depth_bias_level(level: i32));
stub!("grFogTable", gr_fog_table(table: *const u8));
stub!("grDrawLine", gr_draw_line(a: *const core::ffi::c_void, b: *const core::ffi::c_void));
stub!("grDrawPoint", gr_draw_point(a: *const core::ffi::c_void));
stub!("grLoadGammaTable", gr_load_gamma(n: u32, r: *const u32, g: *const u32, b: *const u32));
stub!("guGammaCorrectionRGB", gu_gamma_rgb(r: f32, g: f32, b: f32));

// ---------------------------------------------------------------------------
// Diagnostics (not part of the Glide API)
// ---------------------------------------------------------------------------

/// Probe the Vulkan backend headlessly and copy the selected device name into
/// `out` (NUL-terminated, at most `len` bytes). Returns the string length, or
/// 0 on failure. Lets a Wine harness confirm winevulkan is reachable through
/// the PE boundary without needing a window/display.
#[export_name = "glidevkProbeDevice"]
pub unsafe extern "system" fn glidevkProbeDevice(out: *mut u8, len: i32) -> i32 {
    if out.is_null() || len <= 0 {
        return 0;
    }
    let name = match glidevk_core::vk::Backend::new_headless() {
        Ok(b) => b.device_name(),
        Err(_) => return 0,
    };
    let bytes = name.as_bytes();
    let n = bytes.len().min((len as usize).saturating_sub(1));
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, n);
    *out.add(n) = 0;
    n as i32
}

/// Number of failed `grBufferSwap` presents on the active context. 0 = every
/// frame presented cleanly. Lets a windowed harness verify the swapchain path
/// without inspecting pixels.
#[export_name = "glidevkFrameResult"]
pub extern "system" fn glidevkFrameResult() -> i32 {
    with_ctx(|ctx| ctx.present_errors as i32).unwrap_or(-1)
}

/// Center pixel of the last presented frame, packed `0xRRGGBBAA`. Lets a
/// harness confirm rasterization actually happened (triangle color at center
/// vs. the clear color) without a screen grab.
#[export_name = "glidevkCenterPixel"]
pub extern "system" fn glidevkCenterPixel() -> u32 {
    with_ctx(|ctx| {
        ctx.backend
            .as_ref()
            .map(|b| {
                let [r, g, b_, a] = b.center_pixel_rgba();
                (r as u32) << 24 | (g as u32) << 16 | (b_ as u32) << 8 | a as u32
            })
            .unwrap_or(0)
    })
    .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Enum mapping helpers (numeric GR_* -> core enums). Values per glide.h.
// ---------------------------------------------------------------------------

/// Declare a not-yet-implemented export that links and returns cleanly, so a
/// game importing it does not fail to load. Fill these in per milestone.
#[macro_export]
macro_rules! stub {
    ($name:literal, $rust:ident ( $($arg:ident : $ty:ty),* ) $(-> $ret:ty)?) => {
        #[export_name = $name]
        pub extern "system" fn $rust($(_: $ty),*) $(-> $ret)? {
            Default::default()
        }
    };
}

// A few representative stubs; the full remaining surface lives in glide3x.def.
stub!("grClipWindow", gr_clip_window(minx: u32, miny: u32, maxx: u32, maxy: u32));
stub!("grFogColorValue", gr_fog_color(color: u32));

#[cfg(test)]
mod tests {
    use super::*;
    /// These tests exercise the process-global Glide context and therefore
    /// must not run concurrently under Cargo's default test scheduler.
    static CONTEXT_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn context_lifecycle_and_state_routing() {
        let _guard = CONTEXT_TEST_LOCK.lock().unwrap();
        assert_eq!(grSstWinOpen(0, 0, -1, 0, 0, 2, 1), 1);
        grColorCombine(0x1, 0x0, 0x0, 0x1, 0); // Local / Iterated / Texture
        let hit = with_ctx(|ctx| {
            assert_eq!(ctx.state.color_combine.function, CombineFunction::Local);
            assert_eq!(ctx.state.color_combine.other, CombineOther::Texture);
            true
        });
        assert_eq!(hit, Some(true));

        // grGlideGetState snapshots the live state; a later mutation is undone by
        // grGlideSetState round-tripped through the caller's token buffer.
        let mut token = 0u32;
        unsafe {
            grGlideGetState(&mut token as *mut u32 as *mut core::ffi::c_void);
        }
        grColorCombine(0x0, 0x0, 0x0, 0x0, 0); // clobber -> Zero / Iterated
        assert_eq!(
            with_ctx(|ctx| ctx.state.color_combine.function),
            Some(CombineFunction::Zero)
        );
        unsafe {
            grGlideSetState(&token as *const u32 as *const core::ffi::c_void);
        }
        assert_eq!(
            with_ctx(|ctx| ctx.state.color_combine.function),
            Some(CombineFunction::Local)
        );

        grSstWinClose(1);
        assert_eq!(with_ctx(|_| ()), None);
    }

    #[test]
    fn tex_clamp_mode_toggles_flag() {
        let _guard = CONTEXT_TEST_LOCK.lock().unwrap();
        assert_eq!(grSstWinOpen(0, 0, -1, 0, 0, 2, 1), 1);
        grTexClampMode(0, 1, 0); // s = CLAMP
        assert_eq!(with_ctx(|ctx| ctx.state.tex_clamp), Some(true));
        grTexClampMode(0, 0, 0); // both WRAP
        assert_eq!(with_ctx(|ctx| ctx.state.tex_clamp), Some(false));
        grSstWinClose(1);
    }

    #[test]
    fn tex_mem_matches_dimensions() {
        // largeLod log2 = 8 -> 256px, aspect 0 -> 256x256, RGB565 (2 bpp).
        assert_eq!(grTexCalcMemRequired(8, 8, 0, 0x0a), 256 * 256 * 2);
        let info = GrTexInfo {
            small_lod_log2: 8,
            large_lod_log2: 8,
            aspect_ratio_log2: 1, // 2:1 -> 256x128
            format: 0x0c,         // ARGB4444, 2 bpp
            data: core::ptr::null(),
        };
        assert_eq!(
            unsafe { grTexTextureMemRequired(0, &info as *const _) },
            256 * 128 * 2
        );
    }

    #[test]
    fn fog_table_is_monotonic_and_saturates() {
        let mut t = [0u8; GR_FOG_TABLE_SIZE];
        unsafe { guFogGenerateExp(t.as_mut_ptr(), 1.0) };
        assert_eq!(t[GR_FOG_TABLE_SIZE - 1], 255, "far entry is fully fogged");
        assert!(
            t.windows(2).all(|w| w[0] <= w[1]),
            "exp fog is non-decreasing with distance"
        );
        assert!(
            guFogTableIndexToW(20) > guFogTableIndexToW(0),
            "w grows with index"
        );
    }
}
