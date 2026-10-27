//! `glide2x` — the 3dfx Glide 2.x API entry points.
//!
//! A thin shell over [`glidevk_core`], sibling to `glide3x`. The rendering core
//! (state machine, combine ubershader, textures, blend, depth, LFB) is shared;
//! this crate provides the Glide2-specific surface:
//!
//! * a **fixed `GrVertex`** struct read at hard-coded offsets (Glide2 has no
//!   `grVertexLayout`),
//! * Glide2's **inverted texture LOD** / aspect encoding,
//! * `grSstQueryHardware` board enumeration,
//! * the `gu*` high-level combine helpers most Glide2 games use.
//!
//! Targets the import sets of real games (Unreal-engine `GlideDrv.dll`, POD's
//! `PODX3Dfx.exe`). Signatures follow the Glide 2.x `glide.h` from the open SDK.
#![allow(non_snake_case, clippy::missing_safety_doc)]

use std::sync::Mutex;

use glidevk_core::state::DepthMode;
use glidevk_core::types::*;
use glidevk_core::vk::{BlendConfig, CombineParams, DepthConfig, DrawVertex};
use glidevk_core::Context;

static CONTEXT: Mutex<Option<Context>> = Mutex::new(None);
/// Frame counter (bumped each `grBufferSwap`) for frame-gated debug logging.
static FRAME: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// HWND backing the Vulkan surface when Glide's fullscreen API supplies NULL.
/// The Boolean records whether GlideVK created the window itself.
#[cfg(windows)]
static LAST_GLIDE_HWND: Mutex<Option<(isize, bool)>> = Mutex::new(None);
/// HWND to which the persistent backend's Vulkan surface is actually bound.
#[cfg(windows)]
static BACKEND_HWND: std::sync::atomic::AtomicIsize = std::sync::atomic::AtomicIsize::new(0);
#[cfg(windows)]
static WINDOWED_RESTORE: Mutex<Option<glidevk_core::win32::WindowedWindowState>> = Mutex::new(None);

/// `GLIDEVK_WINDOWED=1`: keep the desktop mode and make the game window
/// movable without changing its client size.
#[cfg(windows)]
fn windowed() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("GLIDEVK_WINDOWED")
            .map(|value| {
                let value = value.to_string_lossy();
                !value.is_empty() && value != "0"
            })
            .unwrap_or(false)
    })
}

/// Glide is fullscreen-only: switch the display to the requested mode like a
/// 3dfx driver did. Under Wine this is emulated by scaling the fullscreen
/// window and its mouse input, which is what makes resolution changes and
/// "fullscreen" work. Unreal sizes its window to `grSstScreenWidth/Height`
/// right after `grSstWinOpen`, so the window then covers the (emulated) screen.
#[cfg(windows)]
fn enter_display_mode(hwnd: usize, w: u32, h: u32) {
    if windowed() {
        let (old, already_saved) = match WINDOWED_RESTORE.lock() {
            Ok(mut state) => {
                let old = if state
                    .as_ref()
                    .is_some_and(|saved| saved.hwnd() != hwnd as isize)
                {
                    state.take()
                } else {
                    None
                };
                (old, state.is_some())
            }
            Err(_) => {
                glidevk_log("GLIDEVK_WINDOWED: window-state lock failed");
                return;
            }
        };
        if let Some(old) = old {
            let _ = glidevk_core::win32::restore_windowed_window(old);
        }
        match glidevk_core::win32::make_windowed_window(hwnd as isize) {
            Ok(Some(original)) => {
                if !already_saved {
                    if let Ok(mut state) = WINDOWED_RESTORE.lock() {
                        *state = Some(original);
                    }
                }
                glidevk_log(&format!(
                    "GLIDEVK_WINDOWED=1: added window decorations (hwnd={hwnd:#x}, logical={w}x{h})"
                ));
            }
            Ok(None) => glidevk_log(&format!(
                "GLIDEVK_WINDOWED=1: window already decorated (hwnd={hwnd:#x}, logical={w}x{h})"
            )),
            Err(code) => glidevk_log(&format!(
                "GLIDEVK_WINDOWED=1: could not decorate hwnd={hwnd:#x} (Win32 error {code})"
            )),
        }
        match glidevk_core::win32::resize_window_client(hwnd as isize, w, h) {
            Ok(()) => {
                let actual = glidevk_core::win32::window_info(hwnd as isize);
                glidevk_log(&format!(
                    "GLIDEVK_WINDOWED: requested client={w}x{h}, actual={actual:?}"
                ));
            }
            Err(code) => glidevk_log(&format!(
                "GLIDEVK_WINDOWED: could not resize hwnd={hwnd:#x} (Win32 error {code})"
            )),
        }
        return;
    }
    match glidevk_core::win32::set_display_mode(w, h) {
        Ok(()) => {
            let actual = glidevk_core::win32::make_fullscreen_window(hwnd as isize, w, h);
            glidevk_log(&format!(
                "display mode {w}x{h} set, fullscreen window={actual:?}"
            ));
        }
        Err(code) => {
            // Wine frequently rejects legacy modes such as 1600x1200 on macOS
            // (DISP_CHANGE_BADMODE). The game may still have its initial
            // 640x480 window, so retaining it produces a black square in the
            // upper-left corner. Cover the current monitor instead and keep the
            // requested mode as Glide's logical coordinate space; Vulkan scales
            // that space to the surface.
            let actual = glidevk_core::win32::make_fullscreen_window(hwnd as isize, w, h);
            glidevk_log(&format!(
                "display mode {w}x{h} FAILED (DISP_CHANGE {code}); \
                 borderless monitor fallback window={actual:?}"
            ));
        }
    }
}

/// Glide 2 fullscreen callers may pass NULL for `grSstWinOpen`'s HWND. NULL
/// has fullscreen semantics for older Glide hardware, but Vulkan still
/// needs a real HWND. Reuse the active game window when available so output
/// stays in the game's existing window; otherwise create a private per-process
/// window. Wine cannot create a Win32 Vulkan surface from HWND 0.
#[cfg(windows)]
fn resolve_surface_hwnd(requested: usize, hinstance: isize, w: u32, h: u32) -> Option<isize> {
    if requested != 0 {
        let hwnd = requested as isize;
        *LAST_GLIDE_HWND.lock().ok()? = Some((hwnd, false));
        return Some(hwnd);
    }
    let cached = {
        let mut last = LAST_GLIDE_HWND.lock().ok()?;
        match *last {
            Some((hwnd, owned)) if glidevk_core::win32::is_window(hwnd) => Some((hwnd, owned)),
            _ => {
                *last = None;
                None
            }
        }
    };
    if let Some((hwnd, owned)) = cached {
        if owned {
            glidevk_core::win32::show_glide_window(hwnd);
            glidevk_core::win32::resize_glide_window(hwnd, w, h);
        }
        return Some(hwnd);
    }

    let active = glidevk_core::win32::active_window();
    if active != 0 {
        *LAST_GLIDE_HWND.lock().ok()? = Some((active, false));
        glidevk_log(&format!(
            "grSstWinOpen: NULL HWND; using active game window {active:#x}"
        ));
        return Some(active);
    }

    let hwnd = glidevk_core::win32::create_glide_window(hinstance, w, h);
    if hwnd != 0 {
        *LAST_GLIDE_HWND.lock().ok()? = Some((hwnd, true));
        glidevk_log(&format!(
            "grSstWinOpen: created private Glide surface window {hwnd:#x} for NULL HWND"
        ));
        Some(hwnd)
    } else {
        None
    }
}

/// Whether `GLIDEVK_LOG` is set — cached, since this is consulted per draw.
fn log_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GLIDEVK_LOG").is_some())
}

/// Keep lifecycle, slow-call and performance logs without the high-volume
/// texture/vertex dumps and call trace used for rendering investigations.
fn profile_only() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GLIDEVK_PROFILE_ONLY").is_some())
}

fn detail_log_enabled() -> bool {
    log_enabled() && !profile_only()
}

fn texture_dump_dir() -> Option<&'static str> {
    use std::sync::OnceLock;
    static DIR: OnceLock<Option<String>> = OnceLock::new();
    DIR.get_or_init(|| std::env::var("GLIDEVK_TEXDUMP").ok())
        .as_deref()
}

/// `GLIDEVK_LOG=1` writes to `glidevk.log` in the game's working directory;
/// any other value is used as the path.
fn glidevk_log(msg: &str) {
    use std::io::Write;
    let path = match std::env::var("GLIDEVK_LOG") {
        Ok(v) if v == "1" || v.is_empty() => "glidevk.log".to_string(), // game's cwd
        Ok(v) => v,                                                     // explicit path
        Err(_) => return,
    };
    // Prefix with seconds since the first log line so lifecycle events
    // (re-opens, captures) can be placed against the game's own timeline.
    let t = log_elapsed();
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "[{t:8.3}] {msg}");
    }
}

use std::sync::atomic::Ordering;

/// Log `msg` the first time this exact `key` is seen — for enumerating all the
/// distinct render-state configurations a game uses without per-call spam.
/// `GLIDEVK_TRACE_FRAMES=<n>` or `<start>..<end>`: log every call to the traced
/// entry points for frames `[0, n)` or `[start, end)`, in order, so call
/// sequencing (texture download vs palette vs bind vs draw) can be read
/// directly.
fn trace_on() -> bool {
    if profile_only() {
        return false;
    }
    use std::sync::OnceLock;
    static RANGE: OnceLock<(u32, u32)> = OnceLock::new();
    static TIME: OnceLock<Option<(f32, f32)>> = OnceLock::new();
    let (start, end) = *RANGE.get_or_init(|| {
        let v = std::env::var("GLIDEVK_TRACE_FRAMES").unwrap_or_default();
        match v.split_once("..") {
            Some((a, b)) => (a.parse().unwrap_or(0), b.parse().unwrap_or(0)),
            None => (0, v.parse().unwrap_or(0)),
        }
    });
    let f = FRAME.load(Ordering::Relaxed);
    if f >= start && f < end {
        return true;
    }
    // `GLIDEVK_TRACE_TIME=<start>..<end>` in seconds since the first log line,
    // for when the frame number at the moment of interest is not predictable
    // (e.g. after a key press that loads the menu).
    let time = *TIME.get_or_init(|| {
        let v = std::env::var("GLIDEVK_TRACE_TIME").ok()?;
        let (a, b) = v.split_once("..")?;
        Some((a.parse().ok()?, b.parse().ok()?))
    });
    match time {
        Some((a, b)) => {
            let t = log_elapsed();
            t >= a && t < b
        }
        None => false,
    }
}

/// Seconds since the first log line (the timestamp the log prefix shows).
fn log_elapsed() -> f32 {
    use std::sync::OnceLock;
    static T0: OnceLock<std::time::Instant> = OnceLock::new();
    T0.get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f32()
}

static PRESENT_ERROR_LOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static DRAW_ERROR_LOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static CLEAR_ERROR_LOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn log_backend_error(
    operation: &str,
    error: &glidevk_core::vk::Error,
    count: &std::sync::atomic::AtomicU32,
) {
    if log_enabled() && count.fetch_add(1, Ordering::Relaxed) < 20 {
        glidevk_log(&format!(
            "{operation} FAILED frame={}: {error:?}",
            FRAME.load(Ordering::Relaxed)
        ));
    }
}
static DROPPED_DRAW_LOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static LFB_LOCK_LOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static LFB_WRITE_LOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn log_slow(operation: &str, elapsed: std::time::Duration, detail: impl FnOnce() -> String) {
    if !log_enabled() || elapsed < std::time::Duration::from_millis(100) {
        return;
    }
    glidevk_log(&format!(
        "slow {operation}: {:.1}ms {}",
        elapsed.as_secs_f64() * 1000.0,
        detail()
    ));
}

fn trace(msg: &str) {
    if trace_on() {
        glidevk_log(&format!("trace f{} {msg}", FRAME.load(Ordering::Relaxed)));
    }
}

fn log_distinct(key: u64, msg: &str) {
    use std::collections::HashSet;
    static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    if !log_enabled() {
        return;
    }
    if let Ok(mut g) = SEEN.lock() {
        if g.get_or_insert_with(HashSet::new).insert(key) {
            glidevk_log(msg);
        }
    }
}

/// Start address of the last texture bound via `grTexSource` — logged alongside
/// the texture-coordinate breakdown so a garbled overlay can be traced to its
/// exact TMU texture.
static LAST_TEX_ADDR: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Distinct textured primitives logged so far by [`log_texcoords`], bounded by
/// `TEXLOG_MAX` so an animating overlay can't flood the log.
static TEXLOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Binds that found no downloaded texture, counted per `perf` window.
static TEX_MISSES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Glide's texture-coordinate range on a texture's major axis (mirrors the
/// shader's `GLIDE_ST_RANGE`).
const GLIDE_ST_RANGE: f32 = 256.0;
const TEXLOG_MAX: u32 = 1500;

/// When `GLIDEVK_LOG` is set, log each *distinct* textured primitive once: the
/// raw Glide TMU0 coordinates (`sow`, `tow`), the `1/w` used for the perspective
/// divide, the bound texture size, and the resulting texel + normalized UV the
/// shader actually samples (`combine.frag::texCoord`). This is the ground-truth
/// data needed to root-cause the affine-overlay ST bug — coordinates landing far
/// outside `[0,1]` (the invisible Eidos logo, the dot-grid intro logos). A `1×1`
/// `texSize` here means `grTexSource` missed and the 1×1 white default is bound.
/// Compact one-line description of the combine/blend/const-colour state a draw
/// uses — appended to each `texdraw` line so a mis-shaded quad can be traced to
/// its exact pipeline state (e.g. is the green-dot backdrop modulating the green
/// constant colour, or is the texture itself green?).
fn shade_desc(s: &glidevk_core::state::State) -> String {
    let c = &s.color_combine;
    let a = &s.alpha_combine;
    format!(
        "colorC={:?}/{:?}/{:?}{} alphaC={:?}/{:?}/{:?} blend={:?}->{:?} const=({:.2},{:.2},{:.2},{:.2}) chroma={} atf={}",
        c.function, c.local, c.other,
        if c.invert { "!" } else { "" },
        a.function, a.local, a.other,
        s.blend_src_rgb, s.blend_dst_rgb,
        s.constant_color[0], s.constant_color[1], s.constant_color[2], s.constant_color[3],
        s.chromakey_enabled as u8, s.alpha_test_func
    )
}

fn log_texcoords(tris: &[DrawVertex], tex_size: [f32; 2], shade: &str) {
    use std::collections::HashSet;
    use std::hash::{Hash, Hasher};
    static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    if !detail_log_enabled() || tris.len() < 3 {
        return;
    }
    if TEXLOG_COUNT.load(Ordering::Relaxed) >= TEXLOG_MAX {
        return;
    }
    let addr = LAST_TEX_ADDR.load(Ordering::Relaxed);
    // Stable key = bound texture + quantized coords + a coarse frame bucket, so a
    // repeated overlay logs once per ~2s window. The bucket lets us sample across
    // the whole session (intro AND menu) instead of exhausting the cap on the
    // first animated seconds.
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (FRAME.load(Ordering::Relaxed) / 60).hash(&mut h);
    addr.hash(&mut h);
    tex_size[0].to_bits().hash(&mut h);
    tex_size[1].to_bits().hash(&mut h);
    for v in &tris[..3] {
        ((v.tex[0] * 16.0) as i64).hash(&mut h);
        ((v.tex[1] * 16.0) as i64).hash(&mut h);
        ((v.oow * 4096.0) as i64).hash(&mut h);
    }
    let key = h.finish();
    let fresh = match SEEN.lock() {
        Ok(mut g) => g.get_or_insert_with(HashSet::new).insert(key),
        Err(_) => return, // lock poisoned
    };
    if !fresh {
        return; // already logged this primitive
    }
    TEXLOG_COUNT.fetch_add(1, Ordering::Relaxed);

    let frame = FRAME.load(Ordering::Relaxed);
    let [tw, th] = tex_size;
    let mut msg = format!(
        "texdraw frame={frame} addr={addr:#x} texSize={tw}x{th} nverts={} {shade}",
        tris.len()
    );
    // Show the primitive's first three vertices, reproducing the shader's
    // mapping: st = sow/oow in Glide's [0,256) space (major axis), then
    // uv = st * major / (256 * texSize).
    for (i, v) in tris.iter().take(3).enumerate() {
        let w = if v.oow != 0.0 { v.oow } else { 1.0 };
        let (s, t) = (v.tex[0] / w, v.tex[1] / w);
        let major = tw.max(th).max(1.0);
        let (u, vcoord) = (
            s * major / (GLIDE_ST_RANGE * tw.max(1.0)),
            t * major / (GLIDE_ST_RANGE * th.max(1.0)),
        );
        msg.push_str(&format!(
            "\n  v{i} pos=({:.1},{:.1}) sow={:.3} tow={:.3} oow={:.6e} -> texel=({:.2},{:.2}) uv=({:.4},{:.4})",
            v.pos[0], v.pos[1], v.tex[0], v.tex[1], v.oow, s, t, u, vcoord
        ));
    }
    glidevk_log(&msg);
}

static RAWLOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
const RAWLOG_MAX: u32 = 400;

/// Dump the *raw* `GrVertex` fields (before `read_vertex` normalizes anything)
/// for the first vertex of a primitive, deduped + capped. This is the ground
/// truth for the ST-coordinate investigation: it shows every candidate `1/w`
/// slot — the vertex `oow` (offset 32) vs each TMU's `oow` (offset 44/56) — so we
/// can tell which one carries a valid perspective W for the 2D overlays, plus
/// `ooz`, colour, and both TMU coordinate pairs.
unsafe fn log_raw_vertex(base: *const u8, nverts: i32) {
    use std::collections::HashSet;
    use std::hash::{Hash, Hasher};
    static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    if !detail_log_enabled() {
        return;
    }
    if RAWLOG_COUNT.load(Ordering::Relaxed) >= RAWLOG_MAX {
        return;
    }
    let f = |off: usize| f32_at(base, off);
    let (x, y) = (f(V_X), f(V_Y));
    let addr = LAST_TEX_ADDR.load(Ordering::Relaxed);
    // Cap per bound texture so one busy texture cannot exhaust the budget
    // before the next one (e.g. a mesh skin) ever shows up.
    static PER_ADDR: Mutex<Option<std::collections::HashMap<u32, u32>>> = Mutex::new(None);
    const PER_ADDR_MAX: u32 = 4;
    match PER_ADDR.lock() {
        Ok(mut g) => {
            let n = g
                .get_or_insert_with(std::collections::HashMap::new)
                .entry(addr)
                .or_insert(0);
            if *n >= PER_ADDR_MAX {
                return;
            }
            *n += 1;
        }
        Err(_) => return,
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    addr.hash(&mut h);
    ((x * 16.0) as i64).hash(&mut h);
    ((y * 16.0) as i64).hash(&mut h);
    ((f(V_SOW0) * 64.0) as i64).hash(&mut h);
    ((f(V_TOW0) * 64.0) as i64).hash(&mut h);
    let key = h.finish();
    let fresh = match SEEN.lock() {
        Ok(mut g) => g.get_or_insert_with(HashSet::new).insert(key),
        Err(_) => return,
    };
    if !fresh {
        return;
    }
    RAWLOG_COUNT.fetch_add(1, Ordering::Relaxed);
    let frame = FRAME.load(Ordering::Relaxed);
    // Layout: x y z r g b ooz a oow | tmu0 sow tow oow | tmu1 sow tow oow
    glidevk_log(&format!(
        "rawvert frame={frame} addr={addr:#x} nverts={nverts} \
         x={x:.2} y={y:.2} z={:.4} rgb=({:.1},{:.1},{:.1}) ooz={:.4} a={:.1} oow={:.6e} | \
         tmu0 sow={:.4} tow={:.4} oow={:.6e} | tmu1 sow={:.4} tow={:.4} oow={:.6e}",
        f(8), // z
        f(V_R),
        f(V_G),
        f(V_B),
        f(V_OOZ),
        f(V_A),
        f(V_OOW),
        f(V_SOW0),
        f(V_TOW0),
        f(V_TMU_OOW0),
        f(48),
        f(52),
        f(56), // tmu1 sow/tow/oow
    ));
}

/// `GrColorFormat_t` from `grSstWinOpen` (0=ARGB 1=ABGR 2=RGBA 3=BGRA). Governs
/// how `GrColor_t` words (constant/clear/chroma) are byte-ordered.
static COLOR_FORMAT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// Unpack a `GrColor_t` to normalized RGBA per the active color format.
fn unpack_color(c: u32) -> [f32; 4] {
    let byte = |shift: u32| ((c >> shift) & 0xff) as f32 / 255.0;
    match COLOR_FORMAT.load(Ordering::Relaxed) {
        1 => [byte(0), byte(8), byte(16), byte(24)], // ABGR: 0xAABBGGRR
        2 => [byte(24), byte(16), byte(8), byte(0)], // RGBA: 0xRRGGBBAA
        3 => [byte(8), byte(16), byte(24), byte(0)], // BGRA: 0xBBGGRRAA
        _ => [byte(16), byte(8), byte(0), byte(24)], // ARGB: 0xAARRGGBB
    }
}

/// Unpack the argument of `grConstantColorValue`. Deus Ex's GlideDrv declares an
/// ABGR framebuffer (`grSstWinOpen` cformat=1) but supplies the *constant color*
/// packed as `0xRRGGBBAA` (RGBA) regardless — a well-known GlideDrv quirk that
/// real 3dfx hardware / nGlide honour. Interpreting it with the declared ABGR
/// order swaps R↔B and renders the Ion Storm / Eidos accent sprites dim blue
/// instead of their intended green. Confirmed empirically: the constant color
/// ramps `0x004e2818`, whose dominant byte (`0x4e`) sits in the green slot only
/// under RGBA. When the game declares RGBA/BGRA already, defer to `unpack_color`.
fn unpack_constant_color(c: u32) -> [f32; 4] {
    let byte = |shift: u32| ((c >> shift) & 0xff) as f32 / 255.0;
    match COLOR_FORMAT.load(Ordering::Relaxed) {
        1 => [byte(24), byte(16), byte(8), byte(0)], // ABGR-declared → RGBA-packed const
        _ => unpack_color(c),
    }
}

fn with_ctx<R>(f: impl FnOnce(&mut Context) -> R) -> Option<R> {
    let mut guard = CONTEXT.lock().ok()?;
    guard.as_mut().map(f)
}

/// Attribute rare draw stalls without emitting a line for every draw.
fn profile_draw_stage<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    static LINES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if !log_enabled() || !profile_only() {
        return f();
    }
    let start = std::time::Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    if elapsed >= std::time::Duration::from_millis(50) && LINES.fetch_add(1, Ordering::Relaxed) < 40
    {
        glidevk_log(&format!(
            "draw slow {name}: {:.1}ms frame={}",
            elapsed.as_secs_f64() * 1000.0,
            FRAME.load(Ordering::Relaxed)
        ));
    }
    result
}

fn with_draw_ctx<R>(f: impl FnOnce(&mut Context) -> R) -> Option<R> {
    let mut guard = profile_draw_stage("context_lock", || CONTEXT.lock()).ok()?;
    guard.as_mut().map(f)
}

// ---------------------------------------------------------------------------
// Fixed GrVertex layout (Glide 2.x, GLIDE_NUM_TMU = 2)
// ---------------------------------------------------------------------------
// struct GrVertex { float x,y,z, r,g,b, ooz, a, oow; GrTmuVertex tmuvtx[2]; }
// where GrTmuVertex { float sow, tow, oow; }.
const V_X: usize = 0;
const V_Y: usize = 4;
const V_R: usize = 12;
const V_G: usize = 16;
const V_B: usize = 20;
const V_OOZ: usize = 24;
const V_A: usize = 28;
const V_OOW: usize = 32; // vertex 1/w
const V_SOW0: usize = 36;
const V_TOW0: usize = 40;
const V_TMU_OOW0: usize = 44; // tmuvtx[0].oow (0 when the W hint routes to vertex oow)
/// sizeof(GrVertex) with GLIDE_NUM_TMU = 2 (36 + 2*12). Stride for vertex lists.
const GRVERTEX_SIZE: usize = 60;

#[inline]
unsafe fn f32_at(base: *const u8, off: usize) -> f32 {
    (base.add(off) as *const f32).read_unaligned()
}

/// Glide's sub-pixel snapping bias, `(3 << 18)` = 786432.0. Many Glide drivers
/// (incl. Unreal's GlideDrv) add this to vertex X/Y so the float mantissa aligns
/// for the hardware's fixed-point conversion; the real screen coordinate is the
/// value minus the bias. Coordinates below the threshold are unbiased (raw).
const SNAP_BIAS: f32 = (3 << 18) as f32;

#[inline]
fn unsnap(v: f32) -> f32 {
    if v >= (1 << 18) as f32 {
        v - SNAP_BIAS
    } else {
        v
    }
}

/// Read a fixed `GrVertex` into the backend's normalized form. Colors are Glide
/// floats in [0, 255]; ST come from TMU0's (s/w, t/w). Depth is derived from
/// `ooz` (65535/w): larger ooz = nearer, mapped to a smaller [0,1] depth.
unsafe fn read_vertex(base: *const u8) -> DrawVertex {
    let ooz = f32_at(base, V_OOZ);
    DrawVertex {
        pos: [unsnap(f32_at(base, V_X)), unsnap(f32_at(base, V_Y))],
        color: [
            f32_at(base, V_R) / 255.0,
            f32_at(base, V_G) / 255.0,
            f32_at(base, V_B) / 255.0,
            f32_at(base, V_A) / 255.0,
        ],
        // sow/tow (= s/w, t/w) + oow (= 1/w); the shader recovers + normalizes.
        tex: [f32_at(base, V_SOW0), f32_at(base, V_TOW0)],
        // Provisional Z-buffer depth from `ooz`; `apply_depth_mode` replaces it
        // from `oow` when the game runs a W-buffer (Unreal never sets `ooz`).
        depth: DepthMode::ZBuffer.vertex_depth(ooz, 0.0),
        // Perspective W (1/w) for the texture divide. In single-texture Glide the
        // vertex `oow` and `tmuvtx[0].oow` are the *same* perspective 1/w — but
        // Deus Ex's GlideDrv leaves `tmuvtx[0].oow` **uninitialized** for its draws.
        // That slot holds whatever was on the stack: sometimes the denormal
        // `1.345e-42` (raw int 960 = screen width), sometimes normal-looking junk
        // like `5.9e3` or `-1.7e38` — and it *varies run to run*. The old code
        // preferred that slot (only skipping an exact `0.0`), so it divided `sow`
        // by random memory, exploding texture coordinates to ±inf and making the
        // intro/menu render as garbage that changed on every launch.
        //
        // The vertex `oow` (offset 32, = 1/z) is the field GlideDrv actually
        // populates, so trust it. Fall back to the per-TMU slot only if the vertex
        // one is unusable (0/denormal), and to `1.0` (affine, no divide) if neither
        // is a normal float. Since the two are equal whenever a game sets both,
        // this can't regress correctly-written Glide programs.
        oow: {
            let vertex = f32_at(base, V_OOW);
            let tmu = f32_at(base, V_TMU_OOW0);
            if vertex.is_normal() {
                vertex
            } else if tmu.is_normal() {
                tmu
            } else {
                1.0
            }
        },
    }
}

/// Give each vertex the depth the active `grDepthBufferMode` implies (see
/// `DepthMode::vertex_depth`). W is interpolated per vertex, which is exact
/// for small primitives and close enough for the rest until depth moves to the
/// fragment shader.
fn apply_depth_mode(mode: DepthMode, tris: &mut [DrawVertex]) {
    if mode == DepthMode::WBuffer {
        for v in tris.iter_mut() {
            v.depth = mode.vertex_depth(0.0, v.oow);
        }
    }
}

/// Bind the texture selected by `grTexSource` for a draw (see
/// `Context::bind_for_draw`), logging each freshly decoded palette variant.
fn bind_draw_texture(ctx: &mut Context) {
    let t0 = log_enabled().then(std::time::Instant::now);
    let up = ctx.bind_for_draw();
    let elapsed = t0.map(|t| t.elapsed());
    if let Some(elapsed) = elapsed {
        log_slow("texture bind", elapsed, || {
            format!(
                "frame={} addr={:#x}",
                FRAME.load(Ordering::Relaxed),
                ctx.bound_texture
            )
        });
    }
    let Some(up) = up else {
        return;
    };
    if let Some(elapsed) = elapsed {
        perf_upload(elapsed.as_nanos() as u64);
    }
    let tex = &up.texture;
    if detail_log_enabled() {
        let n = (tex.rgba.len() / 4).max(1) as u64;
        let avg = |c: usize| {
            tex.rgba
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| p[c] as u64)
                .sum::<u64>()
                / n
        };
        log_distinct(
            8 << 40 | (up.addr as u64) << 8 | avg(0),
            &format!(
                "p8 variant frame={} addr={:#x} palette={} {}x{} avgRGB=({},{},{})",
                FRAME.load(Ordering::Relaxed),
                up.addr,
                up.palette_id,
                tex.width,
                tex.height,
                avg(0),
                avg(1),
                avg(2)
            ),
        );
    }
    if let Some(dir) = texture_dump_dir() {
        write_ppm(
            &format!(
                "{dir}/p8_{:#07x}_pal{}_{}x{}.ppm",
                up.addr, up.palette_id, tex.width, tex.height
            ),
            tex.width,
            tex.height,
            &tex.rgba,
        );
    }
}

/// Per-frame performance counters, reported every `PERF_EVERY` frames when
/// logging is on: fps, draws and triangles per frame, and the split between
/// CPU time spent in draw submission and time spent in `grBufferSwap` (which
/// includes the GPU wait).
struct Perf {
    draws: u64,
    tris: u64,
    uploads: u64,
    upload_ns: u64,
    submit_ns: u64,
    present_ns: u64,
    max_draw_ns: u64,
    max_upload_ns: u64,
    max_present_ns: u64,
    clears: u64,
    clears_after_draw: u64,
    frames_without_clear: u32,
    frame_had_clear: bool,
    frame_had_draw: bool,
    frames: u32,
    since: Option<std::time::Instant>,
}
impl Perf {
    fn start(now: std::time::Instant) -> Self {
        Self {
            draws: 0,
            tris: 0,
            uploads: 0,
            upload_ns: 0,
            submit_ns: 0,
            present_ns: 0,
            max_draw_ns: 0,
            max_upload_ns: 0,
            max_present_ns: 0,
            clears: 0,
            clears_after_draw: 0,
            frames_without_clear: 0,
            frame_had_clear: false,
            frame_had_draw: false,
            frames: 0,
            since: Some(now),
        }
    }
}
static PERF: Mutex<Perf> = Mutex::new(Perf {
    draws: 0,
    tris: 0,
    uploads: 0,
    upload_ns: 0,
    submit_ns: 0,
    present_ns: 0,
    max_draw_ns: 0,
    max_upload_ns: 0,
    max_present_ns: 0,
    clears: 0,
    clears_after_draw: 0,
    frames_without_clear: 0,
    frame_had_clear: false,
    frame_had_draw: false,
    frames: 0,
    since: None,
});
const PERF_EVERY: u32 = 120;

/// Record one texture decode+upload (`grTexDownload*`, or a palette variant).
fn perf_upload(ns: u64) {
    if !log_enabled() {
        return;
    }
    if let Ok(mut p) = PERF.lock() {
        p.uploads += 1;
        p.upload_ns += ns;
        p.max_upload_ns = p.max_upload_ns.max(ns);
    }
}

fn perf_draw(tris: u64, ns: u64) {
    if !log_enabled() {
        return;
    }
    if let Ok(mut p) = PERF.lock() {
        p.draws += 1;
        p.frame_had_draw = true;
        p.tris += tris;
        p.submit_ns += ns;
        p.max_draw_ns = p.max_draw_ns.max(ns);
    }
}

/// Count clears and whether they occur after geometry in the same frame.
fn perf_clear() {
    if !log_enabled() {
        return;
    }
    if let Ok(mut p) = PERF.lock() {
        p.clears += 1;
        p.clears_after_draw += u64::from(p.frame_had_draw);
        p.frame_had_clear = true;
    }
}

fn perf_frame(present_ns: u64) {
    if !log_enabled() {
        return;
    }
    let Ok(mut p) = PERF.lock() else {
        return;
    };
    if p.since.is_none() {
        // Discard startup work before the first swap; otherwise it is charged
        // to a later 120-frame sample whose timer starts only at that swap.
        *p = Perf::start(std::time::Instant::now());
        TEX_MISSES.store(0, Ordering::Relaxed);
        return;
    }
    p.present_ns += present_ns;
    p.max_present_ns = p.max_present_ns.max(present_ns);
    p.frames += 1;
    p.frames_without_clear += u32::from(!p.frame_had_clear);
    p.frame_had_clear = false;
    p.frame_had_draw = false;
    let since = p.since.expect("initialized above");
    if p.frames >= PERF_EVERY {
        let secs = since.elapsed().as_secs_f64().max(1e-6);
        let f = p.frames as f64;
        glidevk_log(&format!(
            "perf fps={:.1} draws/frame={:.0} tris/frame={:.0} submit={:.2}ms/frame present={:.2}ms/frame \
             uploads/frame={:.0} upload={:.2}ms/frame texmiss/frame={:.1} max_draw={:.2}ms max_upload={:.2}ms max_present={:.2}ms clears={} no_clear_frames={} clears_after_draw={}",
            f / secs,
            p.draws as f64 / f,
            p.tris as f64 / f,
            p.submit_ns as f64 / f / 1e6,
            p.present_ns as f64 / f / 1e6,
            p.uploads as f64 / f,
            p.upload_ns as f64 / f / 1e6,
            TEX_MISSES.swap(0, Ordering::Relaxed) as f64 / f,
            p.max_draw_ns as f64 / 1e6,
            p.max_upload_ns as f64 / 1e6,
            p.max_present_ns as f64 / 1e6,
            p.clears,
            p.frames_without_clear,
            p.clears_after_draw
        ));
        *p = Perf::start(std::time::Instant::now());
    }
}

fn submit_fan(verts: &[DrawVertex]) {
    if verts.len() < 3 {
        return;
    }
    let t0 = log_enabled().then(std::time::Instant::now);
    let mut tris = profile_draw_stage("fan_assembly", || {
        let mut tris = Vec::with_capacity((verts.len() - 2) * 3);
        for i in 1..verts.len() - 1 {
            tris.extend_from_slice(&[verts[0], verts[i], verts[i + 1]]);
        }
        tris
    });
    let mut tex_size = [1.0, 1.0];
    let mut shade = String::new();
    let mut draw_error = None;
    with_draw_ctx(|ctx| {
        let combine = CombineParams::from_state(&ctx.state);
        let blend = BlendConfig::from_state(&ctx.state);
        let depth = DepthConfig::from_state(&ctx.state);
        let chroma_enable = ctx.state.chromakey_enabled;
        let chroma = [
            ctx.state.chroma_key[0],
            ctx.state.chroma_key[1],
            ctx.state.chroma_key[2],
        ];
        let (atf, atr) = (ctx.state.alpha_test_func, ctx.state.alpha_test_ref);
        let tracing = trace_on();
        if tracing || (detail_log_enabled() && TEXLOG_COUNT.load(Ordering::Relaxed) < TEXLOG_MAX) {
            shade = shade_desc(&ctx.state);
        }
        if tracing {
            let v = &verts[0];
            let (mut zmin, mut zmax) = (f32::MAX, f32::MIN);
            for v in verts {
                zmin = zmin.min(v.depth);
                zmax = zmax.max(v.depth);
            }
            trace(&format!(
                "draw n={} tex={:#x} v0=({:.0},{:.0}) rgb=({:.0},{:.0},{:.0}) a={:.0} depth={zmin:.4}..{zmax:.4} dmode={:?} dwrite={} dcmp={} {shade}",
                verts.len(),
                LAST_TEX_ADDR.load(Ordering::Relaxed),
                v.pos[0],
                v.pos[1],
                v.color[0] * 255.0,
                v.color[1] * 255.0,
                v.color[2] * 255.0,
                v.color[3] * 255.0,
                ctx.state.depth_mode,
                ctx.state.depth_write,
                ctx.state.depth_compare
            ));
        }
        apply_depth_mode(ctx.state.depth_mode, &mut tris);
        bind_draw_texture(ctx);
        if let Some(b) = ctx.backend.as_mut() {
            b.set_combine(combine);
            b.set_blend(blend);
            b.set_depth(depth);
            b.set_chroma(chroma_enable, chroma);
            b.set_alpha_test(atf, atr);
            b.set_clamp(ctx.state.tex_clamp);
            if let Err(error) = profile_draw_stage("backend_draw", || b.draw_triangles(&tris)) {
                draw_error = Some(error);
            }
            tex_size = b.current_tex_size();
        }
    });
    if let Some(error) = draw_error {
        log_backend_error("grDrawPolygonVertexList", &error, &DRAW_ERROR_LOG_COUNT);
    }
    if let Some(t0) = t0 {
        perf_draw((tris.len() / 3) as u64, t0.elapsed().as_nanos() as u64);
    }
    // Diagnostics only (GLIDEVK_LOG); file I/O deliberately outside the lock.
    log_texcoords(&tris, tex_size, &shade);
}

// ---------------------------------------------------------------------------
// Lifecycle + hardware query
// ---------------------------------------------------------------------------

#[export_name = "grGlideInit"]
pub extern "system" fn grGlideInit() {}

#[export_name = "grGlideShutdown"]
pub extern "system" fn grGlideShutdown() {
    glidevk_log("grGlideShutdown");
    // Keep the Vulkan backend alive: Unreal calls this mid-way through its
    // focus-loss reset dance and immediately re-opens. Destroying the device
    // here (and again on re-open) is what crashed the game on alt-tab. The
    // device is reclaimed on process exit. See grSstWinOpen's reuse path.
}

#[export_name = "grGlideGetVersion"]
pub unsafe extern "system" fn grGlideGetVersion(version: *mut u8) {
    if !version.is_null() {
        let s = b"Glide2x\0";
        std::ptr::copy_nonoverlapping(s.as_ptr(), version, s.len());
    }
}

/// Fill a caller-owned `GrHwConfiguration` describing one Voodoo board with a
/// single 4 MB TMU. We write only the leading fields we're sure of (by offset),
/// never assuming the struct's full size, so we can't overflow the caller.
unsafe fn fill_hwconfig(config: *mut i32) {
    if config.is_null() {
        return;
    }
    // num_sst=1; SSTs[0].type=GR_SSTTYPE_VOODOO(0); then GrVoodooConfig:
    // fbRam, fbiRev, nTexelfx, sliDetect, tmuConfig[2]{tmuRev,tmuRam}.
    let fields = [1, 0, 4, 2, 1, 0, 1, 4, 1, 4];
    std::ptr::copy_nonoverlapping(fields.as_ptr(), config, fields.len());
}

#[export_name = "grSstQueryHardware"]
pub unsafe extern "system" fn grSstQueryHardware(config: *mut i32) -> FxBool {
    fill_hwconfig(config);
    1
}

#[export_name = "grSstQueryBoards"]
pub unsafe extern "system" fn grSstQueryBoards(config: *mut i32) -> FxBool {
    if !config.is_null() {
        *config = 1; // num_sst
    }
    1
}

#[export_name = "grSstSelect"]
pub extern "system" fn grSstSelect(_which: i32) {}

/// `grSstWinOpen(hwnd, res, refresh, cformat, origin, nColBuffers, nAuxBuffers)`
/// -> FxBool. Binds a winevulkan swapchain to `hwnd` on the Windows/Wine build.
#[export_name = "grSstWinOpen"]
pub extern "system" fn grSstWinOpen(
    hwnd: usize,
    screen_resolution: u32,
    _refresh_rate: i32,
    _color_format: i32,
    origin_location: i32,
    _num_color_buffers: i32,
    _num_aux_buffers: i32,
) -> FxBool {
    COLOR_FORMAT.store(_color_format, Ordering::Relaxed);
    let requested_hwnd = hwnd;
    #[cfg(windows)]
    let (hwnd, w, h, hinstance) = {
        let (w, h) = resolution_to_size(screen_resolution);
        // SAFETY: GetModuleHandleW(NULL) returns this process's base HINSTANCE.
        let hinstance = unsafe { GetModuleHandleW(core::ptr::null()) };
        let hwnd = match resolve_surface_hwnd(hwnd, hinstance, w, h) {
            Some(hwnd) => hwnd,
            None => {
                glidevk_log("grSstWinOpen: could not resolve or create an HWND for Vulkan");
                return 0;
            }
        };
        // GR_RESOLUTION_NONE asks to use the existing window dimensions.
        let (w, h) = if screen_resolution == 0xff {
            glidevk_core::win32::window_info(hwnd)
                .map(|(_, w, h)| (w, h))
                .filter(|&(w, h)| w > 0 && h > 0)
                .unwrap_or((w, h))
        } else {
            (w, h)
        };
        (hwnd as usize, w, h, hinstance)
    };
    #[cfg(windows)]
    {
        glidevk_log("GlideVK diagnostic build: draw timing and frame clear counts");
        glidevk_log(&format!(
            "grSstWinOpen: requested_hwnd={requested_hwnd:#x} surface_hwnd={hwnd:#x} res={screen_resolution} cformat={_color_format} -> {w}x{h} (0=ARGB 1=ABGR 2=RGBA 3=BGRA)"
        ));
        // Window style changes can synchronously dispatch window messages; do
        // them before taking CONTEXT so a game callback cannot deadlock it.
        enter_display_mode(hwnd, w, h);
    }
    let mut guard = match CONTEXT.lock() {
        Ok(g) => g,
        Err(_) => return 0,
    };
    #[cfg(windows)]
    if guard.as_ref().is_some_and(|ctx| ctx.backend.is_some())
        && BACKEND_HWND.load(Ordering::Relaxed) != hwnd as isize
    {
        glidevk_log(&format!(
            "grSstWinOpen: surface window changed from {:#x} to {hwnd:#x}; rebuilding backend",
            BACKEND_HWND.load(Ordering::Relaxed)
        ));
        // Vulkan surfaces are bound to an HWND and cannot follow a new one.
        *guard = None;
        BACKEND_HWND.store(0, Ordering::Relaxed);
    }
    // Unreal re-opens the render device on every focus loss / mode change
    // (grSstWinClose -> grGlideShutdown -> grSstWinOpen). Tearing down and
    // recreating the whole Vulkan device on that dance is what crashes the game
    // on alt-tab (real Glide drivers reset a persistent device). If a backend is
    // already live for this process, reuse it — its swapchain re-creates itself
    // on the next present when the window size changed.
    if let Some(existing) = guard.as_mut() {
        if let Some(b) = existing.backend.as_mut() {
            #[cfg(not(windows))]
            let (w, h) = resolution_to_size(screen_resolution);
            b.set_origin_upper_left(origin_location == 0);
            // The re-open may ask for a different mode (Unreal re-inits at the
            // windowed size after a focus loss); the coordinate space and what
            // grSstScreenWidth/Height report must follow it.
            b.set_logical_size(w, h);
            glidevk_log(&format!(
                "grSstWinOpen: reusing persistent backend (re-open) requested_hwnd={requested_hwnd:#x} surface_hwnd={hwnd:#x} res={screen_resolution} -> {w}x{h}"
            ));
            return 1;
        }
    }
    #[allow(unused_mut)]
    let mut ctx = Context::new();

    #[cfg(windows)]
    {
        // The window may not be fully mapped the instant Glide is initialized;
        // retry briefly so a transient surface/swapchain failure doesn't abort
        // the whole render-device init ("3d hardware initialization failed").
        let mut attempt = 0;
        let result = loop {
            match glidevk_core::vk::Backend::new_windowed(hwnd as isize, hinstance, w, h) {
                Ok(b) => break Ok(b),
                Err(e) if attempt < 10 => {
                    glidevk_log(&format!(
                        "grSstWinOpen: attempt {attempt} failed ({e:?}), retrying"
                    ));
                    attempt += 1;
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                Err(e) => break Err(e),
            }
        };
        match result {
            Ok(mut backend) => {
                glidevk_log(&format!(
                    "grSstWinOpen: backend OK extent={:?} ({})",
                    backend.extent(),
                    backend.device_name()
                ));
                backend.set_origin_upper_left(origin_location == 0);
                // GLIDEVK_CAPTURE=<path> dumps a rendered frame to a PPM for
                // offline verification. Frame index via GLIDEVK_CAPTURE_FRAME.
                if std::env::var_os("GLIDEVK_CAPTURE").is_some() {
                    let frame = std::env::var("GLIDEVK_CAPTURE_FRAME")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(400);
                    backend.set_capture(glidevk_core::vk::CaptureMode::from_frame_arg(frame));
                }
                ctx.backend = Some(backend);
                BACKEND_HWND.store(hwnd as isize, Ordering::Relaxed);
            }
            Err(e) => {
                glidevk_log(&format!("grSstWinOpen: new_windowed FAILED: {e:?}"));
                return 0;
            }
        }
    }
    #[cfg(not(windows))]
    let _ = (hwnd, screen_resolution, origin_location);

    *guard = Some(ctx);
    1
}

#[cfg(windows)]
extern "system" {
    fn GetModuleHandleW(module_name: *const u16) -> isize;
}

#[export_name = "grSstWinClose"]
pub extern "system" fn grSstWinClose() {
    glidevk_log("grSstWinClose");
    // Give the desktop back (Unreal closes on focus loss and drops to its
    // windowed software renderer until the window is restored).
    #[cfg(windows)]
    {
        let original = WINDOWED_RESTORE
            .lock()
            .ok()
            .and_then(|mut state| state.take());
        if let Some(original) = original {
            if let Err(code) = glidevk_core::win32::restore_windowed_window(original) {
                glidevk_log(&format!(
                    "grSstWinClose: could not restore window style ({code})"
                ));
            }
        }
        let owned_hwnd = LAST_GLIDE_HWND
            .lock()
            .ok()
            .and_then(|last| last.and_then(|(hwnd, owned)| owned.then_some(hwnd)));
        if let Some(hwnd) = owned_hwnd {
            glidevk_core::win32::hide_glide_window(hwnd);
        }
        if !windowed() {
            glidevk_core::win32::restore_display_mode();
        }
    }
    // Keep the backend alive across Unreal's close/re-open focus-loss dance
    // (see grSstWinOpen / grGlideShutdown). Full teardown here crashed alt-tab.
}

#[export_name = "grSstControl"]
pub extern "system" fn grSstControl(_code: i32) -> FxBool {
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

// ---------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------

/// Bounded profile samples of the clear arguments and write masks. Unlike a
/// per-call trace, repeated clears do not cause file I/O on every frame.
#[derive(Clone, Copy, PartialEq, Eq)]
struct ClearSample {
    color: u32,
    alpha: u8,
    depth: u32,
    rgb_write: bool,
    alpha_write: bool,
    depth_write: bool,
    depth_mode: DepthMode,
    last_requested_clip: Option<[u32; 4]>,
}

static LAST_REQUESTED_CLIP: Mutex<Option<[u32; 4]>> = Mutex::new(None);
static CLEAR_SAMPLES: Mutex<Vec<ClearSample>> = Mutex::new(Vec::new());

fn profile_clear_sample(sample: ClearSample) {
    let fresh = match CLEAR_SAMPLES.lock() {
        Ok(mut seen) if seen.len() < 32 && !seen.contains(&sample) => {
            seen.push(sample);
            true
        }
        _ => false,
    };
    if fresh {
        glidevk_log(&format!(
            "clear_sample frame={} color={:#010x} alpha={} depth={:#x} rgb_write={} alpha_write={} depth_write={} depth_mode={:?} last_requested_clip={:?}",
            FRAME.load(Ordering::Relaxed), sample.color, sample.alpha, sample.depth,
            sample.rgb_write, sample.alpha_write, sample.depth_write, sample.depth_mode,
            sample.last_requested_clip
        ));
    }
}

#[export_name = "grBufferClear"]
pub extern "system" fn grBufferClear(color: u32, alpha: u8, depth: u32) {
    perf_clear();
    if trace_on() {
        trace(&format!("grBufferClear {color:#010x}"));
    }
    let mut c = unpack_color(color);
    c[3] = alpha as f32 / 255.0;
    let profile = log_enabled() && profile_only();
    let last_requested_clip = if profile {
        LAST_REQUESTED_CLIP.lock().ok().and_then(|clip| *clip)
    } else {
        None
    };
    let mut clear_error = None;
    let sample = with_ctx(|ctx| {
        ctx.clear_color = c;
        if let Some(b) = ctx.backend.as_mut() {
            if let Err(error) = b.clear_masked(
                c,
                depth,
                ctx.state.color_mask_rgb,
                ctx.state.color_mask_alpha,
                ctx.state.depth_write,
            ) {
                ctx.present_errors += 1;
                clear_error = Some(error);
            }
        }
        profile.then_some(ClearSample {
            color,
            alpha,
            depth,
            rgb_write: ctx.state.color_mask_rgb,
            alpha_write: ctx.state.color_mask_alpha,
            depth_write: ctx.state.depth_write,
            depth_mode: ctx.state.depth_mode,
            last_requested_clip,
        })
    });
    if let Some(error) = clear_error {
        log_backend_error("grBufferClear", &error, &CLEAR_ERROR_LOG_COUNT);
    }
    if let Some(Some(sample)) = sample {
        profile_clear_sample(sample);
    }
}

#[export_name = "grBufferSwap"]
pub extern "system" fn grBufferSwap(_swap_interval: i32) {
    trace("grBufferSwap");
    let frame = FRAME.fetch_add(1, Ordering::Relaxed);
    let t0 = log_enabled().then(std::time::Instant::now);
    let mut present_error = None;
    let mut dropped_draws = 0;
    let capture = with_ctx(|ctx| {
        if let Some(b) = ctx.backend.as_mut() {
            dropped_draws = b.take_dropped_draws();
            if let Err(error) = b.present() {
                ctx.present_errors += 1;
                present_error = Some(error);
            }
            b.take_capture()
        } else {
            None
        }
    })
    .flatten();
    if dropped_draws > 0
        && log_enabled()
        && DROPPED_DRAW_LOG_COUNT.fetch_add(1, Ordering::Relaxed) < 20
    {
        glidevk_log(&format!(
            "grBufferSwap: frame={frame} dropped_draws={dropped_draws} (vertex buffer capacity)"
        ));
    }
    if let Some(t0) = t0 {
        let elapsed = t0.elapsed();
        perf_frame(elapsed.as_nanos() as u64);
        log_slow("present", elapsed, || format!("frame={frame}"));
    }
    if let Some(error) = present_error {
        if log_enabled() && PRESENT_ERROR_LOG_COUNT.fetch_add(1, Ordering::Relaxed) < 20 {
            glidevk_log(&format!(
                "grBufferSwap: present FAILED frame={frame}: {error:?}"
            ));
        }
    }
    #[cfg(windows)]
    if frame == 0 && windowed() && log_enabled() {
        let hwnd = LAST_GLIDE_HWND
            .lock()
            .ok()
            .and_then(|last| last.map(|(hwnd, _)| hwnd));
        if let Some((style, width, height)) = hwnd.and_then(glidevk_core::win32::window_info) {
            glidevk_log(&format!(
                "GLIDEVK_WINDOWED: first swap window style={style:#010x} client={width}x{height}"
            ));
        }
    }
    // Write the captured frame outside the context lock (file I/O). For interval
    // capture the frame index is appended so a whole intro can be scanned.
    if let Some((frame, w, h, rgba)) = capture {
        if let Ok(base) = std::env::var("GLIDEVK_CAPTURE") {
            let interval = std::env::var("GLIDEVK_CAPTURE_FRAME")
                .ok()
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap_or(0);
            let path = if interval < 0 {
                base.replace(".ppm", &format!("_{frame:05}.ppm"))
            } else {
                base
            };
            write_ppm(&path, w, h, &rgba);
            glidevk_log(&format!("captured frame {frame} {w}x{h} -> {path}"));
        }
    }
}

/// Write an RGBA8 buffer as a binary PPM (P6, RGB) — trivial + viewable.
fn write_ppm(path: &str, w: u32, h: u32, rgba: &[u8]) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::File::create(path) {
        let _ = write!(f, "P6\n{w} {h}\n255\n");
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        let (pixels, remainder) = rgba.as_chunks::<4>();
        debug_assert!(remainder.is_empty());
        for pixel in pixels {
            rgb.extend_from_slice(&pixel[..3]);
        }
        let _ = f.write_all(&rgb);
    }
}

#[export_name = "grBufferNumPending"]
pub extern "system" fn grBufferNumPending() -> i32 {
    0
}

// ---------------------------------------------------------------------------
// Combine / blend / test state
// ---------------------------------------------------------------------------

#[export_name = "grColorCombine"]
pub extern "system" fn grColorCombine(func: i32, factor: i32, local: i32, other: i32, inv: FxBool) {
    log_distinct(
        1 << 40 | (func as u64) << 24 | (factor as u64) << 16 | (local as u64) << 8 | other as u64,
        &format!(
            "grColorCombine func={func} factor={factor} local={local} other={other} inv={inv}"
        ),
    );
    set_color_combine(func, factor, local, other, inv);
}

#[export_name = "grAlphaCombine"]
pub extern "system" fn grAlphaCombine(func: i32, factor: i32, local: i32, other: i32, inv: FxBool) {
    log_distinct(
        2 << 40 | (func as u64) << 24 | (factor as u64) << 16 | (local as u64) << 8 | other as u64,
        &format!("grAlphaCombine func={func} factor={factor} local={local} other={other}"),
    );
    set_alpha_combine(func, factor, local, other, inv);
}

#[export_name = "grAlphaBlendFunction"]
pub extern "system" fn grAlphaBlendFunction(rs: i32, rd: i32, as_: i32, ad: i32) {
    log_distinct(
        3 << 40 | (rs as u64) << 8 | rd as u64,
        &format!("grAlphaBlendFunction rgb src={rs} dst={rd}"),
    );
    with_ctx(|ctx| {
        ctx.state.blend_src_rgb = BlendFactor::from_glide(rs, true);
        ctx.state.blend_dst_rgb = BlendFactor::from_glide(rd, false);
        ctx.state.blend_src_alpha = BlendFactor::from_glide(as_, true);
        ctx.state.blend_dst_alpha = BlendFactor::from_glide(ad, false);
    });
}

#[export_name = "grConstantColorValue"]
pub extern "system" fn grConstantColorValue(color: u32) {
    log_distinct(
        4 << 40 | color as u64,
        &format!("grConstantColorValue raw={color:#010x}"),
    );
    let c = unpack_constant_color(color);
    with_ctx(|ctx| ctx.state.constant_color = c);
}

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
    with_ctx(|ctx| ctx.state.depth_compare = function);
}

#[export_name = "grDepthMask"]
pub extern "system" fn grDepthMask(enable: FxBool) {
    with_ctx(|ctx| ctx.state.depth_write = enable != 0);
}

#[export_name = "grChromakeyMode"]
pub extern "system" fn grChromakeyMode(mode: i32) {
    with_ctx(|ctx| ctx.state.chromakey_enabled = mode != 0);
}

#[export_name = "grChromakeyValue"]
pub extern "system" fn grChromakeyValue(value: u32) {
    let c = unpack_color(value);
    with_ctx(|ctx| ctx.state.chroma_key = c);
}

#[export_name = "grAlphaTestFunction"]
pub extern "system" fn grAlphaTestFunction(function: i32) {
    with_ctx(|ctx| ctx.state.alpha_test_func = function);
}

#[export_name = "grAlphaTestReferenceValue"]
pub extern "system" fn grAlphaTestReferenceValue(value: u8) {
    with_ctx(|ctx| ctx.state.alpha_test_ref = value as f32 / 255.0);
}

#[export_name = "grFogMode"]
pub extern "system" fn grFogMode(mode: i32) {
    with_ctx(|ctx| ctx.state.fog_enabled = mode != 0);
}

// ---------------------------------------------------------------------------
// gu* high-level combine helpers (used by real Glide2 games)
// ---------------------------------------------------------------------------

fn set_color_combine(func: i32, factor: i32, local: i32, other: i32, inv: FxBool) {
    with_ctx(|ctx| {
        let c = &mut ctx.state.color_combine;
        c.function = CombineFunction::from_glide(func);
        c.factor = CombineFactor::from_glide(factor);
        c.local = CombineLocal::from_glide(local);
        c.other = CombineOther::from_glide(other);
        c.invert = inv != 0;
    });
}

fn set_alpha_combine(func: i32, factor: i32, local: i32, other: i32, inv: FxBool) {
    with_ctx(|ctx| {
        let a = &mut ctx.state.alpha_combine;
        a.function = CombineFunction::from_glide(func);
        a.factor = CombineFactor::from_glide(factor);
        a.local = CombineLocal::from_glide(local);
        a.other = CombineOther::from_glide(other);
        a.invert = inv != 0;
    });
}

/// `guColorCombineFunction(GrColorCombineFnc_t)`: the exact `grColorCombine`
/// each code expands to in the 3dfx Glide 2.x SDK (`gu.c`), as
/// (function, factor, local, other, invert) `GR_COMBINE_*` codes. The DELTA0
/// variants (3, 7) additionally flat-shade from the first vertex on hardware;
/// we treat them as their smooth counterparts. Verified against nGlide by
/// `examples/wine_gucombine.c`, which walks every code (Deus Ex draws all its
/// UI text with TEXTURE_TIMES_CCRGB, which used to fall through to modulate
/// with the iterated colour — black text).
#[export_name = "guColorCombineFunction"]
pub extern "system" fn guColorCombineFunction(func: i32) {
    log_distinct(
        5 << 40 | func as u64,
        &format!("guColorCombineFunction func={func}"),
    );
    // Function: ZERO 0, LOCAL 1, SCALE_OTHER 3, SCALE_OTHER_ADD_LOCAL 4,
    // SCALE_OTHER_ADD_LOCAL_ALPHA 5, SCALE_OTHER_MINUS_LOCAL 6, BLEND 7.
    // Factor: NONE 0, LOCAL 1, LOCAL_ALPHA 3, TEXTURE_ALPHA 4, ONE 8.
    // Local: ITERATED 0, CONSTANT (= NONE) 1. Other: ITERATED 0, TEXTURE 1,
    // CONSTANT (= NONE) 2.
    let (f, fac, loc, oth, inv) = match func {
        0x0 => (0, 0, 1, 2, 0),  // ZERO
        0x1 => (1, 0, 1, 2, 0),  // CCRGB
        0x2 => (1, 0, 0, 2, 0),  // ITRGB
        0x3 => (1, 0, 0, 2, 0),  // ITRGB_DELTA0
        0x4 => (3, 8, 1, 1, 0),  // DECAL_TEXTURE
        0x5 => (3, 1, 1, 1, 0),  // TEXTURE_TIMES_CCRGB
        0x6 => (3, 1, 0, 1, 0),  // TEXTURE_TIMES_ITRGB
        0x7 => (3, 1, 0, 1, 0),  // TEXTURE_TIMES_ITRGB_DELTA0
        0x8 => (5, 1, 0, 1, 0),  // TEXTURE_TIMES_ITRGB_ADD_ALPHA
        0x9 => (3, 3, 1, 1, 0),  // TEXTURE_TIMES_ALPHA
        0xa => (4, 3, 0, 1, 0),  // TEXTURE_TIMES_ALPHA_ADD_ITRGB
        0xb => (4, 8, 0, 1, 0),  // TEXTURE_ADD_ITRGB
        0xc => (6, 8, 0, 1, 0),  // TEXTURE_SUB_ITRGB
        0xd => (7, 4, 1, 0, 0),  // CCRGB_BLEND_ITRGB_ON_TEXALPHA
        0xe => (4, 3, 0, 1, 0),  // DIFF_SPEC_A
        0xf => (5, 1, 0, 1, 0),  // DIFF_SPEC_B
        0x10 => (0, 0, 1, 2, 1), // ONE (inverted zero)
        _ => (3, 1, 0, 1, 0),    // unknown: modulate
    };
    set_color_combine(f, fac, loc, oth, inv);
}

/// `guAlphaSource(GrAlphaSource_t)`: the SDK's `grAlphaCombine` expansions.
/// CC_ALPHA 0, ITERATED_ALPHA 1, TEXTURE_ALPHA 2,
/// TEXTURE_ALPHA_TIMES_ITERATED_ALPHA 3.
#[export_name = "guAlphaSource"]
pub extern "system" fn guAlphaSource(src: i32) {
    log_distinct(6 << 40 | src as u64, &format!("guAlphaSource src={src}"));
    let (f, fac, loc, oth) = match src {
        0 => (1, 0, 1, 2),
        1 => (1, 0, 0, 2),
        2 => (3, 8, 1, 1),
        _ => (3, 1, 0, 1),
    };
    set_alpha_combine(f, fac, loc, oth, 0);
}

/// `guFogTableIndexToW(int i)` -> float. Fog-table build helper; approximate.
#[export_name = "guFogTableIndexToW"]
pub extern "system" fn guFogTableIndexToW(i: i32) -> f32 {
    // Glide's canonical mapping is pow(2, i/4) / 8; adequate for table setup.
    let e = (i.max(0) as f32) / 4.0;
    2.0_f32.powf(e) / 8.0
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

/// Glide2 `GrTexInfo` — note LOD/aspect differ from Glide3 (see `tex_dims`).
#[repr(C)]
pub struct GrTexInfo {
    pub small_lod: u32,
    pub large_lod: u32,
    pub aspect_ratio: u32,
    pub format: u32,
    pub data: *const core::ffi::c_void,
}

fn map_texfmt(v: u32) -> Option<TexFormat> {
    match v {
        0x02 => Some(TexFormat::Alpha8),
        0x03 => Some(TexFormat::Intensity8),
        0x05 => Some(TexFormat::P8),
        0x0a => Some(TexFormat::Rgb565),
        0x0b => Some(TexFormat::Argb1555),
        0x0c => Some(TexFormat::Argb4444),
        _ => None,
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

/// Glide2 texture dimensions. Unlike Glide3, `large_lod` counts DOWN from 256
/// (`GR_LOD_256`=0 … `GR_LOD_1`=8) and aspect runs `8x1`=0 … `1x1`=3 … `1x8`=6.
fn tex_dims(info: &GrTexInfo) -> (u32, u32) {
    let dim = (256u32 >> info.large_lod.min(8)).max(1);
    let aspect = info.aspect_ratio.min(6) as i32;
    if aspect <= 3 {
        (dim, (dim >> (3 - aspect)).max(1)) // square or wide
    } else {
        ((dim >> (aspect - 3)).max(1), dim) // tall
    }
}

#[export_name = "grTexDownloadMipMap"]
pub unsafe extern "system" fn grTexDownloadMipMap(
    _tmu: u32,
    start_address: u32,
    _even_odd: u32,
    info: *const GrTexInfo,
) {
    download_texture(start_address, info);
}

#[export_name = "grTexDownloadMipMapLevel"]
pub unsafe extern "system" fn grTexDownloadMipMapLevel(
    _tmu: u32,
    start_address: u32,
    this_lod: u32,
    large_lod: u32,
    aspect: u32,
    format: u32,
    _even_odd: u32,
    data: *const core::ffi::c_void,
) {
    // We don't mipmap; keep only the base (largest) level. In Glide2 LOD counts
    // DOWN from 256, so the base level is `this_lod == large_lod`. Uploading a
    // smaller level with base-level dimensions would over-read `data`.
    if this_lod != large_lod {
        return;
    }
    let info = GrTexInfo {
        small_lod: large_lod,
        large_lod,
        aspect_ratio: aspect,
        format,
        data,
    };
    download_texture(start_address, &info);
}

#[export_name = "grTexDownloadMipMapLevelPartial"]
pub unsafe extern "system" fn grTexDownloadMipMapLevelPartial(
    tmu: u32,
    start_address: u32,
    lod: u32,
    large_lod: u32,
    aspect: u32,
    format: u32,
    even_odd: u32,
    data: *const core::ffi::c_void,
    start_row: i32,
    end_row: i32,
) {
    // Unreal uploads lightmaps this way with rows [0, VClamp-1] only. We still
    // decode the full level (the source buffer is level-sized); log the range so
    // a partial-row mismatch can be recognised.
    log_distinct(
        7 << 40 | (large_lod as u64) << 32 | (aspect as u64) << 24 | (start_row as u64) << 12 | (end_row as u64 & 0xfff),
        &format!(
            "grTexDownloadMipMapLevelPartial lod={lod} largeLod={large_lod} aspect={aspect} fmt={format:#x} rows={start_row}..{end_row}"
        ),
    );
    grTexDownloadMipMapLevel(
        tmu,
        start_address,
        lod,
        large_lod,
        aspect,
        format,
        even_odd,
        data,
    );
}

unsafe fn download_texture(start_address: u32, info: *const GrTexInfo) {
    if info.is_null() {
        return;
    }
    let t0 = log_enabled().then(std::time::Instant::now);
    let info = &*info;
    if trace_on() {
        trace(&format!(
            "grTexDownload addr={start_address:#x} fmt={:#x} lod={} aspect={}",
            info.format, info.large_lod, info.aspect_ratio
        ));
    }
    if info.data.is_null() {
        return;
    }
    let fmt = match map_texfmt(info.format) {
        Some(f) => f,
        None => {
            glidevk_log(&format!(
                "grTexDownload: UNHANDLED format={:#x} largeLod={} aspect={}",
                info.format, info.large_lod, info.aspect_ratio
            ));
            return;
        }
    };
    let (w, h) = tex_dims(info);
    let src = std::slice::from_raw_parts(
        info.data as *const u8,
        (w as usize) * (h as usize) * texfmt_bpp(fmt),
    );
    with_ctx(|ctx| {
        // Palettised textures keep only their indices here: Glide applies the
        // palette at draw time (Unreal binds, THEN uploads the texture's palette,
        // then draws), so they are decoded per draw against the live palette —
        // see `bind_draw_texture`. Any GPU variants of a previous texture at this
        // address are stale.
        if matches!(fmt, TexFormat::P8) {
            ctx.download_p8(start_address, w, h, src);
            if !detail_log_enabled() && texture_dump_dir().is_none() {
                return;
            }
        }
        match ctx.textures.decode(fmt, w, h, src) {
            Ok(tex) => {
                // Log each distinct texture's decoded content once: its format,
                // dims, and average RGBA + alpha coverage. Tells us whether a
                // garbled overlay (e.g. the green-dot backdrop at some addr) comes
                // from the texture itself being green/masked vs. the combine.
                log_texture_content(start_address, info.format, w, h, &tex.rgba, src);
                if !matches!(fmt, TexFormat::P8) {
                    ctx.clear_p8(start_address);
                    if let Some(b) = ctx.backend.as_mut() {
                        // A failed upload leaves nothing at this address, so the
                        // next bind falls back to white: never swallow it.
                        if let Err(e) = b.upload_texture(start_address as u64, w, h, &tex.rgba) {
                            glidevk_log(&format!(
                                "grTexDownload: upload FAILED addr={start_address:#x} {w}x{h}: {e:?}"
                            ));
                        } else if ctx.bound_texture == start_address {
                            b.set_texture(start_address as u64);
                        }
                    }
                }
            }
            Err(e) => glidevk_log(&format!(
                "grTexDownload: decode failed {e:?} fmt={:#x}",
                info.format
            )),
        }
    });
    if let Some(t0) = t0 {
        let elapsed = t0.elapsed();
        perf_upload(elapsed.as_nanos() as u64);
        log_slow("texture download", elapsed, || {
            format!(
                "frame={} addr={start_address:#x} format={:#x} size={w}x{h}",
                FRAME.load(Ordering::Relaxed),
                info.format
            )
        });
    }
}

/// Log a decoded texture's average colour + alpha coverage (deduped by address),
/// so we can tell what a mis-rendered quad's texture actually contains.
fn log_texture_content(addr: u32, format: u32, w: u32, h: u32, rgba: &[u8], raw: &[u8]) {
    use std::collections::HashSet;
    use std::hash::{Hash, Hasher};
    static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    static DUMPED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    const DUMP_MAX: u32 = 400;
    let detail = detail_log_enabled();
    let dump_dir = texture_dump_dir();
    if (!detail && dump_dir.is_none()) || rgba.len() < 4 {
        return;
    }
    // Unreal's texture cache reuses TMU addresses, so key on the content too
    // (a coarse hash of the raw bytes), not just the address.
    let mut hs = std::collections::hash_map::DefaultHasher::new();
    (addr, format, w, h).hash(&mut hs);
    raw.hash(&mut hs);
    let fresh = match SEEN.lock() {
        Ok(mut g) => g.get_or_insert_with(HashSet::new).insert(hs.finish()),
        Err(_) => return,
    };
    if !fresh {
        return;
    }
    let frame = FRAME.load(Ordering::Relaxed);
    // GLIDEVK_TEXDUMP=<dir>: write each distinct decoded texture as a PPM so
    // what the game uploaded can be inspected directly.
    if let Some(dir) = dump_dir {
        if DUMPED.fetch_add(1, Ordering::Relaxed) < DUMP_MAX {
            write_ppm(
                &format!("{dir}/tex_f{frame:05}_{addr:#07x}_{w}x{h}_fmt{format:x}.ppm"),
                w,
                h,
                rgba,
            );
        }
    }
    if !detail {
        return;
    }
    // Palettised textures: show the index range so an all-black decode can be
    // attributed to the indices vs the palette.
    let idx = if format == 0x5 && !raw.is_empty() {
        let (mn, mx) = raw
            .iter()
            .fold((255u8, 0u8), |(a, b), &v| (a.min(v), b.max(v)));
        format!(" idx={mn}..{mx}")
    } else {
        String::new()
    };
    let (mut r, mut gg, mut b, mut a, mut opaque) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let n = rgba.len() / 4;
    let (pixels, remainder) = rgba.as_chunks::<4>();
    debug_assert!(remainder.is_empty());
    for pixel in pixels {
        r += pixel[0] as u64;
        gg += pixel[1] as u64;
        b += pixel[2] as u64;
        a += pixel[3] as u64;
        if pixel[3] >= 128 {
            opaque += 1;
        }
    }
    let d = n.max(1) as u64;
    glidevk_log(&format!(
        "texcontent frame={frame} addr={addr:#x} fmt={format:#x} {w}x{h} avgRGBA=({},{},{},{}) opaque={}%{idx}",
        r / d,
        gg / d,
        b / d,
        a / d,
        opaque * 100 / d
    ));
}

#[export_name = "grTexSource"]
pub extern "system" fn grTexSource(
    _tmu: u32,
    start_address: u32,
    _even_odd: u32,
    _info: *const GrTexInfo,
) {
    LAST_TEX_ADDR.store(start_address, Ordering::Relaxed);
    if trace_on() {
        trace(&format!("grTexSource addr={start_address:#x}"));
    }
    with_ctx(|ctx| {
        if !ctx.bind_texture(start_address) {
            // Nothing downloaded at that address: the backend falls back to a
            // 1x1 white texture, which reads on screen as an untextured surface.
            TEX_MISSES.fetch_add(1, Ordering::Relaxed);
            log_distinct(
                9 << 40 | start_address as u64,
                &format!("grTexSource addr={start_address:#x} -> NO texture (default white)"),
            );
        }
    });
}

#[export_name = "grTexDownloadTable"]
pub unsafe extern "system" fn grTexDownloadTable(
    _tmu: u32,
    _table_type: i32,
    data: *const core::ffi::c_void,
) {
    if data.is_null() {
        return;
    }
    let src = std::slice::from_raw_parts(data as *const u32, 256);
    let mut palette = [0u32; 256];
    palette.copy_from_slice(src);
    let tracing = trace_on();
    if tracing || detail_log_enabled() {
        let nonzero = palette.iter().filter(|&&c| c & 0x00ff_ffff != 0).count();
        if tracing {
            trace(&format!(
                "grTexDownloadTable nonzero={nonzero} p[1]={:#08x} p[29]={:#08x}",
                palette[1] & 0xff_ffff,
                palette[29] & 0xff_ffff
            ));
        }
        if detail_log_enabled() {
            log_distinct(
                6 << 40 | (nonzero as u64) << 8 | (palette[1] & 0xff) as u64,
                &format!(
                    "grTexDownloadTable type={_table_type} nonzero={nonzero}/256 p[0..8]={:06x?} p[16]={:#08x} p[29]={:#08x} p[255]={:#08x}",
                    palette[..8].iter().map(|c| c & 0xff_ffff).collect::<Vec<_>>(),
                    palette[16] & 0xff_ffff,
                    palette[29] & 0xff_ffff,
                    palette[255] & 0xff_ffff
                ),
            );
        }
    }
    // Deus Ex animates its intro logos by cycling this palette. Just record it
    // (a cheap version bump); the affected P8 textures re-decode lazily when
    // next bound for drawing, so the hundreds of redundant per-frame uploads
    // collapse to at most one re-decode per texture per frame.
    with_ctx(|ctx| ctx.textures.set_palette(palette));
}

/// Texels occupied by each SST-1 mip level, indexed by wide aspect and LOD.
///
/// These are the exact `_grMipMapSize` values from 3Dfx's Glide 2 source.
/// They are not always the mathematical `width * height`: narrow levels honor
/// the TMU's fetch granularity (for example, 4x1 occupies eight texels).
const GLIDE2_MIP_TEXELS: [[u64; 9]; 4] = [
    [8_192, 2_048, 512, 128, 32, 16, 8, 4, 4],         // 8x1
    [16_384, 4_096, 1_024, 256, 64, 16, 8, 4, 4],      // 4x1
    [32_768, 8_192, 2_048, 512, 128, 32, 8, 4, 4],     // 2x1
    [65_536, 16_384, 4_096, 1_024, 256, 64, 16, 4, 4], // 1x1
];

/// Calculate the TMU bytes occupied by a Glide 2 mipmap chain.
///
/// Glide 2 encodes LODs in descending size order: 0 is 256 texels and 8 is
/// 1 texel. The original 3Dfx implementation sums every level in the inclusive
/// `large_lod..=small_lod` range, gives each level a four-texel minimum, and
/// rounds the final byte count up to the TMU's eight-byte alignment.
fn tex_mem_required(small_lod: u32, large_lod: u32, aspect: u32, format: u32) -> u32 {
    let large_lod = large_lod.min(8);
    let small_lod = small_lod.min(8).max(large_lod);

    // The SST-1 mip-size table stores only wide aspects and mirrors tall
    // aspects onto their transposed representation.
    let aspect = match aspect.min(6) {
        4..=6 => 6 - aspect.min(6),
        aspect => aspect,
    };

    let texels = GLIDE2_MIP_TEXELS[aspect as usize][large_lod as usize..=small_lod as usize]
        .iter()
        .sum::<u64>();

    // Glide 2's format encoding, rather than a validated format enum, decides
    // the texel width. Values >= GR_TEXFMT_16BIT are two bytes; all others,
    // including malformed game-supplied values, are one byte.
    let bytes_per_texel = if format >= 0x08 { 2 } else { 1 };
    let bytes = texels.saturating_mul(bytes_per_texel);
    let aligned = bytes.saturating_add(7) & !7;
    aligned.min(u32::MAX as u64) as u32
}

/// `grTexCalcMemRequired(smallLod, largeLod, aspect, format)` -> bytes.
#[export_name = "grTexCalcMemRequired"]
pub extern "system" fn grTexCalcMemRequired(
    small_lod: u32,
    large_lod: u32,
    aspect: u32,
    format: u32,
) -> u32 {
    tex_mem_required(small_lod, large_lod, aspect, format)
}

const TMU_MEMORY_BYTES: u32 = 4 * 1024 * 1024;

#[export_name = "grTexMinAddress"]
pub extern "system" fn grTexMinAddress(_tmu: u32) -> u32 {
    0
}

#[export_name = "grTexMaxAddress"]
pub extern "system" fn grTexMaxAddress(_tmu: u32) -> u32 {
    TMU_MEMORY_BYTES - 1
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

#[export_name = "grDrawTriangle"]
pub unsafe extern "system" fn grDrawTriangle(
    a: *const core::ffi::c_void,
    b: *const core::ffi::c_void,
    c: *const core::ffi::c_void,
) {
    let t0 = log_enabled().then(std::time::Instant::now);
    log_raw_vertex(a as *const u8, 3);
    let mut tris = profile_draw_stage("triangle_vertex_read", || {
        [
            read_vertex(a as *const u8),
            read_vertex(b as *const u8),
            read_vertex(c as *const u8),
        ]
    });
    let mut tex_size = [1.0, 1.0];
    let mut shade = String::new();
    let mut draw_error = None;
    with_draw_ctx(|ctx| {
        let combine = CombineParams::from_state(&ctx.state);
        let blend = BlendConfig::from_state(&ctx.state);
        let depth = DepthConfig::from_state(&ctx.state);
        let ce = ctx.state.chromakey_enabled;
        let ck = [
            ctx.state.chroma_key[0],
            ctx.state.chroma_key[1],
            ctx.state.chroma_key[2],
        ];
        let (atf, atr) = (ctx.state.alpha_test_func, ctx.state.alpha_test_ref);
        if detail_log_enabled() && TEXLOG_COUNT.load(Ordering::Relaxed) < TEXLOG_MAX {
            shade = shade_desc(&ctx.state);
        }
        apply_depth_mode(ctx.state.depth_mode, &mut tris);
        bind_draw_texture(ctx);
        if let Some(bk) = ctx.backend.as_mut() {
            bk.set_combine(combine);
            bk.set_blend(blend);
            bk.set_depth(depth);
            bk.set_chroma(ce, ck);
            bk.set_alpha_test(atf, atr);
            bk.set_clamp(ctx.state.tex_clamp);
            if let Err(error) = profile_draw_stage("backend_draw", || bk.draw_triangles(&tris)) {
                draw_error = Some(error);
            }
            tex_size = bk.current_tex_size();
        }
    });
    if let Some(error) = draw_error {
        log_backend_error("grDrawTriangle", &error, &DRAW_ERROR_LOG_COUNT);
    }
    if let Some(t0) = t0 {
        perf_draw(1, t0.elapsed().as_nanos() as u64);
    }
    log_texcoords(&tris, tex_size, &shade);
}

#[export_name = "grDrawPolygonVertexList"]
pub unsafe extern "system" fn grDrawPolygonVertexList(
    nverts: i32,
    vlist: *const core::ffi::c_void,
) {
    if nverts < 3 || vlist.is_null() {
        return;
    }
    let base = vlist as *const u8;
    log_raw_vertex(base, nverts);
    let verts: Vec<DrawVertex> = profile_draw_stage("fan_vertex_read", || {
        (0..nverts as usize)
            .map(|i| read_vertex(base.add(i * GRVERTEX_SIZE)))
            .collect()
    });
    submit_fan(&verts);
}

#[export_name = "grDrawPlanarPolygonVertexList"]
pub unsafe extern "system" fn grDrawPlanarPolygonVertexList(
    nverts: i32,
    vlist: *const core::ffi::c_void,
) {
    grDrawPolygonVertexList(nverts, vlist);
}

// ---------------------------------------------------------------------------
// LFB
// ---------------------------------------------------------------------------

/// Glide2 `GrLfbInfo_t`.
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
    lock_type: u32,
    buffer: u32,
    write_mode: i32,
    origin: i32,
    pixel_pipeline: FxBool,
    info: *mut GrLfbInfo,
) -> FxBool {
    if log_enabled() && LFB_LOCK_LOG_COUNT.fetch_add(1, Ordering::Relaxed) < 10 {
        glidevk_log(&format!("grLfbLock frame={} type={lock_type} buffer={buffer} mode={write_mode} origin={origin} pixel_pipeline={pixel_pipeline}", FRAME.load(Ordering::Relaxed)));
    }
    if info.is_null() {
        return 0;
    }
    with_ctx(|ctx| {
        if let Some(b) = ctx.backend.as_mut() {
            let (ptr, stride) = b.lfb_lock(write_mode);
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
        if let Some(b) = ctx.backend.as_mut() {
            b.lfb_unlock();
        }
    });
    1
}

/// `grLfbReadRegion`: copy a rectangle of the framebuffer to caller memory as
/// 16-bit RGB565 (the only format Glide 2 reads back in).
///
/// Reads come from the **last presented frame** (the front buffer), which the
/// backend keeps a GPU-side copy of every frame; only the requested pixels are
/// converted. Before the first present the region reads as zeros. Coordinates
/// are in the game's logical space and scaled to the swapchain extent when the
/// two differ.
///
/// # Safety
/// `dst_data` must point to at least `src_h * dst_stride` writable bytes.
#[export_name = "grLfbReadRegion"]
pub unsafe extern "system" fn grLfbReadRegion(
    _src_buffer: u32,
    src_x: u32,
    src_y: u32,
    src_w: u32,
    src_h: u32,
    dst_stride: u32,
    dst_data: *mut core::ffi::c_void,
) -> FxBool {
    if dst_data.is_null() || src_w == 0 || src_h == 0 {
        return 0;
    }
    let dst_len = (src_h as usize) * (dst_stride as usize);
    let dst = std::slice::from_raw_parts_mut(dst_data as *mut u8, dst_len);
    with_ctx(|ctx| {
        let Some(b) = ctx.backend.as_mut() else {
            return 0;
        };
        if !b.read_region_rgb565(src_x, src_y, src_w, src_h, dst_stride, dst) {
            glidevk_log("grLfbReadRegion: nothing presented yet, returning zeros");
        }
        1
    })
    .unwrap_or(0)
}

#[export_name = "grLfbWriteRegion"]
pub extern "system" fn grLfbWriteRegion(
    dst_buffer: u32,
    dst_x: u32,
    dst_y: u32,
    fmt: u32,
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    _src_data: *const core::ffi::c_void,
) -> FxBool {
    if log_enabled() && LFB_WRITE_LOG_COUNT.fetch_add(1, Ordering::Relaxed) < 10 {
        glidevk_log(&format!("grLfbWriteRegion UNIMPLEMENTED frame={} buffer={dst_buffer} x={dst_x} y={dst_y} format={fmt} size={src_w}x{src_h} stride={src_stride}", FRAME.load(Ordering::Relaxed)));
    }
    1
}

// ---------------------------------------------------------------------------
// GR_* -> core enum maps (values shared with Glide3)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Accepted-but-minimal entry points (state stored elsewhere or not yet used)
// ---------------------------------------------------------------------------

macro_rules! stub {
    ($name:literal, $rust:ident ( $($arg:ident : $ty:ty),* ) $(-> $ret:ty)?) => {
        #[export_name = $name]
        pub extern "system" fn $rust($(_: $ty),*) $(-> $ret)? { Default::default() }
    };
}

/// `grColorMask(rgb, alpha)` — enable/disable writing the colour / alpha channels
/// to the framebuffer (e.g. depth-only passes, preserving destination alpha).
#[export_name = "grColorMask"]
pub extern "system" fn grColorMask(rgb: FxBool, alpha: FxBool) {
    with_ctx(|ctx| {
        ctx.state.color_mask_rgb = rgb != 0;
        ctx.state.color_mask_alpha = alpha != 0;
    });
}
stub!("grDitherMode", gr_dither_mode(mode: i32));
/// Limit subsequent rasterization to the logical Glide clip window.
#[export_name = "grClipWindow"]
pub extern "system" fn gr_clip_window(minx: u32, miny: u32, maxx: u32, maxy: u32) {
    if log_enabled() && profile_only() {
        if let Ok(mut last) = LAST_REQUESTED_CLIP.lock() {
            *last = Some([minx, miny, maxx, maxy]);
        }
    }
    with_ctx(|ctx| {
        if let Some(backend) = ctx.backend.as_mut() {
            backend.set_clip_window([minx, miny, maxx, maxy]);
        }
    });
}
stub!("grCullMode", gr_cull_mode(mode: i32));
stub!("grDepthBiasLevel", gr_depth_bias(level: i16));
stub!("grDrawLine", gr_draw_line(a: *const core::ffi::c_void, b: *const core::ffi::c_void));
stub!("grErrorSetCallback", gr_error_cb(cb: *const core::ffi::c_void));
stub!("grFogColorValue", gr_fog_color(color: u32));
stub!("grFogTable", gr_fog_table(table: *const u8));
/// `grGammaCorrectionValue(float)` — the 3dfx RAMDAC output gamma. Real Voodoo
/// hardware applies this to the whole framebuffer at scan-out, which lifts dim
/// content (Unreal's flat-shaded intro logos) into the visible range; without it
/// they stay near-black. We forward it to the backend, which applies it in the
/// combiner's final output.
#[export_name = "grGammaCorrectionValue"]
pub extern "system" fn grGammaCorrectionValue(value: f32) {
    log_distinct(
        8 << 40 | value.to_bits() as u64,
        &format!("grGammaCorrectionValue value={value}"),
    );
    with_ctx(|ctx| {
        if let Some(b) = ctx.backend.as_mut() {
            b.set_gamma(value);
        }
    });
}
stub!("grHints", gr_hints(hint: i32, mask: u32));
/// `grTexClampMode(tmu, s_mode, t_mode)`: GR_TEXTURECLAMP_WRAP (0) / _CLAMP (1)
/// (Glide 3 adds _MIRROR_EXTEND, mapped to clamp). Either axis non-wrap selects
/// the clamped sampler for subsequent draws.
#[export_name = "grTexClampMode"]
pub extern "system" fn grTexClampMode(_tmu: u32, s_mode: i32, t_mode: i32) {
    with_ctx(|ctx| ctx.state.tex_clamp = s_mode != 0 || t_mode != 0);
}
stub!("grTexCombine", gr_tex_combine(tmu: u32, rf: i32, rfa: i32, af: i32, afa: i32, ri: FxBool, ai: FxBool));
stub!("grTexCombineFunction", gr_tex_combine_fn(tmu: u32, func: i32));
stub!("grTexFilterMode", gr_tex_filter(tmu: u32, minf: i32, magf: i32));
stub!("grTexLodBiasValue", gr_tex_lod_bias(tmu: u32, bias: f32));
stub!("grTexMipMapMode", gr_tex_mipmap(tmu: u32, mode: i32, lod_blend: FxBool));
/// `grAlphaControlsITRGBLighting(FxBool)` — when enabled, the MSB of the texture
/// alpha selects the color-combine local source per texel (MSB=1 → constant
/// color, MSB=0 → iterated vertex RGB), a "specular alpha paint" trick. The
/// combiner shader applies it via the `alphaLighting` push constant.
#[export_name = "grAlphaControlsITRGBLighting"]
pub extern "system" fn grAlphaControlsITRGBLighting(enable: FxBool) {
    with_ctx(|ctx| ctx.state.alpha_it_lighting = enable != 0);
}
stub!("guTexCombineFunction", gu_tex_combine_fn(tmu: u32, func: i32));

// ---------------------------------------------------------------------------
// Diagnostics (not part of the Glide API) — shared with the glide3x harnesses.
// ---------------------------------------------------------------------------

#[export_name = "glidevkFrameResult"]
pub extern "system" fn glidevkFrameResult() -> i32 {
    with_ctx(|ctx| ctx.present_errors as i32).unwrap_or(-1)
}

#[export_name = "glidevkCenterPixel"]
pub extern "system" fn glidevkCenterPixel() -> u32 {
    with_ctx(|ctx| {
        ctx.backend
            .as_ref()
            .map(|b| {
                let [r, g, bl, a] = b.center_pixel_rgba();
                (r as u32) << 24 | (g as u32) << 16 | (bl as u32) << 8 | a as u32
            })
            .unwrap_or(0)
    })
    .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_vertex_read() {
        // Build a GrVertex: x=100, y=50, rgb=(255,0,0), a=255, sow=0.5, tow=0.25.
        let mut v = [0f32; 16];
        v[0] = 100.0; // x
        v[1] = 50.0; // y
        v[3] = 255.0; // r  (offset 12 = index 3)
        v[7] = 255.0; // a  (offset 28 = index 7)
        v[9] = 0.5; // sow (offset 36 = index 9)
        v[10] = 0.25; // tow (offset 40 = index 10)
        let dv = unsafe { read_vertex(v.as_ptr() as *const u8) };
        assert_eq!(dv.pos, [100.0, 50.0]);
        assert_eq!(dv.color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(dv.tex, [0.5, 0.25]);
    }

    #[test]
    fn uninitialized_tmu_oow_is_ignored_for_vertex_oow() {
        // Regression for the Deus Ex intro/menu garble: GlideDrv leaves
        // tmuvtx[0].oow (offset 44 = idx 11) uninitialized, so read_vertex must
        // take the vertex oow (offset 32 = idx 8, = 1/w) and never divide texture
        // coordinates by that stack garbage.
        let mut v = [0f32; 16];
        v[0] = 100.0; // x
        v[1] = 50.0; // y
        v[8] = 0.5; // vertex oow = 1/w
        v[9] = 2.0; // sow

        // The exact denormal garbage seen in the wild: the int 960 (screen width)
        // reinterpreted as f32 (~1.345e-42). Must be ignored.
        v[11] = f32::from_bits(960);
        assert_eq!(unsafe { read_vertex(v.as_ptr() as *const u8) }.oow, 0.5);

        // Normal-looking garbage (5919.0 was also observed) must lose to the
        // vertex oow too — the whole per-TMU slot is untrustworthy for GlideDrv.
        v[11] = 5919.0;
        assert_eq!(unsafe { read_vertex(v.as_ptr() as *const u8) }.oow, 0.5);

        // Vertex oow unset -> fall back to a usable per-TMU oow.
        v[8] = 0.0;
        v[11] = 4.0;
        assert_eq!(unsafe { read_vertex(v.as_ptr() as *const u8) }.oow, 4.0);

        // Neither usable -> affine (1.0, no perspective divide), never inf.
        v[11] = 0.0;
        assert_eq!(unsafe { read_vertex(v.as_ptr() as *const u8) }.oow, 1.0);
    }

    #[test]
    fn glide2_lod_is_inverted() {
        // GR_LOD_256 = 0 -> 256px; aspect 1x1 = 3 -> square.
        let info = GrTexInfo {
            small_lod: 0,
            large_lod: 0,
            aspect_ratio: 3,
            format: 0x0a,
            data: core::ptr::null(),
        };
        assert_eq!(tex_dims(&info), (256, 256));
        let info8 = GrTexInfo {
            large_lod: 5,
            ..info
        };
        assert_eq!(tex_dims(&info8), (8, 8)); // GR_LOD_8 = 5
    }

    #[test]
    fn tex_mem_required_includes_whole_mipmap_chain() {
        // 256..1 square, RGB565. Every texel count is from the original
        // SST-1 table, including its four-texel minimum for both final levels.
        let expected_texels = 65_536 + 16_384 + 4_096 + 1_024 + 256 + 64 + 16 + 4 + 4;
        assert_eq!(
            grTexCalcMemRequired(8, 0, 3, 0x0a),
            (expected_texels * 2) as u32
        );

        // One level remains the old base-level result: 256x256 RGB565.
        assert_eq!(grTexCalcMemRequired(0, 0, 3, 0x0a), 256 * 256 * 2);
    }

    #[test]
    fn tex_mem_required_matches_aspects_formats_and_padding() {
        // 8x1 and 1x8 mirror to the same SST-1 mip-size table.
        let wide = grTexCalcMemRequired(8, 0, 0, 0x0a);
        let tall = grTexCalcMemRequired(8, 0, 6, 0x0a);
        assert_eq!(wide, tall);
        assert_eq!(wide, (8_192 + 2_048 + 512 + 128 + 32 + 16 + 8 + 4 + 4) * 2);

        // P8 is one byte per texel. This inclusive range includes every level
        // from 64x64 through 8x8, not just the base level.
        assert_eq!(
            grTexCalcMemRequired(5, 2, 3, 0x05),
            (4_096 + 1_024 + 256 + 64) as u32
        );

        // Even a 1x1 level occupies the SST-1's four-texel minimum, and a
        // one-byte format therefore still consumes an aligned eight bytes.
        assert_eq!(grTexCalcMemRequired(8, 8, 3, 0x05), 8);

        // The original API classifies unknown formats by encoding, so this
        // malformed large value must be treated as 16-bit rather than fault.
        assert_eq!(
            grTexCalcMemRequired(8, 8, 3, u32::MAX),
            grTexCalcMemRequired(8, 8, 3, 0x0a)
        );
    }

    #[test]
    fn tex_mem_required_is_bounded_for_invalid_dimensions() {
        // These arguments are outside the documented API. The important ABI
        // compatibility property is that a bad caller gets a bounded aligned
        // size instead of indexing out of bounds or panicking across the FFI.
        let full_chain = grTexCalcMemRequired(u32::MAX, u32::MAX, u32::MAX, u32::MAX);
        assert_eq!(full_chain % 8, 0);
    }
}
