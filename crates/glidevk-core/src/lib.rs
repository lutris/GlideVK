//! GlideVK core — backend-agnostic translation of the 3dfx Glide pipeline.
//!
//! This crate holds everything that is *not* tied to the `glide2x` / `glide3x`
//! DLL entry points or to a particular OS: the Glide state machine, texture
//! format conversion, linear-framebuffer emulation, and the
//! state-hash -> fragment-shader pipeline cache that targets Vulkan.
//!
//! The thin per-API shells (`glide3x`, and later `glide2x`) call into here.
//!
//! ## Architecture
//!
//! ```text
//! glide3x.dll entry ──▶ Context ──▶ State ──▶ PipelineCache ──▶ Vulkan
//!                                     │
//!                                     ├─ TextureCache  (format conversion)
//!                                     └─ Lfb           (framebuffer r/w)
//! ```
//!
//! Reference semantics come from the open 3dfx Glide SDK (SPDX: `Glide`).

pub mod lfb;
pub mod pipeline;
pub mod state;
pub mod texture;
pub mod types;
pub mod win32;

#[cfg(feature = "vulkan")]
pub mod vk;

pub use state::State;
pub use types::*;

/// A single Glide rendering context — one per `grSstWinOpen`.
///
/// Owns the Glide state machine plus the caches that turn that state into
/// Vulkan work. The Vulkan device itself lives behind the `vulkan` feature so
/// the core can be exercised (state transitions, format conversion) on hosts
/// without a Vulkan loader.
#[derive(Default)]
pub struct Context {
    pub state: State,
    pub textures: texture::TextureCache,
    pub pipelines: pipeline::PipelineCache,
    /// Last `grBufferClear` color; presented by the next `grBufferSwap`.
    pub clear_color: [f32; 4],
    /// Count of failed presents — 0 means every `grBufferSwap` succeeded.
    /// Exposed for diagnostics/self-test.
    pub present_errors: u32,
    #[cfg(feature = "vulkan")]
    pub backend: Option<vk::Backend>,
    /// TMU address selected by the last `grTexSource`.
    pub bound_texture: u32,
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }
}

/// A palettised texture freshly decoded and uploaded by
/// [`Context::bind_for_draw`], returned for diagnostics.
pub struct P8Uploaded {
    pub addr: u32,
    pub palette_id: u64,
    pub texture: texture::Texture,
}

#[cfg(feature = "vulkan")]
impl Context {
    /// `grTexSource`: remember the selection. Plain textures bind now;
    /// palettised ones are resolved at draw time ([`Context::bind_for_draw`]),
    /// because Glide's TMU applies its palette LUT when drawing and games
    /// (Unreal) upload a texture's palette *after* binding it. Returns false
    /// when a plain texture is unknown (the backend falls back to white).
    pub fn bind_texture(&mut self, addr: u32) -> bool {
        self.bound_texture = addr;
        if self.textures.is_p8(addr) {
            return true;
        }
        match self.backend.as_mut() {
            Some(b) => b.set_texture(addr as u64),
            None => false,
        }
    }

    /// Download of a `GR_TEXFMT_P_8` texture: keep its indices for draw-time
    /// decoding and drop the GPU variants of whatever was at `addr` before.
    pub fn download_p8(&mut self, addr: u32, width: u32, height: u32, indices: &[u8]) {
        let stale = self.textures.store_p8_source(addr, width, height, indices);
        if let Some(b) = self.backend.as_mut() {
            for key in stale {
                b.release_texture(key);
            }
        }
    }

    /// Forget any P8 data replaced by a non-palettised texture download.
    pub fn clear_p8(&mut self, addr: u32) {
        let stale = self.textures.remove_p8_source(addr);
        if let Some(b) = self.backend.as_mut() {
            for key in stale {
                b.release_texture(key);
            }
        }
    }

    /// Call before every draw: if the bound texture is palettised, bind the
    /// variant decoded against the palette live *now*, uploading it first if
    /// this (texture, palette) pair has no GPU copy yet.
    pub fn bind_for_draw(&mut self) -> Option<P8Uploaded> {
        let addr = self.bound_texture;
        let b = self.backend.as_mut()?;
        let resolved = self.textures.p8_draw(addr)?;
        let palette_id = self.textures.palette_id();
        for key in resolved.release {
            b.release_texture(key);
        }
        let uploaded = if let Some(tex) = resolved.upload {
            if b.upload_texture(resolved.key, tex.width, tex.height, &tex.rgba)
                .is_err()
            {
                self.textures.forget_p8_variant(addr, palette_id);
                b.set_texture(resolved.key);
                return None;
            }
            Some(P8Uploaded {
                addr,
                palette_id,
                texture: tex,
            })
        } else {
            None
        };
        b.set_texture(resolved.key);
        uploaded
    }
}
