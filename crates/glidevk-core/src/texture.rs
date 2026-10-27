//! Texture format conversion and caching.
//!
//! Games upload textures via `grTexDownloadMipMap` in one of the `GR_TEXFMT_*`
//! formats. Several of these have no direct `VkFormat` equivalent (P8
//! palettized, NCC/YIQ compressed) and must be decoded on the CPU (or via a
//! compute shader) into an RGBA image before upload. This is one of the
//! historically crash-prone areas — hence it lives in safe Rust with explicit
//! bounds.

use crate::types::TexFormat;
use std::collections::{HashMap, VecDeque};

/// A decoded, upload-ready texture (RGBA8), keyed by its Glide TMU address.
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    // pub image: Option<vk::Image>,  // populated on upload once VK is wired
}

/// Cache of decoded textures, indexed by the TMU start address the game uses as
/// a texture handle in `grTexSource`.
pub struct TextureCache {
    textures: HashMap<u32, Texture>,
    /// Current palette for `GR_TEXFMT_P_8` (set by `grTexDownloadTable`).
    palette: [u32; 256],
    /// Hash of `palette`'s content and the id assigned to that content. Ids
    /// are stable per content, so a game alternating between two palettes
    /// (Unreal uploads each texture's own palette right after binding it) maps
    /// back to the same GPU variants instead of re-uploading.
    palette_hash: u64,
    palette_id: u64,
    palette_ids: HashMap<u64, u64>,
    next_palette_id: u64,
    /// Raw index data of downloaded P8 textures, keyed by TMU address. Glide
    /// applies the palette at *draw* time (the TMU has one live LUT), so the
    /// indices are kept and decoded against whatever palette is current when a
    /// draw happens — see [`TextureCache::p8_draw`].
    p8_sources: HashMap<u32, (u32, u32, Vec<u8>)>,
    /// Palette ids that have a GPU copy for each P8 address, oldest first.
    p8_variants: HashMap<u32, VecDeque<u64>>,
}

/// GPU variants kept per palettised texture; older palettes are evicted. A
/// palette fade creates one variant per frame, so this bounds the churn.
pub const MAX_P8_VARIANTS: usize = 8;
/// Distinct palette contents remembered before the id map is reset (ids stay
/// unique; only the content→id memo is dropped).
const MAX_PALETTE_IDS: usize = 4096;

/// Backend texture key for a palettised texture decoded against one palette.
/// Plain textures use `addr as u64`, so variant keys never collide with them.
pub fn variant_key(addr: u32, palette_id: u64) -> u64 {
    (palette_id << 32) | addr as u64
}

/// What a draw with a P8 texture bound must do before drawing.
pub struct P8Draw {
    /// Backend key to bind.
    pub key: u64,
    /// Freshly decoded texture to upload under `key`, if not resident yet.
    pub upload: Option<Texture>,
    /// Keys evicted to make room; the backend should release them.
    pub release: Vec<u64>,
}

impl Default for TextureCache {
    fn default() -> Self {
        Self {
            textures: HashMap::new(),
            palette: [0; 256],
            palette_hash: 0,
            palette_id: 0,
            palette_ids: HashMap::new(),
            next_palette_id: 1,
            p8_sources: HashMap::new(),
            p8_variants: HashMap::new(),
        }
    }
}

impl TextureCache {
    /// Decode a source mip level into RGBA8. Returns the decoded texture without
    /// inserting it, so callers can associate it with a TMU address.
    ///
    /// Only the linear (non-compressed) formats are handled here for P0–P2;
    /// NCC/YIQ decode is a TODO gated behind P4 fidelity work.
    pub fn decode(
        &self,
        fmt: TexFormat,
        width: u32,
        height: u32,
        src: &[u8],
    ) -> Result<Texture, DecodeError> {
        let texels = (width as usize) * (height as usize);
        let mut rgba = vec![0u8; texels * 4];

        match fmt {
            TexFormat::Rgb565 => {
                let src16 = as_u16_slice(src, texels)?;
                for (i, &p) in src16.iter().enumerate() {
                    let r = ((p >> 11) & 0x1f) as u8;
                    let g = ((p >> 5) & 0x3f) as u8;
                    let b = (p & 0x1f) as u8;
                    rgba[i * 4] = (r << 3) | (r >> 2);
                    rgba[i * 4 + 1] = (g << 2) | (g >> 4);
                    rgba[i * 4 + 2] = (b << 3) | (b >> 2);
                    rgba[i * 4 + 3] = 0xff;
                }
            }
            TexFormat::Argb1555 => {
                let src16 = as_u16_slice(src, texels)?;
                for (i, &p) in src16.iter().enumerate() {
                    let a = if p & 0x8000 != 0 { 0xff } else { 0 };
                    let r = ((p >> 10) & 0x1f) as u8;
                    let g = ((p >> 5) & 0x1f) as u8;
                    let b = (p & 0x1f) as u8;
                    rgba[i * 4] = (r << 3) | (r >> 2);
                    rgba[i * 4 + 1] = (g << 3) | (g >> 2);
                    rgba[i * 4 + 2] = (b << 3) | (b >> 2);
                    rgba[i * 4 + 3] = a;
                }
            }
            TexFormat::Argb4444 => {
                let src16 = as_u16_slice(src, texels)?;
                for (i, &p) in src16.iter().enumerate() {
                    let a = ((p >> 12) & 0xf) as u8;
                    let r = ((p >> 8) & 0xf) as u8;
                    let g = ((p >> 4) & 0xf) as u8;
                    let b = (p & 0xf) as u8;
                    rgba[i * 4] = (r << 4) | r;
                    rgba[i * 4 + 1] = (g << 4) | g;
                    rgba[i * 4 + 2] = (b << 4) | b;
                    rgba[i * 4 + 3] = (a << 4) | a;
                }
            }
            TexFormat::P8 => {
                if src.len() < texels {
                    return Err(DecodeError::ShortBuffer);
                }
                for (i, &idx) in src[..texels].iter().enumerate() {
                    let c = self.palette[idx as usize];
                    rgba[i * 4] = (c >> 16) as u8;
                    rgba[i * 4 + 1] = (c >> 8) as u8;
                    rgba[i * 4 + 2] = c as u8;
                    // GR_TEXFMT_P_8 is 24-bit RGB indices; the palette's high
                    // byte is unused, so treat texels as opaque.
                    rgba[i * 4 + 3] = 0xff;
                }
            }
            TexFormat::Alpha8 | TexFormat::Intensity8 => {
                if src.len() < texels {
                    return Err(DecodeError::ShortBuffer);
                }
                for (i, &v) in src[..texels].iter().enumerate() {
                    let alpha = matches!(fmt, TexFormat::Alpha8);
                    rgba[i * 4] = if alpha { 0xff } else { v };
                    rgba[i * 4 + 1] = if alpha { 0xff } else { v };
                    rgba[i * 4 + 2] = if alpha { 0xff } else { v };
                    rgba[i * 4 + 3] = if alpha { v } else { 0xff };
                }
            }
            // TODO(P4): Argb8332, AlphaIntensity88, NCC (Yiq422 / Ayiq8422).
            other => return Err(DecodeError::Unsupported(other)),
        }

        Ok(Texture {
            width,
            height,
            rgba,
        })
    }

    /// `grTexDownloadTable`: make `palette` the live palette. Identical content
    /// is a no-op; otherwise the content gets a stable id used to key the GPU
    /// variants of palettised textures.
    pub fn set_palette(&mut self, palette: [u32; 256]) {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        palette.hash(&mut h);
        let hash = h.finish();
        if hash == self.palette_hash {
            return;
        }
        self.palette = palette;
        self.palette_hash = hash;
        if self.palette_ids.len() >= MAX_PALETTE_IDS {
            self.palette_ids.clear();
        }
        let next = &mut self.next_palette_id;
        self.palette_id = *self.palette_ids.entry(hash).or_insert_with(|| {
            let id = *next;
            *next += 1;
            id
        });
    }

    /// Id of the live palette's content (0 until the first upload).
    pub fn palette_id(&self) -> u64 {
        self.palette_id
    }

    /// Remember P8 index data for draw-time decoding. An identical download
    /// preserves resident variants; changed data invalidates them.
    pub fn store_p8_source(&mut self, addr: u32, width: u32, height: u32, src: &[u8]) -> Vec<u64> {
        if self
            .p8_sources
            .get(&addr)
            .is_some_and(|(w, h, old)| *w == width && *h == height && old.as_slice() == src)
        {
            return Vec::new();
        }
        self.p8_sources.insert(addr, (width, height, src.to_vec()));
        self.remove_p8_variants(addr)
    }

    /// A non-P8 download replaces any P8 source at the same address.
    pub fn remove_p8_source(&mut self, addr: u32) -> Vec<u64> {
        self.p8_sources.remove(&addr);
        self.remove_p8_variants(addr)
    }

    fn remove_p8_variants(&mut self, addr: u32) -> Vec<u64> {
        self.p8_variants
            .remove(&addr)
            .map(|ids| ids.into_iter().map(|id| variant_key(addr, id)).collect())
            .unwrap_or_default()
    }

    /// A failed GPU upload must be retried on the next draw.
    pub fn forget_p8_variant(&mut self, addr: u32, palette_id: u64) {
        if let Some(variants) = self.p8_variants.get_mut(&addr) {
            variants.retain(|&id| id != palette_id);
        }
    }

    /// Whether `addr` holds a palettised texture.
    pub fn is_p8(&self, addr: u32) -> bool {
        self.p8_sources.contains_key(&addr)
    }

    /// Resolve the P8 texture at `addr` for a draw under the live palette:
    /// returns the variant key to bind, plus a decoded texture to upload first
    /// if that (texture, palette) pair has no GPU copy yet. `None` if `addr`
    /// is not palettised.
    pub fn p8_draw(&mut self, addr: u32) -> Option<P8Draw> {
        let (w, h, src) = self.p8_sources.get(&addr)?;
        let id = self.palette_id;
        let key = variant_key(addr, id);
        if self.p8_variants.get(&addr).is_some_and(|v| v.contains(&id)) {
            return Some(P8Draw {
                key,
                upload: None,
                release: Vec::new(),
            });
        }
        let (w, h, src) = (*w, *h, src.clone());
        let tex = self.decode(TexFormat::P8, w, h, &src).ok()?;
        let variants = self.p8_variants.entry(addr).or_default();
        let mut release = Vec::new();
        while variants.len() >= MAX_P8_VARIANTS {
            if let Some(old) = variants.pop_front() {
                release.push(variant_key(addr, old));
            }
        }
        variants.push_back(id);
        Some(P8Draw {
            key,
            upload: Some(tex),
            release,
        })
    }

    pub fn insert(&mut self, tmu_addr: u32, tex: Texture) {
        self.textures.insert(tmu_addr, tex);
    }

    pub fn get(&self, tmu_addr: u32) -> Option<&Texture> {
        self.textures.get(&tmu_addr)
    }
}

fn as_u16_slice(src: &[u8], texels: usize) -> Result<Vec<u16>, DecodeError> {
    if src.len() < texels * 2 {
        return Err(DecodeError::ShortBuffer);
    }
    let (chunks, remainder) = src[..texels * 2].as_chunks::<2>();
    debug_assert!(remainder.is_empty());
    Ok(chunks
        .iter()
        .map(|chunk| u16::from_le_bytes(*chunk))
        .collect())
}

#[derive(Debug, PartialEq, Eq)]
pub enum DecodeError {
    ShortBuffer,
    Unsupported(TexFormat),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p8_resolves_against_the_palette_live_at_draw_time() {
        // Unreal's order: download indices, bind, THEN upload the palette, then
        // draw. The decode must use the palette current at the draw.
        let mut c = TextureCache::default();
        let mut zero = [0u32; 256];
        zero[1] = 0;
        c.set_palette(zero);
        assert!(c.store_p8_source(0x40, 2, 2, &[1, 1, 1, 1]).is_empty());
        let mut gray = [0u32; 256];
        gray[1] = 0x80_80_80;
        c.set_palette(gray);
        let d = c.p8_draw(0x40).unwrap();
        assert_eq!(d.key, variant_key(0x40, c.palette_id()));
        assert_eq!(d.upload.as_ref().unwrap().rgba[..3], [0x80, 0x80, 0x80]);
        // Same palette again: resident, no upload.
        assert!(c.p8_draw(0x40).unwrap().upload.is_none());
        // Alternating back to the first palette gives its own resident variant
        // after one upload, then no more churn.
        c.set_palette(zero);
        assert!(c.p8_draw(0x40).unwrap().upload.is_some());
        assert!(c.p8_draw(0x40).unwrap().upload.is_none());
        c.set_palette(gray);
        assert!(c.p8_draw(0x40).unwrap().upload.is_none());
        // A new texture at the address invalidates its variants.
        let released = c.store_p8_source(0x40, 2, 2, &[0; 4]);
        assert_eq!(released.len(), 2);
    }

    #[test]
    fn p8_repeated_download_preserves_only_identical_sources() {
        let mut cache = TextureCache::default();
        let src = [1, 2, 3, 4];
        cache.store_p8_source(7, 2, 2, &src);
        assert!(cache.p8_draw(7).unwrap().upload.is_some());
        cache.forget_p8_variant(7, cache.palette_id());
        assert!(cache.p8_draw(7).unwrap().upload.is_some());
        assert!(cache.store_p8_source(7, 2, 2, &src).is_empty());
        assert!(cache.p8_draw(7).unwrap().upload.is_none());
        assert_eq!(cache.store_p8_source(7, 2, 2, &[1, 2, 3, 5]).len(), 1);
        assert!(cache.p8_draw(7).unwrap().upload.is_some());
        assert_eq!(cache.store_p8_source(7, 1, 4, &[1, 2, 3, 5]).len(), 1);
        assert!(cache.p8_draw(7).unwrap().upload.is_some());
        assert_eq!(cache.remove_p8_source(7).len(), 1);
        assert!(!cache.is_p8(7));
        assert!(cache.p8_draw(7).is_none());
    }

    #[test]
    fn rgb565_opaque_white() {
        let cache = TextureCache::default();
        let src = 0xffff_u16.to_le_bytes();
        let t = cache.decode(TexFormat::Rgb565, 1, 1, &src).unwrap();
        assert_eq!(t.rgba, [0xff, 0xff, 0xff, 0xff]);
    }

    #[test]
    fn argb1555_transparent_bit() {
        let cache = TextureCache::default();
        let src = 0x0000_u16.to_le_bytes(); // top bit clear => alpha 0
        let t = cache.decode(TexFormat::Argb1555, 1, 1, &src).unwrap();
        assert_eq!(t.rgba[3], 0x00);
    }

    #[test]
    fn p8_uses_palette() {
        let mut cache = TextureCache::default();
        let mut pal = [0u32; 256];
        pal[5] = 0xff_11_22_33; // A R G B
        cache.set_palette(pal);
        let t = cache.decode(TexFormat::P8, 1, 1, &[5]).unwrap();
        assert_eq!(t.rgba, [0x11, 0x22, 0x33, 0xff]);
    }

    #[test]
    fn short_buffer_is_error() {
        let cache = TextureCache::default();
        let err = cache.decode(TexFormat::Rgb565, 4, 4, &[0, 0]).err();
        assert_eq!(err, Some(DecodeError::ShortBuffer));
    }
}
