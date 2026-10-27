//! Vulkan backend (feature = "vulkan").
//!
//! Thin wrapper over `ash`, dynamically loading the Vulkan loader at runtime:
//! under Wine that resolves `vulkan-1.dll` -> `winevulkan` (exactly the path
//! DXVK takes); natively it loads `libvulkan.so.1`.
//!
//! Entry points:
//! * [`Backend::new_headless`] — instance + device + queue, no surface. Lets
//!   the Vulkan bring-up be exercised on a normal Linux host with a plain
//!   `#[test]`, independent of Wine's 32-bit Vulkan stack.
//! * [`Backend::new_windowed`] — adds a Win32 surface, swapchain, render pass,
//!   the combiner graphics pipeline, and texture/descriptor machinery
//!   (Windows/Wine only).
//!
//! Frame model (mirrors Glide's clear/draw/swap):
//! `set_clear_color` (grBufferClear) -> `draw_triangles`* (grDrawTriangle /
//! grDrawVertexArray) -> `present` (grBufferSwap). The frame's render pass is
//! begun lazily on the first draw (or at present, for a bare clear).

mod frame_pacer;

use std::collections::HashMap;
use std::ffi::CStr;
use std::io::Cursor;

use ash::vk;

fn clear_aspects(rgb_write: bool, alpha_write: bool, depth_write: bool) -> vk::ImageAspectFlags {
    let mut aspects = vk::ImageAspectFlags::empty();
    if rgb_write || alpha_write {
        aspects |= vk::ImageAspectFlags::COLOR;
    }
    if depth_write {
        aspects |= vk::ImageAspectFlags::DEPTH;
    }
    aspects
}

fn clear_color_write_mask(rgb_write: bool, alpha_write: bool) -> u32 {
    (if rgb_write { 7 } else { 0 }) | (if alpha_write { 8 } else { 0 })
}

/// Map logical Glide clip edges to Vulkan pixel-center bounds. Glide's minimum
/// is inclusive and its maximum exclusive; lower-left origin reverses Y.
fn clip_scissor(
    bounds: Option<[u32; 4]>,
    logical: (u32, u32),
    extent: vk::Extent2D,
    lower_left: bool,
) -> vk::Rect2D {
    let (width, height) = (logical.0.max(1), logical.1.max(1));
    let [x0, y0, x1, y1] = bounds.unwrap_or([0, 0, width, height]);
    let (x0, x1) = (x0.min(width), x1.min(width));
    let (y0, y1) = (y0.min(height), y1.min(height));
    let (y0, y1) = if lower_left {
        (height - y1, height - y0)
    } else {
        (y0, y1)
    };
    // A pixel is inside when its center is within the scaled clip bounds.
    let edge = |value: u32, src: u32, dst: u32| -> u32 {
        let n = (2 * value as u128 * dst as u128).saturating_sub(src as u128);
        n.div_ceil(2 * src as u128).min(dst as u128) as u32
    };
    let (left, right) = (edge(x0, width, extent.width), edge(x1, width, extent.width));
    let (top, bottom) = (
        edge(y0, height, extent.height),
        edge(y1, height, extent.height),
    );
    vk::Rect2D {
        offset: vk::Offset2D {
            x: left as i32,
            y: top as i32,
        },
        extent: vk::Extent2D {
            width: right.saturating_sub(left),
            height: bottom.saturating_sub(top),
        },
    }
}

/// P2 combiner shaders, precompiled to SPIR-V (see `shaders/build.sh`). The
/// fragment shader is an ubershader implementing the Glide combine units,
/// parameterized by the push constants below.
const VERT_SPV: &[u8] = include_bytes!("shaders/combine.vert.spv");
const FRAG_SPV: &[u8] = include_bytes!("shaders/combine.frag.spv");

/// A triangle vertex in the layout the pipeline expects: window-space (pixel)
/// position + normalized RGBA + TMU0 texcoord. `grDraw*` normalizes the game's
/// vertices (per `grVertexLayout`) into this.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DrawVertex {
    pub pos: [f32; 2],
    pub color: [f32; 4],
    /// TMU0 texture coords as Glide delivers them: `sow = s/w`, `tow = t/w`
    /// (texel space). The fragment shader recovers `s = sow/oow` and normalizes
    /// by the texture size.
    pub tex: [f32; 2],
    pub depth: f32,
    /// `oow = 1/w` for perspective-correct texturing. 1.0 = affine (no
    /// perspective), which is correct for 2D/screen-space geometry.
    pub oow: f32,
}

impl Default for DrawVertex {
    fn default() -> Self {
        Self {
            pos: [0.0; 2],
            color: [0.0; 4],
            tex: [0.0; 2],
            depth: 0.0,
            oow: 1.0,
        }
    }
}

const VERTEX_STRIDE: u64 = std::mem::size_of::<DrawVertex>() as u64; // 40
/// Per-frame vertex-buffer capacity (~29k verts). Draws beyond this are dropped
/// rather than reallocating mid-frame (revisit in P4).
const VERTEX_CAPACITY_BYTES: u64 = 1 << 20;

/// Vulkan update commands accept at most 64 KiB per four-byte-aligned chunk.
fn vertex_upload_chunks(vertices: &[DrawVertex]) -> impl Iterator<Item = (u64, &[u8])> {
    let size = std::mem::size_of_val(vertices);
    debug_assert_eq!(size % 4, 0);
    // SAFETY: DrawVertex is repr(C), consists solely of initialized f32s, and
    // its borrowed bytes cannot outlive the source slice.
    let bytes = unsafe { std::slice::from_raw_parts(vertices.as_ptr().cast::<u8>(), size) };
    bytes
        .chunks(65_536)
        .enumerate()
        .map(|(i, chunk)| ((i * 65_536) as u64, chunk))
}

/// Combine-unit configuration for a draw, as the shader consumes it. Each
/// `[i32; 5]` is `[function, factor, local, other, invert]` using the
/// `types::Combine*` discriminants.
#[derive(Clone, Copy, Default)]
pub struct CombineParams {
    pub color: [i32; 5],
    pub alpha: [i32; 5],
    pub const_color: [f32; 4],
    /// `grAlphaControlsITRGBLighting` — texture alpha MSB picks the local color.
    pub alpha_it_lighting: i32,
}

impl CombineParams {
    /// Encode the current color/alpha combine state for the shader. Shared by
    /// the `glide2x` / `glide3x` shells.
    pub fn from_state(s: &crate::state::State) -> Self {
        let c = &s.color_combine;
        let a = &s.alpha_combine;
        Self {
            color: [
                c.function as i32,
                c.factor as i32,
                c.local as i32,
                c.other as i32,
                c.invert as i32,
            ],
            alpha: [
                a.function as i32,
                a.factor as i32,
                a.local as i32,
                a.other as i32,
                a.invert as i32,
            ],
            const_color: s.constant_color,
            alpha_it_lighting: s.alpha_it_lighting as i32,
        }
    }
}

/// Alpha-blend configuration (`grAlphaBlendFunction`). Factor codes are
/// `types::BlendFactor` discriminants. Baked into the pipeline (part of
/// [`PipeKey`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendConfig {
    pub enable: bool,
    pub src_rgb: i32,
    pub dst_rgb: i32,
    pub src_alpha: i32,
    pub dst_alpha: i32,
    /// `VkColorComponentFlags` bits (R=1,G=2,B=4,A=8) from `grColorMask`.
    pub write_mask: u32,
}

impl Default for BlendConfig {
    fn default() -> Self {
        // GR_BLEND_ONE / GR_BLEND_ZERO == opaque.
        Self {
            enable: false,
            src_rgb: 1,
            dst_rgb: 0,
            src_alpha: 1,
            dst_alpha: 0,
            write_mask: 0xf, // RGBA
        }
    }
}

impl BlendConfig {
    /// Blend config from state; blending is off for the opaque identity.
    pub fn from_state(s: &crate::state::State) -> Self {
        use crate::types::BlendFactor;
        let opaque = s.blend_src_rgb == BlendFactor::One
            && s.blend_dst_rgb == BlendFactor::Zero
            && s.blend_src_alpha == BlendFactor::One
            && s.blend_dst_alpha == BlendFactor::Zero;
        // VkColorComponentFlags: R=1, G=2, B=4, A=8.
        let mut write_mask = 0u32;
        if s.color_mask_rgb {
            write_mask |= 0x1 | 0x2 | 0x4;
        }
        if s.color_mask_alpha {
            write_mask |= 0x8;
        }
        Self {
            enable: !opaque,
            src_rgb: s.blend_src_rgb as i32,
            dst_rgb: s.blend_dst_rgb as i32,
            src_alpha: s.blend_src_alpha as i32,
            dst_alpha: s.blend_dst_alpha as i32,
            write_mask,
        }
    }
}

/// Depth-test configuration (`grDepthBufferMode`/`Function`/`grDepthMask`).
/// `compare` is a `GR_CMP_*` code. Baked into the pipeline.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct DepthConfig {
    pub test: bool,
    pub write: bool,
    pub compare: i32,
}

impl DepthConfig {
    /// Depth config from state.
    pub fn from_state(s: &crate::state::State) -> Self {
        Self {
            test: s.depth_mode != crate::state::DepthMode::Disable,
            write: s.depth_write,
            compare: s.depth_compare,
        }
    }
}

/// The baked-in state that selects a `VkPipeline` from the cache. Everything
/// else (combine, chroma, alpha-test) is push-constant-driven and shares one
/// pipeline.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PipeKey {
    blend: BlendConfig,
    depth: DepthConfig,
    wireframe: bool,
}

/// Push-constant block shared by `combine.vert` / `combine.frag`. Layout must
/// match the GLSL `Push` block exactly (std430-compatible scalar packing;
/// `vec4`s sit on 16-byte offsets).
#[repr(C)]
#[derive(Clone, Copy)]
struct PushConstants {
    viewport: [f32; 2],    // 0
    y_sign: f32,           // 8
    _pad0: f32,            // 12
    const_color: [f32; 4], // 16
    chroma_key: [f32; 4],  // 32
    c_func: i32,           // 48
    c_factor: i32,
    c_local: i32,
    c_other: i32,
    c_invert: i32,
    a_func: i32, // 68
    a_factor: i32,
    a_local: i32,
    a_other: i32,
    a_invert: i32,
    chroma_enable: i32,   // 88
    alpha_test_func: i32, // 92
    alpha_ref: f32,       // 96
    gamma: f32,           // 100
    tex_size: [f32; 2],   // 104
    alpha_lighting: i32,  // 112 (grAlphaControlsITRGBLighting)
    wireframe: i32,       // 116 (GLIDEVK_WIREFRAME: force a bright output colour)
}

/// Errors from backend bring-up, resource creation, or presentation.
#[derive(Debug)]
pub enum Error {
    Loading(String),
    Vk(vk::Result),
    NoPhysicalDevice,
    NoMemoryType,
}

impl From<vk::Result> for Error {
    fn from(r: vk::Result) -> Self {
        Error::Vk(r)
    }
}

/// A texture resident on the GPU: image + backing memory + view + the
/// descriptor set that binds it (with the shared sampler) to the shader.
struct GpuTexture {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    /// Descriptor bound with the REPEAT (wrap) sampler.
    descriptor: vk::DescriptorSet,
    /// Descriptor bound with the CLAMP sampler; selected when the game sets
    /// grTexClampMode to clamp.
    descriptor_clamp: vk::DescriptorSet,
    width: u32,
    height: u32,
    /// Frame this texture was last bound or drawn. A re-upload may only overwrite
    /// the image in place when it was not used in the frame being recorded;
    /// earlier frames are already complete (their fence is waited on).
    used_frame: u32,
}

fn mark_texture_drawn(textures: &mut HashMap<u64, GpuTexture>, key: Option<u64>, frame: u32) {
    if let Some(texture) = key.and_then(|key| textures.get_mut(&key)) {
        texture.used_frame = frame;
    }
}

/// Upload resources kept alive until a later render submission has waited for
/// the upload's semaphore. Texture downloads can arrive in large batches
/// before the game's first `grBufferSwap`, so waiting for the queue after each
/// individual upload stalls startup badly on Wine/MoltenVK.
struct PendingUpload {
    /// TMU address lets the cache avoid two deferred writes to the same image.
    addr: Option<u64>,
    command_buffer: vk::CommandBuffer,
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
}

/// One queue submission for all texture transfers accumulated before a frame.
struct SubmittedUploadBatch {
    uploads: Vec<PendingUpload>,
    ready: vk::Semaphore,
}

/// Surface + swapchain + per-image render targets.
struct Swapchain {
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    /// Presentation waits are indexed by image: reacquiring an image ensures
    /// its previous present wait has completed before this semaphore is reused.
    render_finished: Vec<vk::Semaphore>,
    color_image: vk::Image,
    color_memory: vk::DeviceMemory,
    color_view: vk::ImageView,
    framebuffer: vk::Framebuffer,
    targets_initialized: bool,
}

/// When the backend copies a presented frame to the host for
/// [`Backend::take_capture`] (the `GLIDEVK_CAPTURE` diagnostic).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CaptureMode {
    /// No capture. Must stay the default: a capture costs a full-image copy to
    /// host memory, a GPU wait and a per-pixel CPU conversion every time.
    #[default]
    Off,
    /// Only this frame.
    Frame(u32),
    /// Every n-th frame.
    Every(u32),
}

impl CaptureMode {
    /// Decode a `GLIDEVK_CAPTURE_FRAME` value: `n > 0` captures frame `n`,
    /// `n < 0` every `|n|` frames, `0` nothing.
    pub fn from_frame_arg(n: i32) -> Self {
        match n {
            0 => CaptureMode::Off,
            n if n > 0 => CaptureMode::Frame(n as u32),
            n => CaptureMode::Every(n.unsigned_abs()),
        }
    }

    /// Whether frame `frame` is captured.
    pub fn wants(self, frame: u32) -> bool {
        match self {
            CaptureMode::Off => false,
            CaptureMode::Frame(n) => frame == n,
            CaptureMode::Every(n) => frame > 0 && frame.is_multiple_of(n),
        }
    }
}

/// A Vulkan device bound to one Glide context.
pub struct Backend {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    physical: vk::PhysicalDevice,
    queue_family: u32,
    queue: vk::Queue,

    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    vertex_upload_command: vk::CommandBuffer,
    image_available: vk::Semaphore,
    in_flight: vk::Fence,

    // Graphics: render pass, a pipeline cache keyed by baked state (blend +
    // depth), the shared pipeline layout + shader modules, a depth buffer, a
    // vertex buffer, and a 1-pixel readback for the tests.
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipelines: HashMap<PipeKey, vk::Pipeline>,
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,
    render_extent: vk::Extent2D,
    logical_size: [f32; 2],
    depth_image: vk::Image,
    depth_memory: vk::DeviceMemory,
    depth_view: vk::ImageView,
    vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    /// CPU-side vertices for the frame being recorded.
    vertices: Vec<DrawVertex>,
    /// Draw calls dropped because the current frame exceeded the vertex buffer.
    dropped_draws: u64,
    readback_buffer: vk::Buffer,
    readback_memory: vk::DeviceMemory,
    readback_mapped: *mut u8,

    // P2 texturing: shared sampler + descriptor layout/pool, a texture cache
    // keyed by Glide TMU start address, and a 1x1 white default.
    sampler: vk::Sampler,
    /// CLAMP-address sampler, selected when the game sets grTexClampMode=clamp.
    sampler_clamp: vk::Sampler,
    desc_set_layout: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,
    /// Keyed by TMU address for plain textures, or `texture::variant_key` for
    /// palettised ones (one GPU copy per palette content).
    textures: HashMap<u64, GpuTexture>,
    /// Textures replaced this frame (by a re-upload). They can still be
    /// referenced by draws already recorded into the current command buffer, so
    /// destroying them immediately faults the GPU — hold them until the frame's
    /// fence signals in `present_inner`, then destroy.
    retired_textures: Vec<GpuTexture>,
    /// Ordered upload submissions not yet consumed by a frame submission.
    pending_uploads: Vec<PendingUpload>,
    /// Submitted upload batches waiting to be consumed by a render submission.
    submitted_uploads: Vec<SubmittedUploadBatch>,
    default_texture: GpuTexture,
    current_desc: vk::DescriptorSet,
    /// Cache key for the selected texture, if it is not the default texture.
    current_texture_key: Option<u64>,
    /// The clamp-sampler descriptor for the currently bound texture; `draw`
    /// selects between this and `current_desc` per `clamp_mode`.
    current_desc_clamp: vk::DescriptorSet,
    /// Live grTexClampMode: true = CLAMP, false = REPEAT (wrap).
    clamp_mode: bool,
    current_tex_size: [f32; 2],

    // Per-draw state pushed / baked at draw time.
    combine: CombineParams,
    blend: BlendConfig,
    depth_cfg: DepthConfig,
    chroma: (bool, [f32; 3]),
    alpha_test: (i32, f32),
    clip_window: Option<[u32; 4]>,
    scissor_dirty: bool,
    /// 3dfx output gamma (`grGammaCorrectionValue`); 1.0 = identity. Applied in
    /// the combiner's final output to emulate the RAMDAC gamma ramp.
    gamma: f32,
    /// `GLIDEVK_WIREFRAME` debug mode: draw geometry as bright outlines,
    /// overriding the (possibly dark/invisible) combined colour so all geometry
    /// is visible regardless of shading.
    wireframe: bool,
    /// `GLIDEVK_FLAT` debug mode: draw every primitive as solid opaque red with
    /// no texture, no combine, no blend, no depth test, and no alpha/chroma
    /// discard — so the raw *geometry* (where each quad/triangle actually lands)
    /// is visible independent of the texture-coordinate / combine pipeline. This
    /// isolates "is the geometry right?" from "is the texturing right?".
    debug_flat: bool,
    /// `GLIDEVK_GAMMA=<v>` overrides the game's output gamma — a brightness knob
    /// to lift dark, dimly-lit geometry (e.g. Deus Ex's untextured 3D logos) to
    /// match a reference renderer, without guessing the value in code.
    gamma_override: Option<f32>,

    // Linear framebuffer (`grLfbLock`): the game writes 16/32-bit pixels into
    // `lfb_buffer`; on the next present those are unpacked into `lfb_staging`
    // (BGRA8) and copied into the presented image.
    lfb_buffer: Vec<u8>,
    /// Dimensions the LFB buffer was locked at (the logical surface size the game
    /// writes, which can differ from the render extent). Used to read the buffer
    /// back with the right row stride on upload.
    lfb_dims: (u32, u32),
    lfb_staging: vk::Buffer,
    lfb_staging_memory: vk::DeviceMemory,
    lfb_staging_mapped: *mut u8,
    lfb_write_mode: i32,
    lfb_dirty: bool,

    // Frame capture for offline verification: copy the presented image to a
    // host buffer on a target frame so a diagnostic can dump it to disk.
    capture_buffer: vk::Buffer,
    capture_memory: vk::DeviceMemory,
    capture_mapped: *mut u8,
    /// Size of `capture_buffer` in bytes; a full-image copy is only recorded
    /// when the image fits, so a stale buffer can never be overrun.
    capture_bytes: u64,
    capture: CaptureMode,
    frame_num: u32,
    frame_pacer: frame_pacer::FramePacer,
    last_capture: Option<(u32, u32, u32, Vec<u8>)>, // (frame, w, h, rgba)
    /// Size of the frame `capture_buffer` currently holds: every presented
    /// frame is copied there on the GPU (cheap), so a framebuffer read always
    /// sees the front buffer. The CPU only touches it for a read or a capture.
    capture_extent: Option<(u32, u32)>,

    swapchain: Option<Swapchain>,

    // Frame state.
    pending_clear: [f32; 4],
    y_sign: f32,
    current_image: Option<u32>,
    vertex_count: u32,
    last_center_pixel: [u8; 4], // RGBA
}

/// Depth attachment format used throughout.
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

// SAFETY: the raw `*mut u8` mappings make `Backend` `!Send` automatically, but
// Glide is single-threaded and every access goes through the `glide3x` context
// Mutex. Asserting `Send` lets the context live in a `static Mutex`.
unsafe impl Send for Backend {}

impl Backend {
    /// Bring up instance + device + queue with no surface. Host-testable.
    pub fn new_headless() -> Result<Self, Error> {
        // SAFETY: loads the platform Vulkan loader; unsafe purely by ash's API.
        let entry = unsafe { ash::Entry::load() }.map_err(|e| Error::Loading(e.to_string()))?;
        let instance = create_instance(&entry, &[])?;
        Self::finish(entry, instance, None)
    }

    /// Bring up instance + device + Win32 surface + swapchain + pipeline.
    ///
    /// `hwnd`/`hinstance` are the Win32 handles from `grSstWinOpen`'s window.
    #[cfg(windows)]
    pub fn new_windowed(
        hwnd: isize,
        hinstance: isize,
        width: u32,
        height: u32,
    ) -> Result<Self, Error> {
        use ash::khr;

        let entry = unsafe { ash::Entry::load() }.map_err(|e| Error::Loading(e.to_string()))?;
        let exts = [
            khr::surface::NAME.as_ptr(),
            khr::win32_surface::NAME.as_ptr(),
        ];
        let instance = create_instance(&entry, &exts)?;

        // SAFETY: hinstance/hwnd are valid Win32 handles owned by the caller.
        let surface = unsafe {
            let win32_loader = khr::win32_surface::Instance::new(&entry, &instance);
            let info = vk::Win32SurfaceCreateInfoKHR::default()
                .hinstance(hinstance)
                .hwnd(hwnd);
            win32_loader.create_win32_surface(&info, None)?
        };
        let surface_loader = khr::surface::Instance::new(&entry, &instance);

        Self::finish(
            entry,
            instance,
            Some((surface_loader, surface, vk::Extent2D { width, height })),
        )
    }

    /// Shared tail: pick a physical device + graphics/present queue, create the
    /// logical device, per-frame sync, and — for windowed contexts — the
    /// swapchain, render pass, pipeline, and buffers.
    fn finish(
        entry: ash::Entry,
        instance: ash::Instance,
        surface: Option<(ash::khr::surface::Instance, vk::SurfaceKHR, vk::Extent2D)>,
    ) -> Result<Self, Error> {
        let has_surface = surface.is_some();
        // SAFETY: `instance` is valid for the duration of these queries.
        let physicals = unsafe { instance.enumerate_physical_devices()? };
        let surface_ref = surface.as_ref().map(|(l, s, _)| (l, *s));

        let (physical, queue_family) = physicals
            .iter()
            .find_map(|&pd| pick_queue_family(&instance, pd, surface_ref).map(|qf| (pd, qf)))
            .ok_or(Error::NoPhysicalDevice)?;

        let device_exts = if surface.is_some() {
            vec![ash::khr::swapchain::NAME.as_ptr()]
        } else {
            vec![]
        };
        let priorities = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities);
        let queue_infos = [queue_info];
        // fillModeNonSolid enables VK_POLYGON_MODE_LINE for the GLIDEVK_WIREFRAME
        // debug mode (supported by all desktop drivers we target).
        let features = vk::PhysicalDeviceFeatures::default().fill_mode_non_solid(true);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_exts)
            .enabled_features(&features);
        // SAFETY: physical device + queue family were validated above.
        let device = unsafe { instance.create_device(physical, &device_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let command_pool = unsafe {
            let info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            device.create_command_pool(&info, None)?
        };
        let (command_buffer, vertex_upload_command) = unsafe {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(2);
            let commands = device.allocate_command_buffers(&info)?;
            (commands[0], commands[1])
        };
        let (image_available, in_flight) = unsafe {
            let sem = vk::SemaphoreCreateInfo::default();
            let fence = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            (
                device.create_semaphore(&sem, None)?,
                device.create_fence(&fence, None)?,
            )
        };

        // Vertex + readback buffers exist regardless of surface (harmless for
        // headless; the draw path just never runs there).
        let (vertex_buffer, vertex_memory, _vertex_mapped) = unsafe {
            create_mapped_buffer(
                &instance,
                &device,
                physical,
                VERTEX_CAPACITY_BYTES,
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                HostAccess::Upload,
            )?
        };
        let (readback_buffer, readback_memory, readback_mapped) = unsafe {
            create_mapped_buffer(
                &instance,
                &device,
                physical,
                4,
                vk::BufferUsageFlags::TRANSFER_DST,
                HostAccess::Readback,
            )?
        };

        // Two shared samplers (REPEAT / CLAMP) selected per-draw by the game's
        // grTexClampMode; each texture gets a descriptor bound to each. Plus a
        // 1x1 white default texture (bound whenever no game texture is selected).
        let sampler = unsafe { create_sampler(&device, vk::SamplerAddressMode::REPEAT)? };
        let sampler_clamp =
            unsafe { create_sampler(&device, vk::SamplerAddressMode::CLAMP_TO_EDGE)? };
        let desc_set_layout = unsafe { create_desc_set_layout(&device)? };
        let desc_pool = unsafe { create_desc_pool(&device)? };
        let (default_texture, initial_upload) = unsafe {
            create_gpu_texture(
                &instance,
                &device,
                physical,
                command_pool,
                queue,
                sampler,
                sampler_clamp,
                desc_pool,
                desc_set_layout,
                1,
                1,
                &[255, 255, 255, 255],
                0,
                has_surface,
                None,
            )?
        };
        let pending_uploads = initial_upload.into_iter().collect();
        let current_desc = default_texture.descriptor;
        let default_texture_desc_clamp = default_texture.descriptor_clamp;

        // Shader modules + pipeline layout are shared by every cached pipeline.
        let vert_module = unsafe { load_shader(&device, VERT_SPV)? };
        let frag_module = unsafe { load_shader(&device, FRAG_SPV)? };
        let pipeline_layout = unsafe { create_pipeline_layout(&device, desc_set_layout)? };

        // Swapchain + depth buffer + render pass, when there is a surface.
        // Pipelines are built lazily per (blend, depth) state.
        let mut render_extent = vk::Extent2D {
            width: 0,
            height: 0,
        };
        let mut depth_image = vk::Image::null();
        let mut depth_memory = vk::DeviceMemory::null();
        let mut depth_view = vk::ImageView::null();
        // The resolution the game asked for (its coordinate space), which may
        // differ from the swapchain's actual size (the window). Geometry maps to
        // this; the swapchain scales it to fill the window.
        let mut logical_size = [640.0_f32, 480.0];
        let (swapchain, render_pass) = match surface {
            Some((surface_loader, surface, extent)) => {
                logical_size = [extent.width as f32, extent.height as f32];
                let mut sc = create_swapchain(
                    &instance,
                    &device,
                    physical,
                    surface_loader,
                    surface,
                    extent,
                )?;
                render_extent = sc.extent;
                let (di, dm, dv) =
                    unsafe { create_depth_buffer(&instance, &device, physical, sc.extent)? };
                depth_image = di;
                depth_memory = dm;
                depth_view = dv;
                let render_pass = create_render_pass(&device, sc.format)?;
                create_framebuffers(
                    &instance,
                    &device,
                    physical,
                    render_pass,
                    depth_view,
                    &mut sc,
                )?;
                (Some(sc), render_pass)
            }
            None => (None, vk::RenderPass::null()),
        };

        // LFB staging: BGRA8 upload buffer sized to the render target.
        let lfb_bytes = ((render_extent.width * render_extent.height * 4) as u64).max(4);
        let (lfb_staging, lfb_staging_memory, lfb_staging_mapped) = unsafe {
            create_mapped_buffer(
                &instance,
                &device,
                physical,
                lfb_bytes,
                vk::BufferUsageFlags::TRANSFER_SRC,
                HostAccess::Upload,
            )?
        };
        let (capture_buffer, capture_memory, capture_mapped) = unsafe {
            create_mapped_buffer(
                &instance,
                &device,
                physical,
                lfb_bytes,
                vk::BufferUsageFlags::TRANSFER_DST,
                HostAccess::Readback,
            )?
        };

        Ok(Self {
            _entry: entry,
            instance,
            device,
            physical,
            queue_family,
            queue,
            command_pool,
            command_buffer,
            vertex_upload_command,
            image_available,
            in_flight,
            render_pass,
            pipeline_layout,
            pipelines: HashMap::new(),
            vert_module,
            frag_module,
            render_extent,
            logical_size,
            depth_image,
            depth_memory,
            depth_view,
            vertex_buffer,
            vertex_memory,
            vertices: Vec::new(),
            dropped_draws: 0,
            readback_buffer,
            readback_memory,
            readback_mapped,
            sampler,
            sampler_clamp,
            desc_set_layout,
            desc_pool,
            textures: HashMap::new(),
            retired_textures: Vec::new(),
            pending_uploads,
            submitted_uploads: Vec::new(),
            default_texture,
            current_desc,
            current_texture_key: None,
            current_desc_clamp: default_texture_desc_clamp,
            clamp_mode: false,
            current_tex_size: [1.0, 1.0],
            combine: CombineParams::default(),
            blend: BlendConfig::default(),
            depth_cfg: DepthConfig::default(),
            chroma: (false, [0.0; 3]),
            alpha_test: (7, 0.0), // GR_CMP_ALWAYS
            clip_window: None,
            scissor_dirty: true,
            gamma: 1.0,
            wireframe: std::env::var_os("GLIDEVK_WIREFRAME").is_some(),
            debug_flat: std::env::var_os("GLIDEVK_FLAT").is_some(),
            gamma_override: std::env::var("GLIDEVK_GAMMA")
                .ok()
                .and_then(|s| s.parse().ok()),
            lfb_buffer: Vec::new(),
            lfb_dims: (0, 0),
            lfb_staging,
            lfb_staging_memory,
            lfb_staging_mapped,
            lfb_write_mode: 0,
            lfb_dirty: false,
            capture_buffer,
            capture_memory,
            capture_mapped,
            capture_bytes: lfb_bytes,
            capture: CaptureMode::Off,
            frame_num: 0,
            frame_pacer: frame_pacer::FramePacer::from_env(),
            last_capture: None,
            capture_extent: None,
            swapchain,
            pending_clear: [0.0, 0.0, 0.0, 1.0],
            y_sign: 1.0,
            current_image: None,
            vertex_count: 0,
            last_center_pixel: [0, 0, 0, 0],
        })
    }

    /// Human-readable name of the selected physical device.
    pub fn device_name(&self) -> String {
        // SAFETY: `physical` belongs to `instance`.
        let props = unsafe { self.instance.get_physical_device_properties(self.physical) };
        let raw = props.device_name.as_ptr();
        unsafe { CStr::from_ptr(raw) }
            .to_string_lossy()
            .into_owned()
    }

    pub fn queue_family(&self) -> u32 {
        self.queue_family
    }

    /// Swapchain dimensions, or `None` for a headless context.
    pub fn extent(&self) -> Option<(u32, u32)> {
        self.swapchain
            .as_ref()
            .map(|s| (s.extent.width, s.extent.height))
    }

    /// The game's requested resolution (its coordinate space) — what
    /// `grSstScreenWidth/Height` should report so its UI layout matches.
    pub fn logical_size(&self) -> (u32, u32) {
        (self.logical_size[0] as u32, self.logical_size[1] as u32)
    }

    /// Change the game's coordinate space without touching the swapchain — for
    /// a `grSstWinOpen` re-open at a different resolution on a live backend.
    pub fn set_logical_size(&mut self, width: u32, height: u32) {
        self.logical_size = [width.max(1) as f32, height.max(1) as f32];
        self.scissor_dirty = true;
    }

    /// Dimensions (px) of the currently-bound texture, exactly as fed to the
    /// shader's `texSize` push constant (`[1.0, 1.0]` when the 1×1 white default
    /// is bound because `grTexSource` referenced an unknown address). Exposed so
    /// the shell can log the real ST→UV mapping for coordinate diagnostics.
    pub fn current_tex_size(&self) -> [f32; 2] {
        self.current_tex_size
    }

    /// Set the color the next frame's render pass clears to (`grBufferClear`).
    pub fn set_clear_color(&mut self, color: [f32; 4]) {
        self.pending_clear = color;
    }

    /// Clear the current Glide clip rectangle at the point the game requests it.
    pub fn clear_masked(
        &mut self,
        color: [f32; 4],
        depth: u32,
        rgb_write: bool,
        alpha_write: bool,
        depth_write: bool,
    ) -> Result<(), Error> {
        self.pending_clear = color;
        let aspects = clear_aspects(rgb_write, alpha_write, depth_write);
        if self.swapchain.is_none() || aspects.is_empty() {
            return Ok(());
        }
        self.ensure_frame()?;
        let rect = clip_scissor(
            self.clip_window,
            self.logical_size(),
            self.swapchain.as_ref().unwrap().extent,
            self.y_sign < 0.0,
        );
        if rect.extent.width == 0 || rect.extent.height == 0 {
            return Ok(());
        }
        let mut attachments = Vec::with_capacity(2);
        if rgb_write != alpha_write {
            // vkCmdClearAttachments ignores color write masks. A clipped quad
            // with a masked pipeline preserves the unselected channels.
            self.clear_color_masked(color, rgb_write, alpha_write)?;
        } else if rgb_write && alpha_write {
            attachments.push(vk::ClearAttachment {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                color_attachment: 0,
                clear_value: vk::ClearValue {
                    color: vk::ClearColorValue { float32: color },
                },
            });
        }
        if aspects.contains(vk::ImageAspectFlags::DEPTH) {
            attachments.push(vk::ClearAttachment {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                color_attachment: 0,
                clear_value: vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: depth.min(65535) as f32 / 65535.0,
                        stencil: 0,
                    },
                },
            });
        }
        let rect = vk::ClearRect {
            rect,
            base_array_layer: 0,
            layer_count: 1,
        };
        // SAFETY: ensure_frame began a render pass using a framebuffer that
        // contains both attachments and rect is clamped to its extent.
        if !attachments.is_empty() {
            unsafe {
                self.device
                    .cmd_clear_attachments(self.command_buffer, &attachments, &[rect]);
            }
        }
        Ok(())
    }

    /// Draw an unblended constant-color quad for RGB-only or alpha-only clears.
    /// All game state is restored before returning, including after an error.
    fn clear_color_masked(
        &mut self,
        color: [f32; 4],
        rgb_write: bool,
        alpha_write: bool,
    ) -> Result<(), Error> {
        let needed = (self.vertex_count as u64 + 6) * VERTEX_STRIDE;
        if needed > VERTEX_CAPACITY_BYTES {
            return Err(Error::Loading(
                "no vertex space for masked framebuffer clear".into(),
            ));
        }
        let saved = (
            self.blend,
            self.depth_cfg,
            self.combine,
            self.chroma,
            self.alpha_test,
            self.gamma,
            self.gamma_override,
            self.wireframe,
            self.debug_flat,
        );
        self.blend = BlendConfig {
            write_mask: clear_color_write_mask(rgb_write, alpha_write),
            ..BlendConfig::default()
        };
        self.depth_cfg = DepthConfig::default();
        self.combine = CombineParams {
            color: [1, 0, 1, 0, 0], // LOCAL / CONSTANT
            alpha: [1, 0, 1, 0, 0],
            const_color: color,
            alpha_it_lighting: 0,
        };
        self.chroma = (false, [0.0; 3]);
        self.alpha_test = (7, 0.0);
        self.gamma = 1.0;
        self.gamma_override = None;
        self.wireframe = false;
        self.debug_flat = false;
        let [width, height] = self.logical_size;
        let v = |x, y| DrawVertex {
            pos: [x, y],
            ..DrawVertex::default()
        };
        let quad = [
            v(0.0, 0.0),
            v(width, 0.0),
            v(width, height),
            v(0.0, 0.0),
            v(width, height),
            v(0.0, height),
        ];
        let result = self.draw_triangles(&quad);
        (
            self.blend,
            self.depth_cfg,
            self.combine,
            self.chroma,
            self.alpha_test,
            self.gamma,
            self.gamma_override,
            self.wireframe,
            self.debug_flat,
        ) = saved;
        result
    }

    /// `true` for GR_ORIGIN_UPPER_LEFT (Vulkan-native), `false` for LOWER_LEFT.
    pub fn set_origin_upper_left(&mut self, upper_left: bool) {
        self.y_sign = if upper_left { 1.0 } else { -1.0 };
        self.scissor_dirty = true;
    }

    /// Set a Glide clip rectangle (minimum inclusive, maximum exclusive).
    /// Out-of-range or reversed bounds leave the previous window unchanged.
    pub fn set_clip_window(&mut self, bounds: [u32; 4]) {
        let (width, height) = self.logical_size();
        if bounds[0] > bounds[2] || bounds[1] > bounds[3] || bounds[2] > width || bounds[3] > height
        {
            return;
        }
        self.clip_window = Some(bounds);
        self.scissor_dirty = true;
    }

    /// Last presented frame's center-pixel color as RGBA. Smoke-test aid.
    pub fn center_pixel_rgba(&self) -> [u8; 4] {
        self.last_center_pixel
    }

    /// Set the combine-unit configuration used by subsequent draws.
    pub fn set_combine(&mut self, combine: CombineParams) {
        self.combine = combine;
    }

    /// Set the 3dfx output gamma (`grGammaCorrectionValue`). Clamped to a sane
    /// range; 1.0 disables correction.
    pub fn set_gamma(&mut self, gamma: f32) {
        self.gamma = gamma.clamp(0.25, 4.0);
    }

    /// Set alpha-blend state (`grAlphaBlendFunction`). Baked into the pipeline.
    pub fn set_blend(&mut self, blend: BlendConfig) {
        self.blend = blend;
    }

    /// Set depth-test state (`grDepthBufferMode`/`Function`/`grDepthMask`).
    pub fn set_depth(&mut self, depth: DepthConfig) {
        self.depth_cfg = depth;
    }

    /// Set chroma-key state (`grChromakeyMode`/`Value`); `key` is RGB in 0..1.
    pub fn set_chroma(&mut self, enable: bool, key: [f32; 3]) {
        self.chroma = (enable, key);
    }

    /// Set alpha-test state (`grAlphaTestFunction`/`ReferenceValue`); `func` is
    /// a `GR_CMP_*` code (7 = ALWAYS = disabled), `reference` in 0..1.
    pub fn set_alpha_test(&mut self, func: i32, reference: f32) {
        self.alpha_test = (func, reference);
    }

    /// Lock the linear framebuffer for the given `write_mode` (`GR_LFBWRITEMODE_*`):
    /// return a writable staging pointer + row stride sized to the render target.
    /// The game writes packed pixels here; [`Backend::lfb_unlock`] flags them for
    /// upload on the next present.
    ///
    /// TODO: read locks should first read back the current image; unwritten
    /// pixels on a write lock currently retain the previous buffer contents
    /// rather than the live framebuffer.
    pub fn lfb_lock(&mut self, write_mode: i32) -> (*mut u8, u32) {
        // Size to the *logical* surface (what the game sees via grSstScreenWidth/
        // Height and writes into), which can exceed the render extent; using the
        // smaller extent would let the game's writes overrun the buffer.
        let (lw, lh) = (self.logical_size[0] as u32, self.logical_size[1] as u32);
        let (ew, eh) = self.extent().unwrap_or((640, 480));
        let (w, h) = (lw.max(ew).max(1), lh.max(eh).max(1));
        let bpp = if lfb_mode_is_32bit(write_mode) { 4 } else { 2 };
        let stride = w * bpp;
        let need = (stride as usize) * (h as usize);
        if self.lfb_buffer.len() < need {
            self.lfb_buffer.resize(need, 0);
        }
        self.lfb_write_mode = write_mode;
        self.lfb_dims = (w, h);
        (self.lfb_buffer.as_mut_ptr(), stride)
    }

    /// Unlock the linear framebuffer, flagging its contents for upload into the
    /// image at the next `grBufferSwap`.
    pub fn lfb_unlock(&mut self) {
        self.lfb_dirty = true;
    }

    /// Unpack `lfb_buffer` (in the current write mode) into `lfb_staging` as the
    /// swapchain's BGRA8, ready for a buffer→image copy.
    fn pack_lfb_to_staging(&mut self, w: u32, h: u32) {
        let mode = self.lfb_write_mode;
        let bpp = if lfb_mode_is_32bit(mode) { 4 } else { 2 };
        // The game wrote the buffer at the lock dimensions (`lfb_dims`); read it
        // back with that row stride and map 1:1 into the extent-sized staging,
        // clamping to whichever is smaller so neither side overruns.
        let (lw, lh) = self.lfb_dims;
        let src_stride = (lw.max(w) * bpp) as usize;
        let pixels = (w * h) as usize;
        // SAFETY: `lfb_staging_mapped` maps at least w*h*4 bytes (sized in finish).
        let dst = unsafe { std::slice::from_raw_parts_mut(self.lfb_staging_mapped, pixels * 4) };
        let (cw, ch) = (w.min(lw.max(1)) as usize, h.min(lh.max(1)) as usize);
        for y in 0..ch {
            for x in 0..cw {
                let si = y * src_stride + x * bpp as usize;
                let di = (y * w as usize + x) * 4;
                let (r, g, b, a) = decode_lfb_pixel(&self.lfb_buffer, si, mode);
                // Swapchain is B8G8R8A8_UNORM.
                dst[di] = b;
                dst[di + 1] = g;
                dst[di + 2] = r;
                dst[di + 3] = a;
            }
        }
    }

    /// Return the cached pipeline for `key`, building + caching it on first use.
    fn get_or_build_pipeline(&mut self, key: PipeKey) -> Result<vk::Pipeline, Error> {
        if let Some(&p) = self.pipelines.get(&key) {
            return Ok(p);
        }
        let pipeline = profile_vk_call("build_graphics_pipeline", || {
            build_pipeline(
                &self.device,
                self.render_pass,
                self.pipeline_layout,
                self.vert_module,
                self.frag_module,
                key,
            )
        })?;
        self.pipelines.insert(key, pipeline);
        Ok(pipeline)
    }

    /// Submit all recorded texture copies as one batch. The render submission
    /// waits on the batch semaphore, so many Glide downloads can be collected
    /// without blocking the game's thread or creating one queue dependency per
    /// texture.
    fn submit_pending_uploads(&mut self) -> Result<(), Error> {
        if self.pending_uploads.is_empty() {
            return Ok(());
        }
        let ready = unsafe {
            self.device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
        };
        let commands: Vec<_> = self
            .pending_uploads
            .iter()
            .map(|upload| upload.command_buffer)
            .collect();
        let signals = [ready];
        let submit = vk::SubmitInfo::default()
            .command_buffers(&commands)
            .signal_semaphores(&signals);
        if let Err(error) = profile_vk_call("upload_queue_submit", || unsafe {
            self.device
                .queue_submit(self.queue, &[submit], vk::Fence::null())
        }) {
            unsafe { self.device.destroy_semaphore(ready, None) };
            return Err(error.into());
        }
        self.submitted_uploads.push(SubmittedUploadBatch {
            uploads: std::mem::take(&mut self.pending_uploads),
            ready,
        });
        Ok(())
    }

    /// Decode + upload a texture, keyed by its Glide TMU start address. Replaces
    /// any previous texture at that address.
    pub fn upload_texture(
        &mut self,
        addr: u64,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Result<(), Error> {
        // Re-upload of the same texture at the same size: overwrite the image
        // in place. Games re-download constantly (Unreal refreshes lightmaps
        // every frame), and making a new image + descriptors each time both
        // costs milliseconds and used to exhaust the descriptor pool. Only safe
        // when the texture is not already bound in the frame being recorded;
        // earlier frames have completed (their fence is waited on at present).
        let frame = self.frame_num;
        let deferred = self.swapchain.is_some();
        let upload_already_pending = self
            .pending_uploads
            .iter()
            .any(|upload| upload.addr == Some(addr))
            || self
                .submitted_uploads
                .iter()
                .any(|batch| batch.uploads.iter().any(|upload| upload.addr == Some(addr)));
        if let Some(existing) = self.textures.get(&addr) {
            if existing.width == width
                && existing.height == height
                && existing.used_frame != frame
                && !upload_already_pending
            {
                let image = existing.image;
                if deferred {
                    // The image is not bound in this frame. Record its update
                    // for the next batched transfer submission.
                    let upload = unsafe {
                        record_image_upload(
                            &self.instance,
                            &self.device,
                            self.physical,
                            self.command_pool,
                            image,
                            width,
                            height,
                            rgba,
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            Some(addr),
                        )?
                    };
                    self.pending_uploads.push(upload);
                } else {
                    // Headless backends have no render submission to batch with.
                    unsafe {
                        upload_image_pixels_sync(
                            &self.instance,
                            &self.device,
                            self.physical,
                            self.command_pool,
                            self.queue,
                            image,
                            width,
                            height,
                            rgba,
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        )?;
                    }
                }
                if let Some(t) = self.textures.get_mut(&addr) {
                    t.used_frame = frame;
                }
                return Ok(());
            }
        }
        // SAFETY: all handles are owned by `self` and valid.
        let (tex, pending) = profile_vk_call("create_gpu_texture_total", || unsafe {
            create_gpu_texture(
                &self.instance,
                &self.device,
                self.physical,
                self.command_pool,
                self.queue,
                self.sampler,
                self.sampler_clamp,
                self.desc_pool,
                self.desc_set_layout,
                width,
                height,
                rgba,
                frame,
                deferred,
                Some(addr),
            )
        })?;
        if let Some(upload) = pending {
            self.pending_uploads.push(upload);
        }
        if let Some(old) = self.textures.insert(addr, tex) {
            // Do NOT destroy eagerly: a re-upload can happen mid-frame (P8
            // palette re-decode) while draws already recorded this frame still
            // reference `old`. Retire it and destroy once the frame's fence
            // signals (see `present_inner`).
            self.retired_textures.push(old);
            if self.current_texture_key == Some(addr) {
                // Future draws must use the replacement descriptor; previous
                // draws retain the old descriptor until the frame completes.
                self.set_texture(addr);
            }
        }
        Ok(())
    }

    /// Select the texture at `addr` for subsequent draws (`grTexSource`).
    /// Returns `false` (and binds the 1x1 white default) for an unknown address.
    /// Drop the texture under `key` once the current frame no longer needs it.
    pub fn release_texture(&mut self, key: u64) {
        if let Some(old) = self.textures.remove(&key) {
            if self.current_texture_key == Some(key) {
                // The selected descriptor is about to be retired. Switch to
                // the default before it can be freed after the frame fence.
                self.set_texture(key);
            }
            self.retired_textures.push(old);
        }
    }

    pub fn set_texture(&mut self, addr: u64) -> bool {
        let frame = self.frame_num;
        match self.textures.get_mut(&addr) {
            Some(t) => {
                t.used_frame = frame;
                self.current_texture_key = Some(addr);
                self.current_desc = t.descriptor;
                self.current_desc_clamp = t.descriptor_clamp;
                self.current_tex_size = [t.width as f32, t.height as f32];
                true
            }
            None => {
                self.current_texture_key = None;
                self.current_desc = self.default_texture.descriptor;
                self.current_desc_clamp = self.default_texture.descriptor_clamp;
                self.current_tex_size = [1.0, 1.0];
                false
            }
        }
    }

    /// Set the texture address mode (`grTexClampMode`): true = CLAMP, false =
    /// REPEAT. Applied per-draw by selecting the matching descriptor.
    pub fn set_clamp(&mut self, clamp: bool) {
        self.clamp_mode = clamp;
    }

    /// Return and reset the number of draw calls dropped for lack of space.
    pub fn take_dropped_draws(&mut self) -> u64 {
        std::mem::take(&mut self.dropped_draws)
    }

    /// Record `verts` (a triangle list) into the in-progress frame, beginning
    /// the frame if needed. No-op for a headless context.
    pub fn draw_triangles(&mut self, verts: &[DrawVertex]) -> Result<(), Error> {
        if self.swapchain.is_none() || verts.is_empty() {
            return Ok(());
        }
        profile_vk_call("draw_ensure_frame", || self.ensure_frame())?;

        if self.scissor_dirty {
            let extent = self.swapchain.as_ref().unwrap().extent;
            let scissor = clip_scissor(
                self.clip_window,
                self.logical_size(),
                extent,
                self.y_sign < 0.0,
            );
            // SAFETY: the command buffer is recording inside the current render pass.
            unsafe {
                profile_vk_call("draw_set_scissor", || {
                    self.device
                        .cmd_set_scissor(self.command_buffer, 0, &[scissor])
                })
            };
            self.scissor_dirty = false;
        }

        let first = self.vertex_count;
        let byte_off = first as u64 * VERTEX_STRIDE;
        let bytes = std::mem::size_of_val(verts) as u64;
        if byte_off + bytes > VERTEX_CAPACITY_BYTES {
            // Over budget for this frame; drop rather than corrupt.
            self.dropped_draws = self.dropped_draws.saturating_add(1);
            return Ok(());
        }

        // GLIDEVK_FLAT: force opaque, depth-test-off pipeline so every primitive
        // is drawn (nothing blended away or depth-culled) — the raw geometry.
        let key = if self.debug_flat {
            PipeKey {
                blend: BlendConfig::default(),
                depth: DepthConfig::default(),
                wireframe: self.wireframe,
            }
        } else {
            PipeKey {
                blend: self.blend,
                depth: self.depth_cfg,
                wireframe: self.wireframe,
            }
        };
        let pipeline = self.get_or_build_pipeline(key)?;

        let (chroma_enable, chroma_key) = self.chroma;
        let mut push = PushConstants {
            // Map vertices in the game's coordinate space; the VkViewport scales
            // that to the actual swapchain size.
            viewport: self.logical_size,
            y_sign: self.y_sign,
            _pad0: 0.0,
            const_color: self.combine.const_color,
            chroma_key: [chroma_key[0], chroma_key[1], chroma_key[2], 1.0],
            c_func: self.combine.color[0],
            c_factor: self.combine.color[1],
            c_local: self.combine.color[2],
            c_other: self.combine.color[3],
            c_invert: self.combine.color[4],
            a_func: self.combine.alpha[0],
            a_factor: self.combine.alpha[1],
            a_local: self.combine.alpha[2],
            a_other: self.combine.alpha[3],
            a_invert: self.combine.alpha[4],
            chroma_enable: chroma_enable as i32,
            alpha_test_func: self.alpha_test.0,
            alpha_ref: self.alpha_test.1,
            gamma: self.gamma_override.unwrap_or(self.gamma),
            tex_size: self.current_tex_size,
            alpha_lighting: self.combine.alpha_it_lighting,
            wireframe: self.wireframe as i32,
        };
        if self.debug_flat {
            // Solid opaque red, no texture (combine LOCAL/CONSTANT), no discards.
            // GR_COMBINE_FUNCTION_LOCAL = 1, GR_COMBINE_LOCAL_CONSTANT = 1,
            // GR_CMP_ALWAYS = 7. The shader then outputs `constColor` unmodified.
            push.const_color = [1.0, 0.0, 0.0, 1.0];
            push.c_func = 1;
            push.c_local = 1;
            push.c_invert = 0;
            push.a_func = 1;
            push.a_local = 1;
            push.a_invert = 0;
            push.chroma_enable = 0;
            push.alpha_test_func = 7;
            push.alpha_lighting = 0;
            push.gamma = 1.0;
        }
        // Keep draw data in ordinary CPU memory. Upload all vertices before
        // submitting this frame, without writing to mapped Vulkan memory.
        profile_vk_call("draw_vertex_copy", || {
            self.vertices.extend_from_slice(verts)
        });
        // SAFETY: the descriptor/pipeline handles are owned by `self`.
        unsafe {
            profile_vk_call("draw_bind_pipeline", || {
                self.device.cmd_bind_pipeline(
                    self.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline,
                )
            });
            let desc = if self.clamp_mode {
                self.current_desc_clamp
            } else {
                self.current_desc
            };
            profile_vk_call("draw_bind_descriptors", || {
                self.device.cmd_bind_descriptor_sets(
                    self.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[desc],
                    &[],
                )
            });
            let push_bytes = std::slice::from_raw_parts(
                &push as *const PushConstants as *const u8,
                std::mem::size_of::<PushConstants>(),
            );
            profile_vk_call("draw_push_constants", || {
                self.device.cmd_push_constants(
                    self.command_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    push_bytes,
                )
            });
            profile_vk_call("draw_command", || {
                self.device
                    .cmd_draw(self.command_buffer, verts.len() as u32, 1, first, 0)
            });
        }
        // Glide retains its texture binding across frames. Mark the texture
        // actually referenced by this draw even if there was no grTexSource
        // in the current frame, so later uploads cannot overwrite its image
        // before the recorded commands execute.
        mark_texture_drawn(&mut self.textures, self.current_texture_key, self.frame_num);
        self.vertex_count += verts.len() as u32;
        Ok(())
    }

    /// Destroy a GPU texture's resources (not its descriptor set, which returns
    /// to the pool on pool reset/destroy).
    unsafe fn destroy_texture(&self, tex: &GpuTexture) {
        // The pool is created with FREE_DESCRIPTOR_SET so these go back to it.
        let _ = self
            .device
            .free_descriptor_sets(self.desc_pool, &[tex.descriptor, tex.descriptor_clamp]);
        self.device.destroy_image_view(tex.view, None);
        self.device.destroy_image(tex.image, None);
        self.device.free_memory(tex.memory, None);
    }

    /// End the in-progress frame (or a bare clear) and present it.
    /// Present the current frame, recreating the swapchain on out-of-date /
    /// suboptimal (window resize, alt-tab, minimize) instead of failing.
    pub fn present(&mut self) -> Result<(), Error> {
        if self.swapchain.is_none() {
            return Ok(());
        }
        match self.present_inner() {
            Err(Error::Vk(r))
                if r == vk::Result::ERROR_OUT_OF_DATE_KHR || r == vk::Result::SUBOPTIMAL_KHR =>
            {
                self.current_image = None;
                // Recreate to the new window size; skip this frame.
                self.recreate_swapchain()
            }
            other => other,
        }
    }

    /// Tear down and rebuild the swapchain (and depth + framebuffers + LFB
    /// staging) for the surface's current size. Reuses the existing surface.
    fn recreate_swapchain(&mut self) -> Result<(), Error> {
        let old = match self.swapchain.take() {
            Some(s) => s,
            None => return Ok(()),
        };
        // SAFETY: wait until the device is idle before destroying in-use objects.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_framebuffer(old.framebuffer, None);
            self.device.destroy_image_view(old.color_view, None);
            self.device.destroy_image(old.color_image, None);
            self.device.free_memory(old.color_memory, None);
            for &semaphore in &old.render_finished {
                self.device.destroy_semaphore(semaphore, None);
            }
            old.swapchain_loader.destroy_swapchain(old.swapchain, None);
            self.device.destroy_image_view(self.depth_view, None);
            self.device.destroy_image(self.depth_image, None);
            self.device.free_memory(self.depth_memory, None);
            // Drop must not destroy these handles again if recreation fails.
            self.depth_view = vk::ImageView::null();
            self.depth_image = vk::Image::null();
            self.depth_memory = vk::DeviceMemory::null();
        }
        // Reuse the surface; create_swapchain re-queries the current extent.
        let surface_loader = old.surface_loader;
        let mut sc = match create_swapchain(
            &self.instance,
            &self.device,
            self.physical,
            surface_loader.clone(),
            old.surface,
            self.render_extent,
        ) {
            Ok(sc) => sc,
            Err(error) => {
                unsafe { surface_loader.destroy_surface(old.surface, None) };
                return Err(error);
            }
        };
        let result = (|| -> Result<(), Error> {
            // SAFETY: fresh device/physical handles.
            let (di, dm, dv) = unsafe {
                create_depth_buffer(&self.instance, &self.device, self.physical, sc.extent)?
            };
            self.depth_image = di;
            self.depth_memory = dm;
            self.depth_view = dv;
            create_framebuffers(
                &self.instance,
                &self.device,
                self.physical,
                self.render_pass,
                dv,
                &mut sc,
            )?;

            // Resize LFB staging to the new extent.
            let need = ((sc.extent.width * sc.extent.height * 4) as u64).max(4);
            // SAFETY: old staging is no longer referenced by any in-flight command.
            unsafe {
                let (b, m, p) = create_mapped_buffer(
                    &self.instance,
                    &self.device,
                    self.physical,
                    need,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    HostAccess::Upload,
                )?;
                self.device.destroy_buffer(self.lfb_staging, None);
                self.device.free_memory(self.lfb_staging_memory, None);
                self.lfb_staging = b;
                self.lfb_staging_memory = m;
                self.lfb_staging_mapped = p;

                // The capture buffer receives full-image copies: it must follow the
                // swapchain too, or a larger mode overruns it on the GPU and the
                // host read past its mapping.
                let (b, m, p) = create_mapped_buffer(
                    &self.instance,
                    &self.device,
                    self.physical,
                    need,
                    vk::BufferUsageFlags::TRANSFER_DST,
                    HostAccess::Readback,
                )?;
                self.device.destroy_buffer(self.capture_buffer, None);
                self.device.free_memory(self.capture_memory, None);
                self.capture_buffer = b;
                self.capture_memory = m;
                self.capture_mapped = p;
                self.capture_bytes = need;
                self.capture_extent = None;
            }
            Ok(())
        })();
        if let Err(error) = result {
            unsafe {
                self.device.destroy_framebuffer(sc.framebuffer, None);
                self.device.destroy_image_view(sc.color_view, None);
                self.device.destroy_image(sc.color_image, None);
                self.device.free_memory(sc.color_memory, None);
                for &semaphore in &sc.render_finished {
                    self.device.destroy_semaphore(semaphore, None);
                }
                sc.swapchain_loader.destroy_swapchain(sc.swapchain, None);
                sc.surface_loader.destroy_surface(sc.surface, None);
                self.device.destroy_image_view(self.depth_view, None);
                self.device.destroy_image(self.depth_image, None);
                self.device.free_memory(self.depth_memory, None);
            }
            self.depth_view = vk::ImageView::null();
            self.depth_image = vk::Image::null();
            self.depth_memory = vk::DeviceMemory::null();
            return Err(error);
        }
        self.render_extent = sc.extent;
        self.swapchain = Some(sc);
        self.current_image = None;
        Ok(())
    }

    /// Record all vertex data outside the render pass, in a command buffer
    /// submitted immediately before the frame's rendering commands.
    fn record_vertex_upload(&self) -> Result<bool, Error> {
        if self.vertices.is_empty() {
            return Ok(false);
        }
        let size = std::mem::size_of_val(self.vertices.as_slice());
        debug_assert_eq!(size % 4, 0);
        unsafe {
            self.device.reset_command_buffer(
                self.vertex_upload_command,
                vk::CommandBufferResetFlags::empty(),
            )?;
            self.device.begin_command_buffer(
                self.vertex_upload_command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            for (offset, chunk) in vertex_upload_chunks(&self.vertices) {
                profile_vk_call("vertex_command_update", || {
                    self.device.cmd_update_buffer(
                        self.vertex_upload_command,
                        self.vertex_buffer,
                        offset,
                        chunk,
                    )
                });
            }
            let barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::VERTEX_ATTRIBUTE_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.vertex_buffer)
                .offset(0)
                .size(size as u64);
            self.device.cmd_pipeline_barrier(
                self.vertex_upload_command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::VERTEX_INPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
            self.device.end_command_buffer(self.vertex_upload_command)?;
        }
        Ok(true)
    }

    fn present_inner(&mut self) -> Result<(), Error> {
        if self.swapchain.is_none() {
            return Ok(());
        }
        // A swap with no draws still shows the clear color.
        self.ensure_frame()?;
        let index = self.current_image.take().expect("frame begun above");
        let sc = self.swapchain.as_ref().unwrap();
        let image = sc.color_image;
        let swap_image = sc.images[index as usize];
        let render_finished = sc.render_finished[index as usize];
        let extent = sc.extent;

        let color_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        // SAFETY: handles are owned by `self`; command buffer is recording.
        unsafe {
            self.device.cmd_end_render_pass(self.command_buffer);

            // LFB composite: upload the game's framebuffer writes over the
            // rendered image. The render pass left the image in
            // TRANSFER_SRC_OPTIMAL; move it to DST, copy, then back to SRC.
            if self.lfb_dirty {
                self.pack_lfb_to_staging(extent.width, extent.height);
                image_layout_barrier(
                    &self.device,
                    self.command_buffer,
                    image,
                    color_range,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                let copy = vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    },
                };
                self.device.cmd_copy_buffer_to_image(
                    self.command_buffer,
                    self.lfb_staging,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy],
                );
                image_layout_barrier(
                    &self.device,
                    self.command_buffer,
                    image,
                    color_range,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                );
                self.lfb_dirty = false;
            }

            // Copy the center pixel out for the smoke test. The image is in
            // TRANSFER_SRC_OPTIMAL (render-pass finalLayout, or restored above).
            let region = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D {
                    x: (extent.width / 2) as i32,
                    y: (extent.height / 2) as i32,
                    z: 0,
                },
                image_extent: vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            };
            self.device.cmd_copy_image_to_buffer(
                self.command_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback_buffer,
                &[region],
            );

            // Copy the whole image to the host buffer every frame (a GPU-side
            // copy, no CPU work) so framebuffer reads see the front buffer from
            // the very first call — only if it fits the buffer.
            let capturing = self.capture.wants(self.frame_num);
            let image_bytes = extent.width as u64 * extent.height as u64 * 4;
            let copy_full = image_bytes <= self.capture_bytes;
            if copy_full {
                let full = vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    },
                };
                self.device.cmd_copy_image_to_buffer(
                    self.command_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.capture_buffer,
                    &[full],
                );
            }

            // Copy the preserved logical framebuffer to the acquired swapchain image.
            let to_dst = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(swap_image)
                .subresource_range(color_range);
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_dst],
            );
            let subresource = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };
            let copy = vk::ImageCopy {
                src_subresource: subresource,
                src_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                dst_subresource: subresource,
                dst_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                extent: vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
            };
            self.device.cmd_copy_image(
                self.command_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                swap_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );
            // TRANSFER_DST_OPTIMAL -> PRESENT_SRC_KHR for the presentation engine.
            let barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(swap_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
            self.device.end_command_buffer(self.command_buffer)?;

            let has_vertex_upload = self.record_vertex_upload()?;
            self.submit_pending_uploads()?;
            let mut wait = vec![self.image_available];
            let mut stages = vec![vk::PipelineStageFlags::TRANSFER];
            // All recorded texture copies were submitted as batches. Waiting
            // here makes their image writes visible before the render commands.
            for upload_batch in &self.submitted_uploads {
                wait.push(upload_batch.ready);
                stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
            }
            let signal = [render_finished];
            let cmd = if has_vertex_upload {
                vec![self.vertex_upload_command, self.command_buffer]
            } else {
                vec![self.command_buffer]
            };
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait)
                .wait_dst_stage_mask(&stages)
                .command_buffers(&cmd)
                .signal_semaphores(&signal);
            self.device.reset_fences(&[self.in_flight])?;
            profile_vk_call("render_queue_submit", || {
                self.device
                    .queue_submit(self.queue, &[submit], self.in_flight)
            })?;
            // The recorded initialization only becomes effective after submit.
            self.swapchain.as_mut().unwrap().targets_initialized = true;

            // Wait so the readback pixel is valid (and single-frame-in-flight).
            profile_vk_call("render_fence_wait", || {
                self.device
                    .wait_for_fences(&[self.in_flight], true, u64::MAX)
            })?;
            // The render submission waited on every batch semaphore, so its
            // fence proves the staging resources are no longer in use.
            destroy_submitted_uploads(&self.device, self.command_pool, &mut self.submitted_uploads);
            // The frame that referenced any textures retired during recording is
            // now complete on the GPU; it's safe to destroy them.
            for old in std::mem::take(&mut self.retired_textures) {
                self.destroy_texture(&old);
            }
            // Swapchain format is B8G8R8A8_UNORM: bytes are B,G,R,A.
            let px = std::slice::from_raw_parts(self.readback_mapped, 4);
            self.last_center_pixel = [px[2], px[1], px[0], px[3]];

            self.capture_extent = copy_full.then_some((extent.width, extent.height));
            // Full-frame CPU conversion only on a requested capture frame.
            if copy_full && capturing {
                let (w, h) = (extent.width as usize, extent.height as usize);
                let src = std::slice::from_raw_parts(self.capture_mapped, w * h * 4);
                let mut rgba = vec![0u8; w * h * 4];
                for i in 0..w * h {
                    // BGRA -> RGBA
                    rgba[i * 4] = src[i * 4 + 2];
                    rgba[i * 4 + 1] = src[i * 4 + 1];
                    rgba[i * 4 + 2] = src[i * 4];
                    rgba[i * 4 + 3] = 255;
                }
                self.last_capture = Some((self.frame_num, extent.width, extent.height, rgba));
            }

            let sc = self.swapchain.as_ref().unwrap();
            let swapchains = [sc.swapchain];
            let indices = [index];
            let present = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal)
                .swapchains(&swapchains)
                .image_indices(&indices);
            self.frame_pacer.wait();
            profile_vk_call("queue_present", || {
                sc.swapchain_loader.queue_present(self.queue, &present)
            })?;
        }
        self.frame_num += 1;
        Ok(())
    }

    /// Choose which presented frames are copied to the host for
    /// [`Backend::take_capture`].
    pub fn set_capture(&mut self, mode: CaptureMode) {
        self.capture = mode;
    }

    /// Take the last captured frame (frame index, width, height, RGBA8), if any.
    pub fn take_capture(&mut self) -> Option<(u32, u32, u32, Vec<u8>)> {
        self.last_capture.take()
    }

    /// `grLfbReadRegion`: copy a rectangle of the front buffer (the last
    /// presented frame) into `dst` as RGB565 rows of `dst_stride` bytes.
    /// Coordinates are in the game's logical space and scaled to the frame's
    /// extent. Only the requested pixels are read from the host buffer. Before
    /// the first present the region is zero-filled and this returns false.
    pub fn read_region_rgb565(
        &mut self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        dst_stride: u32,
        dst: &mut [u8],
    ) -> bool {
        let (lw, lh) = self.logical_size();
        let Some((fw, fh)) = self.capture_extent else {
            dst.fill(0);
            return false;
        };
        // SAFETY: `capture_extent` is only set after the frame's fence signalled
        // and when the image fitted `capture_bytes`, the mapping's size.
        let bgra =
            unsafe { std::slice::from_raw_parts(self.capture_mapped, (fw * fh * 4) as usize) };
        for row in 0..h {
            let fy = ((y + row) * fh / lh.max(1)).min(fh - 1) as usize;
            for col in 0..w {
                let fx = ((x + col) * fw / lw.max(1)).min(fw - 1) as usize;
                let i = (fy * fw as usize + fx) * 4;
                // Swapchain bytes are B,G,R,A.
                let px = crate::lfb::pack_rgb565(bgra[i + 2], bgra[i + 1], bgra[i]);
                let o = (row * dst_stride + col * 2) as usize;
                if o + 2 <= dst.len() {
                    dst[o..o + 2].copy_from_slice(&px.to_le_bytes());
                }
            }
        }
        true
    }

    /// Begin a frame's render pass if one is not already in progress: wait for
    /// the previous frame, acquire an image, and start recording with the
    /// pipeline, vertex buffer, viewport, and push constants bound.
    fn ensure_frame(&mut self) -> Result<(), Error> {
        if self.current_image.is_some() {
            return Ok(());
        }
        // Follow the window: games resize it after grSstWinOpen (Unreal moves
        // it to the Glide resolution) and Wine does not always report a size
        // change as OUT_OF_DATE, so compare the surface extent ourselves.
        if let Some(sc) = self.swapchain.as_ref() {
            // SAFETY: valid physical device and surface owned by `sc`.
            let caps = profile_vk_call("surface_capabilities", || unsafe {
                sc.surface_loader
                    .get_physical_device_surface_capabilities(self.physical, sc.surface)
            })?;
            let cur = caps.current_extent;
            if cur.width != u32::MAX
                && cur.width > 0
                && cur.height > 0
                && (cur.width != sc.extent.width || cur.height != sc.extent.height)
            {
                self.recreate_swapchain()?;
            }
        }
        let index;
        let sc = self.swapchain.as_mut().unwrap();
        // SAFETY: owned handles; single-frame-in-flight enforced by the fence.
        unsafe {
            // Wait for the previous frame; the fence is reset just before submit
            // in `present_inner`, so an out-of-date acquire here leaves it
            // signaled (no hang) and the swapchain gets recreated.
            profile_vk_call("previous_frame_fence_wait", || {
                self.device
                    .wait_for_fences(&[self.in_flight], true, u64::MAX)
            })?;

            let (idx, _) = profile_vk_call("acquire_next_image", || {
                sc.swapchain_loader.acquire_next_image(
                    sc.swapchain,
                    u64::MAX,
                    self.image_available,
                    vk::Fence::null(),
                )
            })?;
            index = idx;

            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(self.command_buffer, &begin)?;

            if !sc.targets_initialized {
                let color_range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let depth_range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let barriers = [
                    vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(sc.color_image)
                        .subresource_range(color_range),
                    vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(self.depth_image)
                        .subresource_range(depth_range),
                ];
                self.device.cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers,
                );
                self.device.cmd_clear_color_image(
                    self.command_buffer,
                    sc.color_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                    &[color_range],
                );
                self.device.cmd_clear_depth_stencil_image(
                    self.command_buffer,
                    self.depth_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                    &[depth_range],
                );
                let barriers = [
                    vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(sc.color_image)
                        .subresource_range(color_range),
                    vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(
                            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        )
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(self.depth_image)
                        .subresource_range(depth_range),
                ];
                self.device.cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers,
                );
            }

            let clears = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: self.pending_clear,
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];
            let rp_begin = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(sc.framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: sc.extent,
                })
                .clear_values(&clears);
            self.device.cmd_begin_render_pass(
                self.command_buffer,
                &rp_begin,
                vk::SubpassContents::INLINE,
            );

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: sc.extent.width as f32,
                height: sc.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: sc.extent,
            };
            self.device
                .cmd_set_viewport(self.command_buffer, 0, &[viewport]);
            self.device
                .cmd_set_scissor(self.command_buffer, 0, &[scissor]);
            // The next draw applies any Glide clip window, including one saved
            // before this frame or before a swapchain resize.
            self.scissor_dirty = true;
            self.device.cmd_bind_vertex_buffers(
                self.command_buffer,
                0,
                &[self.vertex_buffer],
                &[0],
            );
            // Viewport, scissor, and vertex buffer are bound once per frame; the
            // pipeline (blend/depth state), descriptor set, and push constants
            // are (re)bound per draw, since they can change between draws.
        }
        self.current_image = Some(index);
        self.vertex_count = 0;
        self.vertices.clear();
        Ok(())
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        // SAFETY: destroy in reverse creation order after the device is idle.
        unsafe {
            let _ = self.device.device_wait_idle();
            destroy_pending_uploads(&self.device, self.command_pool, &mut self.pending_uploads);
            destroy_submitted_uploads(&self.device, self.command_pool, &mut self.submitted_uploads);
            if let Some(sc) = &self.swapchain {
                self.device.destroy_framebuffer(sc.framebuffer, None);
                self.device.destroy_image_view(sc.color_view, None);
                self.device.destroy_image(sc.color_image, None);
                self.device.free_memory(sc.color_memory, None);
                for &semaphore in &sc.render_finished {
                    self.device.destroy_semaphore(semaphore, None);
                }
                sc.swapchain_loader.destroy_swapchain(sc.swapchain, None);
                sc.surface_loader.destroy_surface(sc.surface, None);
            }
            for &pipeline in self.pipelines.values() {
                self.device.destroy_pipeline(pipeline, None);
            }
            self.device.destroy_shader_module(self.vert_module, None);
            self.device.destroy_shader_module(self.frag_module, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            if self.render_pass != vk::RenderPass::null() {
                self.device.destroy_render_pass(self.render_pass, None);
            }
            if self.depth_view != vk::ImageView::null() {
                self.device.destroy_image_view(self.depth_view, None);
                self.device.destroy_image(self.depth_image, None);
                self.device.free_memory(self.depth_memory, None);
            }
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_memory, None);
            self.device.destroy_buffer(self.readback_buffer, None);
            self.device.free_memory(self.readback_memory, None);
            self.device.destroy_buffer(self.lfb_staging, None);
            self.device.free_memory(self.lfb_staging_memory, None);
            self.device.destroy_buffer(self.capture_buffer, None);
            self.device.free_memory(self.capture_memory, None);
            // Textures (descriptor sets are freed with the pool below).
            for tex in self.textures.values() {
                self.destroy_texture(tex);
            }
            for tex in &self.retired_textures {
                self.destroy_texture(tex);
            }
            self.destroy_texture(&self.default_texture);
            self.device.destroy_descriptor_pool(self.desc_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.desc_set_layout, None);
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_sampler(self.sampler_clamp, None);
            self.device.destroy_semaphore(self.image_available, None);
            self.device.destroy_fence(self.in_flight, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn create_instance(
    entry: &ash::Entry,
    extensions: &[*const std::os::raw::c_char],
) -> Result<ash::Instance, Error> {
    let app_name = c"GlideVK";
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .engine_name(app_name)
        .api_version(vk::make_api_version(0, 1, 1, 0));
    let info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(extensions);
    // SAFETY: `entry` is a valid loader; pointers in `extensions` are static.
    Ok(unsafe { entry.create_instance(&info, None)? })
}

/// Find a graphics-capable queue family; if a surface is given, require present
/// support on the same family (sufficient for the common single-queue case).
fn pick_queue_family(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    surface: Option<(&ash::khr::surface::Instance, vk::SurfaceKHR)>,
) -> Option<u32> {
    // SAFETY: `pd` comes from this `instance`.
    let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
    families.iter().enumerate().find_map(|(i, f)| {
        let i = i as u32;
        if !f.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            return None;
        }
        if let Some((loader, surf)) = surface {
            // SAFETY: valid pd/family/surface handles.
            let present =
                unsafe { loader.get_physical_device_surface_support(pd, i, surf) }.unwrap_or(false);
            if !present {
                return None;
            }
        }
        Some(i)
    })
}

fn create_swapchain(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    desired: vk::Extent2D,
) -> Result<Swapchain, Error> {
    // SAFETY: handles are valid and owned by the caller for this scope.
    let caps =
        unsafe { surface_loader.get_physical_device_surface_capabilities(physical, surface)? };
    if !caps
        .supported_usage_flags
        .contains(vk::ImageUsageFlags::TRANSFER_DST)
    {
        return Err(Error::Loading(
            "swapchain surface cannot receive framebuffer copies".into(),
        ));
    }
    let formats = unsafe { surface_loader.get_physical_device_surface_formats(physical, surface)? };
    // An invalid/being-destroyed surface (e.g. during a focus-loss teardown) can
    // report no formats; bail cleanly instead of indexing `formats[0]`.
    let format = match formats
        .iter()
        .copied()
        .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
        .or_else(|| formats.first().copied())
    {
        Some(f) => f,
        None => return Err(Error::Vk(vk::Result::ERROR_SURFACE_LOST_KHR)),
    };

    let mut extent = if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: desired
                .width
                .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: desired
                .height
                .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    };
    // A minimized window reports 0x0; a swapchain must be at least 1x1.
    extent.width = extent.width.max(1);
    extent.height = extent.height.max(1);
    let mut image_count = caps.min_image_count + 1;
    if caps.max_image_count > 0 {
        image_count = image_count.min(caps.max_image_count);
    }

    let info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(vk::PresentModeKHR::FIFO)
        .clipped(true);

    let swapchain_loader = ash::khr::swapchain::Device::new(instance, device);
    // SAFETY: `info` references live handles; loader belongs to instance/device.
    let swapchain = unsafe { swapchain_loader.create_swapchain(&info, None)? };
    let images = match unsafe { swapchain_loader.get_swapchain_images(swapchain) } {
        Ok(images) => images,
        Err(error) => {
            unsafe { swapchain_loader.destroy_swapchain(swapchain, None) };
            return Err(error.into());
        }
    };
    let mut render_finished = Vec::with_capacity(images.len());
    for _ in &images {
        match unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) } {
            Ok(semaphore) => render_finished.push(semaphore),
            Err(error) => {
                unsafe {
                    for semaphore in render_finished {
                        device.destroy_semaphore(semaphore, None);
                    }
                    swapchain_loader.destroy_swapchain(swapchain, None);
                }
                return Err(error.into());
            }
        }
    }

    Ok(Swapchain {
        surface_loader,
        surface,
        swapchain_loader,
        swapchain,
        format: format.format,
        extent,
        images,
        render_finished,
        color_image: vk::Image::null(),
        color_memory: vk::DeviceMemory::null(),
        color_view: vk::ImageView::null(),
        framebuffer: vk::Framebuffer::null(),
        targets_initialized: false,
    })
}

fn create_render_pass(device: &ash::Device, format: vk::Format) -> Result<vk::RenderPass, Error> {
    let color = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        // Leave the image ready for the center-pixel copy after the pass.
        .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
    let depth = vk::AttachmentDescription::default()
        .format(DEPTH_FORMAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let depth_ref = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let color_refs = [color_ref];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs)
        .depth_stencil_attachment(&depth_ref);
    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::TRANSFER
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        )
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        )
        .src_access_mask(
            vk::AccessFlags::TRANSFER_READ
                | vk::AccessFlags::TRANSFER_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );
    let attachments = [color, depth];
    let subpasses = [subpass];
    let to_transfer = vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        )
        .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
        .src_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
    let dependencies = [dependency, to_transfer];
    let info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    // SAFETY: all referenced arrays outlive the call.
    Ok(unsafe { device.create_render_pass(&info, None)? })
}

fn create_framebuffers(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    render_pass: vk::RenderPass,
    depth_view: vk::ImageView,
    sc: &mut Swapchain,
) -> Result<(), Error> {
    unsafe {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(sc.format)
            .extent(vk::Extent3D {
                width: sc.extent.width,
                height: sc.extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = device.create_image(&info, None)?;
        let mut memory = vk::DeviceMemory::null();
        let mut view = vk::ImageView::null();
        let result: Result<vk::Framebuffer, Error> = (|| {
            let req = device.get_image_memory_requirements(image);
            let memory_type = find_memory_type(
                instance,
                physical,
                req.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(memory_type),
                None,
            )?;
            device.bind_image_memory(image, memory, 0)?;
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(sc.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            view = device.create_image_view(&view_info, None)?;
            let attachments = [view, depth_view];
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(sc.extent.width)
                .height(sc.extent.height)
                .layers(1);
            Ok(device.create_framebuffer(&fb_info, None)?)
        })();
        let fb = match result {
            Ok(fb) => fb,
            Err(error) => {
                if view != vk::ImageView::null() {
                    device.destroy_image_view(view, None);
                }
                device.destroy_image(image, None);
                if memory != vk::DeviceMemory::null() {
                    device.free_memory(memory, None);
                }
                return Err(error);
            }
        };
        sc.color_image = image;
        sc.color_memory = memory;
        sc.color_view = view;
        sc.framebuffer = fb;
    }
    Ok(())
}

/// Create the shared depth buffer (image + memory + view) for `extent`.
unsafe fn create_depth_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    extent: vk::Extent2D,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), Error> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(DEPTH_FORMAT)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = device.create_image(&info, None)?;
    let req = device.get_image_memory_requirements(image);
    let mem_type = find_memory_type(
        instance,
        physical,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_type);
    let memory = device.allocate_memory(&alloc, None)?;
    device.bind_image_memory(image, memory, 0)?;
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(DEPTH_FORMAT)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let view = device.create_image_view(&view_info, None)?;
    Ok((image, memory, view))
}

fn map_blend_factor(code: i32) -> vk::BlendFactor {
    match code {
        0 => vk::BlendFactor::ZERO,
        1 => vk::BlendFactor::ONE,
        2 => vk::BlendFactor::SRC_COLOR,
        3 => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        4 => vk::BlendFactor::SRC_ALPHA,
        5 => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        6 => vk::BlendFactor::DST_COLOR,
        7 => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        8 => vk::BlendFactor::DST_ALPHA,
        9 => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        10 => vk::BlendFactor::SRC_ALPHA_SATURATE,
        _ => vk::BlendFactor::ONE,
    }
}

/// GR_CMP_* -> VkCompareOp.
fn map_compare_op(code: i32) -> vk::CompareOp {
    match code {
        0 => vk::CompareOp::NEVER,
        1 => vk::CompareOp::LESS,
        2 => vk::CompareOp::EQUAL,
        3 => vk::CompareOp::LESS_OR_EQUAL,
        4 => vk::CompareOp::GREATER,
        5 => vk::CompareOp::NOT_EQUAL,
        6 => vk::CompareOp::GREATER_OR_EQUAL,
        _ => vk::CompareOp::ALWAYS,
    }
}

unsafe fn create_pipeline_layout(
    device: &ash::Device,
    desc_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, Error> {
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(std::mem::size_of::<PushConstants>() as u32);
    let push_ranges = [push_range];
    let set_layouts = [desc_set_layout];
    let info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);
    Ok(device.create_pipeline_layout(&info, None)?)
}

/// Build one pipeline for a given baked state ([`PipeKey`]), reusing the shared
/// shader modules + layout. Called on cache miss.
fn build_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert: vk::ShaderModule,
    frag: vk::ShaderModule,
    key: PipeKey,
) -> Result<vk::Pipeline, Error> {
    let entry = c"main";
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert)
            .name(entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag)
            .name(entry),
    ];

    let binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(VERTEX_STRIDE as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let attrs = [
        vk::VertexInputAttributeDescription::default()
            .location(0)
            .binding(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .location(1)
            .binding(0)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(8),
        vk::VertexInputAttributeDescription::default()
            .location(2)
            .binding(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(24),
        vk::VertexInputAttributeDescription::default()
            .location(3)
            .binding(0)
            .format(vk::Format::R32_SFLOAT)
            .offset(32),
        vk::VertexInputAttributeDescription::default()
            .location(4)
            .binding(0)
            .format(vk::Format::R32_SFLOAT)
            .offset(36),
    ];
    let bindings = [binding];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attrs);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    // GLIDEVK_WIREFRAME renders geometry as outlines — a debug aid for seeing
    // what is drawn regardless of its (possibly invisible) shading.
    let polygon_mode = if key.wireframe {
        vk::PolygonMode::LINE
    } else {
        vk::PolygonMode::FILL
    };
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(polygon_mode)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(key.depth.test)
        .depth_write_enable(key.depth.write)
        .depth_compare_op(map_compare_op(key.depth.compare));

    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::from_raw(key.blend.write_mask))
        .blend_enable(key.blend.enable)
        .src_color_blend_factor(map_blend_factor(key.blend.src_rgb))
        .dst_color_blend_factor(map_blend_factor(key.blend.dst_rgb))
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(map_blend_factor(key.blend.src_alpha))
        .dst_alpha_blend_factor(map_blend_factor(key.blend.dst_alpha))
        .alpha_blend_op(vk::BlendOp::ADD);
    let blend_attachments = [blend_attachment];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);
    // SAFETY: all referenced state outlives the call; modules/layout are owned
    // by the backend and kept alive for the cache's lifetime.
    let pipeline = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_, e)| Error::Vk(e))?[0]
    };
    Ok(pipeline)
}

unsafe fn load_shader(device: &ash::Device, bytes: &[u8]) -> Result<vk::ShaderModule, Error> {
    let code = ash::util::read_spv(&mut Cursor::new(bytes))
        .map_err(|e| Error::Loading(format!("SPIR-V: {e}")))?;
    let info = vk::ShaderModuleCreateInfo::default().code(&code);
    Ok(device.create_shader_module(&info, None)?)
}

/// How the CPU uses a mapped buffer, which decides its memory type.
#[derive(Clone, Copy, PartialEq, Eq)]
enum HostAccess {
    /// CPU writes, GPU reads (vertices, uploads): any coherent host memory,
    /// which on desktop drivers is write-combined — fast to write, very slow
    /// to read back.
    Upload,
    /// GPU writes, CPU reads (readbacks): prefer HOST_CACHED so CPU reads run
    /// at memory speed instead of uncached PCIe/write-combined speed.
    Readback,
}

/// Time individual Vulkan calls only in explicitly enabled profiling mode.
/// Keep file writes bounded so diagnostics do not become the bottleneck.
fn profile_vk_call<T>(name: &'static str, call: impl FnOnce() -> T) -> T {
    use std::sync::{
        atomic::{AtomicU32, Ordering},
        OnceLock,
    };
    use std::time::{Duration, Instant};
    static LOG_PATH: OnceLock<Option<String>> = OnceLock::new();
    static LINES: AtomicU32 = AtomicU32::new(0);
    static DRAW_LINES: AtomicU32 = AtomicU32::new(0);
    let path = LOG_PATH.get_or_init(|| {
        std::env::var_os("GLIDEVK_PROFILE_ONLY")?;
        std::env::var("GLIDEVK_LOG").ok().map(|value| {
            if value.is_empty() || value == "1" {
                "glidevk.log".to_owned()
            } else {
                value
            }
        })
    });
    let Some(path) = path else { return call() };
    let start = Instant::now();
    let result = call();
    let elapsed = start.elapsed();
    let counter = if name.starts_with("draw_") {
        &DRAW_LINES
    } else {
        &LINES
    };
    if elapsed >= Duration::from_millis(50) && counter.fetch_add(1, Ordering::Relaxed) < 40 {
        use std::io::Write;
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(
                file,
                "vk slow {name}: {:.1}ms",
                elapsed.as_secs_f64() * 1000.0
            );
        }
    }
    result
}

/// Allocate a host-visible, coherent buffer and leave it persistently mapped.
unsafe fn create_mapped_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    size: u64,
    usage: vk::BufferUsageFlags,
    access: HostAccess,
) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), Error> {
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = profile_vk_call("create_buffer", || device.create_buffer(&info, None))?;
    let req = profile_vk_call("get_buffer_memory_requirements", || {
        device.get_buffer_memory_requirements(buffer)
    });
    let props = profile_vk_call("get_physical_device_memory_properties", || {
        instance.get_physical_device_memory_properties(physical)
    });
    let base = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let find = |want: vk::MemoryPropertyFlags| {
        (0..props.memory_type_count).find(|&i| {
            (req.memory_type_bits & (1 << i)) != 0
                && props.memory_types[i as usize].property_flags.contains(want)
        })
    };
    let preferred = match access {
        HostAccess::Readback => find(base | vk::MemoryPropertyFlags::HOST_CACHED),
        HostAccess::Upload => None,
    };
    let mem_type = preferred
        .or_else(|| find(base))
        .ok_or(Error::NoMemoryType)?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_type);
    let memory = profile_vk_call("allocate_buffer_memory", || {
        device.allocate_memory(&alloc, None)
    })?;
    profile_vk_call("bind_buffer_memory", || {
        device.bind_buffer_memory(buffer, memory, 0)
    })?;
    let ptr = profile_vk_call("map_buffer_memory", || {
        device.map_memory(memory, 0, req.size, vk::MemoryMapFlags::empty())
    })? as *mut u8;
    Ok((buffer, memory, ptr))
}

/// Allocate an unmapped intermediate buffer for vkCmdUpdateBuffer uploads.
unsafe fn create_upload_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    size: u64,
) -> Result<(vk::Buffer, vk::DeviceMemory), Error> {
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = device.create_buffer(&info, None)?;
    let result = (|| {
        let req = device.get_buffer_memory_requirements(buffer);
        let mem_type = find_memory_type(
            instance,
            physical,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .or_else(|_| {
            find_memory_type(
                instance,
                physical,
                req.memory_type_bits,
                vk::MemoryPropertyFlags::empty(),
            )
        })?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_type);
        let memory = device.allocate_memory(&alloc, None)?;
        if let Err(error) = device.bind_buffer_memory(buffer, memory, 0) {
            device.free_memory(memory, None);
            return Err(error.into());
        }
        Ok(memory)
    })();
    match result {
        Ok(memory) => Ok((buffer, memory)),
        Err(error) => {
            device.destroy_buffer(buffer, None);
            Err(error)
        }
    }
}

/// Vulkan limits a single vkCmdUpdateBuffer to 65536 bytes and requires its
/// size to be a multiple of four. Never use it for an incomplete source.
fn use_command_update(size: u64, source_len: usize) -> bool {
    size > 0 && size <= 65_536 && size.is_multiple_of(4) && source_len as u64 >= size
}

/// A TRANSFER-stage image layout transition (used between copy operations).
unsafe fn image_layout_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range);
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[barrier],
    );
}

/// `GR_LFBWRITEMODE_888`(4) / `_8888`(5) are 32-bit; the rest are 16-bit.
fn lfb_mode_is_32bit(mode: i32) -> bool {
    matches!(mode, 4 | 5)
}

/// Decode one packed LFB pixel at `off` in `buf` to RGBA8, per `GR_LFBWRITEMODE_*`.
fn decode_lfb_pixel(buf: &[u8], off: usize, mode: i32) -> (u8, u8, u8, u8) {
    let ex5 = |c: u16| ((c << 3) | (c >> 2)) as u8;
    let ex6 = |c: u16| ((c << 2) | (c >> 4)) as u8;
    match mode {
        5 => {
            // ARGB8888 (0xAARRGGBB), little-endian bytes [B,G,R,A].
            if off + 4 > buf.len() {
                return (0, 0, 0, 0);
            }
            (buf[off + 2], buf[off + 1], buf[off], buf[off + 3])
        }
        4 => {
            // RGB888 in a 32-bit slot.
            if off + 4 > buf.len() {
                return (0, 0, 0, 255);
            }
            (buf[off + 2], buf[off + 1], buf[off], 255)
        }
        1 | 2 => {
            // RGB555 (1) / ARGB1555 (2).
            if off + 2 > buf.len() {
                return (0, 0, 0, 0);
            }
            // Present ignores framebuffer alpha, so treat as opaque.
            let p = u16::from_le_bytes([buf[off], buf[off + 1]]);
            (
                ex5((p >> 10) & 0x1f),
                ex5((p >> 5) & 0x1f),
                ex5(p & 0x1f),
                255,
            )
        }
        _ => {
            // RGB565 (default).
            if off + 2 > buf.len() {
                return (0, 0, 0, 255);
            }
            let p = u16::from_le_bytes([buf[off], buf[off + 1]]);
            (
                ex5((p >> 11) & 0x1f),
                ex6((p >> 5) & 0x3f),
                ex5(p & 0x1f),
                255,
            )
        }
    }
}

unsafe fn find_memory_type(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    type_bits: u32,
    props: vk::MemoryPropertyFlags,
) -> Result<u32, Error> {
    let mp = instance.get_physical_device_memory_properties(physical);
    (0..mp.memory_type_count)
        .find(|&i| {
            (type_bits & (1 << i)) != 0
                && mp.memory_types[i as usize].property_flags.contains(props)
        })
        .ok_or(Error::NoMemoryType)
}

unsafe fn create_sampler(
    device: &ash::Device,
    address_mode: vk::SamplerAddressMode,
) -> Result<vk::Sampler, Error> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(address_mode)
        .address_mode_v(address_mode)
        .address_mode_w(address_mode)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR);
    Ok(device.create_sampler(&info, None)?)
}

unsafe fn create_desc_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout, Error> {
    let binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);
    let bindings = [binding];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    Ok(device.create_descriptor_set_layout(&info, None)?)
}

/// Descriptor pool for texture image-samplers. Sized generously for a level's
/// worth of textures; a real game rarely exceeds a few hundred.
const MAX_TEXTURES: u32 = 4096;

unsafe fn create_desc_pool(device: &ash::Device) -> Result<vk::DescriptorPool, Error> {
    // Two descriptor sets per texture (REPEAT + CLAMP sampler variants).
    let size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(MAX_TEXTURES * 2);
    let sizes = [size];
    let info = vk::DescriptorPoolCreateInfo::default()
        // Sets are freed with their texture; without this the pool runs dry
        // after a few thousand uploads and every later upload fails.
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
        .max_sets(MAX_TEXTURES * 2)
        .pool_sizes(&sizes);
    Ok(device.create_descriptor_pool(&info, None)?)
}

/// Record a copy of `rgba` into `image`, leaving it in
/// SHADER_READ_ONLY_OPTIMAL. `old_layout` is UNDEFINED for a fresh image or
/// SHADER_READ_ONLY_OPTIMAL when overwriting one already in use.
#[allow(clippy::too_many_arguments)]
unsafe fn record_image_upload(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    image: vk::Image,
    width: u32,
    height: u32,
    rgba: &[u8],
    old_layout: vk::ImageLayout,
    addr: Option<u64>,
) -> Result<PendingUpload, Error> {
    let extent = vk::Extent3D {
        width,
        height,
        depth: 1,
    };
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let staging_size = (width as u64) * (height as u64) * 4;
    let inline = use_command_update(staging_size, rgba.len()) && rgba.as_ptr().align_offset(4) == 0;
    let (staging, staging_mem) = if inline {
        profile_vk_call("staging_buffer_total", || {
            create_upload_buffer(instance, device, physical, staging_size)
        })?
    } else {
        let (staging, memory, ptr) = profile_vk_call("staging_buffer_total", || {
            create_mapped_buffer(
                instance,
                device,
                physical,
                staging_size.max(4),
                vk::BufferUsageFlags::TRANSFER_SRC,
                HostAccess::Upload,
            )
        })?;
        let n = rgba.len().min(staging_size as usize);
        profile_vk_call("staging_copy", || {
            std::ptr::copy_nonoverlapping(rgba.as_ptr(), ptr, n)
        });
        (staging, memory)
    };

    let allocated = profile_vk_call("allocate_upload_command_buffer", || {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    });
    let cmd = match allocated {
        Ok(commands) => commands[0],
        Err(error) => {
            device.destroy_buffer(staging, None);
            device.free_memory(staging_mem, None);
            return Err(error.into());
        }
    };
    if let Err(error) = profile_vk_call("begin_upload_command_buffer", || {
        device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
    }) {
        device.free_command_buffers(command_pool, &[cmd]);
        device.destroy_buffer(staging, None);
        device.free_memory(staging_mem, None);
        return Err(error.into());
    }
    if inline {
        profile_vk_call("upload_command_update", || {
            device.cmd_update_buffer(cmd, staging, 0, &rgba[..staging_size as usize]);
        });
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(staging)
            .offset(0)
            .size(staging_size);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
    }
    let src_access = if old_layout == vk::ImageLayout::UNDEFINED {
        vk::AccessFlags::empty()
    } else {
        vk::AccessFlags::SHADER_READ
    };
    let src_stage = if old_layout == vk::ImageLayout::UNDEFINED {
        vk::PipelineStageFlags::TOP_OF_PIPE
    } else {
        vk::PipelineStageFlags::FRAGMENT_SHADER
    };
    let to_dst = vk::ImageMemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(old_layout)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range);
    profile_vk_call("upload_barrier_to_dst", || {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_dst],
        )
    });
    let region = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: extent,
    };
    profile_vk_call("upload_copy_buffer_to_image", || {
        device.cmd_copy_buffer_to_image(
            cmd,
            staging,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        )
    });
    let to_read = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range);
    profile_vk_call("upload_barrier_to_read", || {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_read],
        )
    });
    if let Err(error) = profile_vk_call("end_upload_command_buffer", || {
        device.end_command_buffer(cmd)
    }) {
        device.free_command_buffers(command_pool, &[cmd]);
        device.destroy_buffer(staging, None);
        device.free_memory(staging_mem, None);
        return Err(error.into());
    }
    Ok(PendingUpload {
        addr,
        command_buffer: cmd,
        staging_buffer: staging,
        staging_memory: staging_mem,
    })
}

/// Synchronous upload for headless backends, which do not have a later render
/// submission to batch with the texture transfer.
#[allow(clippy::too_many_arguments)]
unsafe fn upload_image_pixels_sync(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    image: vk::Image,
    width: u32,
    height: u32,
    rgba: &[u8],
    old_layout: vk::ImageLayout,
) -> Result<(), Error> {
    let upload = record_image_upload(
        instance,
        device,
        physical,
        command_pool,
        image,
        width,
        height,
        rgba,
        old_layout,
        None,
    )?;
    let commands = [upload.command_buffer];
    let submit = vk::SubmitInfo::default().command_buffers(&commands);
    if let Err(error) = device.queue_submit(queue, &[submit], vk::Fence::null()) {
        let mut uploads = vec![upload];
        destroy_pending_uploads(device, command_pool, &mut uploads);
        return Err(error.into());
    }
    device.queue_wait_idle(queue)?;
    let mut uploads = vec![upload];
    destroy_pending_uploads(device, command_pool, &mut uploads);
    Ok(())
}

fn destroy_pending_uploads(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    uploads: &mut Vec<PendingUpload>,
) {
    unsafe {
        for upload in uploads.drain(..) {
            device.free_command_buffers(command_pool, &[upload.command_buffer]);
            device.destroy_buffer(upload.staging_buffer, None);
            device.free_memory(upload.staging_memory, None);
        }
    }
}

fn destroy_submitted_uploads(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    uploads: &mut Vec<SubmittedUploadBatch>,
) {
    for mut batch in uploads.drain(..) {
        destroy_pending_uploads(device, command_pool, &mut batch.uploads);
        unsafe { device.destroy_semaphore(batch.ready, None) };
    }
}

/// Upload an RGBA8 texture to a device-local image and build its descriptor set.
#[allow(clippy::too_many_arguments)]
unsafe fn create_gpu_texture(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    sampler: vk::Sampler,
    sampler_clamp: vk::Sampler,
    desc_pool: vk::DescriptorPool,
    desc_set_layout: vk::DescriptorSetLayout,
    width: u32,
    height: u32,
    rgba: &[u8],
    used_frame: u32,
    defer_upload: bool,
    addr: Option<u64>,
) -> Result<(GpuTexture, Option<PendingUpload>), Error> {
    let format = vk::Format::R8G8B8A8_UNORM;
    let extent = vk::Extent3D {
        width,
        height,
        depth: 1,
    };
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(extent)
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = profile_vk_call("create_image", || device.create_image(&image_info, None))?;
    let req = profile_vk_call("get_image_memory_requirements", || {
        device.get_image_memory_requirements(image)
    });
    let mem_type = profile_vk_call("find_image_memory_type", || {
        find_memory_type(
            instance,
            physical,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
    })?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_type);
    let memory = profile_vk_call("allocate_image_memory", || {
        device.allocate_memory(&alloc, None)
    })?;
    profile_vk_call("bind_image_memory", || {
        device.bind_image_memory(image, memory, 0)
    })?;

    let pending_upload = if defer_upload {
        Some(profile_vk_call("record_image_upload_total", || {
            record_image_upload(
                instance,
                device,
                physical,
                command_pool,
                image,
                width,
                height,
                rgba,
                vk::ImageLayout::UNDEFINED,
                addr,
            )
        })?)
    } else {
        upload_image_pixels_sync(
            instance,
            device,
            physical,
            command_pool,
            queue,
            image,
            width,
            height,
            rgba,
            vk::ImageLayout::UNDEFINED,
        )?;
        None
    };

    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };

    // View + descriptor set.
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(range);
    let view = profile_vk_call("create_image_view", || {
        device.create_image_view(&view_info, None)
    })?;

    // One descriptor per sampler (REPEAT + CLAMP), so a draw can pick the
    // address mode the game requested via grTexClampMode.
    let layouts = [desc_set_layout, desc_set_layout];
    let sets = profile_vk_call("allocate_descriptor_sets", || {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(desc_pool)
                .set_layouts(&layouts),
        )
    })?;
    let (descriptor, descriptor_clamp) = (sets[0], sets[1]);
    for (set, samp) in [(descriptor, sampler), (descriptor_clamp, sampler_clamp)] {
        let image_info = vk::DescriptorImageInfo::default()
            .sampler(samp)
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let image_infos = [image_info];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos);
        profile_vk_call("update_descriptor_sets", || {
            device.update_descriptor_sets(&[write], &[])
        });
    }

    Ok((
        GpuTexture {
            image,
            memory,
            view,
            descriptor,
            descriptor_clamp,
            width,
            height,
            used_frame,
        },
        pending_upload,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_respects_color_and_depth_write_masks() {
        assert!(clear_aspects(false, false, false).is_empty());
        assert!(clear_aspects(false, false, true) == vk::ImageAspectFlags::DEPTH);
        assert!(clear_aspects(true, false, false) == vk::ImageAspectFlags::COLOR);
        assert!(
            clear_aspects(false, true, true)
                == vk::ImageAspectFlags::COLOR | vk::ImageAspectFlags::DEPTH
        );
        assert_eq!(clear_color_write_mask(true, false), 7);
        assert_eq!(clear_color_write_mask(false, true), 8);
        assert_eq!(clear_color_write_mask(true, true), 15);
    }

    #[test]
    fn clip_scissor_scales_edges_and_obeys_origin() {
        let extent = vk::Extent2D {
            width: 1280,
            height: 960,
        };
        let full = clip_scissor(None, (640, 480), extent, false);
        assert_eq!(
            (
                full.offset.x,
                full.offset.y,
                full.extent.width,
                full.extent.height
            ),
            (0, 0, 1280, 960)
        );
        let top_left = clip_scissor(Some([10, 20, 110, 120]), (640, 480), extent, false);
        assert_eq!(
            (
                top_left.offset.x,
                top_left.offset.y,
                top_left.extent.width,
                top_left.extent.height
            ),
            (20, 40, 200, 200)
        );
        let bottom_left = clip_scissor(Some([10, 20, 110, 120]), (640, 480), extent, true);
        assert_eq!(
            (
                bottom_left.offset.x,
                bottom_left.offset.y,
                bottom_left.extent.width,
                bottom_left.extent.height
            ),
            (20, 720, 200, 200)
        );
        let empty = clip_scissor(Some([2, 4, 2, 4]), (640, 480), extent, false);
        assert_eq!((empty.extent.width, empty.extent.height), (0, 0));
        let scaled = clip_scissor(
            Some([1, 1, 3, 3]),
            (4, 4),
            vk::Extent2D {
                width: 6,
                height: 6,
            },
            false,
        );
        assert_eq!(
            (
                scaled.offset.x,
                scaled.offset.y,
                scaled.extent.width,
                scaled.extent.height
            ),
            (1, 1, 3, 3)
        );
    }

    #[test]
    fn vertex_upload_chunks_cover_frame_in_aligned_vulkan_updates() {
        let vertices = vec![DrawVertex::default(); 2_000];
        let chunks: Vec<_> = vertex_upload_chunks(&vertices).collect();
        assert_eq!(chunks.len(), 2);
        assert_eq!((chunks[0].0, chunks[0].1.len()), (0, 65_536));
        assert_eq!((chunks[1].0, chunks[1].1.len()), (65_536, 14_464));
        assert!(chunks
            .iter()
            .all(|(offset, bytes)| offset % 4 == 0 && bytes.len() % 4 == 0));
        assert_eq!(vertex_upload_chunks(&[]).count(), 0);
    }

    #[test]
    fn command_update_obeys_vulkan_size_limits_and_requires_full_input() {
        assert!(use_command_update(4, 4));
        assert!(use_command_update(65_536, 65_536));
        assert!(!use_command_update(0, 0));
        assert!(!use_command_update(3, 3));
        assert!(!use_command_update(65_540, 65_540));
        assert!(!use_command_update(4096, 4095));
    }

    #[test]
    fn persistent_texture_binding_counts_as_used_in_new_frame() {
        let texture = GpuTexture {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            descriptor: vk::DescriptorSet::null(),
            descriptor_clamp: vk::DescriptorSet::null(),
            width: 16,
            height: 16,
            used_frame: 4,
        };
        let mut textures = HashMap::from([(42, texture)]);
        // The game draws with a binding carried over from the previous frame,
        // then re-downloads it: in-place overwrite must no longer be allowed.
        mark_texture_drawn(&mut textures, Some(42), 5);
        assert_eq!(textures[&42].used_frame, 5);
        mark_texture_drawn(&mut textures, None, 6);
        assert_eq!(textures[&42].used_frame, 5);
    }

    #[test]
    fn alpha_blending_is_not_disabled_by_opaque_rgb_factors() {
        use crate::types::BlendFactor;
        let mut state = crate::state::State::default();
        assert!(!BlendConfig::from_state(&state).enable);
        state.blend_dst_alpha = BlendFactor::One;
        let blend = BlendConfig::from_state(&state);
        assert!(blend.enable);
        assert_eq!(blend.src_rgb, BlendFactor::One as i32);
        assert_eq!(blend.dst_rgb, BlendFactor::Zero as i32);
        assert_eq!(blend.dst_alpha, BlendFactor::One as i32);
    }

    #[test]
    fn capture_is_off_by_default_and_decodes_the_frame_argument() {
        // The default must never capture: it used to be -1 = "every frame",
        // which copied and converted the whole image on every present.
        let off = CaptureMode::default();
        assert!((0..1000).all(|f| !off.wants(f)));
        assert_eq!(CaptureMode::from_frame_arg(0), CaptureMode::Off);
        let one = CaptureMode::from_frame_arg(400);
        assert!(one.wants(400) && !one.wants(399) && !one.wants(800));
        let every = CaptureMode::from_frame_arg(-100);
        assert!(every.wants(100) && every.wants(200) && !every.wants(150) && !every.wants(0));
    }

    /// Exercises the non-surface Vulkan path against the host ICD. Skips (rather
    /// than fails) when no loader/device is present, so CI without a GPU stays
    /// green.
    #[test]
    fn headless_brings_up_a_device() {
        match Backend::new_headless() {
            Ok(mut b) => {
                assert!(!b.device_name().is_empty());
                // Exercise command-update staging and the larger mapped fallback.
                b.upload_texture(1, 2, 2, &[255; 16]).unwrap();
                b.upload_texture(2, 256, 256, &vec![255; 256 * 256 * 4])
                    .unwrap();
                eprintln!("GlideVK headless device: {}", b.device_name());
            }
            Err(e) => eprintln!("skipping: no usable Vulkan on host ({e:?})"),
        }
    }
}
