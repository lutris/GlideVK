//! Thin Win32 bindings the shells need outside Vulkan.
//!
//! Glide is a fullscreen API: on 3dfx hardware `grSstWinOpen` switched the
//! display to the requested mode and games size their window to the screen
//! afterwards. Under Wine a display-mode change is emulated by scaling the
//! fullscreen window and its mouse input, which is what makes "fullscreen" and
//! resolution changes work for Glide games there — the same path the D3D-based
//! wrappers get from Direct3D fullscreen.

/// `DEVMODEW` from `wingdi.h`, exact layout (220 bytes).
#[repr(C)]
pub struct DevModeW {
    pub device_name: [u16; 32],
    pub spec_version: u16,
    pub driver_version: u16,
    pub size: u16,
    pub driver_extra: u16,
    pub fields: u32,
    pub position_x: i32,
    pub position_y: i32,
    pub display_orientation: u32,
    pub display_fixed_output: u32,
    pub color: i16,
    pub duplex: i16,
    pub y_resolution: i16,
    pub tt_option: i16,
    pub collate: i16,
    pub form_name: [u16; 32],
    pub log_pixels: u16,
    pub bits_per_pel: u32,
    pub pels_width: u32,
    pub pels_height: u32,
    pub display_flags: u32,
    pub display_frequency: u32,
    pub icm_method: u32,
    pub icm_intent: u32,
    pub media_type: u32,
    pub dither_type: u32,
    pub reserved1: u32,
    pub reserved2: u32,
    pub panning_width: u32,
    pub panning_height: u32,
}

/// `sizeof(DEVMODEW)`.
pub const DEVMODEW_SIZE: usize = 220;
const DM_BITSPERPEL: u32 = 0x0004_0000;
const DM_PELSWIDTH: u32 = 0x0008_0000;
const DM_PELSHEIGHT: u32 = 0x0010_0000;
#[cfg(windows)]
const CDS_FULLSCREEN: u32 = 0x4;
#[cfg(not(windows))]
#[allow(dead_code)]
const _UNUSED_ON_HOST: (i32, u32, u32, isize, u32, u32) = (
    GWL_STYLE,
    WS_POPUP,
    WS_VISIBLE,
    HWND_TOP,
    SWP_FRAMECHANGED,
    SWP_SHOWWINDOW,
);
#[cfg(windows)]
const DISP_CHANGE_SUCCESSFUL: i32 = 0;
/// Colour depth requested with the mode; Wine's emulated modes are 32-bit.
const MODE_BITS_PER_PEL: u32 = 32;

impl DevModeW {
    /// A mode request for `width`x`height` at 32 bpp.
    pub fn mode(width: u32, height: u32) -> Self {
        // SAFETY: an all-zero DEVMODEW is a valid (empty) value.
        let mut dm: DevModeW = unsafe { core::mem::zeroed() };
        dm.size = DEVMODEW_SIZE as u16;
        dm.fields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;
        dm.bits_per_pel = MODE_BITS_PER_PEL;
        dm.pels_width = width;
        dm.pels_height = height;
        dm
    }
}

#[cfg(windows)]
extern "system" {
    fn GetActiveWindow() -> isize;
    fn IsWindow(hwnd: isize) -> i32;
    fn GetWindowLongW(hwnd: isize, index: i32) -> i32;
    fn GetWindowRect(hwnd: isize, rect: *mut Rect) -> i32;
    fn GetClientRect(hwnd: isize, rect: *mut Rect) -> i32;
    fn GetMenu(hwnd: isize) -> isize;
    fn AdjustWindowRectEx(rect: *mut Rect, style: u32, menu: i32, ex_style: u32) -> i32;
    fn MonitorFromWindow(hwnd: isize, flags: u32) -> isize;
    fn GetMonitorInfoW(monitor: isize, info: *mut MonitorInfo) -> i32;
    fn SetLastError(error: u32);
    fn GetLastError() -> u32;
    fn CreateWindowExW(
        ex_style: u32,
        class_name: *const u16,
        window_name: *const u16,
        style: u32,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        parent: isize,
        menu: isize,
        instance: isize,
        param: *mut core::ffi::c_void,
    ) -> isize;
    fn ShowWindow(hwnd: isize, command: i32) -> i32;
    fn ChangeDisplaySettingsExW(
        device_name: *const u16,
        dev_mode: *mut DevModeW,
        hwnd: isize,
        flags: u32,
        param: *mut core::ffi::c_void,
    ) -> i32;
    #[cfg(target_pointer_width = "32")]
    fn SetWindowLongW(hwnd: isize, index: i32, value: i32) -> i32;
    #[cfg(target_pointer_width = "64")]
    fn SetWindowLongPtrW(hwnd: isize, index: i32, value: isize) -> isize;
    fn SetWindowPos(hwnd: isize, after: isize, x: i32, y: i32, cx: i32, cy: i32, flags: u32)
        -> i32;
}

const GWL_STYLE: i32 = -16;
#[cfg(windows)]
const GWL_EXSTYLE: i32 = -20;
const WS_POPUP: u32 = 0x8000_0000;
const WS_VISIBLE: u32 = 0x1000_0000;
#[cfg(any(windows, test))]
const WS_CAPTION: u32 = 0x00c0_0000;
#[cfg(any(windows, test))]
const WS_SYSMENU: u32 = 0x0008_0000;
#[cfg(any(windows, test))]
const WS_MINIMIZEBOX: u32 = 0x0002_0000;
#[cfg(windows)]
const WS_CHILD: u32 = 0x4000_0000;
#[cfg(windows)]
const WS_OVERLAPPEDWINDOW: u32 = 0x00cf_0000;
#[cfg(windows)]
const WS_EX_APPWINDOW: u32 = 0x0004_0000;
const HWND_TOP: isize = 0;
const SWP_FRAMECHANGED: u32 = 0x20;
const SWP_SHOWWINDOW: u32 = 0x40;
#[cfg(windows)]
const SWP_NOZORDER: u32 = 0x4;
#[cfg(windows)]
const SWP_NOACTIVATE: u32 = 0x10;
#[cfg(windows)]
const SWP_NOMOVE: u32 = 0x2;
#[cfg(windows)]
const SW_SHOW: i32 = 5;
#[cfg(windows)]
const SW_HIDE: i32 = 0;

#[cfg(any(windows, test))]
fn windowed_style(style: u32) -> u32 {
    (style & !WS_POPUP) | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX
}

#[cfg(any(windows, test))]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Rect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

/// Choose a borderless fullscreen window's position and size.
///
/// A valid monitor rectangle takes precedence: when Wine rejects the requested
/// legacy video mode, covering that monitor still gives the game a fullscreen
/// surface. The requested mode remains the backend's logical coordinate space.
#[cfg(any(windows, test))]
fn fullscreen_bounds(monitor: Option<Rect>, requested: (u32, u32)) -> Rect {
    let Some(monitor) = monitor else {
        return Rect {
            left: 0,
            top: 0,
            right: requested.0 as i32,
            bottom: requested.1 as i32,
        };
    };
    if monitor.right > monitor.left && monitor.bottom > monitor.top {
        return monitor;
    }

    Rect {
        left: monitor.left,
        top: monitor.top,
        right: monitor.left + requested.0 as i32,
        bottom: monitor.top + requested.1 as i32,
    }
}

#[cfg(windows)]
#[repr(C)]
struct MonitorInfo {
    size: u32,
    monitor: Rect,
    work: Rect,
    flags: u32,
}

/// Original style and position, restored when a windowed Glide context closes.
#[cfg(windows)]
pub struct WindowedWindowState {
    hwnd: isize,
    style: u32,
    rect: Rect,
}

#[cfg(windows)]
impl WindowedWindowState {
    pub fn hwnd(&self) -> isize {
        self.hwnd
    }
}

#[cfg(windows)]
unsafe fn set_window_style(hwnd: isize, style: u32) -> Result<(), u32> {
    SetLastError(0);
    #[cfg(target_pointer_width = "32")]
    let previous = SetWindowLongW(hwnd, GWL_STYLE, style as i32) as isize;
    #[cfg(target_pointer_width = "64")]
    let previous = SetWindowLongPtrW(hwnd, GWL_STYLE, style as isize);
    if previous == 0 {
        let error = GetLastError();
        if error != 0 {
            return Err(error);
        }
    }
    Ok(())
}

/// Add a draggable caption to a game's popup window while preserving its client
/// size. Keep the title bar inside the nearest monitor's work area.
#[cfg(windows)]
pub fn make_windowed_window(hwnd: isize) -> Result<Option<WindowedWindowState>, u32> {
    if hwnd == 0 {
        return Err(1400); // ERROR_INVALID_WINDOW_HANDLE
    }
    unsafe {
        let old_style = GetWindowLongW(hwnd, GWL_STYLE) as u32;
        if old_style & WS_CHILD != 0 {
            return Err(87); // ERROR_INVALID_PARAMETER: leave child windows alone
        }
        let new_style = windowed_style(old_style);
        if new_style == old_style {
            return Ok(None);
        }
        let mut old_rect = Rect::default();
        let mut client = Rect::default();
        if GetWindowRect(hwnd, &mut old_rect) == 0 || GetClientRect(hwnd, &mut client) == 0 {
            return Err(GetLastError());
        }
        let mut outer = Rect {
            left: 0,
            top: 0,
            right: (client.right - client.left).max(1),
            bottom: (client.bottom - client.top).max(1),
        };
        let ex_style = GetWindowLongW(hwnd, GWL_EXSTYLE) as u32;
        if AdjustWindowRectEx(&mut outer, new_style, (GetMenu(hwnd) != 0) as i32, ex_style) == 0 {
            return Err(GetLastError());
        }
        let width = (outer.right - outer.left).max(1);
        let height = (outer.bottom - outer.top).max(1);
        let (mut x, mut y) = (old_rect.left, old_rect.top);
        let monitor = MonitorFromWindow(hwnd, 2); // MONITOR_DEFAULTTONEAREST
        let mut info = MonitorInfo {
            size: core::mem::size_of::<MonitorInfo>() as u32,
            monitor: Rect::default(),
            work: Rect::default(),
            flags: 0,
        };
        if monitor != 0
            && GetMonitorInfoW(monitor, &mut info) != 0
            && info.work.right > info.work.left
            && info.work.bottom > info.work.top
        {
            x = x.clamp(
                info.work.left,
                (info.work.right - width).max(info.work.left),
            );
            y = y.clamp(
                info.work.top,
                (info.work.bottom - height).max(info.work.top),
            );
        }
        set_window_style(hwnd, new_style)?;
        if SetWindowPos(
            hwnd,
            HWND_TOP,
            x,
            y,
            width,
            height,
            SWP_FRAMECHANGED | SWP_NOZORDER | SWP_NOACTIVATE,
        ) == 0
        {
            let error = GetLastError();
            let _ = set_window_style(hwnd, old_style);
            SetWindowPos(
                hwnd,
                HWND_TOP,
                old_rect.left,
                old_rect.top,
                old_rect.right - old_rect.left,
                old_rect.bottom - old_rect.top,
                SWP_FRAMECHANGED | SWP_NOZORDER | SWP_NOACTIVATE,
            );
            return Err(error);
        }
        Ok(Some(WindowedWindowState {
            hwnd,
            style: old_style,
            rect: old_rect,
        }))
    }
}

/// Current style and client dimensions for diagnostics after a game has had a
/// chance to process the window style change.
#[cfg(windows)]
pub fn window_info(hwnd: isize) -> Option<(u32, u32, u32)> {
    let mut client = Rect::default();
    if hwnd == 0 || unsafe { GetClientRect(hwnd, &mut client) } == 0 {
        return None;
    }
    let style = unsafe { GetWindowLongW(hwnd, GWL_STYLE) } as u32;
    Some((
        style,
        (client.right - client.left).max(0) as u32,
        (client.bottom - client.top).max(0) as u32,
    ))
}

/// Restore the window's style and bounds after the Glide context closes.
#[cfg(windows)]
pub fn restore_windowed_window(state: WindowedWindowState) -> Result<(), u32> {
    if !is_window(state.hwnd) {
        return Ok(());
    }
    unsafe {
        set_window_style(state.hwnd, state.style)?;
        if SetWindowPos(
            state.hwnd,
            HWND_TOP,
            state.rect.left,
            state.rect.top,
            state.rect.right - state.rect.left,
            state.rect.bottom - state.rect.top,
            SWP_FRAMECHANGED | SWP_NOZORDER | SWP_NOACTIVATE,
        ) == 0
        {
            return Err(GetLastError());
        }
    }
    Ok(())
}

/// Return the active window for the calling game's UI thread, if it has one.
#[cfg(windows)]
pub fn active_window() -> isize {
    // SAFETY: no arguments; GetActiveWindow returns a handle or NULL.
    unsafe { GetActiveWindow() }
}

/// Whether `hwnd` is still a live Win32 window.
#[cfg(windows)]
pub fn is_window(hwnd: isize) -> bool {
    hwnd != 0 && unsafe { IsWindow(hwnd) != 0 }
}

/// Create a per-process fallback window for fullscreen Glide applications that
/// pass NULL as `grSstWinOpen`'s HWND. The app can keep the desktop mode with
/// `GLIDEVK_WINDOWED=1`; otherwise `make_fullscreen_window` turns this into the
/// requested borderless fullscreen surface after the display mode is selected.
#[cfg(windows)]
pub fn create_glide_window(instance: isize, width: u32, height: u32) -> isize {
    const CLASS_NAME: [u16; 7] = [
        b'S' as u16,
        b'T' as u16,
        b'A' as u16,
        b'T' as u16,
        b'I' as u16,
        b'C' as u16,
        0,
    ];
    const WINDOW_NAME: [u16; 8] = [
        b'G' as u16,
        b'l' as u16,
        b'i' as u16,
        b'd' as u16,
        b'e' as u16,
        b'V' as u16,
        b'K' as u16,
        0,
    ];
    let style = WS_OVERLAPPEDWINDOW | WS_VISIBLE;
    let (outer_width, outer_height) =
        window_outer_size(width, height, style, WS_EX_APPWINDOW, false);
    // SAFETY: STATIC is a built-in Win32 window class; the strings and handles
    // remain valid for the duration of CreateWindowExW.
    let hwnd = unsafe {
        CreateWindowExW(
            WS_EX_APPWINDOW,
            CLASS_NAME.as_ptr(),
            WINDOW_NAME.as_ptr(),
            style,
            100,
            100,
            outer_width,
            outer_height,
            0,
            0,
            instance,
            core::ptr::null_mut(),
        )
    };
    if hwnd != 0 {
        unsafe { ShowWindow(hwnd, SW_SHOW) };
    }
    hwnd
}

/// Translate the requested client size into the size expected by Win32's
/// window creation and positioning functions.
#[cfg(windows)]
fn window_outer_size(width: u32, height: u32, style: u32, ex_style: u32, menu: bool) -> (i32, i32) {
    let mut rect = Rect {
        left: 0,
        top: 0,
        right: width.clamp(1, i32::MAX as u32) as i32,
        bottom: height.clamp(1, i32::MAX as u32) as i32,
    };
    // SAFETY: rect is writable and the style values are Win32 window styles.
    if unsafe { AdjustWindowRectEx(&mut rect, style, menu as i32, ex_style) } == 0 {
        return (rect.right, rect.bottom);
    }
    (
        rect.right.saturating_sub(rect.left),
        rect.bottom.saturating_sub(rect.top),
    )
}

/// Resize a surface window's client area while leaving its position and focus alone.
#[cfg(windows)]
pub fn resize_window_client(hwnd: isize, width: u32, height: u32) -> Result<(), u32> {
    if !is_window(hwnd) {
        return Err(1400);
    }
    unsafe {
        let style = GetWindowLongW(hwnd, GWL_STYLE) as u32;
        let ex_style = GetWindowLongW(hwnd, GWL_EXSTYLE) as u32;
        let (width, height) = window_outer_size(width, height, style, ex_style, GetMenu(hwnd) != 0);
        if SetWindowPos(
            hwnd,
            HWND_TOP,
            0,
            0,
            width,
            height,
            SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE,
        ) == 0
        {
            return Err(GetLastError());
        }
    }
    Ok(())
}

/// Show a previously created Glide fallback window when the game reopens its
/// render context after a focus or display-mode change.
#[cfg(windows)]
pub fn show_glide_window(hwnd: isize) {
    if hwnd != 0 {
        unsafe { ShowWindow(hwnd, SW_SHOW) };
    }
}

/// Resize a fallback window for a new Glide mode request. The caller only
/// invokes this for windows created by `create_glide_window`.
#[cfg(windows)]
pub fn resize_glide_window(hwnd: isize, width: u32, height: u32) {
    if hwnd != 0 {
        unsafe {
            let style = GetWindowLongW(hwnd, GWL_STYLE) as u32;
            let ex_style = GetWindowLongW(hwnd, GWL_EXSTYLE) as u32;
            let (outer_width, outer_height) =
                window_outer_size(width, height, style, ex_style, GetMenu(hwnd) != 0);
            SetWindowPos(
                hwnd,
                HWND_TOP,
                0,
                0,
                outer_width,
                outer_height,
                SWP_SHOWWINDOW | SWP_NOMOVE,
            );
        }
    }
}

/// Hide the Glide fallback window when the game closes its render context.
#[cfg(windows)]
pub fn hide_glide_window(hwnd: isize) {
    if hwnd != 0 {
        unsafe { ShowWindow(hwnd, SW_HIDE) };
    }
}

/// Make `hwnd` a borderless popup covering the (possibly emulated) screen at
/// the origin: what a fullscreen Glide game's window must look like for Wine to
/// treat it as fullscreen and scale it. Games do this themselves on real
/// hardware only if their engine bothers (Deus Ex does not).
#[cfg(windows)]
pub fn make_fullscreen_window(hwnd: isize, width: u32, height: u32) -> Option<(u32, u32)> {
    if hwnd == 0 {
        return None;
    }
    let mut info = MonitorInfo {
        size: core::mem::size_of::<MonitorInfo>() as u32,
        monitor: Rect::default(),
        work: Rect::default(),
        flags: 0,
    };
    let monitor = unsafe { MonitorFromWindow(hwnd, 2) }; // MONITOR_DEFAULTTONEAREST
    let monitor_rect = if monitor != 0 && unsafe { GetMonitorInfoW(monitor, &mut info) } != 0 {
        Some(info.monitor)
    } else {
        None
    };
    let bounds = fullscreen_bounds(monitor_rect, (width, height));
    let style = (WS_POPUP | WS_VISIBLE) as i32;
    // SAFETY: `hwnd` is the game's window handle passed to grSstWinOpen.
    unsafe {
        #[cfg(target_pointer_width = "32")]
        SetWindowLongW(hwnd, GWL_STYLE, style);
        #[cfg(target_pointer_width = "64")]
        SetWindowLongPtrW(hwnd, GWL_STYLE, style as isize);
        SetWindowPos(
            hwnd,
            HWND_TOP,
            bounds.left,
            bounds.top,
            bounds.right - bounds.left,
            bounds.bottom - bounds.top,
            SWP_FRAMECHANGED | SWP_SHOWWINDOW,
        );
    }
    Some((
        (bounds.right - bounds.left).max(1) as u32,
        (bounds.bottom - bounds.top).max(1) as u32,
    ))
}

/// Switch the primary display to `width`x`height` for the calling process
/// (`CDS_FULLSCREEN`: restored when the process exits or on
/// [`restore_display_mode`]). Returns the `DISP_CHANGE_*` code on failure.
#[cfg(windows)]
pub fn set_display_mode(width: u32, height: u32) -> Result<(), i32> {
    let mut dm = DevModeW::mode(width, height);
    // SAFETY: `dm` is a fully initialised DEVMODEW with a correct dmSize.
    let r = unsafe {
        ChangeDisplaySettingsExW(
            core::ptr::null(),
            &mut dm,
            0,
            CDS_FULLSCREEN,
            core::ptr::null_mut(),
        )
    };
    if r == DISP_CHANGE_SUCCESSFUL {
        Ok(())
    } else {
        Err(r)
    }
}

/// Return the display to the desktop mode.
#[cfg(windows)]
pub fn restore_display_mode() {
    // SAFETY: a null DEVMODE means "registry/default mode".
    unsafe {
        ChangeDisplaySettingsExW(
            core::ptr::null(),
            core::ptr::null_mut(),
            0,
            0,
            core::ptr::null_mut(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn devmodew_layout_matches_wingdi() {
        assert_eq!(core::mem::size_of::<DevModeW>(), DEVMODEW_SIZE);
        assert_eq!(core::mem::offset_of!(DevModeW, fields), 72);
        assert_eq!(core::mem::offset_of!(DevModeW, bits_per_pel), 168);
        assert_eq!(core::mem::offset_of!(DevModeW, pels_width), 172);
        assert_eq!(core::mem::offset_of!(DevModeW, pels_height), 176);
        let dm = DevModeW::mode(800, 600);
        assert_eq!(dm.size as usize, DEVMODEW_SIZE);
        assert_eq!((dm.pels_width, dm.pels_height), (800, 600));
    }

    #[test]
    fn windowed_style_keeps_existing_flags_and_adds_a_caption() {
        let clip_siblings = 0x0400_0000;
        let style = windowed_style(WS_POPUP | WS_VISIBLE | clip_siblings);
        assert_eq!(style & WS_POPUP, 0);
        assert_eq!(
            style & (WS_VISIBLE | clip_siblings),
            WS_VISIBLE | clip_siblings
        );
        assert_eq!(
            style & (WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX),
            WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX
        );
    }

    #[test]
    fn fullscreen_bounds_prefer_a_valid_monitor() {
        let monitor = Rect {
            left: 2560,
            top: 144,
            right: 6016,
            bottom: 2444,
        };
        assert_eq!(fullscreen_bounds(Some(monitor), (1600, 1200)), monitor);

        // A malformed monitor result must not collapse the game to 1x1; retain
        // its origin but use the requested Glide mode as the fallback size.
        let invalid = Rect {
            left: 100,
            top: 50,
            right: 100,
            bottom: 50,
        };
        assert_eq!(
            fullscreen_bounds(Some(invalid), (1600, 1200)),
            Rect {
                left: 100,
                top: 50,
                right: 1700,
                bottom: 1250,
            }
        );

        assert_eq!(
            fullscreen_bounds(None, (1600, 1200)),
            Rect {
                left: 0,
                top: 0,
                right: 1600,
                bottom: 1200,
            }
        );
    }
}
