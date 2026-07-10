use core_foundation::base::TCFType;
use core_video::pixel_buffer::CVPixelBuffer;
#[allow(deprecated)]
use io_surface::{IOSurface, IOSurfaceDecrementUseCount, IOSurfaceIncrementUseCount, IOSurfaceRef};

use gpui::SurfaceBuffer;

/// Native macOS storage carried by a GPUI surface buffer.
pub(crate) struct MacSurface {
    pixel_buffer: CVPixelBuffer,
    held_io_surface: Option<HeldIOSurface>,
}

// SAFETY: CVPixelBuffer and IOSurface use thread-safe Core Foundation ownership,
// and the contained resources are only sampled by the Metal renderer.
unsafe impl Send for MacSurface {}
unsafe impl Sync for MacSurface {}

impl MacSurface {
    pub(crate) fn pixel_buffer(&self) -> &CVPixelBuffer {
        &self.pixel_buffer
    }

    #[allow(deprecated)]
    pub(crate) fn io_surface_ref(&self) -> Option<IOSurfaceRef> {
        self.held_io_surface
            .as_ref()
            .map(|surface| surface.surface.as_concrete_TypeRef())
    }
}

#[allow(deprecated)]
struct HeldIOSurface {
    surface: IOSurface,
}

#[allow(deprecated)]
impl HeldIOSurface {
    fn new(surface: IOSurface) -> Self {
        unsafe { IOSurfaceIncrementUseCount(surface.as_concrete_TypeRef()) };
        Self { surface }
    }
}

#[allow(deprecated)]
impl Drop for HeldIOSurface {
    fn drop(&mut self) {
        unsafe { IOSurfaceDecrementUseCount(self.surface.as_concrete_TypeRef()) };
    }
}

/// Wrap a CoreVideo pixel buffer as a GPUI surface.
pub fn surface_buffer(pixel_buffer: CVPixelBuffer, cache_key: Option<usize>) -> SurfaceBuffer {
    let width = pixel_buffer.get_width() as u32;
    let height = pixel_buffer.get_height() as u32;
    SurfaceBuffer::from_platform_surface(
        MacSurface {
            pixel_buffer,
            held_io_surface: None,
        },
        width,
        height,
        cache_key,
        None,
    )
}

/// Wrap an IOSurface-backed CoreVideo pixel buffer and retain its producer storage.
#[allow(deprecated)]
pub fn surface_buffer_from_io_surface(
    pixel_buffer: CVPixelBuffer,
    io_surface: IOSurface,
    cache_key: Option<usize>,
) -> SurfaceBuffer {
    let width = pixel_buffer.get_width() as u32;
    let height = pixel_buffer.get_height() as u32;
    let source_surface_id = Some(io_surface.get_id() as u64);
    SurfaceBuffer::from_platform_surface(
        MacSurface {
            pixel_buffer,
            held_io_surface: Some(HeldIOSurface::new(io_surface)),
        },
        width,
        height,
        cache_key,
        source_surface_id,
    )
}
