use crate::{
    App, Bounds, DevicePixels, Element, ElementId, GlobalElementId, InspectorElementId,
    IntoElement, LayoutId, ObjectFit, Pixels, Style, StyleRefinement, Styled, Window,
};
use refineable::Refineable;
use smallvec::SmallVec;
use std::{
    any::Any,
    fmt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
};

static NEXT_SURFACE_CACHE_KEY: AtomicUsize = AtomicUsize::new(1);

/// A damaged pixel-space rectangle within a platform surface frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceDamageRect {
    /// Left pixel coordinate.
    pub x: usize,
    /// Top pixel coordinate.
    pub y: usize,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

/// A compact collection of damaged pixel-space rectangles for a surface frame.
pub type SurfaceDamageRects = SmallVec<[SurfaceDamageRect; 8]>;

/// An opaque platform GPU surface together with renderer-independent frame metadata.
///
/// Platform crates create these buffers and recover their native resource with
/// [`SurfaceBuffer::platform_surface`]. Keeping the resource opaque lets `gpui`
/// carry surfaces through layout and scene construction without depending on a
/// platform graphics API.
#[derive(Clone)]
pub struct SurfaceBuffer(Arc<SurfaceBufferInner>);

struct SurfaceBufferInner {
    platform_surface: Box<dyn Any + Send + Sync>,
    width: u32,
    height: u32,
    cache_key: usize,
    source_surface_id: Option<u64>,
    prefer_retained_copy: AtomicBool,
    generation: AtomicU64,
    dirty_rects: Mutex<SurfaceDamageRects>,
}

impl SurfaceBuffer {
    /// Wrap a native surface resource for transport through GPUI's scene.
    ///
    /// This is intended for GPUI platform crates. Application code should use
    /// the constructor exposed by its active platform integration.
    #[doc(hidden)]
    pub fn from_platform_surface<T>(
        platform_surface: T,
        width: u32,
        height: u32,
        cache_key: Option<usize>,
        source_surface_id: Option<u64>,
    ) -> Self
    where
        T: Any + Send + Sync,
    {
        Self(Arc::new(SurfaceBufferInner {
            platform_surface: Box::new(platform_surface),
            width,
            height,
            cache_key: cache_key
                .unwrap_or_else(|| NEXT_SURFACE_CACHE_KEY.fetch_add(1, Ordering::Relaxed)),
            source_surface_id,
            prefer_retained_copy: AtomicBool::new(false),
            generation: AtomicU64::new(0),
            dirty_rects: Mutex::new(SurfaceDamageRects::new()),
        }))
    }

    /// Recover the native resource stored by a GPUI platform crate.
    #[doc(hidden)]
    pub fn platform_surface<T: Any>(&self) -> Option<&T> {
        self.0.platform_surface.downcast_ref()
    }

    /// Texture width in pixels.
    pub fn get_width(&self) -> u32 {
        self.0.width
    }

    /// Texture height in pixels.
    pub fn get_height(&self) -> u32 {
        self.0.height
    }

    /// Stable identity for renderer-side caches.
    pub fn cache_key(&self) -> usize {
        self.0.cache_key
    }

    /// Stable identity for the underlying producer surface, when available.
    #[doc(hidden)]
    pub fn source_surface_id(&self) -> Option<u64> {
        self.0.source_surface_id
    }

    /// Whether the renderer should copy this frame into retained private storage.
    pub fn prefer_retained_copy(&self) -> bool {
        self.0.prefer_retained_copy.load(Ordering::Relaxed)
    }

    /// Update the retained-copy preference for this frame.
    pub fn set_prefer_retained_copy(&self, prefer_retained_copy: bool) {
        self.0
            .prefer_retained_copy
            .store(prefer_retained_copy, Ordering::Relaxed);
    }

    /// Monotonically increasing frame generation assigned by the producer.
    pub fn generation(&self) -> u64 {
        self.0.generation.load(Ordering::Relaxed)
    }

    /// Update the frame generation associated with this surface buffer.
    pub fn set_generation(&self, generation: u64) {
        self.0.generation.store(generation, Ordering::Relaxed);
    }

    /// Dirty rects, in pixel coordinates, associated with the current frame.
    pub fn dirty_rects(&self) -> SurfaceDamageRects {
        self.0
            .dirty_rects
            .lock()
            .map(|rects| rects.clone())
            .unwrap_or_default()
    }

    /// Update the dirty rects associated with this frame.
    pub fn set_dirty_rects(&self, rects: impl Into<SurfaceDamageRects>) {
        if let Ok(mut dirty_rects) = self.0.dirty_rects.lock() {
            *dirty_rects = rects.into();
        }
    }
}

impl PartialEq for SurfaceBuffer {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for SurfaceBuffer {}

impl fmt::Debug for SurfaceBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SurfaceBuffer")
            .field("width", &self.get_width())
            .field("height", &self.get_height())
            .field("cache_key", &self.cache_key())
            .field("generation", &self.generation())
            .finish()
    }
}

/// A source of a surface's content.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SurfaceSource {
    /// A platform GPU surface buffer.
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    Surface(SurfaceBuffer),
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
impl From<SurfaceBuffer> for SurfaceSource {
    fn from(value: SurfaceBuffer) -> Self {
        SurfaceSource::Surface(value)
    }
}

/// A surface element.
pub struct Surface {
    source: SurfaceSource,
    object_fit: ObjectFit,
    style: StyleRefinement,
}

/// Create a new surface element.
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub fn surface(source: impl Into<SurfaceSource>) -> Surface {
    Surface {
        source: source.into(),
        object_fit: ObjectFit::Contain,
        style: Default::default(),
    }
}

impl Surface {
    /// Set the object fit for the image.
    pub fn object_fit(mut self, object_fit: ObjectFit) -> Self {
        self.object_fit = object_fit;
        self
    }
}

impl Element for Surface {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut style = Style::default();
        style.refine(&self.style);
        let layout_id = window.request_layout(style, [], cx);
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Self::PrepaintState {
    }

    fn paint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        #[cfg_attr(
            not(any(target_os = "macos", target_os = "windows")),
            allow(unused_variables)
        )]
        bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        #[cfg_attr(
            not(any(target_os = "macos", target_os = "windows")),
            allow(unused_variables)
        )]
        window: &mut Window,
        _: &mut App,
    ) {
        match &self.source {
            #[cfg(any(target_os = "macos", target_os = "windows"))]
            SurfaceSource::Surface(surface) => {
                let size = crate::size(
                    DevicePixels(surface.get_width() as i32),
                    DevicePixels(surface.get_height() as i32),
                );
                let new_bounds = self.object_fit.get_bounds(bounds, size);
                window.paint_surface(new_bounds, surface.clone());
            }
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }
}

impl IntoElement for Surface {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Styled for Surface {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_platform_surface_round_trips_with_frame_metadata() {
        let surface = SurfaceBuffer::from_platform_surface(
            String::from("native-resource"),
            640,
            480,
            Some(42),
            Some(7),
        );
        let damage = SurfaceDamageRect {
            x: 3,
            y: 4,
            width: 20,
            height: 10,
        };

        surface.set_prefer_retained_copy(true);
        surface.set_generation(9);
        surface.set_dirty_rects(SurfaceDamageRects::from_slice(&[damage]));

        assert_eq!(
            surface.platform_surface::<String>().map(String::as_str),
            Some("native-resource")
        );
        assert_eq!((surface.get_width(), surface.get_height()), (640, 480));
        assert_eq!(surface.cache_key(), 42);
        assert_eq!(surface.source_surface_id(), Some(7));
        assert!(surface.prefer_retained_copy());
        assert_eq!(surface.generation(), 9);
        assert_eq!(surface.dirty_rects().as_slice(), &[damage]);
    }
}
