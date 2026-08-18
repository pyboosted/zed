// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, Edges, Hsla, Pixels,
    Point, Radians, ScaledPixels, Size, bounds_tree::BoundsTree, point,
};
use scheduler::Instant;
use std::{
    fmt::Debug,
    iter::Peekable,
    ops::{Add, Range, Sub},
    slice,
    time::Duration,
};

#[allow(non_camel_case_types, unused)]
#[expect(missing_docs)]
pub type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

#[expect(missing_docs)]
pub type DrawOrder = u32;

/// A boolean stored as a `u32` so that GPU-facing structs contain no
/// compiler-inserted padding bytes, which would be undefined behavior to
/// reinterpret as `&[u8]` when writing instance buffers. Guaranteed to be
/// `0` or `1` by construction; shaders read it as a `u32`/`uint`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct PaddedBool32(u32);

impl From<bool> for PaddedBool32 {
    fn from(value: bool) -> Self {
        PaddedBool32(value as u32)
    }
}

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<DrawOrder>,
    pub shadows: Vec<Shadow>,
    pub quads: Vec<Quad>,
    pub effects: Vec<EffectQuad>,
    pub paths: Vec<Path<ScaledPixels>>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
    motion_schedule: MotionSchedule,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct MotionSchedule {
    buckets: Vec<MotionScheduleBucket>,
}

/// Why a retained scene cannot use the bounded motion-damage experiment.
///
/// This is exposed for renderer diagnostics. It is not a product-facing
/// rendering contract.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetainedMotionDamageFallback {
    /// The scene has no procedural motion effect.
    NoEffect,
    /// The experiment is deliberately bounded to one effect instance.
    MultipleEffects,
    /// Native or externally-produced surfaces require the ordinary renderer.
    ExternalSurface,
    /// A non-effect primitive is interleaved with or above the effect.
    InterleavedOrder,
    /// The effect is not the initially-supported spinner primitive.
    UnsupportedEffect,
}

/// Conservative pixel damage for an eligible retained-motion scene.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetainedMotionDamage {
    /// Effect bounds clipped by the effect's content mask.
    pub bounds: Bounds<ScaledPixels>,
}

/// Tracks damage and flip-chain coherence for a retained scene.
///
/// A renderer must call [`Self::begin_scene`] after an ordinary scene draw,
/// [`Self::observe_retained_full`] after each fallback full replay, and may use
/// [`Self::next_damage`] only when [`Self::is_coherent`] is true.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct RetainedMotionDamageHistory {
    buffer_bounds: Vec<Option<Bounds<ScaledPixels>>>,
    next_buffer: usize,
}

impl RetainedMotionDamageHistory {
    /// Starts history for a newly drawn scene. Only the current flip-chain
    /// buffer is known to contain that scene.
    pub fn begin_scene(&mut self, bounds: Bounds<ScaledPixels>, buffer_count: usize) {
        self.buffer_bounds = vec![None; buffer_count];
        if let Some(first) = self.buffer_bounds.first_mut() {
            *first = Some(bounds);
            self.next_buffer = 1 % buffer_count;
        } else {
            self.next_buffer = 0;
        }
    }

    /// Records a full replay of the same retained scene into another buffer.
    pub fn observe_retained_full(&mut self, bounds: Bounds<ScaledPixels>, buffer_count: usize) {
        if self.buffer_bounds.len() != buffer_count || self.buffer_bounds.is_empty() {
            self.begin_scene(bounds, buffer_count);
            return;
        }
        self.buffer_bounds[self.next_buffer] = Some(bounds);
        self.next_buffer = (self.next_buffer + 1) % buffer_count;
    }

    /// Returns whether every flip-chain buffer has received a full scene.
    pub fn is_coherent(&self, buffer_count: usize) -> bool {
        buffer_count > 0
            && self.buffer_bounds.len() == buffer_count
            && self.buffer_bounds.iter().all(Option::is_some)
    }

    /// Returns the union of the bounds currently stored in the next
    /// swap-chain buffer and the current effect bounds. This is intentionally
    /// non-mutating: history advances only after a successful presentation.
    pub fn next_damage(&self, current: Bounds<ScaledPixels>) -> Option<Bounds<ScaledPixels>> {
        let previous = self.buffer_bounds.get(self.next_buffer)?.as_ref()?;
        Some(previous.union(&current))
    }

    /// Records a successfully presented damage frame in the current buffer.
    pub fn did_present_damage(&mut self, current: Bounds<ScaledPixels>) {
        if self.buffer_bounds.is_empty() {
            return;
        }
        self.buffer_bounds[self.next_buffer] = Some(current);
        self.next_buffer = (self.next_buffer + 1) % self.buffer_bounds.len();
    }

    /// Invalidates all scene and swapchain history.
    pub fn invalidate(&mut self) {
        *self = Self::default();
    }
}

#[derive(Clone, Copy, Debug)]
struct MotionScheduleBucket {
    frame_interval: Duration,
    latest_end: Option<Instant>,
}

impl MotionSchedule {
    pub(crate) fn register(&mut self, animation: EffectAnimation) {
        if let Some(bucket) = self
            .buckets
            .iter_mut()
            .find(|bucket| bucket.frame_interval == animation.frame_interval)
        {
            bucket.latest_end = match (bucket.latest_end, animation.ends_at) {
                (None, _) | (_, None) => None,
                (Some(current), Some(incoming)) => Some(current.max(incoming)),
            };
        } else {
            self.buckets.push(MotionScheduleBucket {
                frame_interval: animation.frame_interval,
                latest_end: animation.ends_at,
            });
        }
    }

    pub(crate) fn active_frame_interval(&self, now: Instant) -> Option<Duration> {
        self.buckets
            .iter()
            .filter(|bucket| bucket.latest_end.is_none_or(|end| now < end))
            .map(|bucket| bucket.frame_interval)
            .min()
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct EffectAnimation {
    pub(crate) frame_interval: Duration,
    pub(crate) ends_at: Option<Instant>,
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct EffectPrimitive {
    pub(crate) instance: EffectQuad,
    pub(crate) animation: Option<EffectAnimation>,
}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.paths.clear();
        self.shadows.clear();
        self.quads.clear();
        self.effects.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.subpixel_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
        self.motion_schedule = MotionSchedule::default();
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push(order);
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        let mut primitive = primitive.into();
        let clipped_bounds = primitive
            .bounds()
            .intersect(&primitive.content_mask().bounds);

        if clipped_bounds.is_empty() {
            return;
        }

        let order = self
            .layer_stack
            .last()
            .copied()
            .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds));
        match &mut primitive {
            Primitive::Shadow(shadow) => {
                shadow.order = order;
                self.shadows.push(*shadow);
            }
            Primitive::Quad(quad) => {
                quad.order = order;
                self.quads.push(*quad);
            }
            Primitive::Effect(effect) => {
                effect.instance.order = order;
                self.effects.push(effect.instance);
                if let Some(animation) = effect.animation {
                    self.motion_schedule.register(animation);
                }
            }
            Primitive::Path(path) => {
                path.order = order;
                path.id = PathId(self.paths.len());
                self.paths.push(path.clone());
            }
            Primitive::Underline(underline) => {
                underline.order = order;
                self.underlines.push(*underline);
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.order = order;
                self.monochrome_sprites.push(*sprite);
            }
            Primitive::SubpixelSprite(sprite) => {
                sprite.order = order;
                self.subpixel_sprites.push(*sprite);
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.order = order;
                self.polychrome_sprites.push(*sprite);
            }
            Primitive::Surface(surface) => {
                surface.order = order;
                self.surfaces.push(surface.clone());
            }
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive(primitive) => self.insert_primitive(primitive.clone()),
                PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
                PaintOperation::EndLayer => self.pop_layer(),
            }
        }
    }

    pub fn finish(&mut self) {
        self.shadows.sort_by_key(|shadow| shadow.order);
        self.quads.sort_by_key(|quad| quad.order);
        self.effects.sort_by_key(|effect| effect.order);
        self.paths.sort_by_key(|path| path.order);
        self.underlines.sort_by_key(|underline| underline.order);
        self.monochrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.subpixel_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.polychrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.surfaces.sort_by_key(|surface| surface.order);
    }

    pub(crate) fn motion_schedule(&self) -> &MotionSchedule {
        &self.motion_schedule
    }

    /// Classifies the deliberately narrow scene accepted by the DirectX
    /// retained-motion damage experiment.
    #[doc(hidden)]
    pub fn retained_motion_damage(
        &self,
    ) -> Result<RetainedMotionDamage, RetainedMotionDamageFallback> {
        let effect = match self.effects.as_slice() {
            [] => return Err(RetainedMotionDamageFallback::NoEffect),
            [effect] => effect,
            _ => return Err(RetainedMotionDamageFallback::MultipleEffects),
        };
        if !self.surfaces.is_empty() {
            return Err(RetainedMotionDamageFallback::ExternalSurface);
        }
        if effect.kind != 0 {
            return Err(RetainedMotionDamageFallback::UnsupportedEffect);
        }

        let non_effect_max_order = self
            .shadows
            .iter()
            .map(|primitive| primitive.order)
            .chain(self.quads.iter().map(|primitive| primitive.order))
            .chain(self.paths.iter().map(|primitive| primitive.order))
            .chain(self.underlines.iter().map(|primitive| primitive.order))
            .chain(
                self.monochrome_sprites
                    .iter()
                    .map(|primitive| primitive.order),
            )
            .chain(
                self.subpixel_sprites
                    .iter()
                    .map(|primitive| primitive.order),
            )
            .chain(
                self.polychrome_sprites
                    .iter()
                    .map(|primitive| primitive.order),
            )
            .max();
        if non_effect_max_order.is_some_and(|order| order >= effect.order) {
            return Err(RetainedMotionDamageFallback::InterleavedOrder);
        }

        Ok(RetainedMotionDamage {
            bounds: effect.bounds.intersect(&effect.content_mask.bounds),
        })
    }

    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub fn batches(&self) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        BatchIterator {
            shadows_start: 0,
            shadows_iter: self.shadows.iter().peekable(),
            quads_start: 0,
            quads_iter: self.quads.iter().peekable(),
            effects_start: 0,
            effects_iter: self.effects.iter().peekable(),
            paths_start: 0,
            paths_iter: self.paths.iter().peekable(),
            underlines_start: 0,
            underlines_iter: self.underlines.iter().peekable(),
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
            subpixel_sprites_start: 0,
            subpixel_sprites_iter: self.subpixel_sprites.iter().peekable(),
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
            surfaces_start: 0,
            surfaces_iter: self.surfaces.iter().peekable(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveKind {
    Shadow,
    #[default]
    Quad,
    Effect,
    Path,
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    Surface,
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    Effect(EffectPrimitive),
    Path(Path<ScaledPixels>),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
}

#[expect(missing_docs)]
impl Primitive {
    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Effect(effect) => &effect.instance.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::SubpixelSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Effect(effect) => &effect.instance.content_mask,
            Primitive::Path(path) => &path.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
        }
    }
}

#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
struct BatchIterator<'a> {
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    effects_start: usize,
    effects_iter: Peekable<slice::Iter<'a, EffectQuad>>,
    paths_start: usize,
    paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    subpixel_sprites_start: usize,
    subpixel_sprites_iter: Peekable<slice::Iter<'a, SubpixelSprite>>,
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (
                self.effects_iter.peek().map(|effect| effect.order),
                PrimitiveKind::Effect,
            ),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.subpixel_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::SubpixelSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        match batch_kind {
            PrimitiveKind::Shadow => {
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| (shadow.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows(shadows_start..shadows_end))
            }
            PrimitiveKind::Quad => {
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| (quad.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(PrimitiveBatch::Quads(quads_start..quads_end))
            }
            PrimitiveKind::Effect => {
                let effects_start = self.effects_start;
                let mut effects_end = effects_start + 1;
                self.effects_iter.next();
                while self
                    .effects_iter
                    .next_if(|effect| (effect.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    effects_end += 1;
                }
                self.effects_start = effects_end;
                Some(PrimitiveBatch::Effects(effects_start..effects_end))
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| (path.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                Some(PrimitiveBatch::Paths(paths_start..paths_end))
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| (underline.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(underlines_start..underlines_end))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::SubpixelSprite => {
                let texture_id = self.subpixel_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.subpixel_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.subpixel_sprites_iter.next();
                while self
                    .subpixel_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.subpixel_sprites_start = sprites_end;
                Some(PrimitiveBatch::SubpixelSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let texture_id = self.polychrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| (surface.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(surfaces_start..surfaces_end))
            }
        }
    }
}

#[derive(Debug)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
#[allow(missing_docs)]
pub enum PrimitiveBatch {
    Shadows(Range<usize>),
    Quads(Range<usize>),
    Effects(Range<usize>),
    Paths(Range<usize>),
    Underlines(Range<usize>),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    SubpixelSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    Surfaces(Range<usize>),
}

impl PrimitiveBatch {
    #[expect(missing_docs)]
    pub fn label(&self) -> String {
        match self {
            Self::Shadows(range) => format!("shadows ({})", range.len()),
            Self::Quads(range) => format!("quads ({})", range.len()),
            Self::Effects(range) => format!("effects ({})", range.len()),
            Self::Paths(range) => format!("paths ({})", range.len()),
            Self::Underlines(range) => format!("underlines ({})", range.len()),
            Self::MonochromeSprites { texture_id, range } => {
                format!(
                    "monochrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::SubpixelSprites { texture_id, range } => {
                format!(
                    "subpixel sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::PolychromeSprites { texture_id, range } => {
                format!(
                    "polychrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::Surfaces(range) => format!("surfaces ({})", range.len()),
        }
    }
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: Background,
    pub border_color: Hsla,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
}

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct EffectQuad {
    pub order: DrawOrder,
    pub kind: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub accent_color: Hsla,
    pub corner_radii: Corners<ScaledPixels>,
    pub started_at: f32,
    pub duration: f32,
    pub intensity: f32,
    pub thickness: ScaledPixels,
    pub feather: ScaledPixels,
}

impl From<EffectPrimitive> for Primitive {
    fn from(effect: EffectPrimitive) -> Self {
        Primitive::Effect(effect)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Underline {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub thickness: ScaledPixels,
    pub wavy: PaddedBool32,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub element_bounds: Bounds<ScaledPixels>,
    pub element_corner_radii: Corners<ScaledPixels>,
    /// 0 = drop shadow (rendered outside the element), 1 = inset shadow (rendered inside).
    pub inset: u32,
    pub pad: u32, // align to 8 bytes
}

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point
    pub fn translate(mut self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct MonochromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct SubpixelSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<SubpixelSprite> for Primitive {
    fn from(sprite: SubpixelSprite) -> Self {
        Primitive::SubpixelSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PolychromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub grayscale: PaddedBool32,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub tile: AtlasTile,
}

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    pub image_buffer: crate::SurfaceBuffer,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[expect(missing_docs)]
pub struct PathId(pub usize);

/// A line made up of a series of vertices and control points.
#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    pub id: PathId,
    pub order: DrawOrder,
    pub bounds: Bounds<P>,
    pub content_mask: ContentMask<P>,
    pub vertices: Vec<PathVertex<P>>,
    pub color: Background,
    start: Point<P>,
    current: Point<P>,
    contour_count: usize,
}

impl Path<Pixels> {
    /// Create a new path with the given starting point.
    pub fn new(start: Point<Pixels>) -> Self {
        Self {
            id: PathId(0),
            order: DrawOrder::default(),
            vertices: Vec::new(),
            start,
            current: start,
            bounds: Bounds {
                origin: start,
                size: Default::default(),
            },
            content_mask: Default::default(),
            color: Default::default(),
            contour_count: 0,
        }
    }

    /// Scale this path by the given factor.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        Path {
            id: self.id,
            order: self.order,
            bounds: self.bounds.scale(factor),
            content_mask: self.content_mask.scale(factor),
            vertices: self
                .vertices
                .iter()
                .map(|vertex| vertex.scale(factor))
                .collect(),
            start: self.start.map(|start| start.scale(factor)),
            current: self.current.scale(factor),
            contour_count: self.contour_count,
            color: self.color,
        }
    }

    /// Move the start, current point to the given point.
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        self.start = to;
        self.current = to;
    }

    /// Draw a straight line from the current point to the given point.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        self.current = to;
    }

    /// Draw a curve from the current point to the given point, using the given control point.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }

        self.push_triangle(
            (self.current, ctrl, to),
            (point(0., 0.), point(0.5, 0.), point(1., 1.)),
        );
        self.current = to;
    }

    /// Push a triangle to the Path.
    pub fn push_triangle(
        &mut self,
        xy: (Point<Pixels>, Point<Pixels>, Point<Pixels>),
        st: (Point<f32>, Point<f32>, Point<f32>),
    ) {
        self.bounds = self
            .bounds
            .union(&Bounds {
                origin: xy.0,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.1,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.2,
                size: Default::default(),
            });

        self.vertices.push(PathVertex {
            xy_position: xy.0,
            st_position: st.0,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.1,
            st_position: st.1,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.2,
            st_position: st.2,
            content_mask: Default::default(),
        });
    }
}

impl<T> Path<T>
where
    T: Clone + Debug + Default + PartialEq + PartialOrd + Add<T, Output = T> + Sub<Output = T>,
{
    #[allow(unused)]
    #[expect(missing_docs)]
    pub fn clipped_bounds(&self) -> Bounds<T> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

impl From<Path<ScaledPixels>> for Primitive {
    fn from(path: Path<ScaledPixels>) -> Self {
        Primitive::Path(path)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathVertex<P: Clone + Debug + Default + PartialEq> {
    pub xy_position: Point<P>,
    pub st_position: Point<f32>,
    pub content_mask: ContentMask<P>,
}

#[expect(missing_docs)]
impl PathVertex<Pixels> {
    pub fn scale(&self, factor: f32) -> PathVertex<ScaledPixels> {
        PathVertex {
            xy_position: self.xy_position.scale(factor),
            st_position: self.st_position,
            content_mask: self.content_mask.scale(factor),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::size;

    #[test]
    fn motion_schedule_uses_fastest_active_bucket() {
        let now = Instant::now();
        let mut schedule = MotionSchedule::default();
        schedule.register(EffectAnimation {
            frame_interval: Duration::from_millis(33),
            ends_at: None,
        });
        schedule.register(EffectAnimation {
            frame_interval: Duration::from_millis(16),
            ends_at: Some(now + Duration::from_millis(100)),
        });

        assert_eq!(
            schedule.active_frame_interval(now),
            Some(Duration::from_millis(16))
        );
        assert_eq!(
            schedule.active_frame_interval(now + Duration::from_millis(100)),
            Some(Duration::from_millis(33))
        );
    }

    #[test]
    fn motion_schedule_merges_equal_cadence_without_losing_repeating_effects() {
        let now = Instant::now();
        let mut schedule = MotionSchedule::default();
        schedule.register(EffectAnimation {
            frame_interval: Duration::from_millis(33),
            ends_at: Some(now + Duration::from_millis(100)),
        });
        schedule.register(EffectAnimation {
            frame_interval: Duration::from_millis(33),
            ends_at: None,
        });

        assert_eq!(
            schedule.active_frame_interval(now + Duration::from_secs(10)),
            Some(Duration::from_millis(33))
        );
    }

    fn test_bounds(x: f32, y: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(16.0), ScaledPixels(16.0)),
        )
    }

    fn eligible_motion_scene() -> Scene {
        let mut scene = Scene::default();
        let bounds = test_bounds(10.0, 20.0);
        scene.effects.push(EffectQuad {
            order: 3,
            kind: 0,
            bounds,
            content_mask: ContentMask::from_bounds(bounds),
            ..Default::default()
        });
        scene
    }

    #[test]
    fn retained_motion_damage_eligibility_table() {
        let eligible = eligible_motion_scene();
        assert_eq!(
            eligible.retained_motion_damage(),
            Ok(RetainedMotionDamage {
                bounds: test_bounds(10.0, 20.0)
            })
        );

        let mut clipped = eligible_motion_scene();
        let clipped_bounds = Bounds::new(
            point(ScaledPixels(14.0), ScaledPixels(24.0)),
            size(ScaledPixels(4.0), ScaledPixels(5.0)),
        );
        clipped.effects[0].content_mask = ContentMask::from_bounds(clipped_bounds);
        assert_eq!(
            clipped.retained_motion_damage(),
            Ok(RetainedMotionDamage {
                bounds: clipped_bounds
            })
        );

        let mut multiple = eligible_motion_scene();
        multiple.effects.push(multiple.effects[0]);
        let mut unsupported = eligible_motion_scene();
        unsupported.effects[0].kind = 1;
        let mut surface = eligible_motion_scene();
        surface.surfaces.push(PaintSurface {
            order: 1,
            bounds: test_bounds(0.0, 0.0),
            content_mask: ContentMask::from_bounds(test_bounds(0.0, 0.0)),
            #[cfg(any(target_os = "macos", target_os = "windows"))]
            image_buffer: crate::SurfaceBuffer::from_platform_surface((), 1, 1, None, None),
        });
        let mut interleaved = eligible_motion_scene();
        interleaved.quads.push(Quad {
            order: 3,
            ..Default::default()
        });

        for (name, scene, expected) in [
            (
                "multiple",
                multiple,
                RetainedMotionDamageFallback::MultipleEffects,
            ),
            (
                "unsupported",
                unsupported,
                RetainedMotionDamageFallback::UnsupportedEffect,
            ),
            (
                "surface",
                surface,
                RetainedMotionDamageFallback::ExternalSurface,
            ),
            (
                "interleaved",
                interleaved,
                RetainedMotionDamageFallback::InterleavedOrder,
            ),
        ] {
            assert_eq!(scene.retained_motion_damage(), Err(expected), "{name}");
        }
    }

    #[test]
    fn retained_motion_history_tracks_each_flip_chain_buffer() {
        let p0 = test_bounds(0.0, 0.0);
        let p1 = test_bounds(20.0, 0.0);
        let p2 = test_bounds(40.0, 0.0);
        let p3 = test_bounds(60.0, 0.0);
        let p4 = test_bounds(80.0, 0.0);
        let mut history = RetainedMotionDamageHistory::default();
        history.begin_scene(p0, 3);
        assert!(!history.is_coherent(3));
        history.observe_retained_full(p0, 3);
        history.observe_retained_full(p0, 3);
        assert!(history.is_coherent(3));

        assert_eq!(history.next_damage(p1), Some(p0.union(&p1)));
        history.did_present_damage(p1);
        assert_eq!(history.next_damage(p2), Some(p0.union(&p2)));
        history.did_present_damage(p2);
        assert_eq!(history.next_damage(p3), Some(p0.union(&p3)));
        history.did_present_damage(p3);
        assert_eq!(history.next_damage(p4), Some(p1.union(&p4)));
    }

    #[test]
    fn retained_motion_history_does_not_advance_before_present_success() {
        let p0 = test_bounds(0.0, 0.0);
        let p1 = test_bounds(20.0, 0.0);
        let p2 = test_bounds(40.0, 0.0);
        let mut history = RetainedMotionDamageHistory::default();
        history.begin_scene(p0, 3);
        history.observe_retained_full(p0, 3);
        history.observe_retained_full(p0, 3);
        assert!(history.is_coherent(3));

        assert_eq!(history.next_damage(p1), Some(p0.union(&p1)));
        // Simulate a failed Present1 by deliberately not committing p1. The
        // next attempt must still target the same backbuffer's p0 contents.
        assert_eq!(history.next_damage(p2), Some(p0.union(&p2)));
    }

    #[test]
    fn retained_motion_history_invalidation_requires_full_rewarm() {
        let p0 = test_bounds(0.0, 0.0);
        let p1 = test_bounds(20.0, 0.0);
        let mut history = RetainedMotionDamageHistory::default();
        history.begin_scene(p0, 3);
        history.observe_retained_full(p0, 3);
        history.observe_retained_full(p0, 3);
        assert!(history.is_coherent(3));

        history.invalidate();
        assert!(!history.is_coherent(3));
        assert_eq!(history.next_damage(p1), None);
        history.observe_retained_full(p0, 3);
        history.observe_retained_full(p0, 3);
        assert!(!history.is_coherent(3));
        history.observe_retained_full(p0, 3);
        assert!(history.is_coherent(3));
    }
}
