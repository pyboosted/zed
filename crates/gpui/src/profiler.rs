use itertools::Itertools;
use scheduler::{Instant, SpawnTime};
use std::{
    cell::LazyCell,
    collections::{HashMap, VecDeque},
    hash::{DefaultHasher, Hash, Hasher},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::ThreadId,
    time::Duration,
};

mod actions;
pub use actions::{ActionStatistics, ActionTiming, take_action_stats};
pub(crate) use actions::{save_action_timing, update_running_action};

use serde::{Deserialize, Serialize};

use crate::{RetainedMotionDamageFallback, SharedString, TasksIncluded, WindowId};

#[cfg(feature = "profiler")]
#[doc(hidden)]
pub fn get_all_timings(included: gpui::TasksIncluded) -> Vec<gpui::ThreadTaskTimings> {
    let global_thread_timings = GLOBAL_THREAD_TIMINGS.lock();
    ThreadTaskTimings::collect(&global_thread_timings, included)
}

#[cfg(feature = "profiler")]
#[doc(hidden)]
pub fn get_current_thread_timings(included: TasksIncluded) -> gpui::ThreadTaskTimings {
    gpui::profiler::get_current_thread_task_timings(included)
}

#[cfg(feature = "profiler")]
#[doc(hidden)]
pub fn take_all_stats(included: TasksIncluded) -> Vec<gpui::ThreadTaskStatistics> {
    let global_timings = GLOBAL_THREAD_TIMINGS.lock();
    ThreadTaskStatistics::collect_and_reset(&global_timings, included)
}

#[cfg(not(feature = "profiler"))]
#[doc(hidden)]
pub fn get_all_timings(_included: gpui::TasksIncluded) -> Vec<gpui::ThreadTaskTimings> {
    Vec::new()
}
#[cfg(not(feature = "profiler"))]
#[doc(hidden)]
pub fn get_current_thread_timings(_included: TasksIncluded) -> gpui::ThreadTaskTimings {
    gpui::ThreadTaskTimings {
        thread_name: None,
        thread_id: std::thread::current().id(),
        timings: Vec::new(),
        stats: TaskStatistics::default(),
        total_pushed: 0,
    }
}
#[cfg(not(feature = "profiler"))]
#[doc(hidden)]
pub fn take_all_stats(_included: TasksIncluded) -> Vec<gpui::ThreadTaskStatistics> {
    Vec::new()
}

#[doc(hidden)]
#[derive(Debug, Copy, Clone)]
pub struct YieldTime(pub Instant);

#[doc(hidden)]
#[derive(Copy, Clone)]
pub struct TaskTiming {
    pub location: &'static core::panic::Location<'static>,
    pub spawned: SpawnTime,
    pub start: Instant,
    pub end: YieldTime,
}

impl std::fmt::Debug for TaskTiming {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskTiming")
            .field("location", &self.location)
            .field("since_spawned", &self.spawned.0.elapsed())
            .field("last_poll_duration", &self.poll_duration())
            .field("total_runtime", &self.since_spawn())
            .finish()
    }
}

#[doc(hidden)]
#[derive(Debug, Copy, Clone)]
pub struct ActiveTiming {
    pub location: &'static core::panic::Location<'static>,
    pub spawned: SpawnTime,
    pub start: Instant,
}

impl TaskTiming {
    /// A task timing with a duration of zero. Any task will replace this in history.
    pub fn placeholder() -> Self {
        let now = Instant::now();
        Self {
            location: std::panic::Location::caller(),
            spawned: SpawnTime(now),
            start: now,
            end: YieldTime(now),
        }
    }

    #[inline(always)]
    pub fn poll_duration(&self) -> Duration {
        self.end.0 - self.start
    }

    #[inline(always)]
    fn since_spawn(&self) -> Duration {
        self.end.0 - self.spawned.0
    }
}

#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct ThreadTaskTimings {
    pub thread_name: Option<String>,
    pub thread_id: ThreadId,
    pub timings: Vec<TaskTiming>,
    pub stats: TaskStatistics,
    pub total_pushed: u64,
}

impl ThreadTaskTimings {
    /// Convert global thread timings into their structured format.
    pub fn collect(timings: &[GlobalThreadTimings], included: TasksIncluded) -> Vec<Self> {
        timings
            .iter()
            .filter_map(|t| match t.timings.upgrade() {
                Some(timings) => Some((t.thread_id, timings)),
                _ => None,
            })
            .map(|(thread_id, timings)| {
                let timings = timings.lock();
                let thread_name = timings.thread_name.clone();
                let total_pushed = timings.total_pushed;
                let completed = &timings.timings;

                let mut vec = Vec::with_capacity(completed.len() + 1); // +1 for running task
                let (s1, s2) = completed.as_slices();
                vec.extend_from_slice(s1);
                vec.extend_from_slice(s2);
                if let TasksIncluded::CompletedAndRunning = included
                    && let Some(running) = timings.running
                {
                    vec.push(TaskTiming {
                        location: running.location,
                        spawned: running.spawned,
                        start: running.start,
                        end: YieldTime(Instant::now()),
                    })
                }

                ThreadTaskTimings {
                    thread_name,
                    thread_id,
                    timings: vec,
                    stats: timings.stats.clone(),
                    total_pushed,
                }
            })
            .collect()
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct ThreadTaskStatistics {
    pub thread_name: Option<String>,
    pub thread_id: ThreadId,
    pub stats: TaskStatistics,
}

impl ThreadTaskStatistics {
    pub fn collect_and_reset(
        timings: &[GlobalThreadTimings],
        include_running: TasksIncluded,
    ) -> Vec<Self> {
        timings
            .iter()
            .filter_map(|t| match t.timings.upgrade() {
                Some(timings) => Some((t.thread_id, timings)),
                _ => None,
            })
            .map(|(thread_id, timings)| {
                let mut timings = timings.lock();
                let thread_name = timings.thread_name.clone();

                let mut stats = std::mem::take(&mut timings.stats);
                if let TasksIncluded::CompletedAndRunning = include_running
                    && let Some(ActiveTiming {
                        location,
                        spawned,
                        start,
                    }) = timings.running
                {
                    let end = YieldTime(Instant::now());
                    let timing = TaskTiming {
                        location,
                        spawned,
                        start,
                        end,
                    };
                    stats.add_runtime(timing);
                    stats.add_yield_timing(timing);
                }

                Self {
                    thread_name,
                    thread_id,
                    stats,
                }
            })
            .collect()
    }
}

/// Serializable variant of [`core::panic::Location`]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedLocation {
    /// Name of the source file
    pub file: SharedString,
    /// Line in the source file
    pub line: u32,
    /// Column in the source file
    pub column: u32,
}

impl From<&core::panic::Location<'static>> for SerializedLocation {
    fn from(value: &core::panic::Location<'static>) -> Self {
        SerializedLocation {
            file: value.file().into(),
            line: value.line(),
            column: value.column(),
        }
    }
}

/// Serializable variant of [`TaskTiming`]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedTaskTiming {
    /// Location of the timing
    pub location: SerializedLocation,
    /// Time at which the measurement was reported in nanoseconds
    pub start: u128,
    /// Duration of the measurement in nanoseconds
    pub duration: u128,
}

impl SerializedTaskTiming {
    /// Convert an array of [`TaskTiming`] into their serializable format
    ///
    /// # Params
    ///
    /// `anchor` - [`Instant`] that should be earlier than all timings to use as base anchor
    pub fn convert(anchor: Instant, timings: &[TaskTiming]) -> Vec<SerializedTaskTiming> {
        let serialized = timings
            .iter()
            .map(|timing| {
                let start = timing.start.duration_since(anchor).as_nanos();
                let duration = timing.end.0.duration_since(timing.start).as_nanos();
                SerializedTaskTiming {
                    location: timing.location.into(),
                    start,
                    duration,
                }
            })
            .collect::<Vec<_>>();

        serialized
    }

    /// `anchor` - [`Instant`] that should be earlier than all timings to use as base anchor
    pub fn from(anchor: Instant, timing: TaskTiming) -> SerializedTaskTiming {
        let start = timing.start.duration_since(anchor).as_nanos();
        let duration = timing.end.0.duration_since(timing.start).as_nanos();
        SerializedTaskTiming {
            location: timing.location.into(),
            start,
            duration,
        }
    }
}

/// Serializable variant of [`ThreadTaskTimings`]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedThreadTaskTimings {
    /// Thread name
    pub thread_name: Option<String>,
    /// Hash of the thread id
    pub thread_id: u64,
    /// Timing records for this thread
    pub timings: Vec<SerializedTaskTiming>,
}

impl SerializedThreadTaskTimings {
    /// Convert [`ThreadTaskTimings`] into their serializable format
    ///
    /// # Params
    ///
    /// `anchor` - [`Instant`] that should be earlier than all timings to use as base anchor
    pub fn convert(anchor: Instant, timings: ThreadTaskTimings) -> SerializedThreadTaskTimings {
        let serialized_timings = SerializedTaskTiming::convert(anchor, &timings.timings);

        let mut hasher = DefaultHasher::new();
        timings.thread_id.hash(&mut hasher);
        let thread_id = hasher.finish();

        SerializedThreadTaskTimings {
            thread_name: timings.thread_name,
            thread_id,
            timings: serialized_timings,
        }
    }
}

#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct ThreadTimingsDelta {
    /// Hashed thread id
    pub thread_id: u64,
    /// Thread name, if known
    pub thread_name: Option<String>,
    /// New timings since the last call. If the circular buffer wrapped around
    /// since the previous poll, some entries may have been lost.
    pub new_timings: Vec<SerializedTaskTiming>,
}

/// Tracks which timing events have already been seen so that callers can request only unseen events.
#[doc(hidden)]
pub struct ProfilingCollector {
    startup_time: Instant,
    cursors: HashMap<ThreadId, u64>,
}

impl ProfilingCollector {
    pub fn new(startup_time: Instant) -> Self {
        Self {
            startup_time,
            cursors: HashMap::default(),
        }
    }

    pub fn startup_time(&self) -> Instant {
        self.startup_time
    }

    pub fn collect_unseen(
        &mut self,
        all_timings: Vec<ThreadTaskTimings>,
    ) -> Vec<ThreadTimingsDelta> {
        let mut deltas = Vec::with_capacity(all_timings.len());

        for thread in all_timings {
            let mut hasher = DefaultHasher::new();
            thread.thread_id.hash(&mut hasher);
            let hashed_id = hasher.finish();

            let prev_cursor = self.cursors.get(&thread.thread_id).copied().unwrap_or(0);
            let buffer_len = thread.timings.len() as u64;
            let buffer_start = thread.total_pushed.saturating_sub(buffer_len);

            let mut slice = if prev_cursor < buffer_start {
                // Cursor fell behind the buffer — some entries were evicted.
                // Return everything still in the buffer.
                thread.timings.as_slice()
            } else {
                let skip = (prev_cursor - buffer_start) as usize;
                &thread.timings[skip.min(thread.timings.len())..]
            };

            let cursor_advance = thread.total_pushed;
            self.cursors.insert(thread.thread_id, cursor_advance);

            if slice.is_empty() {
                continue;
            }

            let new_timings = SerializedTaskTiming::convert(self.startup_time, slice);

            deltas.push(ThreadTimingsDelta {
                thread_id: hashed_id,
                thread_name: thread.thread_name,
                new_timings,
            });
        }

        deltas
    }

    pub fn reset(&mut self) {
        self.cursors.clear();
    }
}

// Allow 16MiB of task timing entries.
// VecDeque grows by doubling its capacity when full, so keep this a power of 2 to avoid wasting
// memory.
#[cfg(feature = "profiler")]
const MAX_TASK_TIMINGS: usize = (16 * 1024 * 1024) / core::mem::size_of::<TaskTiming>();

#[doc(hidden)]
pub(crate) type TaskTimings = VecDeque<TaskTiming>;

#[doc(hidden)]
pub type GuardedTaskTimings = spin::Mutex<ThreadTimings>;

#[doc(hidden)]
pub struct GlobalThreadTimings {
    pub thread_id: ThreadId,
    pub timings: std::sync::Weak<GuardedTaskTimings>,
}

#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct TaskStatistics {
    pub poll_time_to_beat: Duration,
    pub runtime_to_beat: Duration,
    pub longest_poll_times: [TaskTiming; 5],
    pub longest_runtimes: [TaskTiming; 5],
}

impl std::fmt::Display for TaskStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Tasks that blocked the longest before yielding\n")?;
        for timing in self.longest_poll_times {
            f.write_fmt(format_args!(
                "{:<20} - {}:{}\n",
                format!("{:?}", timing.poll_duration()),
                timing.location.file(),
                timing.location.column()
            ))?;
        }
        f.write_str("Tasks that ran the longest\n")?;
        for timing in self.longest_runtimes {
            f.write_fmt(format_args!(
                "{:<20} - {}:{}\n",
                format!("{:?}", timing.since_spawn()),
                timing.location.file(),
                timing.location.column()
            ))?;
        }
        Ok(())
    }
}

impl Default for TaskStatistics {
    fn default() -> Self {
        Self {
            // Do not track polls that are not problematic
            // this keeps more calls on the fast path
            poll_time_to_beat: Duration::from_micros(100),
            runtime_to_beat: Duration::from_micros(100),
            longest_poll_times: [TaskTiming::placeholder(); 5],
            longest_runtimes: [TaskTiming::placeholder(); 5],
        }
    }
}

impl TaskStatistics {
    #[inline(always)]
    fn add_yield_timing(&mut self, task: TaskTiming) {
        let yielded_after = task.poll_duration();
        if yielded_after >= self.poll_time_to_beat {
            std::hint::cold_path(); // most tasks are not the worst, optimize for that
            let to_replace = self
                .longest_poll_times
                .iter()
                .position_min_by_key(|task| task.since_spawn())
                .expect("guarded by the comparison with nth_longest_yield_time");
            self.longest_poll_times[to_replace] = task;

            self.poll_time_to_beat = self
                .longest_poll_times
                .iter()
                .map(|task| task.since_spawn())
                .min()
                .expect("never empty");
        }
    }

    #[inline(always)]
    fn add_runtime(&mut self, task: TaskTiming) {
        let runtime = task.since_spawn();
        if runtime >= self.runtime_to_beat {
            std::hint::cold_path(); // most tasks are not the worst, optimize for that
            let to_replace = self
                .longest_runtimes
                .iter()
                .position_min_by_key(|task| task.since_spawn())
                .expect("guarded by the comparison with nth_longest_yield_time");
            self.longest_runtimes[to_replace] = task;

            self.runtime_to_beat = self
                .longest_runtimes
                .iter()
                .map(|task| task.since_spawn())
                .min()
                .expect("never empty");
        }
    }
}

#[doc(hidden)]
pub static GLOBAL_THREAD_TIMINGS: spin::Mutex<Vec<GlobalThreadTimings>> =
    spin::Mutex::new(Vec::new());

thread_local! {
    #[doc(hidden)]
    pub static THREAD_TIMINGS: LazyCell<Arc<GuardedTaskTimings>> = LazyCell::new(|| {
        let current_thread = std::thread::current();
        let thread_name = current_thread.name();
        let thread_id = current_thread.id();
        let timings = ThreadTimings::new(thread_name.map(|e| e.to_string()), thread_id);
        let timings = Arc::new(spin::Mutex::new(timings));

        {
            let timings = Arc::downgrade(&timings);
            let global_timings = GlobalThreadTimings {
                thread_id: std::thread::current().id(),
                timings,
            };
            GLOBAL_THREAD_TIMINGS.lock().push(global_timings);
        }

        timings
    });
}

#[doc(hidden)]
pub struct ThreadTimings {
    pub thread_name: Option<String>,
    pub thread_id: ThreadId,
    pub timings: TaskTimings,
    pub running: Option<ActiveTiming>,
    pub stats: TaskStatistics,
    pub total_pushed: u64,
}

impl ThreadTimings {
    pub fn new(thread_name: Option<String>, thread_id: ThreadId) -> Self {
        ThreadTimings {
            thread_name,
            thread_id,
            timings: TaskTimings::new(),
            stats: TaskStatistics::default(),
            total_pushed: 0,
            running: None,
        }
    }

    #[cfg(feature = "profiler")]
    pub fn update_running_task(
        &mut self,
        spawned: SpawnTime,
        location: &'static std::panic::Location<'_>,
    ) {
        let start = Instant::now();
        self.running = Some(ActiveTiming {
            spawned,
            location,
            start,
        });
    }
    #[cfg(not(feature = "profiler"))]
    pub fn update_running_task(&mut self, _: SpawnTime, _: &'static std::panic::Location<'_>) {}

    #[cfg(feature = "profiler")]
    pub fn save_task_timing(&mut self, ended: YieldTime) {
        let ActiveTiming {
            location,
            start,
            spawned,
        } = self
            .running
            .take()
            .expect("this function is only ever called after register_task_start");

        let timing = TaskTiming {
            location,
            spawned,
            start,
            end: ended,
        };
        self.stats.add_yield_timing(timing);
        self.stats.add_runtime(timing);

        if trace_enabled() {
            std::hint::cold_path(); // optimize for when the profiling is off
            if self.timings.len() >= MAX_TASK_TIMINGS {
                self.timings.pop_front();
            }
            self.timings.push_back(timing);
            self.total_pushed += 1;
        }
    }
    #[cfg(not(feature = "profiler"))]
    pub fn save_task_timing(&mut self, _: YieldTime) {}

    // Running tasks are included in the reliability trace, which is written
    // whenever the foreground executor makes no progress for > n seconds
    pub fn get_thread_task_timings(&self, includes: TasksIncluded) -> ThreadTaskTimings {
        ThreadTaskTimings {
            thread_name: self.thread_name.clone(),
            thread_id: self.thread_id,
            timings: self
                .timings
                .iter()
                .cloned()
                .chain(
                    self.running
                        .filter(|_| matches!(includes, TasksIncluded::CompletedAndRunning))
                        .map(|running| TaskTiming {
                            spawned: running.spawned,
                            location: running.location,
                            start: running.start,
                            end: YieldTime(Instant::now()),
                        }),
                )
                .collect(),
            stats: self.stats.clone(),
            total_pushed: self.total_pushed,
        }
    }
}

impl Drop for ThreadTimings {
    fn drop(&mut self) {
        let mut thread_timings = GLOBAL_THREAD_TIMINGS.lock();

        let Some((index, _)) = thread_timings
            .iter()
            .enumerate()
            .find(|(_, t)| t.thread_id == self.thread_id)
        else {
            return;
        };
        thread_timings.swap_remove(index);
    }
}

#[doc(hidden)]
pub fn update_running_task(spawned: SpawnTime, location: &'static std::panic::Location<'_>) {
    THREAD_TIMINGS.with(|timings| {
        timings.lock().update_running_task(spawned, location);
    });
}

#[doc(hidden)]
pub fn save_task_timing() {
    let yielded_at = YieldTime(Instant::now());
    THREAD_TIMINGS.with(|timings| {
        timings.lock().save_task_timing(yielded_at);
    });
}

#[doc(hidden)]
pub fn get_current_thread_task_timings(include_running: TasksIncluded) -> ThreadTaskTimings {
    THREAD_TIMINGS.with(|timings| timings.lock().get_thread_task_timings(include_running))
}

static PROFILER_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enables or disables task timing trace collection at runtime.
///
/// When transitioning from enabled to disabled, `add_task_timing` becomes a
/// cheaper since only cheap statistics are gathered. The existing per-thread
/// buffers for traces are cleared so stale data isn't reported after a later
/// re-enable. Calls with the current value are a no-op.
pub fn set_trace_enabled(enabled: bool) -> bool {
    if PROFILER_ENABLED.swap(enabled, Ordering::AcqRel) == enabled {
        return false;
    }

    if !enabled {
        for global in GLOBAL_THREAD_TIMINGS.lock().iter() {
            if let Some(timings) = global.timings.upgrade() {
                let mut timings = timings.lock();
                timings.timings.clear();
                timings.timings.shrink_to_fit();
                timings.total_pushed = 0;
            }
        }
    }
    true
}

/// Returns whether task timing tracing is enabled.
pub fn trace_enabled() -> bool {
    PROFILER_ENABLED.load(Ordering::Relaxed)
}

/// Timing for a single drawn window frame.
#[derive(Debug, Copy, Clone)]
pub struct FrameTiming {
    /// The window that was drawn.
    pub window_id: WindowId,
    /// When the frame first became dirty (its first invalidation). `None` if
    /// frame tracing was not yet enabled when the invalidation occurred.
    pub dirty_at: Option<Instant>,
    /// Number of invalidations coalesced into this frame.
    pub invalidations: u64,
    /// When `Window::draw` started.
    pub draw_start: Instant,
    /// When `Window::draw` finished.
    pub draw_end: Instant,
}

/// Whether a presented frame rebuilt the retained scene first.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PresentationKind {
    /// The window ran `Window::draw` before presenting.
    Drawn,
    /// The window presented its previously drawn scene without rebuilding it.
    Retained,
}

/// CPU preparation timing for one window presentation.
#[derive(Debug, Copy, Clone)]
pub struct PresentationTiming {
    /// The window that was presented.
    pub window_id: WindowId,
    /// Whether this presentation rebuilt or reused the retained scene.
    pub kind: PresentationKind,
    /// Whether a retained motion deadline requested this presentation.
    pub motion_requested: bool,
    /// When frame preparation started on the window thread.
    pub preparation_start: Instant,
    /// When drawing/submission returned on the window thread.
    pub preparation_end: Instant,
}

/// Renderer work selected for one presentation.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RendererFrameOrigin {
    /// A newly built scene took the ordinary full renderer path.
    Full,
    /// A retained scene took the ordinary full renderer path.
    RetainedFull,
    /// A retained scene used bounded motion damage.
    RetainedDamage,
}

/// Swapchain API used for a renderer submission.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RendererPresentKind {
    /// Ordinary full `Present`.
    Present,
    /// `Present1` with a dirty rectangle.
    Present1,
}

/// Why a retained motion request used the safe full-render fallback.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RendererFallbackReason {
    /// No fallback occurred.
    None,
    /// The experiment is disabled (the production default).
    ExperimentDisabled,
    /// This presentation was not requested solely by retained motion.
    NotMotionOnly,
    /// The window uses a background mode outside the bounded experiment.
    UnsupportedBackground,
    /// Scene structure is outside the bounded eligibility contract.
    Scene(RetainedMotionDamageFallback),
    /// Not every flip-chain buffer contains the current full scene yet.
    HistoryUncertain,
    /// The coherent underlap snapshot was unavailable.
    MissingUnderlap,
    /// Preparing the underlap snapshot failed; this frame still used the
    /// ordinary full presentation path.
    UnderlapCaptureFailed,
    /// The bounded damage path failed before presentation; the same frame was
    /// replayed through the ordinary full renderer.
    DamagePathFailed,
}

/// Primitive counts used by renderer upload and scene diagnostics.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
#[expect(missing_docs)]
pub struct RendererPrimitiveCounts {
    pub shadows: u32,
    pub quads: u32,
    pub effects: u32,
    pub paths: u32,
    pub underlines: u32,
    pub monochrome_sprites: u32,
    pub subpixel_sprites: u32,
    pub polychrome_sprites: u32,
    pub surfaces: u32,
}

impl RendererPrimitiveCounts {
    /// Returns the total number of recorded primitive instances.
    pub fn total(self) -> u32 {
        self.shadows
            .saturating_add(self.quads)
            .saturating_add(self.effects)
            .saturating_add(self.paths)
            .saturating_add(self.underlines)
            .saturating_add(self.monochrome_sprites)
            .saturating_add(self.subpixel_sprites)
            .saturating_add(self.polychrome_sprites)
            .saturating_add(self.surfaces)
    }
}

/// Renderer-boundary diagnostics for one submitted frame.
#[derive(Debug, Copy, Clone)]
#[expect(missing_docs)]
pub struct RendererFrameTiming {
    /// Static renderer identifier, for example `directx11`.
    pub backend: &'static str,
    pub origin: RendererFrameOrigin,
    pub scene: RendererPrimitiveCounts,
    pub uploads: RendererPrimitiveCounts,
    pub uploaded_bytes: u64,
    pub draw_calls: u32,
    pub effect_draw_calls: u32,
    /// `[left, top, right, bottom]` in physical pixels.
    pub damage_rect: Option<[i32; 4]>,
    pub damaged_pixels: u64,
    pub fallback: RendererFallbackReason,
    pub present: RendererPresentKind,
    pub preparation_start: Instant,
    pub preparation_end: Instant,
    /// Best-effort GPU duration. `None` when the backend cannot collect it
    /// without synchronously stalling the render thread.
    pub gpu_duration: Option<Duration>,
}

impl RendererFrameTiming {
    /// CPU time spent inside the renderer boundary.
    pub fn preparation_duration(&self) -> Duration {
        self.preparation_end.duration_since(self.preparation_start)
    }
}

impl PresentationTiming {
    /// CPU time spent preparing and submitting this presentation.
    pub fn preparation_duration(&self) -> Duration {
        self.preparation_end.duration_since(self.preparation_start)
    }
}

impl FrameTiming {
    /// Time spent inside `Window::draw`.
    pub fn draw_duration(&self) -> Duration {
        self.draw_end.duration_since(self.draw_start)
    }

    /// Time from the frame's first invalidation to the end of its draw, if the
    /// first invalidation was observed.
    pub fn dirty_to_draw_duration(&self) -> Option<Duration> {
        self.dirty_at
            .map(|dirty_at| self.draw_end.duration_since(dirty_at))
    }
}

// Allow 16MiB of frame timing entries.
const MAX_FRAME_TIMINGS: usize = (16 * 1024 * 1024) / core::mem::size_of::<FrameTiming>();

struct FrameTimings {
    timings: VecDeque<FrameTiming>,
    total_pushed: u64,
}

static FRAME_TIMINGS: spin::Mutex<FrameTimings> = spin::Mutex::new(FrameTimings {
    timings: VecDeque::new(),
    total_pushed: 0,
});

const MAX_PRESENTATION_TIMINGS: usize =
    (16 * 1024 * 1024) / core::mem::size_of::<PresentationTiming>();

struct PresentationTimings {
    timings: VecDeque<PresentationTiming>,
    total_pushed: u64,
}

const MAX_RENDERER_FRAME_TIMINGS: usize =
    (16 * 1024 * 1024) / core::mem::size_of::<RendererFrameTiming>();

struct RendererFrameTimings {
    timings: VecDeque<RendererFrameTiming>,
    total_pushed: u64,
}

static RENDERER_FRAME_TIMINGS: spin::Mutex<RendererFrameTimings> =
    spin::Mutex::new(RendererFrameTimings {
        timings: VecDeque::new(),
        total_pushed: 0,
    });

static PRESENTATION_TIMINGS: spin::Mutex<PresentationTimings> =
    spin::Mutex::new(PresentationTimings {
        timings: VecDeque::new(),
        total_pushed: 0,
    });

static FRAME_TRACE_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enables or disables frame timing collection at runtime.
///
/// When transitioning from enabled to disabled, the buffered frame timings are
/// cleared so stale data isn't reported after a later re-enable. Returns false
/// if the value was unchanged.
pub fn set_frame_trace_enabled(enabled: bool) -> bool {
    if FRAME_TRACE_ENABLED.swap(enabled, Ordering::AcqRel) == enabled {
        return false;
    }

    if !enabled {
        let mut frames = FRAME_TIMINGS.lock();
        frames.timings.clear();
        frames.timings.shrink_to_fit();
        frames.total_pushed = 0;
        let mut presentations = PRESENTATION_TIMINGS.lock();
        presentations.timings.clear();
        presentations.timings.shrink_to_fit();
        presentations.total_pushed = 0;
        let mut renderer_frames = RENDERER_FRAME_TIMINGS.lock();
        renderer_frames.timings.clear();
        renderer_frames.timings.shrink_to_fit();
        renderer_frames.total_pushed = 0;
    }
    true
}

/// Returns whether frame timing collection is enabled.
pub fn frame_trace_enabled() -> bool {
    FRAME_TRACE_ENABLED.load(Ordering::Relaxed)
}

/// Returns a timestamp in the same clock domain as renderer diagnostics.
#[doc(hidden)]
pub fn renderer_frame_timestamp() -> Instant {
    Instant::now()
}

/// Records the timing of a drawn window frame.
///
/// No-op unless frame tracing is enabled via [`set_frame_trace_enabled`].
pub fn record_frame_timing(timing: FrameTiming) {
    if !frame_trace_enabled() {
        return;
    }
    std::hint::cold_path(); // optimize for when profiling is off

    let mut frames = FRAME_TIMINGS.lock();
    if frames.timings.len() >= MAX_FRAME_TIMINGS {
        frames.timings.pop_front();
    }
    frames.timings.push_back(timing);
    frames.total_pushed += 1;
}

/// Records the CPU preparation time of a presented window frame.
///
/// No-op unless frame tracing is enabled via [`set_frame_trace_enabled`].
pub fn record_presentation_timing(timing: PresentationTiming) {
    if !frame_trace_enabled() {
        return;
    }
    std::hint::cold_path();

    let mut presentations = PRESENTATION_TIMINGS.lock();
    if presentations.timings.len() >= MAX_PRESENTATION_TIMINGS {
        presentations.timings.pop_front();
    }
    presentations.timings.push_back(timing);
    presentations.total_pushed += 1;
}

/// Records renderer-boundary work for a submitted frame.
///
/// No-op unless frame tracing is enabled via [`set_frame_trace_enabled`].
pub fn record_renderer_frame_timing(timing: RendererFrameTiming) {
    if !frame_trace_enabled() {
        return;
    }
    std::hint::cold_path();

    let mut frames = RENDERER_FRAME_TIMINGS.lock();
    if frames.timings.len() >= MAX_RENDERER_FRAME_TIMINGS {
        frames.timings.pop_front();
    }
    frames.timings.push_back(timing);
    frames.total_pushed += 1;
}

/// Drains frame timings recorded after this collector was created, tracking a
/// cursor so each call to [`Self::collect_unseen`] returns only new entries.
pub struct FrameTimingCollector {
    cursor: u64,
}

impl Default for FrameTimingCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameTimingCollector {
    /// Creates a collector that only sees frames recorded from this point on.
    pub fn new() -> Self {
        Self {
            cursor: FRAME_TIMINGS.lock().total_pushed,
        }
    }

    /// Returns frame timings recorded since the previous call (or since the
    /// collector was created). If the ring buffer wrapped around since the
    /// previous poll, the evicted entries are lost.
    pub fn collect_unseen(&mut self) -> Vec<FrameTiming> {
        let frames = FRAME_TIMINGS.lock();
        let buffer_len = frames.timings.len() as u64;
        let buffer_start = frames.total_pushed.saturating_sub(buffer_len);
        let skip = self.cursor.saturating_sub(buffer_start) as usize;
        let unseen = frames
            .timings
            .iter()
            .skip(skip.min(frames.timings.len()))
            .copied()
            .collect();
        self.cursor = frames.total_pushed;
        unseen
    }
}

/// Collects presentation timings recorded after the collector was created.
pub struct PresentationTimingCollector {
    cursor: u64,
}

impl Default for PresentationTimingCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl PresentationTimingCollector {
    /// Creates a collector that only sees presentations recorded from this point on.
    pub fn new() -> Self {
        Self {
            cursor: PRESENTATION_TIMINGS.lock().total_pushed,
        }
    }

    /// Returns presentation timings recorded since the previous call.
    pub fn collect_unseen(&mut self) -> Vec<PresentationTiming> {
        let presentations = PRESENTATION_TIMINGS.lock();
        let buffer_len = presentations.timings.len() as u64;
        let buffer_start = presentations.total_pushed.saturating_sub(buffer_len);
        let skip = self.cursor.saturating_sub(buffer_start) as usize;
        let unseen = presentations
            .timings
            .iter()
            .skip(skip.min(presentations.timings.len()))
            .copied()
            .collect();
        self.cursor = presentations.total_pushed;
        unseen
    }
}

/// Collects renderer-boundary timings recorded after construction.
pub struct RendererFrameTimingCollector {
    cursor: u64,
}

impl Default for RendererFrameTimingCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl RendererFrameTimingCollector {
    /// Creates a collector that only sees later renderer submissions.
    pub fn new() -> Self {
        Self {
            cursor: RENDERER_FRAME_TIMINGS.lock().total_pushed,
        }
    }

    /// Returns renderer submissions recorded since the previous call.
    pub fn collect_unseen(&mut self) -> Vec<RendererFrameTiming> {
        let frames = RENDERER_FRAME_TIMINGS.lock();
        let buffer_len = frames.timings.len() as u64;
        let buffer_start = frames.total_pushed.saturating_sub(buffer_len);
        let skip = self.cursor.saturating_sub(buffer_start) as usize;
        let unseen = frames
            .timings
            .iter()
            .skip(skip.min(frames.timings.len()))
            .copied()
            .collect();
        self.cursor = frames.total_pushed;
        unseen
    }
}
