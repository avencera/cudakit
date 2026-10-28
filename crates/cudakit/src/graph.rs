use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode, cuGraphGetNodes};
use cudarc::driver::{CudaContext, CudaGraph};

use crate::{Error, Gpu};

/// A recorded and instantiated sequence of work on one [`Gpu`]'s stream
///
/// The graph holds raw pointers to every buffer the captured work touched. Creating one
/// is `unsafe` ([`Gpu::capture`]) because nothing ties those buffers' lifetimes to it:
/// they must outlive it and keep their addresses, so allocate them once and refill them
/// instead of reallocating. Replays record no cudarc events, which is safe as long as
/// the buffers are used only on this GPU's stream
pub struct Graph(CudaGraph);

impl std::fmt::Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Graph")
    }
}

impl Graph {
    /// Queues one replay on the stream it was captured from
    pub fn launch(&self) -> Result<(), Error> {
        self.0.launch()?;
        Ok(())
    }

    /// The cudarc graph, for driver calls cudakit does not wrap
    pub fn as_cuda(&self) -> &CudaGraph {
        &self.0
    }
}

impl Gpu {
    /// Records everything `enqueue` queues on this GPU's stream, or `None` when it
    /// queued nothing
    ///
    /// Waits for prior work first, so nothing queued before the capture is pending when
    /// it starts; finish lazy library setup and weight uploads before calling this.
    /// cudarc's per-buffer event tracking is paused during the capture, since waiting
    /// on an event recorded before the capture fails with
    /// `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`. The capture always ends, so a failed
    /// `enqueue` leaves the stream usable
    ///
    /// # Safety
    ///
    /// Every buffer `enqueue` touches must stay allocated at the same address until the
    /// returned graph is dropped, and must be used only on this GPU's stream while the
    /// graph exists; otherwise a replay reads or writes freed device memory. Keep the
    /// graph beside the buffers' owner and drop it first
    pub unsafe fn capture<E: From<Error>>(
        &self,
        enqueue: impl FnOnce() -> Result<(), E>,
    ) -> Result<Option<Graph>, E> {
        let stream = self.stream();
        stream.synchronize().map_err(Error::from)?;
        // SAFETY: every buffer the capture touches belongs to this GPU and is used on
        // its one stream, after the synchronize above, so stream order alone keeps
        // them consistent while tracking is off
        let _tracking = unsafe { EventTrackingPause::new(self.context()) };
        stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(Error::from)?;
        let enqueued = enqueue();
        // always end the capture, so a failed enqueue leaves the stream usable
        let graph = stream.end_capture(CUgraphInstantiate_flags(0));
        enqueued?;

        let Some(graph) = graph.map_err(Error::from)? else {
            return Ok(None);
        };
        // the driver ends an empty capture with a zero-node graph rather than none
        Ok((node_count(&graph)? > 0).then_some(Graph(graph)))
    }

    /// Like [`Self::capture`], but an empty capture is [`Error::EmptyCapture`]
    ///
    /// # Safety
    ///
    /// The same contract as [`Self::capture`]
    pub unsafe fn capture_nonempty<E: From<Error>>(
        &self,
        enqueue: impl FnOnce() -> Result<(), E>,
    ) -> Result<Graph, E> {
        // SAFETY: the caller upholds the buffer contract for the returned graph
        Ok(unsafe { self.capture(enqueue) }?.ok_or(Error::EmptyCapture)?)
    }
}

/// Nodes in the captured graph's top level
fn node_count(graph: &CudaGraph) -> Result<usize, Error> {
    let mut count = 0;
    // SAFETY: the graph is live for the call, and a null node array asks only for the
    // count, which the driver writes to `count`
    unsafe { cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) }.result()?;
    Ok(count)
}

/// Turns off cudarc's per-buffer event tracking on a context until dropped
///
/// cudarc makes every buffer access wait on the event of the buffer's last write and
/// record a new one. Inside a stream capture, waiting on an event recorded before the
/// capture fails with `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`. The flag belongs to the
/// context object and is read at each access
pub struct EventTrackingPause<'a> {
    context: &'a CudaContext,
    was_tracking: bool,
}

impl<'a> EventTrackingPause<'a> {
    /// Pauses tracking on `context`
    ///
    /// # Safety
    ///
    /// While the pause lives, every buffer of `context` that is touched must be used on
    /// one stream only, with no pending work from other streams, so stream order alone
    /// provides the dependencies the events would have
    pub unsafe fn new(context: &'a CudaContext) -> Self {
        let was_tracking = context.is_event_tracking();
        // SAFETY: the caller upholds the single-stream contract above
        unsafe { context.disable_event_tracking() };
        Self {
            context,
            was_tracking,
        }
    }
}

impl Drop for EventTrackingPause<'_> {
    fn drop(&mut self) {
        if self.was_tracking {
            // SAFETY: restores the state from before the pause
            unsafe { self.context.enable_event_tracking() };
        }
    }
}
