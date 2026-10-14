//! The Coachwhip engine: runs mixture-of-experts models bigger than the memory on Apple silicon.
//! The experts stream from the SSD into a small bank on the GPU; the model families sit on top.

pub mod experts;
pub mod gpu_sync;
pub mod delta_kernel;
pub mod attn_kernel;
pub mod fuse_kernel;
pub mod requant;
pub mod model;
pub mod model_next;
pub mod model_gptoss;
mod mv_id;
