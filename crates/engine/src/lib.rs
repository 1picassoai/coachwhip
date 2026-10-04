//! The Coachwhip engine: runs mixture-of-experts models bigger than the memory on Apple silicon.
//! The experts stream from the SSD into a small bank on the GPU; the model families sit on top.

pub mod experts;
pub mod model;
pub mod model_next;
mod mv_id;
