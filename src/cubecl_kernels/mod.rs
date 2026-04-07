//! CubeCL custom kernels for Qwen3.5 inference optimization.
//!
//! This module contains custom GPU kernels written using CubeCL for
//! optimizing decode performance in the Qwen3.5 model.

// Pre-existing stubs (broken, kept for reference)
// pub mod attention;
// pub mod attention_v2;
pub mod flash_decode;

#[cfg(feature = "wgpu")]
pub mod bridge;