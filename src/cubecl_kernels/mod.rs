//! CubeCL custom kernels for Qwen3.5 inference optimization.
//!
//! This module contains custom GPU kernels written using CubeCL for
//! optimizing decode performance in the Qwen3.5 model.

pub mod attention;
pub mod attention_v2;