//
// Copyright (c) 2025 Łukasz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
use crate::Backend;
use crate::Result;

pub use opencl3::context::Context;
pub use opencl3::device::Device;
pub use opencl3::error_codes::ClError;
pub use opencl3::platform::Platform;
pub use opencl3::platform::get_platforms;

pub struct ClBackend
{
}

impl ClBackend
{
    pub fn new() -> Result<ClBackend>
    { Ok(ClBackend {}) }

    
    pub fn new_with_context(context: Context) -> Result<ClBackend>
    { Ok(ClBackend {}) }
}

impl Backend for ClBackend
{
}
