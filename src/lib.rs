//
// Copyright (c) 2025 Łukasz Szpakowski
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
use std::sync::Arc;
use std::sync::Mutex;
use std::result;

#[cfg(feature = "opencl")]
pub mod opencl;

pub trait Backend
{
}

pub enum Error
{
    Mutex,
    #[cfg(feature = "opencl")]
    OpenCl(opencl::ClError),
    Compilation(String),
}

pub type Result<T> = result::Result<T, Error>;

static mut DEFAULT_BACKEND: Mutex<Option<Arc<dyn Backend>>> = Mutex::new(None);
