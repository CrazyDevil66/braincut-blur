pub mod jobs;
pub mod models;
pub mod system;

use std::{sync::Arc, time::{SystemTime, UNIX_EPOCH}};

use axum::{http::StatusCode, Json};
use serde_json::{json, Value};

use crate::{
    blur::CancelFlag,
    config::Config,
    models::InstallMap,
    state::SharedState,
};

#[derive(Clone)]
pub struct App {
    pub status: SharedState,
    pub cancel: CancelFlag,
    pub install_progress: InstallMap,
    pub cfg: Arc<Config>,
}

pub fn json_ok(v: Value) -> (StatusCode, Json<Value>) {
    (StatusCode::OK, Json(v))
}

pub fn json_err(code: StatusCode, msg: &str) -> (StatusCode, Json<Value>) {
    (code, Json(json!({ "error": msg })))
}

pub fn now_unix() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64()
}
