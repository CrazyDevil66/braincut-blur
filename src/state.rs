use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, serde::Serialize)]
pub struct AppStatus {
    #[serde(skip)]
    pub preview_jpeg: Vec<u8>,
    pub state: String,
    pub current: u32,
    pub total: u32,
    pub name: String,
    pub error: String,
    pub frame_current: u64,
    pub frame_total: u64,
    pub frame_pct: u8,
    pub eta_seconds: u64,
    pub face_count: u64,
    pub plate_count: u64,
    pub hw_nvdec: bool,
    pub hw_nvenc: bool,
    pub hw_trt: bool,
    pub sub_state: String,
    pub out_name: String,
    pub started_at: String,
    pub started_at_ts: f64,
    pub log: VecDeque<String>,
}

impl Default for AppStatus {
    fn default() -> Self {
        Self {
            state: "idle".into(),
            current: 0,
            total: 0,
            name: String::new(),
            error: String::new(),
            frame_current: 0,
            frame_total: 0,
            frame_pct: 0,
            eta_seconds: 0,
            face_count: 0,
            plate_count: 0,
            hw_nvdec: false,
            hw_nvenc: false,
            hw_trt: false,
            sub_state: String::new(),
            out_name: String::new(),
            started_at: String::new(),
            started_at_ts: 0.0,
            log: VecDeque::with_capacity(200),
            preview_jpeg: Vec::new(),
        }
    }
}

pub type SharedState = Arc<Mutex<AppStatus>>;

pub fn new_state() -> SharedState {
    Arc::new(Mutex::new(AppStatus::default()))
}

pub fn log(state: &SharedState, msg: &str) {
    let ts = chrono::Local::now().format("%H:%M:%S").to_string();
    let line = format!("[{ts}] {msg}");
    tracing::info!("{}", msg);
    let mut s = state.lock().unwrap();
    s.log.push_back(line);
    if s.log.len() > 200 {
        s.log.pop_front();
    }
}

pub fn set_idle(state: &SharedState) {
    let mut s = state.lock().unwrap();
    s.state = "idle".into();
    s.name = String::new();
    s.frame_current = 0;
    s.frame_total = 0;
    s.frame_pct = 0;
    s.eta_seconds = 0;
}
