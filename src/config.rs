use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    pub n8n_ip: String,
    pub n8n_port: u16,
    pub completion_webhook: String,
    pub media_host_path: String,
    pub container_root: String,
    pub models_path: PathBuf,
    pub centerface_model: PathBuf,
    pub detection_interval: u32,
    pub plate_conf_thresh: f32,
    pub face_conf_thresh: f32,
    pub plate_grid: i32,
    pub frame_buffer: usize,
}

impl Config {
    pub fn from_env() -> Self {
        let n8n_ip = std::env::var("N8N_SERVER_IP").unwrap_or_default();
        let n8n_port: u16 = std::env::var("N8N_SERVER_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5678);

        let completion_webhook = if n8n_ip.is_empty() {
            String::new()
        } else {
            format!("http://{}:{}/webhook/blur-complete", n8n_ip, n8n_port)
        };

        let media_host_path = std::env::var("MEDIA_HOST_PATH")
            .unwrap_or_else(|_| "/mnt/user/n8n_automation/BrainCut".into());
        let container_root = std::env::var("CONTAINER_ROOT")
            .unwrap_or_else(|_| "/data".into());

        let models_path = PathBuf::from(
            std::env::var("MODELS_PATH").unwrap_or_else(|_| "/app/.cache/models".into()),
        );

        // CenterFace ONNX: deface installs it during Docker build
        let centerface_model = PathBuf::from(
            std::env::var("CENTERFACE_MODEL").unwrap_or_else(|_| {
                // deface stores model relative to its package
                "/app/deface_cache/centerface.onnx".into()
            }),
        );

        Self {
            n8n_ip,
            n8n_port,
            completion_webhook,
            media_host_path,
            container_root,
            models_path,
            centerface_model,
            detection_interval: 4,
            plate_conf_thresh: 0.45,
            face_conf_thresh: 0.55,
            plate_grid: 20,
            frame_buffer: 32,
        }
    }
}
