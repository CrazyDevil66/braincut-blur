use anyhow::{Context, Result};
use ort::{
    execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider},
    session::Session,
};
use std::path::Path;

pub fn build_session(model_path: &Path, use_trt: bool) -> Result<Session> {
    let mut b = Session::builder().context("ORT SessionBuilder")?;
    if use_trt {
        let cache = std::env::var("MODELS_PATH").unwrap_or_else(|_| "/app/.cache/models".into());
        b = b
            .with_execution_providers([
                TensorRTExecutionProvider::default()
                    .with_engine_cache(true)
                    .with_engine_cache_path(&cache)
                    .with_fp16(true)
                    .build(),
                CUDAExecutionProvider::default().build(),
            ])
            .map_err(|e| anyhow::anyhow!("set TRT+CUDA providers: {e:?}"))?;
    } else {
        b = b
            .with_execution_providers([CUDAExecutionProvider::default().build()])
            .map_err(|e| anyhow::anyhow!("set CUDA provider: {e:?}"))?;
    }
    b.commit_from_file(model_path).context("load ONNX model")
}

pub fn check_nvenc() -> bool {
    std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("h264_nvenc"))
        .unwrap_or(false)
}

pub fn check_nvdec() -> bool {
    std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-hwaccels"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("cuda"))
        .unwrap_or(false)
}
