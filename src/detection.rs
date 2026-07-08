use anyhow::{bail, Context, Result};
use ort::{
    execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider},
    session::Session,
    value::Tensor,
};
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub x: usize,
    pub y: usize,
    pub x2: usize,
    pub y2: usize,
}

// ── Session factory ──────────────────────────────────────────────────────────

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

// ── Detectors bundle ─────────────────────────────────────────────────────────

pub struct Detectors {
    pub centerface: Option<Session>,
    pub face_yolo: Option<Session>,
    pub face_scrfd: Option<Session>,
    pub plate_yolo: Option<Session>,
    pub in_h: usize,
    pub in_w: usize,
    pub face_conf_thresh: f32,
    // Pre-allocated resize scratch buffers — eliminates ~4 MB malloc/free per Detection-Frame.
    // ORT still needs an owned Vec for tensor data, so only the resize step is saved here.
    cf_resize_buf: Vec<u8>,   // cf_pad_h * cf_pad_w * 3
    yolo_resize_buf: Vec<u8>, // 640 * 640 * 3
}

impl Detectors {
    pub fn load(
        centerface_path: Option<&Path>,
        face_yolo_path: Option<&Path>,
        face_scrfd_path: Option<&Path>,
        plate_yolo_path: Option<&Path>,
        in_h: usize,
        in_w: usize,
        use_trt: bool,
        face_conf_thresh: f32,
    ) -> Result<Self> {
        let centerface = centerface_path
            .map(|p| build_session(p, false))
            .transpose()?;
        let face_yolo = face_yolo_path
            .map(|p| build_session(p, use_trt))
            .transpose()?;
        let face_scrfd = face_scrfd_path
            .map(|p| build_session(p, false))
            .transpose()?;
        let plate_yolo = plate_yolo_path
            .map(|p| build_session(p, use_trt))
            .transpose()?;

        let cf_pad_h = ((in_h + 31) / 32) * 32;
        let cf_pad_w = ((in_w + 31) / 32) * 32;

        Ok(Self {
            centerface, face_yolo, face_scrfd, plate_yolo, in_h, in_w,
            face_conf_thresh,
            cf_resize_buf: vec![0u8; cf_pad_h * cf_pad_w * 3],
            yolo_resize_buf: vec![0u8; 640 * 640 * 3],
        })
    }

    pub fn detect_faces(&mut self, frame: &[u8], fw: usize, fh: usize) -> Vec<BBox> {
        let thresh = self.face_conf_thresh;
        let in_h = self.in_h;
        let in_w = self.in_w;
        if self.face_scrfd.is_some() {
            let sess = self.face_scrfd.as_mut().unwrap();
            scrfd_detect(sess, frame, fw, fh, thresh, &mut self.yolo_resize_buf)
                .unwrap_or_default()
                .into_iter()
                .map(|b| expand_bbox(b, fw, fh, 0.10))
                .collect()
        } else if self.face_yolo.is_some() {
            let sess = self.face_yolo.as_mut().unwrap();
            yolo_detect(sess, frame, fw, fh, thresh, false, &mut self.yolo_resize_buf).unwrap_or_default()
        } else if self.centerface.is_some() {
            let sess = self.centerface.as_mut().unwrap();
            match centerface_detect(sess, frame, fw, fh, in_h, in_w, thresh, &mut self.cf_resize_buf) {
                Ok(boxes) => boxes.into_iter().map(|b| expand_bbox(b, fw, fh, 0.15)).collect(),
                Err(e) => {
                    static ERR_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !ERR_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        eprintln!("[CF-ERROR] CenterFace Fehler (einmalig): {e:?}");
                    }
                    vec![]
                }
            }
        } else {
            vec![]
        }
    }

    pub fn detect_plates(&mut self, frame: &[u8], fw: usize, fh: usize, conf: f32) -> Vec<BBox> {
        if self.plate_yolo.is_some() {
            let sess = self.plate_yolo.as_mut().unwrap();
            yolo_detect(sess, frame, fw, fh, conf, true, &mut self.yolo_resize_buf)
                .unwrap_or_default()
                .into_iter()
                .map(|b| expand_bbox(b, fw, fh, 0.15))
                .collect()
        } else {
            vec![]
        }
    }
}

// ── CenterFace ───────────────────────────────────────────────────────────────

pub fn centerface_detect(
    session: &mut Session,
    frame: &[u8],
    frame_w: usize,
    frame_h: usize,
    in_h: usize,
    in_w: usize,
    threshold: f32,
    resize_buf: &mut [u8],
) -> Result<Vec<BBox>> {
    let pad_h = ((in_h + 31) / 32) * 32;
    let pad_w = ((in_w + 31) / 32) * 32;

    resize_bgr_into(frame, frame_w, frame_h, pad_w, pad_h, resize_buf);

    // BGR mean subtraction (ImageNet-BGR: [104, 117, 123]), CHW layout
    let mean = [104.0f32, 117.0, 123.0];
    let mut input = vec![0f32; 3 * pad_h * pad_w];
    for y in 0..pad_h {
        for x in 0..pad_w {
            let s = (y * pad_w + x) * 3;
            for c in 0..3usize {
                input[c * pad_h * pad_w + y * pad_w + x] = resize_buf[s + c] as f32 - mean[c];
            }
        }
    }

    // ort 2.x rc.12: from_array takes (shape, owned_data)
    let tensor = Tensor::from_array(([1i64, 3, pad_h as i64, pad_w as i64], input))?;
    // Named input (CenterFace input node is "input.1") — inputs!{} returns Vec, not Result
    let outputs = session.run(ort::inputs! { "input.1" => tensor })?;

    // try_extract_tensor returns (&Shape, &[T]) in rc.12
    let hm_t = outputs[0].try_extract_tensor::<f32>()?;
    let sc_t = outputs[1].try_extract_tensor::<f32>()?;
    let off_t = outputs[2].try_extract_tensor::<f32>()?;

    let hm_shape = hm_t.0;
    let feat_h = hm_shape[hm_shape.len() - 2] as usize;
    let feat_w = hm_shape[hm_shape.len() - 1] as usize;
    let feat_n = feat_h * feat_w;

    let hm_d: &[f32] = hm_t.1;
    let sc_d: &[f32] = sc_t.1;
    let off_d: &[f32] = off_t.1;

    // Diagnostic: log output shapes and max heatmap score once
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        let max_raw = hm_d.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_sig = sigmoid(max_raw);
        eprintln!("[CF-DIAG] hm_shape={:?} sc_shape={:?} off_shape={:?} max_hm_raw={:.4} max_hm_sig={:.4} threshold={:.2}",
            hm_t.0, sc_t.0, off_t.0, max_raw, max_sig, threshold);
    }

    let scale_x = frame_w as f32 / pad_w as f32;
    let scale_y = frame_h as f32 / pad_h as f32;

    let mut scored: Vec<(BBox, f32)> = Vec::new();
    for cy in 0..feat_h {
        for cx in 0..feat_w {
            let idx = cy * feat_w + cx;
            let score = sigmoid(hm_d[idx]);
            if score < threshold || !is_local_max(hm_d, feat_h, feat_w, cy, cx) {
                continue;
            }
            let oy = off_d[idx];
            let ox = off_d[feat_n + idx];
            let sh = sc_d[idx].exp() * 4.0;
            let sw = sc_d[feat_n + idx].exp() * 4.0;

            let x1_f = (cx as f32 + ox + 0.5) * 4.0 - sw * 0.5;
            let y1_f = (cy as f32 + oy + 0.5) * 4.0 - sh * 0.5;

            let x1 = ((x1_f * scale_x) as isize).max(0) as usize;
            let y1 = ((y1_f * scale_y) as isize).max(0) as usize;
            let x2 = (((x1_f + sw) * scale_x) as usize).min(frame_w);
            let y2 = (((y1_f + sh) * scale_y) as usize).min(frame_h);

            if x2 > x1 && y2 > y1 {
                scored.push((BBox { x: x1, y: y1, x2, y2 }, score));
            }
        }
    }

    // Greedy IoU-NMS: highest-score boxes zuerst behalten
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut suppressed = vec![false; scored.len()];
    let mut boxes = Vec::new();
    for i in 0..scored.len() {
        if suppressed[i] { continue; }
        boxes.push(scored[i].0);
        for j in (i + 1)..scored.len() {
            if suppressed[j] { continue; }
            if bbox_iou(&scored[i].0, &scored[j].0) > 0.3 {
                suppressed[j] = true;
            }
        }
    }
    boxes.truncate(200);
    Ok(boxes)
}

fn bbox_iou(a: &BBox, b: &BBox) -> f32 {
    let ix1 = a.x.max(b.x);
    let iy1 = a.y.max(b.y);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);
    if ix2 <= ix1 || iy2 <= iy1 { return 0.0; }
    let inter = ((ix2 - ix1) * (iy2 - iy1)) as f32;
    let area_a = ((a.x2 - a.x) * (a.y2 - a.y)) as f32;
    let area_b = ((b.x2 - b.x) * (b.y2 - b.y)) as f32;
    inter / (area_a + area_b - inter)
}

fn is_local_max(hm: &[f32], feat_h: usize, feat_w: usize, cy: usize, cx: usize) -> bool {
    let center = hm[cy * feat_w + cx];
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dy == 0 && dx == 0 {
                continue;
            }
            let ny = cy as i32 + dy;
            let nx = cx as i32 + dx;
            if ny < 0 || ny >= feat_h as i32 || nx < 0 || nx >= feat_w as i32 {
                continue;
            }
            if hm[ny as usize * feat_w + nx as usize] >= center {
                return false;
            }
        }
    }
    true
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn expand_bbox(b: BBox, fw: usize, fh: usize, factor: f32) -> BBox {
    let ex = (((b.x2.saturating_sub(b.x)) as f32 * factor) as usize).max(2);
    let ey = (((b.y2.saturating_sub(b.y)) as f32 * factor) as usize).max(2);
    BBox {
        x: b.x.saturating_sub(ex),
        y: b.y.saturating_sub(ey),
        x2: (b.x2 + ex).min(fw.saturating_sub(1)),
        y2: (b.y2 + ey).min(fh.saturating_sub(1)),
    }
}

// ── SCRFD (InsightFace) ───────────────────────────────────────────────────────

fn scrfd_detect(
    session: &mut Session,
    frame: &[u8],
    frame_w: usize,
    frame_h: usize,
    threshold: f32,
    resize_buf: &mut [u8],
) -> Result<Vec<BBox>> {
    let in_h = 640usize;
    let in_w = 640usize;

    resize_bgr_into(frame, frame_w, frame_h, in_w, in_h, resize_buf);

    // CHW, BGR→RGB, /255 (identisch zu YOLO)
    let mut input = vec![0f32; 3 * in_h * in_w];
    for y in 0..in_h {
        for x in 0..in_w {
            let s = (y * in_w + x) * 3;
            input[y * in_w + x] = resize_buf[s + 2] as f32 / 255.0; // R
            input[in_h * in_w + y * in_w + x] = resize_buf[s + 1] as f32 / 255.0; // G
            input[2 * in_h * in_w + y * in_w + x] = resize_buf[s] as f32 / 255.0; // B
        }
    }

    let tensor = Tensor::from_array(([1i64, 3, in_h as i64, in_w as i64], input))?;
    let outputs = session.run(ort::inputs![tensor])?;

    // SCRFD ONNX (buffalo_l / buffalo_sc) gibt pro Stride [scores, boxes] aus,
    // optional mit KPS: dann [scores, boxes, kps] → 6 oder 9 Outputs.
    // Stride-Reihenfolge: 8, 16, 32; 2 Anchors pro Zelle.
    let has_kps = outputs.len() == 9;
    let step = if has_kps { 3 } else { 2 };
    let strides = [8usize, 16, 32];

    let scale_x = frame_w as f32 / in_w as f32;
    let scale_y = frame_h as f32 / in_h as f32;

    let mut candidates: Vec<(BBox, f32)> = Vec::new();

    for (si, &stride) in strides.iter().enumerate() {
        let out_h = in_h / stride;
        let out_w = in_w / stride;
        let n_anchors = out_h * out_w * 2;

        let (_, scores) = outputs[si * step].try_extract_tensor::<f32>()?;
        let (_, boxes) = outputs[si * step + 1].try_extract_tensor::<f32>()?;

        for i in 0..n_anchors {
            let score = scores[i];
            if score < threshold {
                continue;
            }
            // 2 Anchors teilen sich dasselbe Gitterzentrum
            let cell = i / 2;
            let row = cell / out_w;
            let col = cell % out_w;
            let anchor_cx = col as f32 * stride as f32;
            let anchor_cy = row as f32 * stride as f32;

            // Distances [left, top, right, bottom] in Input-Pixeln
            let b = i * 4;
            let x1 = ((anchor_cx - boxes[b]) * scale_x) as isize;
            let y1 = ((anchor_cy - boxes[b + 1]) * scale_y) as isize;
            let x2 = ((anchor_cx + boxes[b + 2]) * scale_x) as isize;
            let y2 = ((anchor_cy + boxes[b + 3]) * scale_y) as isize;

            let x1 = x1.max(0) as usize;
            let y1 = y1.max(0) as usize;
            let x2 = (x2 as usize).min(frame_w);
            let y2 = (y2 as usize).min(frame_h);

            if x2 > x1 && y2 > y1 {
                candidates.push((BBox { x: x1, y: y1, x2, y2 }, score));
            }
        }
    }

    // Greedy NMS (IoU 0.45)
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<BBox> = Vec::new();
    'outer: for (b, _) in &candidates {
        for kb in &kept {
            let ix1 = b.x.max(kb.x);
            let iy1 = b.y.max(kb.y);
            let ix2 = b.x2.min(kb.x2);
            let iy2 = b.y2.min(kb.y2);
            if ix2 > ix1 && iy2 > iy1 {
                let inter = ((ix2 - ix1) * (iy2 - iy1)) as f32;
                let ua = ((b.x2 - b.x) * (b.y2 - b.y)) as f32
                    + ((kb.x2 - kb.x) * (kb.y2 - kb.y)) as f32
                    - inter;
                if ua > 0.0 && inter / ua > 0.45 {
                    continue 'outer;
                }
            }
        }
        kept.push(*b);
    }
    Ok(kept)
}

// ── YOLOv8 ───────────────────────────────────────────────────────────────────

pub fn yolo_detect(
    session: &mut Session,
    frame: &[u8],
    frame_w: usize,
    frame_h: usize,
    conf_thresh: f32,
    aspect_filter: bool,
    resize_buf: &mut [u8],
) -> Result<Vec<BBox>> {
    let in_h = 640usize;
    let in_w = 640usize;

    resize_bgr_into(frame, frame_w, frame_h, in_w, in_h, resize_buf);

    // CHW layout, BGR→RGB, /255
    let mut input = vec![0f32; 3 * in_h * in_w];
    for y in 0..in_h {
        for x in 0..in_w {
            let s = (y * in_w + x) * 3;
            input[0 * in_h * in_w + y * in_w + x] = resize_buf[s + 2] as f32 / 255.0; // R
            input[1 * in_h * in_w + y * in_w + x] = resize_buf[s + 1] as f32 / 255.0; // G
            input[2 * in_h * in_w + y * in_w + x] = resize_buf[s] as f32 / 255.0;     // B
        }
    }

    // Positional (unnamed) input – no ? on inputs![] in rc.12
    let tensor = Tensor::from_array(([1i64, 3, in_h as i64, in_w as i64], input))?;
    let outputs = session.run(ort::inputs![tensor])?;

    let out_t = outputs[0].try_extract_tensor::<f32>()?;
    let out_shape = out_t.0;

    if out_shape.len() != 3 {
        bail!("YOLO output ndim={} expected 3", out_shape.len());
    }

    // Shape is either [1, features, anchors] or [1, anchors, features]
    let (n_rows, n_cols, transposed) = if out_shape[1] <= out_shape[2] {
        (out_shape[2] as usize, out_shape[1] as usize, true)
    } else {
        (out_shape[1] as usize, out_shape[2] as usize, false)
    };

    let data: &[f32] = out_t.1;

    let mut boxes: Vec<(BBox, f32)> = Vec::new();
    for i in 0..n_rows {
        let (cx, cy, bw, bh, score) = if transposed {
            let cx = data[0 * n_rows + i];
            let cy = data[1 * n_rows + i];
            let bw = data[2 * n_rows + i];
            let bh = data[3 * n_rows + i];
            let mut sc = f32::NEG_INFINITY;
            for c in 4..n_cols {
                let v = data[c * n_rows + i];
                if v > sc { sc = v; }
            }
            (cx, cy, bw, bh, sc)
        } else {
            let base = i * n_cols;
            let mut sc = f32::NEG_INFINITY;
            for c in 4..n_cols {
                let v = data[base + c];
                if v > sc { sc = v; }
            }
            (data[base], data[base + 1], data[base + 2], data[base + 3], sc)
        };

        if score < conf_thresh {
            continue;
        }

        let x1 = (((cx - bw * 0.5) / in_w as f32) * frame_w as f32) as isize;
        let y1 = (((cy - bh * 0.5) / in_h as f32) * frame_h as f32) as isize;
        let x2 = (((cx + bw * 0.5) / in_w as f32) * frame_w as f32) as isize;
        let y2 = (((cy + bh * 0.5) / in_h as f32) * frame_h as f32) as isize;

        let x1 = x1.max(0) as usize;
        let y1 = y1.max(0) as usize;
        let x2 = (x2 as usize).min(frame_w);
        let y2 = (y2 as usize).min(frame_h);

        if x2 <= x1 || y2 <= y1 {
            continue;
        }
        if aspect_filter {
            let wb = (x2 - x1) as f32;
            let hb = (y2 - y1) as f32;
            if hb == 0.0 || wb < 20.0 || hb < 8.0 || !(1.5..=7.0).contains(&(wb / hb)) {
                continue;
            }
        }
        boxes.push((BBox { x: x1, y: y1, x2, y2 }, score));
    }

    // Greedy NMS with IoU threshold 0.45
    boxes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut kept: Vec<BBox> = Vec::new();
    'outer: for (b, _) in &boxes {
        for kb in &kept {
            let ix1 = b.x.max(kb.x);
            let iy1 = b.y.max(kb.y);
            let ix2 = b.x2.min(kb.x2);
            let iy2 = b.y2.min(kb.y2);
            if ix2 > ix1 && iy2 > iy1 {
                let inter = ((ix2 - ix1) * (iy2 - iy1)) as f32;
                let ua = ((b.x2 - b.x) * (b.y2 - b.y)) as f32
                    + ((kb.x2 - kb.x) * (kb.y2 - kb.y)) as f32
                    - inter;
                if ua > 0.0 && inter / ua > 0.45 {
                    continue 'outer;
                }
            }
        }
        kept.push(*b);
    }
    Ok(kept)
}

// ── Nearest-neighbour resize ─────────────────────────────────────────────────

/// Writes the resized image into an existing buffer (no allocation).
fn resize_bgr_into(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, dst: &mut [u8]) {
    for dy in 0..dh {
        let sy = (dy * sh / dh).min(sh - 1);
        for dx in 0..dw {
            let sx = (dx * sw / dw).min(sw - 1);
            let s = (sy * sw + sx) * 3;
            let d = (dy * dw + dx) * 3;
            dst[d] = src[s];
            dst[d + 1] = src[s + 1];
            dst[d + 2] = src[s + 2];
        }
    }
}

pub fn resize_bgr(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let mut dst = vec![0u8; dw * dh * 3];
    resize_bgr_into(src, sw, sh, dw, dh, &mut dst);
    dst
}

// ── Hardware checks ──────────────────────────────────────────────────────────

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
