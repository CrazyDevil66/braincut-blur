use anyhow::Result;
use ort::{session::Session, value::Tensor};

use super::{nms::greedy_nms, preprocess::preprocess_centerface, BBox};

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
    let (input, pad_h, pad_w) = preprocess_centerface(frame, frame_w, frame_h, in_h, in_w, resize_buf);

    let tensor = Tensor::from_array(([1i64, 3, pad_h as i64, pad_w as i64], input))?;
    let outputs = session.run(ort::inputs! { "input.1" => tensor })?;

    let hm_t  = outputs[0].try_extract_tensor::<f32>()?;
    let sc_t  = outputs[1].try_extract_tensor::<f32>()?;
    let off_t = outputs[2].try_extract_tensor::<f32>()?;

    let hm_shape = hm_t.0;
    let feat_h = hm_shape[hm_shape.len() - 2] as usize;
    let feat_w = hm_shape[hm_shape.len() - 1] as usize;
    let feat_n = feat_h * feat_w;
    let hm_d: &[f32] = hm_t.1;
    let sc_d: &[f32] = sc_t.1;
    let off_d: &[f32] = off_t.1;

    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        let max_raw = hm_d.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("[CF-DIAG] hm={:?} sc={:?} off={:?} max_score={:.4} thresh={:.2}",
            hm_t.0, sc_t.0, off_t.0, max_raw, threshold);
    }

    let scale_x = frame_w as f32 / pad_w as f32;
    let scale_y = frame_h as f32 / pad_h as f32;
    let mut candidates: Vec<(BBox, f32)> = Vec::new();

    for cy in 0..feat_h {
        for cx in 0..feat_w {
            let idx = cy * feat_w + cx;
            // Die Heatmap enthält im Modell bereits Wahrscheinlichkeiten (wie in deface ausgewertet).
            let score = hm_d[idx];
            if score < threshold || !is_local_max(hm_d, feat_h, feat_w, cy, cx) { continue; }

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
                candidates.push((BBox { x: x1, y: y1, x2, y2, score }, score));
            }
        }
    }

    Ok(greedy_nms(candidates, 0.3, 200))
}

fn is_local_max(hm: &[f32], feat_h: usize, feat_w: usize, cy: usize, cx: usize) -> bool {
    let center = hm[cy * feat_w + cx];
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dy == 0 && dx == 0 { continue; }
            let ny = cy as i32 + dy; let nx = cx as i32 + dx;
            if ny < 0 || ny >= feat_h as i32 || nx < 0 || nx >= feat_w as i32 { continue; }
            if hm[ny as usize * feat_w + nx as usize] >= center { return false; }
        }
    }
    true
}

