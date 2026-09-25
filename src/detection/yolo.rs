use anyhow::{bail, Result};
use ort::{session::Session, value::Tensor};

use super::{nms::greedy_nms, preprocess::preprocess_yolo, BBox};

pub fn yolo_detect(
    session: &mut Session,
    frame: &[u8],
    frame_w: usize,
    frame_h: usize,
    conf_thresh: f32,
    aspect_filter: bool,
    resize_buf: &mut [u8],
) -> Result<Vec<BBox>> {
    let (in_h, in_w) = (640usize, 640usize);
    let input = preprocess_yolo(frame, frame_w, frame_h, resize_buf);

    let tensor = Tensor::from_array(([1i64, 3, in_h as i64, in_w as i64], input))?;
    let outputs = session.run(ort::inputs![tensor])?;

    let out_t = outputs[0].try_extract_tensor::<f32>()?;
    if out_t.0.len() != 3 {
        bail!("YOLO output ndim={} expected 3", out_t.0.len());
    }

    let boxes = decode_yolo_output(out_t.1, out_t.0, frame_w, frame_h, conf_thresh, aspect_filter);
    Ok(greedy_nms(boxes, 0.45, usize::MAX))
}

fn decode_yolo_output(
    data: &[f32],
    out_shape: &[i64],
    frame_w: usize,
    frame_h: usize,
    conf_thresh: f32,
    aspect_filter: bool,
) -> Vec<(BBox, f32)> {
    let (in_h, in_w) = (640usize, 640usize);
    let (n_rows, n_cols, transposed) = if out_shape[1] <= out_shape[2] {
        (out_shape[2] as usize, out_shape[1] as usize, true)
    } else {
        (out_shape[1] as usize, out_shape[2] as usize, false)
    };

    let mut boxes: Vec<(BBox, f32)> = Vec::new();
    for i in 0..n_rows {
        let (cx, cy, bw, bh, score) = if transposed {
            let mut sc = f32::NEG_INFINITY;
            for c in 4..n_cols { let v = data[c * n_rows + i]; if v > sc { sc = v; } }
            (data[i], data[n_rows + i], data[2 * n_rows + i], data[3 * n_rows + i], sc)
        } else {
            let base = i * n_cols;
            let mut sc = f32::NEG_INFINITY;
            for c in 4..n_cols { let v = data[base + c]; if v > sc { sc = v; } }
            (data[base], data[base + 1], data[base + 2], data[base + 3], sc)
        };

        if score < conf_thresh { continue; }

        let x1 = (((cx - bw * 0.5) / in_w as f32) * frame_w as f32) as isize;
        let y1 = (((cy - bh * 0.5) / in_h as f32) * frame_h as f32) as isize;
        let x2 = (((cx + bw * 0.5) / in_w as f32) * frame_w as f32) as isize;
        let y2 = (((cy + bh * 0.5) / in_h as f32) * frame_h as f32) as isize;

        let x1 = x1.max(0) as usize;
        let y1 = y1.max(0) as usize;
        let x2 = (x2 as usize).min(frame_w);
        let y2 = (y2 as usize).min(frame_h);

        if x2 <= x1 || y2 <= y1 { continue; }
        if aspect_filter {
            let (wb, hb) = ((x2 - x1) as f32, (y2 - y1) as f32);
            if hb == 0.0 || wb < 20.0 || hb < 8.0 || !(1.5..=7.0).contains(&(wb / hb)) {
                continue;
            }
        }
        boxes.push((BBox { x: x1, y: y1, x2, y2 }, score));
    }
    boxes
}
