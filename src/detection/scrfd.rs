use anyhow::{bail, Result};
use ort::{session::Session, value::Tensor};

use super::{nms::greedy_nms, preprocess::preprocess_scrfd, BBox};

pub fn scrfd_detect(
    session: &mut Session,
    frame: &[u8],
    frame_w: usize,
    frame_h: usize,
    threshold: f32,
    resize_buf: &mut [u8],
) -> Result<Vec<BBox>> {
    let (in_h, in_w) = (640usize, 640usize);
    let input = preprocess_scrfd(frame, frame_w, frame_h, resize_buf);

    let tensor = Tensor::from_array(([1i64, 3, in_h as i64, in_w as i64], input))?;
    let outputs = session.run(ort::inputs![tensor])?;

    if outputs.len() != 6 && outputs.len() != 9 {
        bail!("SCRFD: unerwartete Output-Anzahl {} (erwartet 6 oder 9)", outputs.len());
    }
    let step = if outputs.len() == 9 { 3usize } else { 2 };
    let strides = [8usize, 16, 32];
    let scale_x = frame_w as f32 / in_w as f32;
    let scale_y = frame_h as f32 / in_h as f32;

    let mut candidates: Vec<(BBox, f32)> = Vec::new();
    for (si, &stride) in strides.iter().enumerate() {
        let out_h = in_h / stride;
        let out_w = in_w / stride;
        let (_, scores) = outputs[si * step].try_extract_tensor::<f32>()?;
        let (_, boxes)  = outputs[si * step + 1].try_extract_tensor::<f32>()?;

        for i in 0..(out_h * out_w * 2) {
            let score = scores[i];
            if score < threshold { continue; }
            let cell = i / 2;
            let anchor_cx = (cell % out_w) as f32 * stride as f32;
            let anchor_cy = (cell / out_w) as f32 * stride as f32;
            let b = i * 4;
            let x1 = ((anchor_cx - boxes[b])     * scale_x) as isize;
            let y1 = ((anchor_cy - boxes[b + 1]) * scale_y) as isize;
            let x2 = ((anchor_cx + boxes[b + 2]) * scale_x) as isize;
            let y2 = ((anchor_cy + boxes[b + 3]) * scale_y) as isize;
            let x1 = x1.max(0) as usize; let y1 = y1.max(0) as usize;
            let x2 = (x2 as usize).min(frame_w); let y2 = (y2 as usize).min(frame_h);
            if x2 > x1 && y2 > y1 {
                candidates.push((BBox { x: x1, y: y1, x2, y2 }, score));
            }
        }
    }

    Ok(greedy_nms(candidates, 0.45, usize::MAX))
}
