use anyhow::{bail, Result};
use ort::{session::Session, value::Tensor};

use super::{
    nms::greedy_nms,
    preprocess::{letterbox_yolo, tile_regions, Letterbox, Region},
    BBox,
};

const IN: usize = 640;
const TILE_OVERLAP: f32 = 0.15;

/// Erkennung im Gesamtbild (seitenverhältnistreu skaliert).
pub fn yolo_detect(
    session: &mut Session,
    frame: &[u8],
    frame_w: usize,
    frame_h: usize,
    conf_thresh: f32,
    aspect_filter: bool,
    resize_buf: &mut [u8],
) -> Result<Vec<BBox>> {
    yolo_detect_tiled(session, frame, frame_w, frame_h, conf_thresh, aspect_filter, 1, 1, resize_buf)
}

/// Erkennung im Gesamtbild und – bei mehr als einer Kachel – zusätzlich je Kachel.
/// Das Gesamtbild findet große Objekte über Kachelgrenzen hinweg, die Kacheln kleine.
#[allow(clippy::too_many_arguments)]
pub fn yolo_detect_tiled(
    session: &mut Session,
    frame: &[u8],
    frame_w: usize,
    frame_h: usize,
    conf_thresh: f32,
    aspect_filter: bool,
    tile_cols: usize,
    tile_rows: usize,
    resize_buf: &mut [u8],
) -> Result<Vec<BBox>> {
    let full = Region { x0: 0, y0: 0, w: frame_w, h: frame_h };
    let mut candidates = detect_region(session, frame, frame_w, frame_h, &full, conf_thresh, resize_buf)?;
    if tile_cols * tile_rows > 1 {
        for region in tile_regions(frame_w, frame_h, tile_cols, tile_rows, TILE_OVERLAP) {
            candidates.extend(detect_region(session, frame, frame_w, frame_h, &region, conf_thresh, resize_buf)?);
        }
    }
    if aspect_filter {
        candidates.retain(|(b, _)| is_plate_shaped(b));
    }
    Ok(greedy_nms(candidates, 0.45, usize::MAX))
}

fn detect_region(
    session: &mut Session,
    frame: &[u8],
    frame_w: usize,
    frame_h: usize,
    region: &Region,
    conf_thresh: f32,
    resize_buf: &mut [u8],
) -> Result<Vec<(BBox, f32)>> {
    let (input, lb) = letterbox_yolo(frame, frame_w, region, IN, resize_buf);
    let tensor = Tensor::from_array(([1i64, 3, IN as i64, IN as i64], input))?;
    let outputs = session.run(ort::inputs![tensor])?;

    let out_t = outputs[0].try_extract_tensor::<f32>()?;
    if out_t.0.len() != 3 {
        bail!("YOLO output ndim={} expected 3", out_t.0.len());
    }
    Ok(decode_yolo_output(out_t.1, out_t.0, conf_thresh)
        .into_iter()
        .filter_map(|(cx, cy, bw, bh, score)| to_frame_box(&lb, region, frame_w, frame_h, cx, cy, bw, bh).map(|b| (BBox { score, ..b }, score)))
        .collect())
}

/// Liefert (cx, cy, w, h, score) im Koordinatensystem des Modell-Eingangs.
fn decode_yolo_output(data: &[f32], out_shape: &[i64], conf_thresh: f32) -> Vec<(f32, f32, f32, f32, f32)> {
    let (n_rows, n_cols, transposed) = if out_shape[1] <= out_shape[2] {
        (out_shape[2] as usize, out_shape[1] as usize, true)
    } else {
        (out_shape[1] as usize, out_shape[2] as usize, false)
    };

    let mut boxes = Vec::new();
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
        if score >= conf_thresh {
            boxes.push((cx, cy, bw, bh, score));
        }
    }
    boxes
}

#[allow(clippy::too_many_arguments)]
fn to_frame_box(lb: &Letterbox, region: &Region, frame_w: usize, frame_h: usize, cx: f32, cy: f32, bw: f32, bh: f32) -> Option<BBox> {
    let (x1, y1) = lb.to_frame(region, cx - bw * 0.5, cy - bh * 0.5);
    let (x2, y2) = lb.to_frame(region, cx + bw * 0.5, cy + bh * 0.5);
    let x1 = (x1.max(region.x0 as f32) as usize).min(frame_w);
    let y1 = (y1.max(region.y0 as f32) as usize).min(frame_h);
    let x2 = (x2.max(0.0) as usize).min(region.x0 + region.w).min(frame_w);
    let y2 = (y2.max(0.0) as usize).min(region.y0 + region.h).min(frame_h);
    (x2 > x1 && y2 > y1).then_some(BBox { x: x1, y: y1, x2, y2, score: 0.0 })
}

fn is_plate_shaped(b: &BBox) -> bool {
    let (wb, hb) = ((b.x2 - b.x) as f32, (b.y2 - b.y) as f32);
    hb > 0.0 && wb >= 20.0 && hb >= 8.0 && (1.5..=7.0).contains(&(wb / hb))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_aus_kachel_landet_an_der_richtigen_stelle() {
        // Kachel 1440×1160 ab (2400,1000): Skalierung 640/1440, oben/unten Rand
        let region = Region { x0: 2400, y0: 1000, w: 1440, h: 1160 };
        let scale = 640.0 / 1440.0;
        let lb = Letterbox { scale, pad_x: 0.0, pad_y: ((640.0 - 1160.0 * scale) / 2.0f32).floor() };
        // Kennzeichen 90×30 px bei (3000,1500) im Frame → Mitte im Modell-Eingang
        let (cx, cy) = ((3045.0 - 2400.0) * scale, (1515.0 - 1000.0) * scale + lb.pad_y);
        let b = to_frame_box(&lb, &region, 3840, 2160, cx, cy, 90.0 * scale, 30.0 * scale).unwrap();
        assert!((b.x as i32 - 3000).abs() <= 1 && (b.y as i32 - 1500).abs() <= 1);
        assert!((b.x2 as i32 - 3090).abs() <= 1 && (b.y2 as i32 - 1530).abs() <= 1);
        assert!(is_plate_shaped(&b));
    }

    #[test]
    fn boxen_werden_auf_die_kachel_begrenzt() {
        let region = Region { x0: 0, y0: 0, w: 640, h: 640 };
        let lb = Letterbox { scale: 1.0, pad_x: 0.0, pad_y: 0.0 };
        let b = to_frame_box(&lb, &region, 3840, 2160, 630.0, 10.0, 40.0, 40.0).unwrap();
        assert_eq!((b.x, b.y, b.x2, b.y2), (610, 0, 640, 30));
    }

    #[test]
    fn kennzeichen_form_filter() {
        let b = |w, h| BBox { x: 0, y: 0, x2: w, y2: h, score: 0.0 };
        assert!(is_plate_shaped(&b(90, 30)));
        assert!(!is_plate_shaped(&b(30, 30)));
        assert!(!is_plate_shaped(&b(15, 5)));
    }
}
