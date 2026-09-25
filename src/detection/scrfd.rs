use anyhow::{bail, Result};
use ort::{session::Session, value::Tensor};

use super::{
    nms::greedy_nms,
    preprocess::{letterbox_scrfd, tile_regions, Region},
    BBox,
};

const IN: usize = 640;
const STRIDES: [usize; 3] = [8, 16, 32];
/// SCRFD legt zwei Anker pro Rasterzelle an.
const ANCHORS: usize = 2;
const TILE_OVERLAP: f32 = 0.15;
const NMS_IOU: f32 = 0.4;

/// Gesichtserkennung mit SCRFD (InsightFace) im Gesamtbild und – bei mehr als einer Kachel –
/// zusätzlich je Kachel, damit kleine Gesichter beim Verkleinern auf 640 px erhalten bleiben.
#[allow(clippy::too_many_arguments)]
pub fn scrfd_detect_tiled(
    session: &mut Session,
    frame: &[u8],
    frame_w: usize,
    frame_h: usize,
    threshold: f32,
    tile_cols: usize,
    tile_rows: usize,
    resize_buf: &mut [u8],
) -> Result<Vec<BBox>> {
    let full = Region { x0: 0, y0: 0, w: frame_w, h: frame_h };
    let mut candidates = detect_region(session, frame, frame_w, frame_h, &full, threshold, resize_buf)?;
    if tile_cols * tile_rows > 1 {
        for region in tile_regions(frame_w, frame_h, tile_cols, tile_rows, TILE_OVERLAP) {
            candidates.extend(detect_region(session, frame, frame_w, frame_h, &region, threshold, resize_buf)?);
        }
    }
    Ok(greedy_nms(candidates, NMS_IOU, usize::MAX))
}

fn detect_region(
    session: &mut Session,
    frame: &[u8],
    frame_w: usize,
    frame_h: usize,
    region: &Region,
    threshold: f32,
    resize_buf: &mut [u8],
) -> Result<Vec<(BBox, f32)>> {
    let (input, lb) = letterbox_scrfd(frame, frame_w, region, IN, resize_buf);
    let tensor = Tensor::from_array(([1i64, 3, IN as i64, IN as i64], input))?;
    let outputs = session.run(ort::inputs![tensor])?;

    let mut tensors: Vec<(usize, &[f32])> = Vec::with_capacity(outputs.len());
    for i in 0..outputs.len() {
        let (shape, data) = outputs[i].try_extract_tensor::<f32>()?;
        let last = shape.last().copied().unwrap_or(1).max(1) as usize;
        tensors.push((last, data));
    }

    let mut out = Vec::new();
    for (b, score) in decode_scrfd(&tensors, IN, threshold)? {
        let (x1, y1) = lb.to_frame(region, b[0], b[1]);
        let (x2, y2) = lb.to_frame(region, b[2], b[3]);
        let x1 = (x1.max(region.x0 as f32) as usize).min(frame_w);
        let y1 = (y1.max(region.y0 as f32) as usize).min(frame_h);
        let x2 = (x2.max(0.0) as usize).min(region.x0 + region.w).min(frame_w);
        let y2 = (y2.max(0.0) as usize).min(region.y0 + region.h).min(frame_h);
        if x2 > x1 && y2 > y1 {
            out.push((BBox { x: x1, y: y1, x2, y2, score }, score));
        }
    }
    Ok(out)
}

/// Dekodiert die SCRFD-Ausgaben zu Boxen im Modell-Eingang ([x1, y1, x2, y2], Konfidenz).
/// Die Ausgaben werden über ihre Form zugeordnet (letzte Dimension 1 = Konfidenz, 4 = Box,
/// Länge = Anker je Stride), weil die Reihenfolge je nach Export abweicht.
pub fn decode_scrfd(tensors: &[(usize, &[f32])], input: usize, threshold: f32) -> Result<Vec<([f32; 4], f32)>> {
    let mut boxes = Vec::new();
    for stride in STRIDES {
        let grid = input / stride;
        let n = grid * grid * ANCHORS;
        let find = |cols: usize| tensors.iter().find(|(last, d)| *last == cols && d.len() == n * cols).map(|(_, d)| *d);
        let (Some(scores), Some(dist)) = (find(1), find(4)) else {
            bail!("SCRFD: Ausgaben für Stride {stride} nicht gefunden (Modell ist kein SCRFD-Detektor?)");
        };
        let s = stride as f32;
        for i in 0..n {
            let score = scores[i];
            if score < threshold {
                continue;
            }
            let cell = i / ANCHORS;
            let (cx, cy) = ((cell % grid) as f32 * s, (cell / grid) as f32 * s);
            let d = &dist[i * 4..i * 4 + 4];
            boxes.push(([cx - d[0] * s, cy - d[1] * s, cx + d[2] * s, cy + d[3] * s], score));
        }
    }
    Ok(boxes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Baut leere SCRFD-Ausgaben für 640 px in der Reihenfolge des InsightFace-Exports.
    fn leere_ausgaben() -> Vec<(usize, Vec<f32>)> {
        let mut v = Vec::new();
        for cols in [1usize, 4, 10] {
            for stride in STRIDES {
                let n = (IN / stride) * (IN / stride) * ANCHORS;
                v.push((cols, vec![0.0; n * cols]));
            }
        }
        v
    }

    #[test]
    fn box_wird_mit_stride_und_anker_berechnet() {
        let mut t = leere_ausgaben();
        // Stride 16 (Index 1 = Konfidenz, 4 = Box): Zelle x=10, y=5 → Anker (160, 80), zweiter Anker
        let grid = IN / 16;
        let i = (5 * grid + 10) * ANCHORS + 1;
        t[1].1[i] = 0.9;
        t[4].1[i * 4..i * 4 + 4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let refs: Vec<(usize, &[f32])> = t.iter().map(|(c, d)| (*c, d.as_slice())).collect();
        let b = decode_scrfd(&refs, IN, 0.5).unwrap();
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].0, [160.0 - 16.0, 80.0 - 32.0, 160.0 + 48.0, 80.0 + 64.0]);
        assert_eq!(b[0].1, 0.9);
    }

    #[test]
    fn reihenfolge_der_ausgaben_spielt_keine_rolle() {
        let mut t = leere_ausgaben();
        let i = 3;
        t[0].1[i] = 0.8; // Stride 8, Konfidenz
        t[3].1[i * 4..i * 4 + 4].copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        t.reverse();
        let refs: Vec<(usize, &[f32])> = t.iter().map(|(c, d)| (*c, d.as_slice())).collect();
        let b = decode_scrfd(&refs, IN, 0.5).unwrap();
        // Zelle 1 (i=3 → zweiter Anker von Zelle 1) → Anker (8, 0)
        assert_eq!(b[0].0, [0.0, -8.0, 16.0, 8.0]);
    }

    #[test]
    fn falsches_modell_wird_erkannt() {
        let d = vec![0.0f32; 3];
        assert!(decode_scrfd(&[(3, d.as_slice())], IN, 0.5).is_err());
    }

    /// Echter Modelltest: `SCRFD_MODEL=/pfad/det_10g.onnx SCRFD_BILD=/pfad/frame.jpg cargo test -- --ignored`
    #[test]
    #[ignore]
    fn scrfd_modell_auf_bild() {
        let model = std::env::var("SCRFD_MODEL").expect("SCRFD_MODEL setzen");
        let bild = std::env::var("SCRFD_BILD").expect("SCRFD_BILD setzen");
        let tiles: usize = std::env::var("SCRFD_TILES").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let img = image::open(&bild).unwrap().to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let mut bgr = img.into_raw();
        for px in bgr.chunks_exact_mut(3) { px.swap(0, 2); }
        // Nur CPU: der Test soll auch ohne GPU laufen.
        let mut session = Session::builder().unwrap().commit_from_file(&model).unwrap();
        let mut buf = vec![0u8; IN * IN * 3];
        let (cols, rows) = if tiles > 1 { (4, 3) } else { (1, 1) };
        let thresh: f32 = std::env::var("SCRFD_THRESH").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
        let found = scrfd_detect_tiled(&mut session, &bgr, w, h, thresh, cols, rows, &mut buf).unwrap();
        for b in &found {
            println!("GESICHT {} {} {} {} {:.3}", b.x, b.y, b.x2 - b.x, b.y2 - b.y, b.score);
        }
        println!("ANZAHL {}", found.len());
    }
}
