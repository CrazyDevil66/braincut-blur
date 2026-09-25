use std::collections::HashMap;
use std::io::Write;

use crate::detection::{BBox, Detectors};

pub const PLATE_TTL: u32 = 200;
pub const FACE_TTL: u32 = 60;
pub const FACE_GRID: usize = 30;

#[allow(clippy::too_many_arguments)]
pub fn process_detection_frame(
    detectors: &mut Detectors,
    buf: &[u8],
    w: usize, h: usize,
    mode: &str,
    should_detect: bool,
    face_buf: &mut HashMap<(usize, usize, usize, usize), (BBox, u32)>,
    plate_buf: &mut HashMap<(usize, usize, usize, usize), (BBox, u32)>,
    conf_thresh: f32,
    plate_grid: usize,
    frame_idx: u64,
    det_log: &mut Option<std::io::BufWriter<std::fs::File>>,
) -> (u64, u64) {
    let mut new_faces = 0u64;
    let mut new_plates = 0u64;

    if mode == "faces" || mode == "both" {
        face_buf.retain(|_, (_, ttl)| { *ttl = ttl.saturating_sub(1); *ttl > 0 });
        if should_detect {
            let detected = detectors.detect_faces(buf, w, h);
            new_faces = detected.len() as u64;
            for bf in detected {
                let key = (bf.x / FACE_GRID, bf.y / FACE_GRID, bf.x2 / FACE_GRID, bf.y2 / FACE_GRID);
                if let Some(ref mut log) = det_log {
                    let dw = bf.x2 - bf.x; let dh = bf.y2 - bf.y;
                    let applied = dw >= 50 && dh >= 50 && dw <= w / 5 && dh <= h / 5;
                    let _ = writeln!(log, "{frame_idx},face,{},{},{dw},{dh},{:.1},{:.1},{}",
                        bf.x, bf.y,
                        dw as f32 / w as f32 * 100.0, dh as f32 / h as f32 * 100.0,
                        if applied { 1 } else { 0 });
                }
                face_buf.insert(key, (bf, FACE_TTL));
            }
        }
    }

    if mode == "plates" || mode == "both" {
        plate_buf.retain(|_, (_, ttl)| { *ttl = ttl.saturating_sub(1); *ttl > 0 });
        if should_detect {
            let detected = detectors.detect_plates(buf, w, h, conf_thresh);
            new_plates = detected.len() as u64;
            for bp in detected {
                let key = (bp.x / plate_grid, bp.y / plate_grid, bp.x2 / plate_grid, bp.y2 / plate_grid);
                if let Some(ref mut log) = det_log {
                    let dw = bp.x2 - bp.x; let dh = bp.y2 - bp.y;
                    let _ = writeln!(log, "{frame_idx},plate,{},{},{dw},{dh},{:.1},{:.1},1",
                        bp.x, bp.y,
                        dw as f32 / w as f32 * 100.0, dh as f32 / h as f32 * 100.0);
                }
                plate_buf.insert(key, (bp, PLATE_TTL));
            }
        }
    }

    (new_faces, new_plates)
}
