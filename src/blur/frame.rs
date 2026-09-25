use std::io::Write;

use crate::detection::{bbox_iou, BBox, Detectors};

/// Wie viele Frames eine Box ohne neue Erkennung stehen bleibt.
pub const PLATE_TTL: u32 = 200;
pub const FACE_TTL: u32 = 60;
/// Ab dieser Überlappung gilt eine neue Erkennung als dasselbe Objekt und verschiebt die Box.
const TRACK_IOU: f32 = 0.3;
/// Kleinere Gesichtsboxen sind Pixelrauschen; alles darüber wird verpixelt.
pub const MIN_FACE_PX: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Track {
    pub bbox: BBox,
    pub ttl: u32,
}

pub fn face_wird_verpixelt(b: &BBox) -> bool {
    b.x2 - b.x >= MIN_FACE_PX && b.y2 - b.y >= MIN_FACE_PX
}

/// Lässt Boxen ohne neue Erkennung altern und entfernt abgelaufene.
pub fn age_tracks(tracks: &mut Vec<Track>) {
    tracks.retain_mut(|t| {
        t.ttl = t.ttl.saturating_sub(1);
        t.ttl > 0
    });
}

/// Ordnet neue Erkennungen vorhandenen Boxen zu: Überlappen sie ausreichend, wird die Box
/// an die neue Position verschoben, statt eine zusätzliche anzulegen (kein Nachziehen).
pub fn update_tracks(tracks: &mut Vec<Track>, detected: &[BBox], ttl: u32) {
    let mut matched = vec![false; tracks.len()];
    for d in detected {
        let best = tracks
            .iter()
            .enumerate()
            .filter(|(i, _)| !matched[*i])
            .map(|(i, t)| (i, bbox_iou(&t.bbox, d)))
            .filter(|(_, iou)| *iou >= TRACK_IOU)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        match best {
            Some((i, _)) => {
                tracks[i] = Track { bbox: *d, ttl };
                matched[i] = true;
            }
            None => {
                tracks.push(Track { bbox: *d, ttl });
                matched.push(true);
            }
        }
    }
}

fn log_detection(log: &mut Option<std::io::BufWriter<std::fs::File>>, frame_idx: u64, model: &str, b: &BBox, w: usize, h: usize, applied: bool) {
    if let Some(ref mut log) = log {
        let (dw, dh) = (b.x2 - b.x, b.y2 - b.y);
        let _ = writeln!(log, "{frame_idx},{model},{},{},{dw},{dh},{:.1},{:.1},{:.3},{}",
            b.x, b.y,
            dw as f32 / w as f32 * 100.0, dh as f32 / h as f32 * 100.0,
            b.score, if applied { 1 } else { 0 });
    }
}

#[allow(clippy::too_many_arguments)]
pub fn process_detection_frame(
    detectors: &mut Detectors,
    buf: &[u8],
    w: usize, h: usize,
    mode: &str,
    should_detect: bool,
    face_tracks: &mut Vec<Track>,
    plate_tracks: &mut Vec<Track>,
    conf_thresh: f32,
    plate_tiles: (usize, usize),
    frame_idx: u64,
    det_log: &mut Option<std::io::BufWriter<std::fs::File>>,
) -> (u64, u64) {
    let mut new_faces = 0u64;
    let mut new_plates = 0u64;

    if mode == "faces" || mode == "both" {
        age_tracks(face_tracks);
        if should_detect {
            let detected = detectors.detect_faces(buf, w, h);
            new_faces = detected.len() as u64;
            for bf in &detected {
                log_detection(det_log, frame_idx, "face", bf, w, h, face_wird_verpixelt(bf));
            }
            update_tracks(face_tracks, &detected, FACE_TTL);
        }
    }

    if mode == "plates" || mode == "both" {
        age_tracks(plate_tracks);
        if should_detect {
            let detected = detectors.detect_plates(buf, w, h, conf_thresh, plate_tiles.0, plate_tiles.1);
            new_plates = detected.len() as u64;
            for bp in &detected {
                log_detection(det_log, frame_idx, "plate", bp, w, h, true);
            }
            update_tracks(plate_tracks, &detected, PLATE_TTL);
        }
    }

    (new_faces, new_plates)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(x: usize, y: usize, x2: usize, y2: usize) -> BBox {
        BBox { x, y, x2, y2, score: 0.9 }
    }

    #[test]
    fn ueberlappende_erkennung_verschiebt_die_box() {
        let mut t = vec![Track { bbox: b(100, 100, 200, 140), ttl: 5 }];
        update_tracks(&mut t, &[b(110, 102, 210, 142)], 200);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].bbox.x, 110);
        assert_eq!(t[0].ttl, 200);
    }

    #[test]
    fn entfernte_erkennung_legt_neue_box_an() {
        let mut t = vec![Track { bbox: b(100, 100, 200, 140), ttl: 5 }];
        update_tracks(&mut t, &[b(800, 100, 900, 140)], 200);
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].ttl, 5);
    }

    #[test]
    fn zwei_erkennungen_nehmen_nicht_dieselbe_box() {
        let mut t = vec![Track { bbox: b(100, 100, 200, 140), ttl: 5 }];
        update_tracks(&mut t, &[b(105, 100, 205, 140), b(95, 100, 195, 140)], 50);
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn boxen_altern_und_verschwinden() {
        let mut t = vec![Track { bbox: b(0, 0, 10, 10), ttl: 2 }, Track { bbox: b(0, 0, 10, 10), ttl: 1 }];
        age_tracks(&mut t);
        assert_eq!(t.len(), 1);
        age_tracks(&mut t);
        assert!(t.is_empty());
    }

    #[test]
    fn auch_kleine_gesichter_werden_verpixelt() {
        assert!(face_wird_verpixelt(&b(0, 0, 26, 30)));
        assert!(face_wird_verpixelt(&b(0, 0, 900, 1200)));
        assert!(!face_wird_verpixelt(&b(0, 0, 8, 12)));
    }
}
