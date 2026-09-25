mod centerface;
mod nms;
mod preprocess;
mod scrfd;
mod session;
mod yolo;

pub use nms::bbox_iou;
pub use session::{build_session, check_nvdec, check_nvenc};

use centerface::centerface_detect;
use scrfd::scrfd_detect_tiled;
use yolo::{yolo_detect, yolo_detect_tiled};

use anyhow::Result;
use ort::session::Session;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub x: usize,
    pub y: usize,
    pub x2: usize,
    pub y2: usize,
    /// Konfidenz des Modells (0 bei Boxen ohne Bewertung).
    pub score: f32,
}

pub struct Detectors {
    pub centerface: Option<Session>,
    pub face_yolo: Option<Session>,
    pub face_scrfd: Option<Session>,
    pub plate_yolo: Option<Session>,
    pub in_h: usize,
    pub in_w: usize,
    pub face_conf_thresh: f32,
    /// Kacheln für SCRFD (Spalten, Zeilen); (1, 1) = nur Gesamtbild.
    pub face_tiles: (usize, usize),
    /// Kombi-Modus: CenterFace-Schwelle, mit der CenterFace ergänzend zum Hauptmodell läuft.
    pub combo_cf_thresh: Option<f32>,
    cf_resize_buf: Vec<u8>,
    yolo_resize_buf: Vec<u8>,
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
        let centerface = centerface_path.map(|p| build_session(p, false)).transpose()?;
        let face_yolo  = face_yolo_path.map(|p| build_session(p, use_trt)).transpose()?;
        let face_scrfd = face_scrfd_path.map(|p| build_session(p, false)).transpose()?;
        let plate_yolo = plate_yolo_path.map(|p| build_session(p, use_trt)).transpose()?;

        let cf_pad_h = ((in_h + 31) / 32) * 32;
        let cf_pad_w = ((in_w + 31) / 32) * 32;

        Ok(Self {
            centerface, face_yolo, face_scrfd, plate_yolo,
            in_h, in_w, face_conf_thresh,
            face_tiles: (1, 1),
            combo_cf_thresh: None,
            cf_resize_buf: vec![0u8; cf_pad_h * cf_pad_w * 3],
            yolo_resize_buf: vec![0u8; 640 * 640 * 3],
        })
    }

    pub fn detect_faces(&mut self, frame: &[u8], fw: usize, fh: usize) -> Vec<BBox> {
        let main = self.detect_faces_main(frame, fw, fh);
        match self.combo_cf_thresh {
            Some(t) if self.face_scrfd.is_some() || self.face_yolo.is_some() => {
                let extra = self.detect_centerface(frame, fw, fh, t);
                merge_faces(main, extra)
            }
            _ => main,
        }
    }

    fn detect_centerface(&mut self, frame: &[u8], fw: usize, fh: usize, thresh: f32) -> Vec<BBox> {
        let (in_h, in_w) = (self.in_h, self.in_w);
        let Some(session) = self.centerface.as_mut() else { return vec![] };
        match centerface_detect(session, frame, fw, fh, in_h, in_w, thresh, &mut self.cf_resize_buf) {
            Ok(boxes) => boxes.into_iter().map(|b| expand_bbox(b, fw, fh, 0.15)).collect(),
            Err(e) => {
                static ERR_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !ERR_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[CF-ERROR] CenterFace Fehler (einmalig): {e:?}");
                }
                vec![]
            }
        }
    }

    fn detect_faces_main(&mut self, frame: &[u8], fw: usize, fh: usize) -> Vec<BBox> {
        let thresh = self.face_conf_thresh;
        if self.face_scrfd.is_some() {
            let (cols, rows) = self.face_tiles;
            match scrfd_detect_tiled(self.face_scrfd.as_mut().unwrap(), frame, fw, fh, thresh, cols, rows, &mut self.yolo_resize_buf) {
                Ok(boxes) => boxes.into_iter().map(|b| expand_bbox(b, fw, fh, 0.10)).collect(),
                Err(e) => {
                    static ERR_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !ERR_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        eprintln!("[SCRFD-ERROR] SCRFD Fehler (einmalig): {e:?}");
                    }
                    vec![]
                }
            }
        } else if self.face_yolo.is_some() {
            yolo_detect(self.face_yolo.as_mut().unwrap(), frame, fw, fh, thresh, false, &mut self.yolo_resize_buf)
                .unwrap_or_default()
        } else if self.centerface.is_some() {
            self.detect_centerface(frame, fw, fh, thresh)
        } else {
            vec![]
        }
    }

    /// Erkennt Kennzeichen im Gesamtbild und zusätzlich in `tile_cols` × `tile_rows` Kacheln,
    /// damit kleine Kennzeichen nicht beim Verkleinern auf 640 px verloren gehen.
    pub fn detect_plates(&mut self, frame: &[u8], fw: usize, fh: usize, conf: f32, tile_cols: usize, tile_rows: usize) -> Vec<BBox> {
        if self.plate_yolo.is_some() {
            yolo_detect_tiled(self.plate_yolo.as_mut().unwrap(), frame, fw, fh, conf, true, tile_cols, tile_rows, &mut self.yolo_resize_buf)
                .unwrap_or_default()
                .into_iter().map(|b| expand_bbox(b, fw, fh, 0.15)).collect()
        } else {
            vec![]
        }
    }
}

/// Führt die Treffer zweier Gesichtsmodelle zusammen: Boxen auf demselben Gesicht werden
/// zu einer (die mit höherer Konfidenz bleibt), alle übrigen bleiben erhalten.
fn merge_faces(a: Vec<BBox>, b: Vec<BBox>) -> Vec<BBox> {
    let candidates = a.into_iter().chain(b).map(|x| (x, x.score)).collect();
    nms::greedy_nms(candidates, 0.3, usize::MAX)
}

fn expand_bbox(b: BBox, fw: usize, fh: usize, factor: f32) -> BBox {
    let ex = (((b.x2.saturating_sub(b.x)) as f32 * factor) as usize).max(2);
    let ey = (((b.y2.saturating_sub(b.y)) as f32 * factor) as usize).max(2);
    BBox {
        x: b.x.saturating_sub(ex),
        y: b.y.saturating_sub(ey),
        x2: (b.x2 + ex).min(fw.saturating_sub(1)),
        y2: (b.y2 + ey).min(fh.saturating_sub(1)),
        score: b.score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(x: usize, y: usize, s: usize, score: f32) -> BBox {
        BBox { x, y, x2: x + s, y2: y + s, score }
    }

    #[test]
    fn doppelte_treffer_werden_zu_einer_box() {
        let m = merge_faces(vec![b(100, 100, 40, 0.8)], vec![b(104, 102, 44, 0.55)]);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].score, 0.8);
    }

    #[test]
    fn getrennte_treffer_bleiben_erhalten() {
        let m = merge_faces(vec![b(100, 100, 40, 0.8)], vec![b(900, 300, 30, 0.6), b(2000, 500, 50, 0.7)]);
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn ohne_ergaenzung_bleibt_das_hauptergebnis() {
        let m = merge_faces(vec![b(1, 1, 20, 0.9), b(500, 1, 20, 0.6)], vec![]);
        assert_eq!(m.len(), 2);
    }
}
