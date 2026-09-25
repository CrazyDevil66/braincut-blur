pub fn preprocess_centerface(
    frame: &[u8], fw: usize, fh: usize,
    in_h: usize, in_w: usize,
    buf: &mut [u8],
) -> (Vec<f32>, usize, usize) {
    let pad_h = ((in_h + 31) / 32) * 32;
    let pad_w = ((in_w + 31) / 32) * 32;
    resize_bgr_into(frame, fw, fh, pad_w, pad_h, buf);
    let mean = [104.0f32, 117.0, 123.0];
    let mut input = vec![0f32; 3 * pad_h * pad_w];
    for y in 0..pad_h {
        for x in 0..pad_w {
            let s = (y * pad_w + x) * 3;
            for c in 0..3usize {
                input[c * pad_h * pad_w + y * pad_w + x] = buf[s + c] as f32 - mean[c];
            }
        }
    }
    (input, pad_h, pad_w)
}

/// Rechteckiger Bildausschnitt in Frame-Koordinaten.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Region {
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
}

/// Lage des Ausschnitts im quadratischen Modell-Eingang nach dem Letterbox-Skalieren.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

impl Letterbox {
    /// Rechnet einen Punkt aus dem Modell-Eingang zurück in Frame-Koordinaten.
    pub fn to_frame(&self, region: &Region, mx: f32, my: f32) -> (f32, f32) {
        (
            (mx - self.pad_x) / self.scale + region.x0 as f32,
            (my - self.pad_y) / self.scale + region.y0 as f32,
        )
    }
}

/// Teilt das Bild in `cols` × `rows` Kacheln mit `overlap` (Anteil) Überlappung,
/// damit Objekte an den Kachelgrenzen vollständig in mindestens einer Kachel liegen.
pub fn tile_regions(fw: usize, fh: usize, cols: usize, rows: usize, overlap: f32) -> Vec<Region> {
    let (cols, rows) = (cols.max(1), rows.max(1));
    let axis = |len: usize, n: usize| -> Vec<(usize, usize)> {
        let base = len / n;
        let size = ((base as f32 * (1.0 + overlap)) as usize).min(len).max(1);
        (0..n)
            .map(|i| {
                let center = i * base + base / 2;
                let start = center.saturating_sub(size / 2).min(len - size);
                (start, size)
            })
            .collect()
    };
    let xs = axis(fw, cols);
    let ys = axis(fh, rows);
    ys.iter()
        .flat_map(|&(y0, h)| xs.iter().map(move |&(x0, w)| Region { x0, y0, w, h }))
        .collect()
}

/// Skaliert `region` seitenverhältnistreu in ein `size` × `size`-Quadrat (BGR, Rand mit `pad`).
fn letterbox_into(frame: &[u8], fw: usize, region: &Region, size: usize, pad: u8, buf: &mut [u8]) -> Letterbox {
    let scale = (size as f32 / region.w as f32).min(size as f32 / region.h as f32);
    let nw = ((region.w as f32 * scale).round() as usize).clamp(1, size);
    let nh = ((region.h as f32 * scale).round() as usize).clamp(1, size);
    let pad_x = (size - nw) / 2;
    let pad_y = (size - nh) / 2;

    buf[..size * size * 3].fill(pad);
    for dy in 0..nh {
        let sy = region.y0 + (dy * region.h / nh).min(region.h - 1);
        for dx in 0..nw {
            let sx = region.x0 + (dx * region.w / nw).min(region.w - 1);
            let s = (sy * fw + sx) * 3;
            let d = ((dy + pad_y) * size + dx + pad_x) * 3;
            buf[d..d + 3].copy_from_slice(&frame[s..s + 3]);
        }
    }
    Letterbox { scale, pad_x: pad_x as f32, pad_y: pad_y as f32 }
}

/// BGR-Puffer → RGB-CHW-Tensor mit `(wert - mean) / std`.
fn to_rgb_chw(buf: &[u8], size: usize, mean: f32, std: f32) -> Vec<f32> {
    let n = size * size;
    let mut input = vec![0f32; 3 * n];
    for i in 0..n {
        let s = i * 3;
        input[i]         = (buf[s + 2] as f32 - mean) / std;
        input[n + i]     = (buf[s + 1] as f32 - mean) / std;
        input[2 * n + i] = (buf[s]     as f32 - mean) / std;
    }
    input
}

/// YOLO-Eingang: Rand grau (wie beim Training), Werte 0–1.
pub fn letterbox_yolo(frame: &[u8], fw: usize, region: &Region, size: usize, buf: &mut [u8]) -> (Vec<f32>, Letterbox) {
    let lb = letterbox_into(frame, fw, region, size, 114, buf);
    (to_rgb_chw(buf, size, 0.0, 255.0), lb)
}

/// SCRFD-Eingang (InsightFace): Rand schwarz, Werte (x − 127,5) / 128.
pub fn letterbox_scrfd(frame: &[u8], fw: usize, region: &Region, size: usize, buf: &mut [u8]) -> (Vec<f32>, Letterbox) {
    let lb = letterbox_into(frame, fw, region, size, 0, buf);
    (to_rgb_chw(buf, size, 127.5, 128.0), lb)
}

fn resize_bgr_into(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, dst: &mut [u8]) {
    for dy in 0..dh {
        let sy = (dy * sh / dh).min(sh - 1);
        for dx in 0..dw {
            let sx = (dx * sw / dw).min(sw - 1);
            let s = (sy * sw + sx) * 3;
            let d = (dy * dw + dx) * 3;
            dst[d] = src[s]; dst[d + 1] = src[s + 1]; dst[d + 2] = src[s + 2];
        }
    }
}

pub fn resize_bgr(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let mut dst = vec![0u8; dw * dh * 3];
    resize_bgr_into(src, sw, sh, dw, dh, &mut dst);
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kacheln_decken_das_bild_mit_ueberlappung_ab() {
        let t = tile_regions(3840, 2160, 3, 2, 0.15);
        assert_eq!(t.len(), 6);
        assert!(t.iter().all(|r| r.x0 + r.w <= 3840 && r.y0 + r.h <= 2160));
        assert_eq!(t[0].x0, 0);
        assert_eq!(t[2].x0 + t[2].w, 3840);
        assert_eq!(t[5].y0 + t[5].h, 2160);
        // Nachbarkacheln überlappen sich
        assert!(t[0].x0 + t[0].w > t[1].x0);
        assert!(t[0].y0 + t[0].h > t[3].y0);
    }

    #[test]
    fn eine_kachel_ist_das_ganze_bild() {
        assert_eq!(tile_regions(3840, 2160, 1, 1, 0.15), vec![Region { x0: 0, y0: 0, w: 3840, h: 2160 }]);
    }

    #[test]
    fn letterbox_rechnet_punkte_zurueck() {
        let (fw, fh) = (3840usize, 2160usize);
        let frame = vec![0u8; fw * fh * 3];
        let mut buf = vec![0u8; 640 * 640 * 3];
        // Gesamtbild: 16:9 → oben/unten Rand
        let full = Region { x0: 0, y0: 0, w: fw, h: fh };
        let (_, lb) = letterbox_yolo(&frame, fw, &full, 640, &mut buf);
        assert_eq!(lb.pad_x, 0.0);
        assert_eq!(lb.pad_y, 140.0);
        let (x, y) = lb.to_frame(&full, 320.0, 320.0);
        assert!((x - 1920.0).abs() < 1.0 && (y - 1080.0).abs() < 1.0);
        // Kachel rechts unten
        let tile = Region { x0: 2400, y0: 1000, w: 1440, h: 1160 };
        let (_, lb) = letterbox_yolo(&frame, fw, &tile, 640, &mut buf);
        let (x, y) = lb.to_frame(&tile, lb.pad_x, lb.pad_y);
        assert!((x - 2400.0).abs() < 1.0 && (y - 1000.0).abs() < 1.0);
    }
}
