// In-place image operations on raw BGR frame buffers.
// Frames: [H, W, 3] as contiguous &mut [u8], row-major, BGR channels.

/// Gaussian blur via 3-pass box blur (good approximation, fast).
pub fn gaussian_blur_roi(
    frame: &mut [u8],
    frame_w: usize,
    x: usize,
    y: usize,
    x2: usize,
    y2: usize,
) {
    let x = x.min(frame_w.saturating_sub(1));
    let y2 = y2.min(frame.len() / (frame_w * 3) );
    let x2 = x2.min(frame_w);
    if x >= x2 || y >= y2 {
        return;
    }
    let roi_w = x2 - x;
    let roi_h = y2 - y;
    if roi_w < 4 || roi_h < 4 {
        return;
    }
    // Radius: ~30% of the shorter side, clamped, must be odd
    let r = ((roi_w.min(roi_h) / 4).max(3)) | 1;
    // Three box-blur passes approximates Gaussian
    box_blur_roi(frame, frame_w, x, y, x2, y2, r);
    box_blur_roi(frame, frame_w, x, y, x2, y2, r);
    box_blur_roi(frame, frame_w, x, y, x2, y2, r);
}

fn box_blur_roi(
    frame: &mut [u8],
    frame_w: usize,
    x: usize,
    y: usize,
    x2: usize,
    y2: usize,
    r: usize,
) {
    let roi_w = x2 - x;
    let roi_h = y2 - y;
    let mut tmp = vec![0u8; roi_w * roi_h * 3];

    // Horizontal pass with sliding window — O(roi_w × roi_h) instead of O(roi_w × roi_h × r)
    for row in 0..roi_h {
        let row_base = (y + row) * frame_w;
        let init_end = r.min(roi_w - 1);
        let mut sum = [0u32; 3];
        for c in 0..=init_end {
            let px = (row_base + x + c) * 3;
            sum[0] += frame[px] as u32;
            sum[1] += frame[px + 1] as u32;
            sum[2] += frame[px + 2] as u32;
        }
        let mut cnt = (init_end + 1) as u32;
        for col in 0..roi_w {
            let dst = (row * roi_w + col) * 3;
            tmp[dst]     = (sum[0] / cnt) as u8;
            tmp[dst + 1] = (sum[1] / cnt) as u8;
            tmp[dst + 2] = (sum[2] / cnt) as u8;
            let add_col = col + r + 1;
            if add_col < roi_w {
                let px = (row_base + x + add_col) * 3;
                sum[0] += frame[px] as u32;
                sum[1] += frame[px + 1] as u32;
                sum[2] += frame[px + 2] as u32;
                cnt += 1;
            }
            if col >= r {
                let px = (row_base + x + col - r) * 3;
                sum[0] -= frame[px] as u32;
                sum[1] -= frame[px + 1] as u32;
                sum[2] -= frame[px + 2] as u32;
                cnt -= 1;
            }
        }
    }

    // Vertical pass with sliding window — O(roi_w × roi_h) instead of O(roi_w × roi_h × r)
    for col in 0..roi_w {
        let init_end = r.min(roi_h - 1);
        let mut sum = [0u32; 3];
        for rr in 0..=init_end {
            let src = (rr * roi_w + col) * 3;
            sum[0] += tmp[src] as u32;
            sum[1] += tmp[src + 1] as u32;
            sum[2] += tmp[src + 2] as u32;
        }
        let mut cnt = (init_end + 1) as u32;
        for row in 0..roi_h {
            let px = ((y + row) * frame_w + x + col) * 3;
            if px + 2 < frame.len() {
                frame[px]     = (sum[0] / cnt) as u8;
                frame[px + 1] = (sum[1] / cnt) as u8;
                frame[px + 2] = (sum[2] / cnt) as u8;
            }
            let add_row = row + r + 1;
            if add_row < roi_h {
                let src = (add_row * roi_w + col) * 3;
                sum[0] += tmp[src] as u32;
                sum[1] += tmp[src + 1] as u32;
                sum[2] += tmp[src + 2] as u32;
                cnt += 1;
            }
            if row >= r {
                let src = ((row - r) * roi_w + col) * 3;
                sum[0] -= tmp[src] as u32;
                sum[1] -= tmp[src + 1] as u32;
                sum[2] -= tmp[src + 2] as u32;
                cnt -= 1;
            }
        }
    }
}

/// Verpixelt den Ausschnitt mit Blöcken von einem Drittel seiner Höhe (mindestens 8 px),
/// damit Schrift auf Kennzeichen auch bei großen Boxen nicht lesbar bleibt.
pub fn pixelate_roi(
    frame: &mut [u8],
    frame_w: usize,
    x: usize,
    y: usize,
    x2: usize,
    y2: usize,
) {
    let frame_h = frame.len() / (frame_w * 3);
    let x = x.min(frame_w.saturating_sub(1));
    let y = y.min(frame_h.saturating_sub(1));
    let x2 = x2.min(frame_w);
    let y2 = y2.min(frame_h);
    if x >= x2 || y >= y2 {
        return;
    }
    let roi_w = x2 - x;
    let roi_h = y2 - y;
    if roi_w < 2 || roi_h < 2 {
        return;
    }
    let block = (roi_h / 3).max(8);
    let bw = roi_w.div_ceil(block).max(1);
    let bh = roi_h.div_ceil(block).max(1);

    // Downsample: average bw×bh blocks
    let mut small = vec![0u8; bw * bh * 3];
    for by in 0..bh {
        for bx in 0..bw {
            let ry0 = (by * roi_h / bh).min(roi_h - 1);
            let ry1 = ((by + 1) * roi_h / bh).min(roi_h);
            let rx0 = (bx * roi_w / bw).min(roi_w - 1);
            let rx1 = ((bx + 1) * roi_w / bw).min(roi_w);
            let cnt = ((ry1 - ry0) * (rx1 - rx0)) as u32;
            let mut sum = [0u32; 3];
            for ry in ry0..ry1 {
                for rx in rx0..rx1 {
                    let px = ((y + ry) * frame_w + (x + rx)) * 3;
                    if px + 2 < frame.len() {
                        sum[0] += frame[px] as u32;
                        sum[1] += frame[px + 1] as u32;
                        sum[2] += frame[px + 2] as u32;
                    }
                }
            }
            let dst = (by * bw + bx) * 3;
            if cnt > 0 {
                small[dst] = (sum[0] / cnt) as u8;
                small[dst + 1] = (sum[1] / cnt) as u8;
                small[dst + 2] = (sum[2] / cnt) as u8;
            }
        }
    }

    // Upsample: nearest-neighbour back into frame
    for ry in 0..roi_h {
        for rx in 0..roi_w {
            let by = (ry * bh / roi_h).min(bh - 1);
            let bx = (rx * bw / roi_w).min(bw - 1);
            let src = (by * bw + bx) * 3;
            let px = ((y + ry) * frame_w + (x + rx)) * 3;
            if px + 2 < frame.len() && src + 2 < small.len() {
                frame[px] = small[src];
                frame[px + 1] = small[src + 1];
                frame[px + 2] = small[src + 2];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verpixelung_nutzt_bloecke_nach_der_hoehe() {
        let (w, h) = (400usize, 200usize);
        let original: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
        let mut frame = original.clone();
        // Box 300×90 → Blöcke von 30 px
        pixelate_roi(&mut frame, w, 50, 50, 350, 140);
        let px = |f: &[u8], x: usize, y: usize| f[(y * w + x) * 3..(y * w + x) * 3 + 3].to_vec();
        assert_eq!(px(&frame, 50, 50), px(&frame, 79, 79));
        assert_ne!(px(&frame, 50, 50), px(&frame, 80, 50));
        assert_ne!(px(&frame, 50, 50), px(&frame, 50, 80));
        assert_eq!(px(&frame, 10, 10), px(&original, 10, 10));
        assert_eq!(px(&frame, 360, 150), px(&original, 360, 150));
    }
}
