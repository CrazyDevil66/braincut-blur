use std::collections::HashMap;

use crate::detection::BBox;

pub fn draw_rect_rgb(buf: &mut [u8], pw: usize, ph: usize, x1: usize, y1: usize, x2: usize, y2: usize, color: [u8; 3], thickness: usize) {
    for t in 0..thickness {
        let top    = y1.saturating_add(t);
        let bottom = y2.saturating_sub(1).saturating_sub(t);
        let left   = x1.saturating_add(t);
        let right  = x2.saturating_sub(1).saturating_sub(t);
        for x in x1..x2.min(pw) {
            if top < ph    { let i = (top * pw + x) * 3;    buf[i] = color[0]; buf[i+1] = color[1]; buf[i+2] = color[2]; }
            if bottom < ph && bottom != top { let i = (bottom * pw + x) * 3; buf[i] = color[0]; buf[i+1] = color[1]; buf[i+2] = color[2]; }
        }
        for y in y1..y2.min(ph) {
            if left  < pw  { let i = (y * pw + left)  * 3; buf[i] = color[0]; buf[i+1] = color[1]; buf[i+2] = color[2]; }
            if right < pw && right != left { let i = (y * pw + right) * 3; buf[i] = color[0]; buf[i+1] = color[1]; buf[i+2] = color[2]; }
        }
    }
}

pub fn make_preview(
    frame: &[u8],
    fw: usize, fh: usize,
    face_buf: &HashMap<(usize, usize, usize, usize), (BBox, u32)>,
    plate_buf: &HashMap<(usize, usize, usize, usize), (BBox, u32)>,
) -> Vec<u8> {
    let max_w = 960usize;
    let (pw, ph) = if fw > max_w { (max_w, (fh * max_w / fw).max(1)) } else { (fw.max(1), fh.max(1)) };
    let sx = pw as f64 / fw as f64;
    let sy = ph as f64 / fh as f64;

    let mut rgb = vec![0u8; pw * ph * 3];
    for dy in 0..ph {
        let src_y = (dy * fh / ph).min(fh - 1);
        for dx in 0..pw {
            let src_x = (dx * fw / pw).min(fw - 1);
            let s = (src_y * fw + src_x) * 3;
            let d = (dy * pw + dx) * 3;
            rgb[d] = frame[s + 2]; rgb[d+1] = frame[s + 1]; rgb[d+2] = frame[s]; // BGR→RGB
        }
    }
    for (_, (bf, _)) in face_buf {
        draw_rect_rgb(&mut rgb, pw, ph,
            (bf.x as f64 * sx) as usize, (bf.y as f64 * sy) as usize,
            (bf.x2 as f64 * sx) as usize, (bf.y2 as f64 * sy) as usize,
            [255, 60, 60], 2);
    }
    for (_, (bp, _)) in plate_buf {
        draw_rect_rgb(&mut rgb, pw, ph,
            (bp.x as f64 * sx) as usize, (bp.y as f64 * sy) as usize,
            (bp.x2 as f64 * sx) as usize, (bp.y2 as f64 * sy) as usize,
            [255, 200, 0], 2);
    }

    let mut out = Vec::new();
    let _ = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 65)
        .encode(&rgb, pw as u32, ph as u32, image::ColorType::Rgb8);
    out
}
