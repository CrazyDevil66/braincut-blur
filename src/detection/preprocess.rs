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

pub fn preprocess_scrfd(frame: &[u8], fw: usize, fh: usize, buf: &mut [u8]) -> Vec<f32> {
    let (in_h, in_w) = (640usize, 640usize);
    resize_bgr_into(frame, fw, fh, in_w, in_h, buf);
    // BGR→RGB, (pixel − 127.5) / 128  [InsightFace buffalo_l/sc Normalisierung]
    let mut input = vec![0f32; 3 * in_h * in_w];
    for y in 0..in_h {
        for x in 0..in_w {
            let s = (y * in_w + x) * 3;
            input[y * in_w + x]                   = (buf[s + 2] as f32 - 127.5) / 128.0;
            input[in_h * in_w + y * in_w + x]     = (buf[s + 1] as f32 - 127.5) / 128.0;
            input[2 * in_h * in_w + y * in_w + x] = (buf[s]     as f32 - 127.5) / 128.0;
        }
    }
    input
}

pub fn preprocess_yolo(frame: &[u8], fw: usize, fh: usize, buf: &mut [u8]) -> Vec<f32> {
    let (in_h, in_w) = (640usize, 640usize);
    resize_bgr_into(frame, fw, fh, in_w, in_h, buf);
    // BGR→RGB, / 255
    let mut input = vec![0f32; 3 * in_h * in_w];
    for y in 0..in_h {
        for x in 0..in_w {
            let s = (y * in_w + x) * 3;
            input[y * in_w + x]                   = buf[s + 2] as f32 / 255.0;
            input[in_h * in_w + y * in_w + x]     = buf[s + 1] as f32 / 255.0;
            input[2 * in_h * in_w + y * in_w + x] = buf[s]     as f32 / 255.0;
        }
    }
    input
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
