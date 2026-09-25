use super::BBox;

pub fn greedy_nms(mut candidates: Vec<(BBox, f32)>, iou_thresh: f32, max_boxes: usize) -> Vec<BBox> {
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<BBox> = Vec::new();
    'outer: for (b, _) in &candidates {
        for kb in &kept {
            if bbox_iou(b, kb) > iou_thresh {
                continue 'outer;
            }
        }
        kept.push(*b);
        if kept.len() >= max_boxes { break; }
    }
    kept
}

pub fn bbox_iou(a: &BBox, b: &BBox) -> f32 {
    let ix1 = a.x.max(b.x); let iy1 = a.y.max(b.y);
    let ix2 = a.x2.min(b.x2); let iy2 = a.y2.min(b.y2);
    if ix2 <= ix1 || iy2 <= iy1 { return 0.0; }
    let inter = ((ix2 - ix1) * (iy2 - iy1)) as f32;
    let area_a = ((a.x2 - a.x) * (a.y2 - a.y)) as f32;
    let area_b = ((b.x2 - b.x) * (b.y2 - b.y)) as f32;
    inter / (area_a + area_b - inter)
}
