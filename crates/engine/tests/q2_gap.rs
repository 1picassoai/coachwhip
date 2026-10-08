//! How much worse is the GPU shrink's Q2_K than candle's own Q2_K quantiser, on the same expert?
use candle::quantized::{ggml_file::qtensor_from_ggml, gguf_file, GgmlDType, QTensor};
use candle::Device;
use coachwhip_engine::requant::{Requant, SrcKind, Q2K_BLOCK_BYTES, QK_K};
use std::os::unix::fs::FileExt;

fn rel_err(a: &[f32], b: &[f32]) -> f64 {
    let (mut se, mut tr) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x - *y) as f64;
        se += d * d;
        tr += (*y as f64) * (*y as f64);
    }
    (se / tr).sqrt()
}

#[test]
fn gpu_q2_vs_candle_q2() {
    let path = std::env::var("COACHWHIP_JIT_MODEL").unwrap_or_default();
    if path.is_empty() {
        return;
    }
    let dev = Device::new_metal(0).unwrap();
    let Device::Metal(metal) = &dev else { panic!("metal") };
    let rq = Requant::new(metal).unwrap();
    let mut file = std::fs::File::open(&path).unwrap();
    let ct = gguf_file::Content::read(&mut file).unwrap();
    for (layer, m) in [(0, "gate"), (0, "up"), (0, "down"), (20, "gate"), (20, "down")] {
        let info = &ct.tensor_infos[&format!("blk.{layer}.ffn_{m}_exps.weight")];
        let dims = info.shape.dims().to_vec();
        let (n, k) = (dims[1], dims[2]);
        let per_expert = n * k;
        let bytes = per_expert / info.ggml_dtype.block_size() * info.ggml_dtype.type_size();
        let mut buf = vec![0u8; bytes];
        file.read_exact_at(&mut buf, ct.tensor_data_offset + info.offset).unwrap();
        let truth_q = qtensor_from_ggml(info.ggml_dtype, &buf, vec![n, k], &Device::Cpu).unwrap();
        let truth_t = truth_q.dequantize(&Device::Cpu).unwrap();
        let truth: Vec<f32> = truth_t.flatten_all().unwrap().to_vec1().unwrap();
        // candle's own Q2_K on the CPU
        let cq = QTensor::quantize(&truth_t, GgmlDType::Q2K).unwrap();
        let candle_back: Vec<f32> = cq.dequantize(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        // the GPU shrink
        let kind = SrcKind::of(info.ggml_dtype).unwrap();
        let n_blocks = per_expert / QK_K;
        let src = metal.new_buffer_builder().with_size(bytes).with_label("src").build().unwrap();
        unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr(), src.contents() as *mut u8, bytes) };
        let dst = metal.new_buffer_builder().with_size(n_blocks * Q2K_BLOCK_BYTES).with_label("dst").build().unwrap();
        rq.run(kind, &src, 0, &dst, 0, n_blocks).unwrap();
        let q2 = unsafe { std::slice::from_raw_parts(dst.contents() as *const u8, n_blocks * Q2K_BLOCK_BYTES) }.to_vec();
        let gpu_back: Vec<f32> = qtensor_from_ggml(GgmlDType::Q2K, &q2, vec![n, k], &Device::Cpu).unwrap().dequantize(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        // and candle's Q3_K, as the yardstick for what "keep it at Q3" costs
        let c3 = QTensor::quantize(&truth_t, GgmlDType::Q3K).unwrap();
        let c3_back: Vec<f32> = c3.dequantize(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        eprintln!(
            "layer {layer} {m} ({:?}): candle Q2_K error {:.3}, GPU shrink Q2_K error {:.3}, candle Q3_K error {:.3}",
            info.ggml_dtype,
            rel_err(&candle_back, &truth),
            rel_err(&gpu_back, &truth),
            rel_err(&c3_back, &truth)
        );
    }
}
