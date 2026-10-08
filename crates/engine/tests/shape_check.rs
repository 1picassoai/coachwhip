//! Does the shaped shrink reconstruct an expert better than the plain one, measured against the
//! file's own values, weighted by a real token shape? Offline, one expert, no engine run.
use candle::quantized::{ggml_file::qtensor_from_ggml, gguf_file, GgmlDType};
use candle::{Device, Tensor};
use coachwhip_engine::requant::{Requant, SrcKind, Q2K_BLOCK_BYTES, QK_K};
use std::os::unix::fs::FileExt;

fn dequant_bytes(dtype: GgmlDType, bytes: &[u8], dims: Vec<usize>, dev: &Device) -> Vec<f32> {
    let q = qtensor_from_ggml(dtype, bytes, dims, dev).unwrap();
    q.dequantize(dev).unwrap().to_device(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

#[test]
fn shaped_vs_plain_reconstruction() {
    let path = std::env::var("COACHWHIP_JIT_MODEL").unwrap_or_default();
    let shape_path = std::env::var("COACHWHIP_SHAPE_FILE").unwrap_or_default();
    if path.is_empty() || shape_path.is_empty() {
        eprintln!("set COACHWHIP_JIT_MODEL and COACHWHIP_SHAPE_FILE (f32 x^2 over the hidden size, layer 0)");
        return;
    }
    let dev = Device::new_metal(0).unwrap();
    let Device::Metal(metal) = &dev else { panic!("metal") };
    let rq = Requant::new(metal).unwrap();
    let mut file = std::fs::File::open(&path).unwrap();
    let ct = gguf_file::Content::read(&mut file).unwrap();
    let shape: Vec<f32> = std::fs::read(&shape_path).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    for m in ["gate", "up"] {
        let info = &ct.tensor_infos[&format!("blk.0.ffn_{m}_exps.weight")];
        let dims = info.shape.dims().to_vec(); // (experts, n, k)
        let (n, k) = (dims[1], dims[2]);
        assert_eq!(k, shape.len(), "shape length");
        let per_expert = n * k;
        let bytes = per_expert / info.ggml_dtype.block_size() * info.ggml_dtype.type_size();
        let mut buf = vec![0u8; bytes];
        file.read_exact_at(&mut buf, ct.tensor_data_offset + info.offset).unwrap();
        let kind = SrcKind::of(info.ggml_dtype).unwrap();
        let n_blocks = per_expert / QK_K;
        let truth = dequant_bytes(info.ggml_dtype, &buf, vec![n, k], &dev);
        let src = metal.new_buffer_builder().with_size(bytes).with_label("src").build().unwrap();
        unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr(), src.contents() as *mut u8, bytes) };
        let out_bytes = n_blocks * Q2K_BLOCK_BYTES;
        let mut results = Vec::new();
        for shaped in [false, true] {
            let dst = metal.new_buffer_builder().with_size(out_bytes).with_label("dst").build().unwrap();
            if shaped {
                let sb = metal.new_buffer_builder().with_size(k * 4).with_label("shape").build().unwrap();
                unsafe { std::ptr::copy_nonoverlapping(shape.as_ptr() as *const u8, sb.contents() as *mut u8, k * 4) };
                rq.run_shaped(kind, &src, 0, &dst, 0, n_blocks, &sb, k).unwrap();
            } else {
                rq.run(kind, &src, 0, &dst, 0, n_blocks).unwrap();
            }
            let q2 = unsafe { std::slice::from_raw_parts(dst.contents() as *const u8, out_bytes) }.to_vec();
            let back = dequant_bytes(GgmlDType::Q2K, &q2, vec![n, k], &dev);
            // errors: plain squared error, and squared error weighted by the shape per column
            let (mut se, mut wse, mut ref_w) = (0.0f64, 0.0f64, 0.0f64);
            for r in 0..n {
                for c in 0..k {
                    let i = r * k + c;
                    let d = (back[i] - truth[i]) as f64;
                    se += d * d;
                    wse += d * d * shape[c] as f64;
                    ref_w += (truth[i] as f64).powi(2) * shape[c] as f64;
                }
            }
            let tr: f64 = truth.iter().map(|v| (*v as f64).powi(2)).sum();
            results.push((shaped, (se / tr).sqrt(), (wse / ref_w).sqrt()));
        }
        // the output of the expert on a random-ish token: y = W x, compare directions
        let x: Vec<f32> = shape.iter().map(|s| s.sqrt()).collect();
        let xt = Tensor::from_vec(x, (k, 1), &Device::Cpu).unwrap();
        let wt = Tensor::from_vec(truth.clone(), (n, k), &Device::Cpu).unwrap();
        let y_true = wt.matmul(&xt).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (shaped, rel, wrel) in &results {
            eprintln!("{m}: shaped={shaped}: relative error {:.3}, shape-weighted relative error {:.3}", rel, wrel);
        }
        let _ = y_true;
    }
}
