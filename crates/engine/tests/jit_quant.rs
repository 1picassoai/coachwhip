//! Lab timing: how long does it take to shrink one expert of the official 122B file from its
//! Q3_K_M bytes to Q2_K on the CPU (dequantise, then quantise)? Decides whether just-in-time
//! quantisation can ride on the reader threads or needs its own.
use candle::quantized::{ggml_file::qtensor_from_ggml, gguf_file, GgmlDType, QTensor};
use candle::Device;
use std::os::unix::fs::FileExt;
use std::time::Instant;

#[test]
fn jit_quant_one_expert() {
    let path = std::env::var("COACHWHIP_JIT_MODEL").unwrap_or_default();
    if path.is_empty() {
        eprintln!("set COACHWHIP_JIT_MODEL to the official 122B gguf to run this timing");
        return;
    }
    let mut file = std::fs::File::open(&path).unwrap();
    let ct = gguf_file::Content::read(&mut file).unwrap();
    let dev = Device::Cpu;
    let mut total_dq = 0.0;
    let mut total_q = 0.0;
    let mut bytes_in = 0usize;
    let mut bytes_out = 0usize;
    for m in ["gate", "up", "down"] {
        let info = &ct.tensor_infos[&format!("blk.0.ffn_{m}_exps.weight")];
        let dims = info.shape.dims().to_vec();
        let per_expert: usize = dims[1..].iter().product();
        let bytes = per_expert / info.ggml_dtype.block_size() * info.ggml_dtype.type_size();
        let mut buf = vec![0u8; bytes];
        file.read_exact_at(&mut buf, ct.tensor_data_offset + info.offset).unwrap();
        let t0 = Instant::now();
        let q = qtensor_from_ggml(info.ggml_dtype, &buf, dims[1..].to_vec(), &dev).unwrap();
        let f = q.dequantize(&dev).unwrap();
        let dq = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        let q2 = QTensor::quantize(&f, GgmlDType::Q2K).unwrap();
        let qt = t1.elapsed().as_secs_f64() * 1e3;
        let out = per_expert / GgmlDType::Q2K.block_size() * GgmlDType::Q2K.type_size();
        eprintln!("{m}: {:?} {} MB -> Q2_K {} MB: dequantise {dq:.1} ms, quantise {qt:.1} ms", info.ggml_dtype, bytes as f64 / 1e6, out as f64 / 1e6);
        let _ = q2;
        total_dq += dq;
        total_q += qt;
        bytes_in += bytes;
        bytes_out += out;
    }
    eprintln!(
        "one expert: {:.2} MB -> {:.2} MB; dequantise {:.1} ms + quantise {:.1} ms = {:.1} ms on one CPU core",
        bytes_in as f64 / 1e6,
        bytes_out as f64 / 1e6,
        total_dq,
        total_q,
        total_dq + total_q
    );
}
