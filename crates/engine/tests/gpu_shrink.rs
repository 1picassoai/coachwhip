//! Lab timing: one expert of the official 122B file through the GPU shrink (Q3_K/Q5_K -> Q2_K).
use candle::quantized::gguf_file;
use candle::Device;
use coachwhip_engine::requant::{time_one_expert, SrcKind, QK_K};
use std::os::unix::fs::FileExt;

#[test]
fn gpu_shrink_one_expert() {
    let path = std::env::var("COACHWHIP_JIT_MODEL").unwrap_or_default();
    if path.is_empty() {
        eprintln!("set COACHWHIP_JIT_MODEL to the official 122B gguf to run this timing");
        return;
    }
    let dev = Device::new_metal(0).unwrap();
    let Device::Metal(metal) = &dev else { panic!("metal") };
    let mut file = std::fs::File::open(&path).unwrap();
    let ct = gguf_file::Content::read(&mut file).unwrap();
    let mut parts = Vec::new();
    for m in ["gate", "up", "down"] {
        let info = &ct.tensor_infos[&format!("blk.0.ffn_{m}_exps.weight")];
        let dims = info.shape.dims().to_vec();
        let per_expert: usize = dims[1..].iter().product();
        let bytes = per_expert / info.ggml_dtype.block_size() * info.ggml_dtype.type_size();
        let mut buf = vec![0u8; bytes];
        file.read_exact_at(&mut buf, ct.tensor_data_offset + info.offset).unwrap();
        let kind = SrcKind::of(info.ggml_dtype).unwrap_or_else(|| panic!("expert {m} is {:?}", info.ggml_dtype));
        parts.push((kind, buf, per_expert / QK_K));
    }
    // warm the pipelines once, then time
    let _ = time_one_expert(metal, &parts).unwrap();
    let mut total = 0.0;
    let reps = 20;
    let (mut bi, mut bo) = (0, 0);
    for _ in 0..reps {
        let (ms, a, b) = time_one_expert(metal, &parts).unwrap();
        total += ms;
        bi = a;
        bo = b;
    }
    eprintln!("GPU shrink, one expert: {:.2} MB -> {:.2} MB in {:.2} ms (mean of {reps}, includes buffer setup and the wait)", bi as f64 / 1e6, bo as f64 / 1e6, total / reps as f64);
}
