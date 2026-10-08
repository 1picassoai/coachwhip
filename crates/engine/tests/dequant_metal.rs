//! Does Candle's Metal dequantize agree with the CPU for every type the 122B files use?

use candle::quantized::{GgmlDType, QStorage, QTensor};
use candle::{Device, Tensor};

fn check(dtype: GgmlDType, shape: (usize, usize)) -> (f32, f32) {
    let cpu = Device::Cpu;
    let metal = Device::new_metal(0).expect("metal");
    let w = Tensor::randn(0f32, 1.0, shape, &cpu).unwrap();
    let q = QTensor::quantize(&w, dtype).unwrap();
    let truth = q.dequantize(&cpu).unwrap();
    let data = q.data().unwrap();
    let qm = QTensor::new(QStorage::from_data(data, &metal, dtype).unwrap(), q.shape().clone()).unwrap();
    let got = qm.dequantize(&metal).unwrap().to_device(&cpu).unwrap();
    let diff = (&truth - &got).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    let scale = truth.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    (diff, scale)
}

#[test]
fn dequantize_metal_matches_cpu() {
    let mut bad = vec![];
    for (name, dtype) in [("Q2_K", GgmlDType::Q2K), ("Q3_K", GgmlDType::Q3K), ("Q4_K", GgmlDType::Q4K), ("Q5_K", GgmlDType::Q5K), ("Q6_K", GgmlDType::Q6K)] {
        for shape in [(64, 1024), (3072, 8192)] {
            let (d, s) = check(dtype, shape);
            eprintln!("{name} {shape:?}: max diff {d:.5} of {s:.2}");
            if d > s * 1e-3 {
                bad.push(format!("{name} {shape:?}"));
            }
        }
    }
    assert!(bad.is_empty(), "metal dequantize differs from cpu: {bad:?}");
}
