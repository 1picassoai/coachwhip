//! Does Candle's Metal Q2_K matvec agree with the CPU? Q3_K as the control.

use candle::quantized::{GgmlDType, QMatMul, QTensor};
use candle::{Device, Module, Tensor};

fn check(dtype: GgmlDType) -> (f32, f32) {
    let cpu = Device::Cpu;
    let metal = Device::new_metal(0).expect("metal");
    let w = Tensor::randn(0f32, 1.0, (512, 1024), &cpu).unwrap();
    let x = Tensor::randn(0f32, 1.0, (1, 1024), &cpu).unwrap();
    let qw = QTensor::quantize(&w, dtype).unwrap();
    let truth = QMatMul::from_qtensor(qw).unwrap().forward(&x).unwrap();
    let qm = QTensor::quantize(&w.to_device(&metal).unwrap(), dtype).unwrap();
    let got = QMatMul::from_qtensor(qm).unwrap().forward(&x.to_device(&metal).unwrap()).unwrap().to_device(&cpu).unwrap();
    let diff = (&truth - &got).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    let scale = truth.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    (diff, scale)
}

#[test]
fn q2k_metal_matches_cpu() {
    let (d3, s3) = check(GgmlDType::Q3K);
    let (d2, s2) = check(GgmlDType::Q2K);
    eprintln!("Q3_K: max diff {d3:.4} of {s3:.2}; Q2_K: max diff {d2:.4} of {s2:.2}");
    assert!(d3 < s3 * 0.05, "Q3_K metal differs from cpu");
    assert!(d2 < s2 * 0.05, "Q2_K metal differs from cpu");
}
