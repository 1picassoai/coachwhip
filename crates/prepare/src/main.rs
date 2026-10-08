//! Prepare a GGUF for Coachwhip: rewrite the expert tensors for speed on a 16 GB Mac and copy
//! everything else untouched, streaming, so the whole model never sits in memory.
//!
//!   coachwhip-prepare <in.gguf> <out.gguf>

use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use candle::quantized::gguf_file::{Content, Value, ValueType};
use candle::quantized::{GgmlDType, QStorage, QTensor};
use candle::Device;
use coachwhip_engine::requant::{Requant, SrcKind, Q2K_BLOCK_BYTES, QK_K};
use std::borrow::Cow;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::time::Instant;

/// What each tensor becomes. The experts carry the bytes per word; the rest stays as it came.
fn target(name: &str, dtype: GgmlDType) -> GgmlDType {
    if name.ends_with(".ffn_gate_exps.weight") || name.ends_with(".ffn_up_exps.weight") {
        GgmlDType::Q2K
    } else if name.ends_with(".ffn_down_exps.weight") {
        GgmlDType::Q3K
    } else {
        dtype
    }
}

fn value_type_code(t: ValueType) -> u32 {
    match t {
        ValueType::U8 => 0,
        ValueType::I8 => 1,
        ValueType::U16 => 2,
        ValueType::I16 => 3,
        ValueType::U32 => 4,
        ValueType::I32 => 5,
        ValueType::F32 => 6,
        ValueType::Bool => 7,
        ValueType::String => 8,
        ValueType::Array => 9,
        ValueType::U64 => 10,
        ValueType::I64 => 11,
        ValueType::F64 => 12,
    }
}

fn write_string<W: Write>(w: &mut W, s: &str) -> Result<()> {
    w.write_u64::<LittleEndian>(s.len() as u64)?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

fn write_value<W: Write>(w: &mut W, v: &Value) -> Result<()> {
    match v {
        Value::U8(x) => w.write_u8(*x)?,
        Value::I8(x) => w.write_i8(*x)?,
        Value::U16(x) => w.write_u16::<LittleEndian>(*x)?,
        Value::I16(x) => w.write_i16::<LittleEndian>(*x)?,
        Value::U32(x) => w.write_u32::<LittleEndian>(*x)?,
        Value::I32(x) => w.write_i32::<LittleEndian>(*x)?,
        Value::U64(x) => w.write_u64::<LittleEndian>(*x)?,
        Value::I64(x) => w.write_i64::<LittleEndian>(*x)?,
        Value::F32(x) => w.write_f32::<LittleEndian>(*x)?,
        Value::F64(x) => w.write_f64::<LittleEndian>(*x)?,
        Value::Bool(x) => w.write_u8(u8::from(*x))?,
        Value::String(s) => write_string(w, s)?,
        Value::Array(items) => {
            let t = items.first().map(|i| i.value_type()).unwrap_or(ValueType::U32);
            w.write_u32::<LittleEndian>(value_type_code(t))?;
            w.write_u64::<LittleEndian>(items.len() as u64)?;
            for item in items {
                write_value(w, item)?;
            }
        }
    }
    Ok(())
}

/// GGUF's type codes (Candle keeps its own mapping crate-private).
fn dtype_code(d: GgmlDType) -> u32 {
    match d {
        GgmlDType::F32 => 0,
        GgmlDType::F16 => 1,
        GgmlDType::Q4_0 => 2,
        GgmlDType::Q4_1 => 3,
        GgmlDType::Q5_0 => 6,
        GgmlDType::Q5_1 => 7,
        GgmlDType::Q8_0 => 8,
        GgmlDType::Q8_1 => 9,
        GgmlDType::Q2K => 10,
        GgmlDType::Q3K => 11,
        GgmlDType::Q4K => 12,
        GgmlDType::Q5K => 13,
        GgmlDType::Q6K => 14,
        GgmlDType::Q8K => 15,
        GgmlDType::BF16 => 30,
    }
}

fn bytes_for(dtype: GgmlDType, elems: usize) -> usize {
    elems / dtype.block_size() * dtype.type_size()
}

fn pad_to(w: &mut impl Write, pos: u64, align: u64) -> Result<u64> {
    let padding = (align - pos % align) % align;
    if padding > 0 {
        w.write_all(&vec![0u8; padding as usize])?;
    }
    Ok(pos + padding)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        bail!("usage: coachwhip-prepare <in.gguf> <out.gguf>");
    }
    let (src_path, dst_path) = (&args[1], &args[2]);
    let mut src = std::fs::File::open(src_path).with_context(|| format!("opening {src_path}"))?;
    let content = Content::read(&mut src).context("reading the GGUF header")?;
    let align = match content.metadata.get("general.alignment") {
        Some(Value::U32(a)) => *a as u64,
        _ => 32,
    };

    // Tensors in the order they sit in the source file.
    let mut infos: Vec<(&String, &candle::quantized::gguf_file::TensorInfo)> = content.tensor_infos.iter().collect();
    infos.sort_by_key(|(_, i)| i.offset);

    // New offsets, from the new sizes.
    let mut new_offsets = Vec::with_capacity(infos.len());
    let mut offset = 0u64;
    let mut changed = 0usize;
    for (name, info) in &infos {
        let to = target(name, info.ggml_dtype);
        if to != info.ggml_dtype {
            changed += 1;
        }
        new_offsets.push(offset);
        offset += bytes_for(to, info.shape.elem_count()) as u64;
        offset = (offset + align - 1) / align * align;
    }
    eprintln!("coachwhip-prepare: {} tensors, {changed} to rewrite, {:.1} GB out", infos.len(), offset as f64 / 1e9);

    // The GPU shrink for gate and up (Q3_K/Q4_K/Q5_K -> Q2_K); down goes the CPU way to Q3_K.
    let gpu = match Device::new_metal(0) {
        Ok(Device::Metal(m)) => match Requant::new(&m) {
            Ok(rq) => {
                eprintln!("coachwhip-prepare: squeezing gate and up on the GPU");
                Some((m, rq))
            }
            Err(e) => {
                eprintln!("coachwhip-prepare: GPU shrink unavailable ({e}); using the CPU");
                None
            }
        },
        _ => None,
    };
    let out = std::fs::File::create(dst_path).with_context(|| format!("creating {dst_path}"))?;
    let mut w = BufWriter::with_capacity(16 << 20, out);
    // Header.
    w.write_u32::<LittleEndian>(0x4655_4747)?;
    w.write_u32::<LittleEndian>(3)?;
    w.write_u64::<LittleEndian>(infos.len() as u64)?;
    w.write_u64::<LittleEndian>(content.metadata.len() as u64)?;
    let mut keys: Vec<&String> = content.metadata.keys().collect();
    keys.sort();
    for k in keys {
        let v = &content.metadata[k];
        write_string(&mut w, k)?;
        w.write_u32::<LittleEndian>(value_type_code(v.value_type()))?;
        write_value(&mut w, v)?;
    }
    for ((name, info), new_off) in infos.iter().zip(&new_offsets) {
        write_string(&mut w, name)?;
        let dims = info.shape.dims();
        w.write_u32::<LittleEndian>(dims.len() as u32)?;
        for &d in dims.iter().rev() {
            w.write_u64::<LittleEndian>(d as u64)?;
        }
        w.write_u32::<LittleEndian>(dtype_code(target(name, info.ggml_dtype)))?;
        w.write_u64::<LittleEndian>(*new_off)?;
    }
    w.flush()?;
    let pos = w.get_ref().metadata()?.len();
    let data_start = pad_to(&mut w, pos, align)?;

    // Data, tensor by tensor.
    let started = Instant::now();
    let mut written = data_start;
    let total = infos.len();
    for (i, ((name, info), new_off)) in infos.iter().zip(&new_offsets).enumerate() {
        let expect = data_start + new_off;
        if written != expect {
            bail!("offset drift at {name}: at {written}, expected {expect}");
        }
        let elems = info.shape.elem_count();
        let src_len = bytes_for(info.ggml_dtype, elems);
        src.seek(SeekFrom::Start(content.tensor_data_offset + info.offset))?;
        let mut buf = vec![0u8; src_len];
        src.read_exact(&mut buf).with_context(|| format!("reading {name}"))?;
        let to = target(name, info.ggml_dtype);
        if to == info.ggml_dtype {
            w.write_all(&buf)?;
            written += src_len as u64;
        } else if let (GgmlDType::Q2K, Some((metal, rq)), Some(kind)) = (to, gpu.as_ref(), SrcKind::of(info.ggml_dtype)) {
            // On the GPU: the whole tensor in one dispatch, one thread per 256-weight block.
            let n_blocks = elems / QK_K;
            let src_buf = metal.new_buffer_builder().with_size(src_len).with_label("prepare_src").build()?;
            unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr(), src_buf.contents() as *mut u8, src_len) };
            let out_len = n_blocks * Q2K_BLOCK_BYTES;
            let dst_buf = metal.new_buffer_builder().with_size(out_len).with_label("prepare_dst").build()?;
            rq.run(kind, &src_buf, 0, &dst_buf, 0, n_blocks)?;
            let data = unsafe { std::slice::from_raw_parts(dst_buf.contents() as *const u8, out_len) };
            w.write_all(data)?;
            written += out_len as u64;
        } else {
            let storage = QStorage::from_data(Cow::Owned(buf), &Device::Cpu, info.ggml_dtype)?;
            let q = QTensor::new(storage, info.shape.clone())?;
            let f = q.dequantize(&Device::Cpu)?;
            let q2 = QTensor::quantize(&f, to)?;
            let data = q2.data()?;
            w.write_all(&data)?;
            written += data.len() as u64;
        }
        written = pad_to(&mut w, written, align)?;
        if i % 10 == 0 || i + 1 == total {
            eprintln!("coachwhip-prepare: {}/{} tensors, {:.1} GB written, {:.0} s", i + 1, total, (written as f64) / 1e9, started.elapsed().as_secs_f64());
        }
    }
    w.flush()?;
    eprintln!("coachwhip-prepare: done, {:.1} GB in {:.0} s", written as f64 / 1e9, started.elapsed().as_secs_f64());
    Ok(())
}
