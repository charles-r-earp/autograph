use super::Conv2Options;
use crate::{
    learn::neural_network::autograd::{Variable, Variable4},
    tensor::{
        parallel::{
            broadcast, parallel_size, SyncRawArrayViewMut, SyncRawArrayViewMut4,
            SyncRawArrayViewMut5,
        },
        ScalarTensor, ScalarTensor4, ScalarTensorBase, ScalarTensorView4, Tensor,
    },
};
use anyhow::{Error, Result};
use dry::macro_for;
use krnl::{buffer::ScalarData, scalar::ScalarType};
use ndarray::{Array, Array4, ArrayView4, ArrayViewMut4, Dimension, IntoDimension, Ix4};
use once_cell::sync::OnceCell;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::{mem::size_of, sync::Arc};
#[cfg(debug_assertions)]
use unchecked_index::GetUnchecked;
use wide::f32x8;

pub(super) fn conv2_direct(
    input: Variable4,
    weight: Variable4,
    options: &Conv2Options,
) -> Result<Variable4> {
    let (_bs, ic, _ih, _iw) = input.dim();
    let (oc, _ic, fh, fw) = weight.dim();
    debug_assert_eq!(ic, _ic);
    let input_packed = PackedInput::try_from(input.value().clone())?;
    let weight_packed = PackedWeight::try_from(weight.value().clone())?;
    let output = conv2_direct_packed(&input_packed, ic, &weight_packed, options)?;
    if input.node().is_some() || weight.node().is_some() {
        if input.device().is_host() {
            let mut builder = Variable::builder();
            let output_packed_cell = Arc::new(OnceCell::new());
            if let Some(node) = input.node() {
                let options = options.clone();
                let output_packed_cell = output_packed_cell.clone();
                builder.edge(node, move |output_grad| {
                    let output_grad = output_packed_cell
                        .get_or_init(|| PackedOutput::try_from(output_grad).unwrap());
                    conv2_direct_backward_input_packed(weight_packed, ic, output_grad, oc, &options)
                        .map(Into::into)
                });
            }
            if let Some(node) = weight.node() {
                let options = options.clone();
                builder.edge(node, move |output_grad| {
                    let output_grad = output_packed_cell
                        .get_or_init(|| PackedOutput::try_from(output_grad).unwrap());
                    conv2_direct_backward_weight_packed(
                        &input_packed,
                        output_grad,
                        &options,
                        [oc, ic, fh, fw],
                    )
                    .map(Into::into)
                });
            }
            Ok(builder.build(output.unpack2(oc)?.into()))
        } else {
            let mut builder = Variable::builder();
            if let Some(node) = input.node() {
                builder.edge(node, move |_output_grad| todo!());
            }
            if let Some(node) = weight.node() {
                builder.edge(node, move |_output_grad| todo!());
            }
            let output_packed = builder.build(output.into_tensor_packed().unwrap().into());
            let mut builder = Variable::builder();
            builder.edge(output_packed.node().unwrap(), |_output_grad| todo!());
            #[allow(unreachable_code)]
            Ok(builder.build(todo!()))
        }
    } else {
        output.unpack2(oc).map(Into::into)
    }
}

pub(super) fn conv2_direct_forward(
    input: ScalarTensorView4,
    weight: ScalarTensorView4,
    options: &Conv2Options,
) -> Result<ScalarTensor4> {
    let (_bs, ic, _ih, _iw) = input.dim();
    let (oc, _ic, _fh, _fw) = weight.dim();
    debug_assert_eq!(ic, _ic);
    let input = PackedInput::try_from(input)?;
    let weight = PackedWeight::try_from(weight)?;
    conv2_direct_packed(&input, ic, &weight, options)?.unpack2(oc)
}

pub(super) fn conv2_direct_backward_input(
    weight: ScalarTensorView4,
    output_grad: ScalarTensorView4,
    options: &Conv2Options,
) -> Result<ScalarTensor4> {
    let (oc, ic, _fh, _fw) = weight.dim();
    let (_bs, _oc, _oh, _ow) = output_grad.dim();
    debug_assert_eq!(oc, _oc);
    let weight = PackedWeight::try_from(weight)?;
    let output_grad = PackedOutput::try_from(output_grad)?;
    conv2_direct_backward_input_packed(weight, ic, &output_grad, oc, options)
}

pub(super) fn conv2_direct_backward_weight(
    input: ScalarTensorView4,
    output_grad: ScalarTensorView4,
    filter: [usize; 2],
    options: &Conv2Options,
) -> Result<ScalarTensor4> {
    let (_bs1, ic, _ih, _iw) = input.dim();
    let (_bs2, oc, _oh, _ow) = output_grad.dim();
    let [fh, fw] = filter;
    debug_assert_eq!(_bs1, _bs2);
    let input = PackedInput::try_from(input)?;
    let output_grad = PackedOutput::try_from(output_grad)?;
    conv2_direct_backward_weight_packed(&input, &output_grad, options, [oc, ic, fh, fw])
}

fn conv2_direct_packed<S: ScalarData>(
    input: &PackedInput<S, Ix4>,
    ic: usize,
    weight: &PackedWeight<Ix4>,
    options: &Conv2Options,
) -> Result<PackedOutput<Ix4>> {
    match input {
        PackedInput::Tensor(input) => {
            if input.device().is_host() && input.scalar_type() == ScalarType::F32 {
                let x = input.view().try_into_tensor_view::<f32>().unwrap();
                let x: ArrayView4<f32> = x.as_array().unwrap();
                let x: ArrayView4<[f32; 1]> = unsafe { std::mem::transmute(x) };
                let w = weight.array_1xf32x8().unwrap();
                let y = conv2_direct_host_f32(x, ic, w.view(), options);
                return Ok(PackedOutput::Arrayf32x8(y));
            }
            todo!()
        }
        //#[cfg(feature = "device")]
        //PackedInput::TensorPacked(_) => todo!(),
        PackedInput::Array8xf32(input) => {
            let w = weight.array_8xf32x8().unwrap();
            let y = conv2_direct_host_f32(input.view(), ic, w.view(), options);
            Ok(PackedOutput::Arrayf32x8(y))
        }
    }
}

fn conv2_direct_backward_input_packed(
    weight: PackedWeight<Ix4>,
    ic: usize,
    output_grad: &PackedOutput<Ix4>,
    oc: usize,
    options: &Conv2Options,
) -> Result<ScalarTensor4> {
    match weight {
        PackedWeight::Array1xf32x8(weight) => {
            let output_grad = output_grad.array_f32x8().unwrap();
            let input_grad: Array4<[f32; 1]> = conv2_direct_backward_input_host_f32(
                weight.view(),
                output_grad.view(),
                oc,
                options,
            );
            let input_grad: Array4<f32> = unsafe { std::mem::transmute(input_grad) };
            Ok(Tensor::from(input_grad).into())
        }
        PackedWeight::Array8xf32x8(mut weight) => {
            transpose_weight_host_f32(weight.view_mut());
            let output_grad = output_grad.array_f32x8().unwrap();
            let input_grad = conv2_direct_backward_input_host_f32(
                weight.view(),
                output_grad.view(),
                oc,
                options,
            );
            let input_grad = unpack_input2_host_f32(input_grad.view(), ic);
            Ok(Tensor::from(input_grad).into())
        }
        _ => todo!(),
    }
}

fn conv2_direct_backward_weight_packed<S: ScalarData>(
    input: &PackedInput<S, Ix4>,
    output_grad: &PackedOutput<Ix4>,
    options: &Conv2Options,
    weight_shape: [usize; 4],
) -> Result<ScalarTensor4> {
    let [oc, ic, _fh, _fw] = weight_shape;
    match input {
        PackedInput::Tensor(input) => {
            let input = input.view().try_into_tensor_view().unwrap();
            let input: ArrayView4<f32> = input.as_array().unwrap();
            let input: ArrayView4<[f32; 1]> = unsafe { std::mem::transmute(input) };
            let output_grad = output_grad.array_f32x8().unwrap();
            let weight_grad = conv2_direct_backward_weight_host_f32(
                input,
                output_grad.view(),
                options,
                weight_shape,
            );
            let weight_grad = unpack_weight2_host_f32(weight_grad.view(), ic, oc);
            Ok(Tensor::from(weight_grad).into())
        }
        //#[cfg(feature = "device")]
        //PackedInput::TensorPacked(_) => todo!(),
        PackedInput::Array8xf32(input) => {
            let output_grad = output_grad.array_f32x8().unwrap();
            let weight_grad = conv2_direct_backward_weight_host_f32(
                input.view(),
                output_grad.view(),
                options,
                weight_shape,
            );
            let weight_grad = unpack_weight2_host_f32(weight_grad.view(), ic, oc);
            Ok(Tensor::from(weight_grad).into())
        }
    }
}

const fn div_up(a: usize, b: usize) -> usize {
    a / b + (a % b != 0) as usize
}

enum PackedInput<S: ScalarData, D: Dimension> {
    Tensor(ScalarTensorBase<S, D>),
    //#[cfg(feature = "device")]
    //TensorPacked(ScalarTensor<D::Larger>),
    Array8xf32(Array<[f32; 8], D>),
}

/*
impl<S: ScalarData, D: Dimension> PackedInput<S, D> {
    fn tensor_packed(&self) -> Option<&ScalarTensor<D::Larger>> {
        if let Self::TensorPacked(x) = self {
            Some(x)
        } else {
            None
        }
    }
}*/

impl<S: ScalarData> TryFrom<ScalarTensorBase<S, Ix4>> for PackedInput<S, Ix4> {
    type Error = Error;
    fn try_from(input: ScalarTensorBase<S, Ix4>) -> Result<Self> {
        let (_bs, ic, _ih, _iw) = input.dim();
        if ic == 1 {
            return Ok(Self::Tensor(input));
        }
        if input.device().is_host() && input.scalar_type() == ScalarType::F32 {
            let x = input.view().try_into_tensor_view::<f32>().unwrap();
            let x_packed = pack_input2_host_f32(x.as_array().unwrap());
            return Ok(Self::Array8xf32(x_packed));
        }
        todo!()
    }
}

enum PackedWeight<D: Dimension> {
    #[allow(dead_code)]
    TensorPacked(ScalarTensor<D::Larger>),
    Array1xf32x8(Array<[f32x8; 1], D>),
    Array8xf32x8(Array<[f32x8; 8], D>),
}

impl<D: Dimension> PackedWeight<D> {
    /*fn tensor_packed(&self) -> Option<&ScalarTensor<D::Larger>> {
        if let Self::TensorPacked(w) = self {
            Some(w)
        } else {
            None
        }
    }*/
    fn array_1xf32x8(&self) -> Option<&Array<[f32x8; 1], D>> {
        if let Self::Array1xf32x8(w) = self {
            Some(w)
        } else {
            None
        }
    }
    fn array_8xf32x8(&self) -> Option<&Array<[f32x8; 8], D>> {
        if let Self::Array8xf32x8(w) = self {
            Some(w)
        } else {
            None
        }
    }
}

impl<S: ScalarData> TryFrom<ScalarTensorBase<S, Ix4>> for PackedWeight<Ix4> {
    type Error = Error;
    fn try_from(weight: ScalarTensorBase<S, Ix4>) -> Result<Self> {
        let (_oc, ic, _fh, _fw) = weight.dim();
        if weight.device().is_host() && weight.scalar_type() == ScalarType::F32 {
            let w = weight.view().try_into_tensor_view::<f32>().unwrap();
            let w = w.as_array().unwrap();
            let w_packed = if ic == 1 {
                Self::Array1xf32x8(pack_weight2_host_f32(w))
            } else {
                Self::Array8xf32x8(pack_weight2_host_f32(w))
            };
            return Ok(w_packed);
        }
        todo!()
    }
}

enum PackedOutput<D: Dimension> {
    #[allow(dead_code)]
    TensorPacked(ScalarTensor<D::Larger>),
    Arrayf32x8(Array<f32x8, D>),
}

impl<D: Dimension> PackedOutput<D> {
    /*
    fn tensor_packed(&self) -> Option<&ScalarTensor<D::Larger>> {
        if let Self::TensorPacked(y) = self {
            Some(y)
        } else {
            None
        }
    }
    */
    fn into_tensor_packed(self) -> Option<ScalarTensor<D::Larger>> {
        if let Self::TensorPacked(y) = self {
            Some(y)
        } else {
            None
        }
    }
    fn array_f32x8(&self) -> Option<&Array<f32x8, D>> {
        if let Self::Arrayf32x8(y) = self {
            Some(y)
        } else {
            None
        }
    }
}

impl<S: ScalarData> TryFrom<ScalarTensorBase<S, Ix4>> for PackedOutput<Ix4> {
    type Error = Error;
    fn try_from(output: ScalarTensorBase<S, Ix4>) -> Result<Self> {
        if output.device().is_host() && output.scalar_type() == ScalarType::F32 {
            let y = output.view().try_into_tensor_view::<f32>().unwrap();
            return Ok(Self::Arrayf32x8(pack_output2_host_f32(
                y.as_array().unwrap(),
            )));
        }
        todo!()
    }
}

impl PackedOutput<Ix4> {
    fn unpack2(self, oc: usize) -> Result<ScalarTensor4> {
        match self {
            Self::TensorPacked(_) => todo!(),
            Self::Arrayf32x8(y) => Ok(Tensor::from(unpack_output2_host_f32(y.view(), oc)).into()),
        }
    }
}

fn pack_input2_host_f32(input: ArrayView4<f32>) -> Array4<[f32; 8]> {
    unsafe fn inner(
        thread_id: usize,
        threads: usize,
        x: ArrayView4<f32>,
        x_packed: &SyncRawArrayViewMut4<[f32; 8]>,
    ) {
        let (_bs, ic, ih, iw) = x.dim();
        let mut x_packed = x_packed.clone();
        let (_bs, ic_blocks, _ih, _iw) = x_packed.dim();
        let x = x.as_slice().unwrap();
        for (bid, x) in x
            .chunks_exact(ic * ih * iw)
            .enumerate()
            .skip(thread_id)
            .step_by(threads)
        {
            let x_packed = unsafe {
                std::slice::from_raw_parts_mut(
                    x_packed.as_mut_ptr().add(bid * ic_blocks * ih * iw),
                    ic_blocks * ih * iw,
                )
            };
            for (x, x_packed) in x
                .chunks(8 * ih * iw)
                .zip(x_packed.chunks_exact_mut(ih * iw))
            {
                for (hidx, x_packed) in x_packed.chunks_exact_mut(iw).enumerate() {
                    let mut x_packed_chunks = x_packed.chunks_exact_mut(8);
                    for (widx, x_packed) in (0..).step_by(8).zip(x_packed_chunks.by_ref()) {
                        let mut x_tile = [f32x8::default(); 8];
                        for (x, x_tile) in x.chunks_exact(ih * iw).zip(x_tile.iter_mut()) {
                            *x_tile.as_array_mut() = unsafe {
                                x.get_unchecked(hidx * iw + widx..hidx * iw + widx + 8)
                                    .try_into()
                                    .unwrap_unchecked()
                            };
                        }
                        x_tile = f32x8::transpose(x_tile);
                        let x_tile: [[f32; 8]; 8] = unsafe { std::mem::transmute(x_tile) };
                        x_packed.copy_from_slice(&x_tile);
                    }
                    let x_packed = x_packed_chunks.into_remainder();
                    if !x_packed.is_empty() {
                        let mut x_tile = [f32x8::default(); 8];
                        for (x, x_tile) in x.chunks_exact(ih * iw).zip(x_tile.iter_mut()) {
                            let x = unsafe { x.get_unchecked(hidx * iw..(hidx + 1) * iw) };
                            for (x, x_tile) in x
                                .chunks_exact(8)
                                .remainder()
                                .iter()
                                .copied()
                                .zip(x_tile.as_array_mut())
                            {
                                *x_tile = x;
                            }
                        }
                        x_tile = f32x8::transpose(x_tile);
                        let x_tile: [[f32; 8]; 8] = unsafe { std::mem::transmute(x_tile) };
                        x_packed.copy_from_slice(&x_tile[..x_packed.len()]);
                    }
                }
            }
        }
    }
    let (bs, ic, ih, iw) = input.dim();
    let ic_blocks = div_up(ic, 8);
    let mut input_packed = unsafe { Array::uninit([bs, ic_blocks, ih, iw]).assume_init() };
    let sync_input_packed = SyncRawArrayViewMut4::try_from(input_packed.view_mut()).unwrap();
    broadcast(Some(bs), |thread_id, threads| unsafe {
        inner(thread_id, threads, input, &sync_input_packed)
    });
    input_packed
}

fn unpack_input2_host_f32(input_packed: ArrayView4<[f32; 8]>, ic: usize) -> Array4<f32> {
    unsafe fn inner(
        thread_id: usize,
        threads: usize,
        x_packed: ArrayView4<[f32; 8]>,
        x: &SyncRawArrayViewMut4<f32>,
    ) {
        let (_bs, ic_blocks, ih, iw) = x_packed.dim();
        let mut x = x.clone();
        let (_bs, ic, _ih, _iw) = x.dim();
        let x_packed = x_packed.as_slice().unwrap();
        for (bid, x_packed) in x_packed
            .chunks_exact(ic_blocks * ih * iw)
            .enumerate()
            .skip(thread_id)
            .step_by(threads)
        {
            let x = unsafe {
                std::slice::from_raw_parts_mut(x.as_mut_ptr().add(bid * ic * ih * iw), ic * ih * iw)
            };
            for (x_packed, x) in x_packed
                .chunks_exact(ih * iw)
                .zip(x.chunks_mut(8 * ih * iw))
            {
                for (hidx, x_packed) in x_packed.chunks_exact(iw).enumerate() {
                    let mut x_packed_chunks = x_packed.chunks_exact(8);
                    for (widx, x_packed) in (0..).step_by(8).zip(x_packed_chunks.by_ref()) {
                        let x_tile: [[f32; 8]; 8] = x_packed.try_into().unwrap();
                        let x_tile: [f32x8; 8] = unsafe { std::mem::transmute(x_tile) };
                        let x_tile = f32x8::transpose(x_tile);
                        for (x_tile, x) in x_tile.iter().zip(x.chunks_exact_mut(ih * iw)) {
                            let x = unsafe {
                                x.get_unchecked_mut(hidx * iw + widx..hidx * iw + widx + 8)
                            };
                            x.copy_from_slice(x_tile.as_array_ref());
                        }
                    }
                    let x_packed = x_packed_chunks.remainder();
                    if !x_packed.is_empty() {
                        let widx = (0..iw).step_by(8).last().unwrap();
                        let mut x_tile = [[0f32; 8]; 8];
                        x_tile[..x_packed.len()].copy_from_slice(x_packed);
                        let x_tile: [f32x8; 8] = unsafe { std::mem::transmute(x_tile) };
                        let x_tile = f32x8::transpose(x_tile);
                        for (x_tile, x) in x_tile.iter().zip(x.chunks_exact_mut(ih * iw)) {
                            let x =
                                unsafe { x.get_unchecked_mut(hidx * iw + widx..(hidx + 1) * iw) };
                            x.copy_from_slice(&x_tile.as_array_ref()[..x.len()]);
                        }
                    }
                }
            }
        }
    }
    let (bs, _ic_blocks, ih, iw) = input_packed.dim();
    let mut input = unsafe { Array::uninit([bs, ic, ih, iw]).assume_init() };
    let sync_input = SyncRawArrayViewMut4::try_from(input.view_mut()).unwrap();
    broadcast(Some(bs), |thread_id, threads| unsafe {
        inner(thread_id, threads, input_packed, &sync_input)
    });
    input
}

fn pack_weight2_host_f32<const TCX: usize>(weight: ArrayView4<f32>) -> Array4<[f32x8; TCX]> {
    unsafe fn inner<const TCX: usize>(
        thread_id: usize,
        threads: usize,
        w: ArrayView4<f32>,
        w_packed: &SyncRawArrayViewMut4<[f32x8; TCX]>,
    ) {
        let (_oc, ic, fh, fw) = w.dim();
        let (oc_blocks, ic_blocks, _fh, _fw) = w_packed.dim();
        let w = w.as_slice().unwrap();
        let mut w_packed = w_packed.clone();
        for (cidy_block, cidx_block) in ndarray::indices([oc_blocks, ic_blocks])
            .into_iter()
            .skip(thread_id)
            .step_by(threads)
        {
            let w = w.chunks(8 * ic * fh * fw).nth(cidy_block).unwrap();
            let w_packed = unsafe {
                std::slice::from_raw_parts_mut(
                    w_packed
                        .as_mut_ptr()
                        .add((cidy_block * ic_blocks + cidx_block) * fh * fw),
                    fh * fw,
                )
            };
            for (fi, w_packed) in w_packed.chunks_exact_mut(fw).enumerate() {
                for (fj, w_packed) in w_packed.iter_mut().enumerate() {
                    for (ciy, w) in w.chunks_exact(ic * fh * fw).take(8).enumerate() {
                        let w = w.chunks(TCX * fh * fw).nth(cidx_block).unwrap();
                        for (cix, w) in w.chunks_exact(fh * fw).take(TCX).enumerate() {
                            w_packed[cix].as_array_mut()[ciy] =
                                unsafe { *w.get_unchecked(fi * fw + fj) };
                        }
                    }
                }
            }
        }
    }

    let (oc, ic, fh, fw) = weight.dim();
    let ic_blocks = div_up(ic, TCX);
    let oc_blocks = div_up(oc, 8);
    let mut weight_packed =
        unsafe { Array::<[f32x8; TCX], _>::uninit([oc_blocks, ic_blocks, fh, fw]).assume_init() };
    let threads = if weight_packed.len() > 40_000 {
        rayon::current_num_threads().min(oc_blocks * ic_blocks)
    } else {
        1
    };
    let sync_weight_packed = SyncRawArrayViewMut::try_from(weight_packed.view_mut()).unwrap();
    broadcast(Some(threads), |thread_id, threads| unsafe {
        inner(thread_id, threads, weight, &sync_weight_packed);
    });
    weight_packed
}

fn unpack_weight2_host_f32<const TCX: usize>(
    weight_packed: ArrayView4<[f32x8; TCX]>,
    ic: usize,
    oc: usize,
) -> Array4<f32> {
    unsafe fn inner<const TCX: usize>(
        thread_id: usize,
        threads: usize,
        w_packed: ArrayView4<[f32x8; TCX]>,
        w: &SyncRawArrayViewMut4<f32>,
    ) {
        let (oc_blocks, ic_blocks, fh, fw) = w_packed.dim();
        let (oc, ic, _fh, _fw) = w.dim();
        let w_packed = w_packed.as_slice().unwrap();
        let mut w = w.clone();
        for ((cidy_block, cidx_block), w_packed) in ndarray::indices([oc_blocks, ic_blocks])
            .into_iter()
            .zip(w_packed.chunks_exact(fh * fw))
            .skip(thread_id)
            .step_by(threads)
        {
            let tcy = (cidy_block * 8..oc).take(8).len();
            let tcx = (cidx_block * TCX..ic).take(TCX).len();
            for ciy in 0..tcy {
                let w = unsafe {
                    std::slice::from_raw_parts_mut(
                        w.as_mut_ptr()
                            .add(((cidy_block * 8 + ciy) * ic + cidx_block * TCX) * fh * fw),
                        tcx * fh * fw,
                    )
                };
                for (cix, w) in w.chunks_exact_mut(fh * fw).enumerate() {
                    for (fi, w) in w.chunks_exact_mut(fw).enumerate() {
                        for (fj, w) in w.iter_mut().enumerate() {
                            *w = unsafe {
                                w_packed.get_unchecked(fi * fw + fj)[cix].as_array_ref()[ciy]
                            };
                        }
                    }
                }
            }
        }
    }

    let (oc_blocks, ic_blocks, fh, fw) = weight_packed.dim();
    let mut weight = unsafe { Array::<f32, _>::uninit([oc, ic, fh, fw]).assume_init() };
    let threads = if weight.len() > 40_000 {
        rayon::current_num_threads().min(oc_blocks * ic_blocks)
    } else {
        1
    };
    let sync_weight = SyncRawArrayViewMut::try_from(weight.view_mut()).unwrap();
    broadcast(Some(threads), |thread_id, threads| unsafe {
        inner(thread_id, threads, weight_packed, &sync_weight);
    });
    weight
}

fn pack_output2_host_f32(output: ArrayView4<f32>) -> Array4<f32x8> {
    unsafe fn inner(
        thread_id: usize,
        threads: usize,
        y: ArrayView4<f32>,
        y_packed: &SyncRawArrayViewMut4<f32x8>,
    ) {
        let (_bs, oc, oh, ow) = y.dim();
        let (_bs, oc_blocks, _oh, _ow) = y_packed.dim();
        let y = y.as_slice().unwrap();
        let mut y_packed = y_packed.clone();
        for (bid, y) in y
            .chunks_exact(oc * oh * ow)
            .enumerate()
            .skip(thread_id)
            .step_by(threads)
        {
            let y_packed = unsafe {
                std::slice::from_raw_parts_mut(
                    y_packed.as_mut_ptr().add(bid * oc_blocks * oh * ow),
                    oc_blocks * oh * ow,
                )
            };
            for (cidy_block, (y, y_packed)) in y
                .chunks(8 * oh * ow)
                .zip(y_packed.chunks_exact_mut(oh * ow))
                .enumerate()
            {
                let block_oc = (cidy_block * 8..oc).take(8).len();
                for (hidy, y_packed) in y_packed.chunks_exact_mut(ow).enumerate() {
                    let mut y_packed_chunks = y_packed.chunks_exact_mut(8);
                    for (widy, y_packed) in (0..).step_by(8).zip(y_packed_chunks.by_ref()) {
                        let mut y_tile = [f32x8::default(); 8];
                        for (ciy, y_tile) in y_tile.iter_mut().take(block_oc).enumerate() {
                            for (widy, y_tile) in (widy..).zip(y_tile.as_array_mut()) {
                                *y_tile =
                                    unsafe { *y.get_unchecked((ciy * oh + hidy) * ow + widy) };
                            }
                        }
                        y_tile = f32x8::transpose(y_tile);
                        y_packed.copy_from_slice(&y_tile);
                    }
                    let y_packed = y_packed_chunks.into_remainder();
                    if !y_packed.is_empty() {
                        let widy = (0..ow).step_by(8).last().unwrap();
                        let mut y_tile = [f32x8::default(); 8];
                        for (ciy, y_tile) in y_tile.iter_mut().take(block_oc).enumerate() {
                            for (widy, y_tile) in (widy..ow).zip(y_tile.as_array_mut()) {
                                *y_tile =
                                    unsafe { *y.get_unchecked((ciy * oh + hidy) * ow + widy) };
                            }
                        }
                        y_tile = f32x8::transpose(y_tile);
                        for (y_tile, y_packed) in y_tile.iter().copied().zip(y_packed.iter_mut()) {
                            *y_packed = y_tile;
                        }
                    }
                }
            }
        }
    }

    let (bs, oc, oh, ow) = output.dim();
    let oc_blocks = div_up(oc, 8);
    let mut packed_output = unsafe { Array::uninit([bs, oc_blocks, oh, ow]).assume_init() };
    let sync_packed_output = SyncRawArrayViewMut::try_from(packed_output.view_mut()).unwrap();
    broadcast(Some(bs), |thread_id, threads| unsafe {
        inner(thread_id, threads, output, &sync_packed_output);
    });
    packed_output
}

fn unpack_output2_host_f32(output_packed: ArrayView4<f32x8>, oc: usize) -> Array4<f32> {
    unsafe fn inner(
        thread_id: usize,
        threads: usize,
        y_packed: ArrayView4<f32x8>,
        y: &SyncRawArrayViewMut4<f32>,
    ) {
        let (_bs, oc_blocks, oh, ow) = y_packed.dim();
        let (_bs, oc, _oh, _ow) = y.dim();
        let y_packed = y_packed.as_slice().unwrap();
        let mut y = y.clone();
        for (bid, y_packed) in y_packed
            .chunks_exact(oc_blocks * oh * ow)
            .enumerate()
            .skip(thread_id)
            .step_by(threads)
        {
            let y = unsafe {
                std::slice::from_raw_parts_mut(y.as_mut_ptr().add(bid * oc * oh * ow), oc * oh * ow)
            };
            for (cidy_block, (y_packed, y)) in y_packed
                .chunks_exact(oh * ow)
                .zip(y.chunks_mut(8 * oh * ow))
                .enumerate()
            {
                let block_oc = (cidy_block * 8..oc).take(8).len();
                for (hidy, y_packed) in y_packed.chunks_exact(ow).enumerate() {
                    let mut y_packed_chunks = y_packed.chunks_exact(8);
                    for (widy, y_packed) in (0..).step_by(8).zip(y_packed_chunks.by_ref()) {
                        let y_tile = f32x8::transpose(y_packed.try_into().unwrap());
                        for (ciy, y_tile) in y_tile.into_iter().take(block_oc).enumerate() {
                            for (widy, y_tile) in (widy..).zip(y_tile.to_array()) {
                                unsafe {
                                    *y.get_unchecked_mut((ciy * oh + hidy) * ow + widy) = y_tile;
                                }
                            }
                        }
                    }
                    let y_packed = y_packed_chunks.remainder();
                    if !y_packed.is_empty() {
                        let widy = (0..ow).step_by(8).last().unwrap();
                        let mut y_tile = [f32x8::default(); 8];
                        for (y_packed, y_tile) in y_packed.iter().zip(y_tile.iter_mut()) {
                            *y_tile = *y_packed;
                        }
                        y_tile = f32x8::transpose(y_tile);
                        for (ciy, y_tile) in y_tile.into_iter().take(block_oc).enumerate() {
                            for (widy, y_tile) in (widy..ow).zip(y_tile.to_array()) {
                                unsafe {
                                    *y.get_unchecked_mut((ciy * oh + hidy) * ow + widy) = y_tile;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let (bs, _oc_blocks, oh, ow) = output_packed.dim();
    let mut output = unsafe { Array::uninit([bs, oc, oh, ow]).assume_init() };
    let sync_output = SyncRawArrayViewMut::try_from(output.view_mut()).unwrap();
    broadcast(Some(bs), |thread_id, threads| unsafe {
        inner(thread_id, threads, output_packed, &sync_output);
    });
    output
}

fn conv2_direct_host_f32<const TCX: usize>(
    input: ArrayView4<[f32; TCX]>,
    ic: usize,
    weight: ArrayView4<[f32x8; TCX]>,
    options: &Conv2Options,
) -> Array4<f32x8> {
    #[allow(unused_mut, unused_assignments)]
    const fn twy_for_tby(tby: usize) -> usize {
        if cfg!(target_feature = "fma") {
            15 / (tby + 1)
        } else if cfg!(target_feature = "avx") {
            15 / (tby + 2)
        } else {
            15 / (2 * (tby + 2))
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    unsafe fn kernel<const TCX: usize, const TBY: usize, const TWY: usize, const UNROLL: bool>(
        cidx_block: usize,
        widy: usize,
        x: &[[f32; TCX]],
        tcx: usize,
        iw: usize,
        w: &[[f32x8; TCX]],
        tby: usize,
        ic_blocks: usize,
        filter: [usize; 2],
        y: &mut [[f32x8; TWY]; TBY],
        twy: usize,
    ) {
        let twy = if UNROLL { TWY } else { twy.max(1).min(TWY) };
        let [fh, fw] = filter;
        let mut x_packed = [f32x8::default(); TWY];
        let mut w_packed = [f32x8::default(); TBY];
        for fi in 0..fh {
            for fj in 0..fw {
                for cix in (0..tcx).take(TCX) {
                    for (widx, x_packed) in (widy + fj..).zip(x_packed.iter_mut()).take(twy) {
                        unsafe {
                            *x_packed = f32x8::splat(x.get_unchecked(fi * iw + widx)[cix]);
                        }
                    }
                    for (cidy_block, w_packed) in w_packed.iter_mut().take(tby).enumerate() {
                        *w_packed = unsafe {
                            w.get_unchecked(
                                ((cidy_block * ic_blocks + cidx_block) * fh + fi) * fw + fj,
                            )[cix]
                        };
                    }
                    for (w, y) in w_packed.into_iter().zip(y.iter_mut()) {
                        for (x, y) in x_packed.into_iter().zip(y.iter_mut()) {
                            #[cfg(target_feature = "fma")]
                            {
                                *y = x.mul_add(w, *y);
                            }
                            #[cfg(not(target_feature = "fma"))]
                            {
                                *y += x * w;
                            }
                        }
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn inner<const TCX: usize, const TBY: usize, const TWY: usize>(
        thread_id: usize,
        threads: usize,
        x: ArrayView4<[f32; TCX]>,
        ic: usize,
        w: ArrayView4<[f32x8; TCX]>,
        y: &SyncRawArrayViewMut4<f32x8>,
    ) {
        let (_bs, ic_blocks, ih, iw) = x.dim();
        let (oc_blocks, _ic_blocks, fh, fw) = w.dim();
        let (_bs, _oc_blocks, oh, ow) = y.dim();
        let x = x.as_slice().unwrap();
        let w = w.as_slice().unwrap();
        let mut y = y.clone();
        for (bid, x) in x
            .chunks_exact(ic_blocks * ih * iw)
            .enumerate()
            .skip(thread_id)
            .step_by(threads)
        {
            let y = unsafe {
                std::slice::from_raw_parts_mut(
                    y.as_mut_ptr().add(bid * oc_blocks * oh * ow),
                    oc_blocks * oh * ow,
                )
            };
            for (cidy_chunk, (w, y)) in w
                .chunks(TBY * ic_blocks * fh * fw)
                .zip(y.chunks_mut(TBY * oh * ow))
                .enumerate()
            {
                let tby = (cidy_chunk * TBY..oc_blocks).take(TBY).len();
                for (cidx_block, x) in x.chunks_exact(ih * iw).enumerate() {
                    let tcx = (cidx_block * TCX..ic).take(TCX).len();
                    for (hidy, x) in x.windows(fh * iw).step_by(iw).enumerate() {
                        for widy in (0..ow).step_by(TWY) {
                            let twy = (widy..ow).take(TWY).len();
                            let mut y_tile = [[f32x8::default(); TWY]; TBY];
                            if cidx_block > 0 {
                                for (y, y_tile) in y.chunks_exact(oh * ow).zip(y_tile.iter_mut()) {
                                    let y = y.chunks_exact(ow).nth(hidy).unwrap();
                                    let y = unsafe { y.get_unchecked(widy..widy + twy) };
                                    y_tile[..twy].copy_from_slice(y);
                                }
                            }
                            if twy == TWY {
                                unsafe {
                                    kernel::<TCX, TBY, TWY, true>(
                                        cidx_block,
                                        widy,
                                        x,
                                        tcx,
                                        iw,
                                        w,
                                        tby,
                                        ic_blocks,
                                        [fh, fw],
                                        &mut y_tile,
                                        twy,
                                    );
                                }
                            } else {
                                unsafe {
                                    kernel::<TCX, TBY, TWY, false>(
                                        cidx_block,
                                        widy,
                                        x,
                                        tcx,
                                        iw,
                                        w,
                                        tby,
                                        ic_blocks,
                                        [fh, fw],
                                        &mut y_tile,
                                        twy,
                                    );
                                }
                            }
                            for (y_tile, y) in y_tile.iter().zip(y.chunks_exact_mut(oh * ow)) {
                                let y = y.chunks_exact_mut(ow).nth(hidy).unwrap();
                                let y = unsafe { y.get_unchecked_mut(widy..widy + twy) };
                                y.copy_from_slice(&y_tile[..twy]);
                            }
                        }
                    }
                }
            }
        }
    }

    let (bs, _ic_blocks, ih, iw) = input.dim();
    let (oc_blocks, _ic_blocks, fh, fw) = weight.dim();
    let (oh, ow) = options
        .output_shape([ih, iw].into_dimension(), &[fh, fw].into_dimension())
        .unwrap()
        .into_pattern();
    let tby = oc_blocks.min(4);
    let twy = twy_for_tby(tby).min(ow);
    let mut output = unsafe { Array::uninit([bs, oc_blocks, oh, ow]).assume_init() };
    let sync_output = SyncRawArrayViewMut::try_from(output.view_mut()).unwrap();
    macro_for!($TBY in [1, 2, 3, 4] {
        if tby == $TBY {
            macro_for!($TWY in [1, 2, 3, 4, 5, 6, 7] {
                if twy == $TWY {
                    if options.is_default() {
                        broadcast(Some(bs), |thread_id, threads| {
                            inner::<TCX, $TBY, $TWY>(thread_id, threads, input, ic, weight, &sync_output);
                        });
                    } else {
                        todo!("{options:?}");
                    }
                }
            });
        }
    });
    output
}

trait Conv2DirectBackwardInputHostF32Kernel<const TCX: usize> {
    #[allow(clippy::too_many_arguments)]
    unsafe fn conv2_direct_backward_input_host_f32_kernel<const TWY: usize, const UNROLL: bool>(
        widy: usize,
        w: &[[f32x8; TCX]],
        fw: usize,
        dy: &[f32x8; TWY],
        tcy: usize,
        twy: usize,
        dx: &mut [[f32; TCX]],
        iw: usize,
    );
}

impl Conv2DirectBackwardInputHostF32Kernel<8> for () {
    #[inline(never)]
    unsafe fn conv2_direct_backward_input_host_f32_kernel<const TWY: usize, const UNROLL: bool>(
        widy: usize,
        w: &[[f32x8; 8]],
        fw: usize,
        dy: &[f32x8; TWY],
        tcy: usize,
        twy: usize,
        dx: &mut [[f32; 8]],
        iw: usize,
    ) {
        let twy = if UNROLL { TWY } else { twy.max(1).min(TWY) };
        let tcy = tcy.max(1).min(8);
        let mut dy_packed = [f32x8::default(); TWY];
        let mut dx_packed = [f32x8::default(); TWY];
        for (fi, w) in w.chunks_exact(fw).enumerate() {
            for (fj, w) in w.iter().copied().enumerate() {
                for (widx, dx_packed) in (widy + fj..).zip(dx_packed.iter_mut()).take(twy) {
                    unsafe {
                        *dx_packed = f32x8::from(*dx.get_unchecked(fi * iw + widx));
                    }
                }
                for (ciy, w) in (0..tcy).zip(w) {
                    for (dy, dy_packed) in dy.iter().zip(dy_packed.iter_mut()) {
                        *dy_packed = f32x8::splat(dy.as_array_ref()[ciy]);
                    }
                    for (dy, dx) in dy_packed.iter().zip(dx_packed.iter_mut()) {
                        *dx = dy.mul_add(w, *dx);
                    }
                }
                for (widx, dx_packed) in (widy + fj..).zip(dx_packed).take(twy) {
                    unsafe {
                        *dx.get_unchecked_mut(fi * iw + widx) = dx_packed.to_array();
                    }
                }
            }
        }
    }
}

impl Conv2DirectBackwardInputHostF32Kernel<1> for () {
    #[inline(never)]
    unsafe fn conv2_direct_backward_input_host_f32_kernel<const TWY: usize, const UNROLL: bool>(
        widy: usize,
        w: &[[f32x8; 1]],
        fw: usize,
        dy: &[f32x8; TWY],
        tcy: usize,
        twy: usize,
        dx: &mut [[f32; 1]],
        iw: usize,
    ) {
        let twy = if UNROLL { TWY } else { twy.max(1).min(TWY) };
        let tcy = tcy.max(1).min(8);
        let mut dy_packed = f32x8::default();
        let mut dx_packed = f32x8::default();
        for (fi, w) in w.chunks_exact(fw).enumerate() {
            for (fj, w) in w.iter().enumerate() {
                for (widx, dx_packed) in (widy + fj..)
                    .zip(dx_packed.as_array_mut().iter_mut())
                    .take(twy)
                {
                    unsafe {
                        *dx_packed = dx.get_unchecked(fi * iw + widx)[0];
                    }
                }
                let [w] = w;
                for (ciy, w) in (0..tcy).zip(w.to_array().map(f32x8::splat)) {
                    for (dy, dy_packed) in dy.iter().zip(dy_packed.as_array_mut()) {
                        *dy_packed = dy.as_array_ref()[ciy];
                    }
                    dx_packed = dy_packed.mul_add(w, dx_packed);
                }
                for (widx, dx_packed) in (widy + fj..).zip(dx_packed.to_array()).take(twy) {
                    unsafe {
                        dx.get_unchecked_mut(fi * iw + widx)[0] = dx_packed;
                    }
                }
            }
        }
    }
}

fn transpose_weight_host_f32(mut weight: ArrayViewMut4<[f32x8; 8]>) {
    let w = weight.as_slice_mut().unwrap();
    if w.len() * size_of::<f32>() > parallel_size() && rayon::current_num_threads() > 1 {
        w.par_iter_mut().for_each(|w| *w = f32x8::transpose(*w));
    } else {
        w.iter_mut().for_each(|w| *w = f32x8::transpose(*w));
    }
}

fn conv2_direct_backward_input_host_f32<const TCX: usize>(
    weight: ArrayView4<[f32x8; TCX]>,
    output_grad: ArrayView4<f32x8>,
    oc: usize,
    options: &Conv2Options,
) -> Array4<[f32; TCX]>
where
    (): Conv2DirectBackwardInputHostF32Kernel<TCX>,
{
    fn inner<const TCX: usize, const TWY: usize>(
        thread_id: usize,
        threads: usize,
        w: ArrayView4<[f32x8; TCX]>,
        dy: ArrayView4<f32x8>,
        oc: usize,
        dx: &SyncRawArrayViewMut4<[f32; TCX]>,
    ) where
        (): Conv2DirectBackwardInputHostF32Kernel<TCX>,
    {
        let mut dx = dx.clone();
        let (oc_blocks, ic_blocks, fh, fw) = w.dim();
        let (_bs, _oc_blocks, oh, ow) = dy.dim();
        let (_bs, _ic_blocks, ih, iw) = dx.dim();
        let w = w.as_slice().unwrap();
        let dy = dy.as_slice().unwrap();
        let mut dy_tile = [f32x8::default(); TWY];
        for (bid, dy) in dy
            .chunks_exact(oc_blocks * oh * ow)
            .enumerate()
            .skip(thread_id)
            .step_by(threads)
        {
            let dx = unsafe {
                std::slice::from_raw_parts_mut(
                    dx.as_mut_ptr().add(bid * ic_blocks * ih * iw),
                    ic_blocks * ih * iw,
                )
            };
            for (cidx_block, dx) in dx.chunks_exact_mut(ih * iw).enumerate() {
                for (cidy_block, (w, dy)) in w
                    .chunks_exact(ic_blocks * fh * fw)
                    .zip(dy.chunks_exact(oh * ow))
                    .enumerate()
                {
                    let tcy = (cidy_block * 8..oc).take(8).len();
                    let w = w.chunks_exact(fh * fw).nth(cidx_block).unwrap();
                    for (hidy, dy) in dy.chunks_exact(ow).enumerate() {
                        let dx = unsafe { dx.get_unchecked_mut(hidy * iw..(hidy + fh) * iw) };
                        let mut dy_chunks = dy.chunks_exact(TWY);
                        for (widy, dy) in (0..).step_by(TWY).zip(dy_chunks.by_ref()) {
                            dy_tile = dy.try_into().unwrap();
                            unsafe {
                                <()>::conv2_direct_backward_input_host_f32_kernel::<TWY, true>(
                                    widy, w, fw, &dy_tile, tcy, TWY, dx, iw,
                                );
                            }
                        }
                        let dy = dy_chunks.remainder();
                        if !dy.is_empty() {
                            let widy = (0..ow).step_by(TWY).last().unwrap();
                            let twy = dy.len();
                            dy_tile[..dy.len()].copy_from_slice(dy);
                            unsafe {
                                <()>::conv2_direct_backward_input_host_f32_kernel::<TWY, false>(
                                    widy, w, fw, &dy_tile, tcy, twy, dx, iw,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    let (bs, _oc_blocks, oh, ow) = output_grad.dim();
    let (_oc_blocks, ic_blocks, fh, fw) = weight.dim();
    let (ih, iw) = options
        .input_shape([oh, ow].into_dimension(), &[fh, fw].into_dimension())
        .unwrap()
        .into_pattern();
    let twy = if TCX == 8 {
        7.min(ow)
    } else if TCX == 1 {
        8.min(ow)
    } else {
        unreachable!()
    };
    let mut input_grad = Array::from_elem([bs, ic_blocks, ih, iw], [0f32; TCX]);
    let sync_input_grad = SyncRawArrayViewMut::try_from(input_grad.view_mut()).unwrap();
    macro_for!($TWY in [1, 2, 3, 4, 5, 6, 7] {
        if twy == $TWY {
            if options.is_default() {
                broadcast(Some(bs), |thread_id, threads| {
                    inner::<TCX, $TWY>(thread_id, threads, weight, output_grad, oc, &sync_input_grad);
                });
            } else {
                todo!("{options:?}");
            }
        }
    });
    input_grad
}

fn conv2_direct_backward_weight_host_f32<const TCX: usize>(
    input: ArrayView4<[f32; TCX]>,
    output_grad: ArrayView4<f32x8>,
    options: &Conv2Options,
    weight_shape: [usize; 4],
) -> Array4<[f32x8; TCX]> {
    const fn tw_for_tby(tby: usize) -> usize {
        if cfg!(target_feature = "fma") {
            15 / (tby + 1)
        } else if cfg!(target_feature = "avx") {
            15 / (tby + 2)
        } else {
            15 / (2 * tby + 2)
        }
    }

    fn inner<const TCX: usize, const TBY: usize, const TW: usize>(
        thread_id: usize,
        threads: usize,
        x: ArrayView4<[f32; TCX]>,
        ic: usize,
        dy: ArrayView4<f32x8>,
        dw: &SyncRawArrayViewMut5<[f32x8; TCX]>,
    ) {
        let (_bs, ic_blocks, ih, iw) = x.dim();
        let (_bs, _oc_blocks, oh, ow) = dy.dim();
        let (oc_blocks, _ic_blocks, threads_bs, fh, fw) = dw.dim();
        let oc_groups = div_up(oc_blocks, TBY);
        let threads_c = threads / threads_bs;
        let thread_cid = thread_id / threads_bs;
        let thread_bid = thread_id % threads_bs;
        let x = x.as_slice().unwrap();
        let dy = dy.as_slice().unwrap();
        let mut dw = dw.clone();
        for (bid, (x, dy)) in x
            .chunks_exact(ic_blocks * ih * iw)
            .zip(dy.chunks_exact(oc_blocks * oh * ow))
            .enumerate()
            .skip(thread_bid)
            .step_by(threads_bs)
        {
            for (cidy_group, cidx_block) in ndarray::indices([oc_groups, ic_blocks])
                .into_iter()
                .skip(thread_cid)
                .step_by(threads_c)
            {
                let cidy_block = cidy_group * TBY;
                let x = x.chunks_exact(ih * iw).nth(cidx_block).unwrap();
                let tcx = (cidx_block * TCX..ic).take(TCX).len();
                let fhwc = fh * fw * tcx;
                let fijc_indices = |fijc| {
                    let fij = fijc / tcx;
                    let cix = fijc % tcx;
                    let fi = fij / fw;
                    let fj = fij % fw;
                    (fi, fj, cix)
                };
                for fijc in (0..fhwc).step_by(TW) {
                    let mut dw_packed = [[f32x8::default(); TW]; TBY];
                    if bid > thread_bid {
                        for (cidy_block, dw_packed) in
                            (cidy_block..oc_blocks).zip(dw_packed.iter_mut())
                        {
                            let dw = unsafe {
                                std::slice::from_raw_parts(
                                    dw.as_mut_ptr().add(
                                        ((cidy_block * ic_blocks + cidx_block) * threads_bs
                                            + thread_bid)
                                            * fh
                                            * fw,
                                    ),
                                    fh * fw,
                                )
                            };
                            for (fijc, dw_packed) in (fijc..fhwc).zip(dw_packed.iter_mut()) {
                                let (fi, fj, cix) = fijc_indices(fijc);
                                *dw_packed = unsafe { dw.get_unchecked(fi * fw + fj)[cix] };
                            }
                        }
                    }
                    for (hidy, x) in x.windows(fh * iw).step_by(iw).enumerate() {
                        for widy in 0..ow {
                            let mut dy_packed = [f32x8::default(); TBY];
                            for (cidy_block, dy_packed) in
                                (cidy_block..oc_blocks).zip(dy_packed.iter_mut())
                            {
                                *dy_packed = unsafe {
                                    *dy.get_unchecked((cidy_block * oh + hidy) * ow + widy)
                                };
                            }
                            let mut x_packed = [f32x8::default(); TW];
                            for (fijc, x_packed) in (fijc..fhwc).zip(x_packed.iter_mut()) {
                                let (fi, fj, cix) = fijc_indices(fijc);
                                let widx = widy + fj;
                                let x = unsafe { x.get_unchecked(fi * iw + widx)[cix] };
                                *x_packed = f32x8::splat(x);
                            }
                            for (dy, dw) in dy_packed.into_iter().zip(dw_packed.iter_mut()) {
                                for (x, dw) in x_packed.into_iter().zip(dw.iter_mut()) {
                                    *dw = x.mul_add(dy, *dw);
                                }
                            }
                        }
                    }
                    for (cidy_block, dw_packed) in (cidy_block..oc_blocks).zip(dw_packed) {
                        let dw = unsafe {
                            std::slice::from_raw_parts_mut(
                                dw.as_mut_ptr().add(
                                    ((cidy_block * ic_blocks + cidx_block) * threads_bs
                                        + thread_bid)
                                        * fh
                                        * fw,
                                ),
                                fh * fw,
                            )
                        };
                        for (fijc, dw_packed) in (fijc..fhwc).zip(dw_packed) {
                            let (fi, fj, cix) = fijc_indices(fijc);
                            unsafe {
                                dw.get_unchecked_mut(fi * fw + fj)[cix] = dw_packed;
                            }
                        }
                    }
                }
            }
        }
    }

    let (bs, ic_blocks, _ih, _iw) = input.dim();
    let (_bs, oc_blocks, _oh, _ow) = output_grad.dim();
    let [_oc, ic, fh, fw] = weight_shape;
    let tby = oc_blocks.min(4);
    let oc_groups = div_up(oc_blocks, tby);
    let threads = rayon::current_num_threads();
    let threads_c = threads.min(oc_groups * ic_blocks);
    let threads_bs = (threads / threads_c).min(bs);
    let mut weight_grad_tmp =
        unsafe { Array::uninit([oc_blocks, ic_blocks, threads_bs, fh, fw]).assume_init() };
    let sync_weight_grad_tmp = SyncRawArrayViewMut::try_from(weight_grad_tmp.view_mut()).unwrap();
    let tw = tw_for_tby(tby).min(fh * fw);
    macro_for!($TBY in [1, 2, 3, 4] {
        if tby == $TBY {
            macro_for!($TW in [1, 2, 3, 4, 5, 6, 7] {
                if tw == $TW {
                    if options.is_default() {
                        broadcast(Some(threads_c * threads_bs), |thread_id, threads| {
                            inner::<TCX, $TBY, $TW>(thread_id, threads, input, ic, output_grad, &sync_weight_grad_tmp);
                        });
                    } else {
                        todo!("{options:?}");
                    }
                }
            });
        }
    });

    if threads_bs == 1 {
        weight_grad_tmp
            .into_shape([oc_blocks, ic_blocks, fh, fw])
            .unwrap()
    } else {
        let mut weight_grad =
            Array::from_elem([oc_blocks, ic_blocks, fh, fw], [f32x8::default(); TCX]);
        for ((_cidy_block, cidx_block), (x, y)) in
            ndarray::indices([oc_blocks, ic_blocks]).into_iter().zip(
                weight_grad_tmp
                    .as_slice()
                    .unwrap()
                    .chunks_exact(threads_bs * fh * fw)
                    .zip(
                        weight_grad
                            .as_slice_mut()
                            .unwrap()
                            .chunks_exact_mut(fh * fw),
                    ),
            )
        {
            let tcx = (cidx_block * TCX..ic).take(TCX).len();
            for x in x.chunks_exact(fh * fw) {
                for (x, y) in x.iter().zip(y.iter_mut()) {
                    for (x, y) in x.iter().zip(y.iter_mut()).take(tcx) {
                        *y += x;
                    }
                }
            }
        }
        weight_grad
    }
}
