use autograph::{
    backend::Device,
    tensor::{Num, Scalar, Tensor, Unsigned},
    Result,
};
use half::bf16;
use ndarray::{Array, Array1, Array2};
use num_traits::{FromPrimitive, ToPrimitive};

fn array_scaled_cast<T1: ToPrimitive, T2: FromPrimitive>(x: &Array1<T1>, alpha: f64) -> Array1<T2> {
    x.iter()
        .map(|x| T2::from_f64(x.to_f64().unwrap() * alpha).unwrap())
        .collect()
}

fn scaled_cast<T1: Scalar + From<u8> + ToPrimitive, T2: Num + From<u8> + FromPrimitive>(
) -> Result<()> {
    let n = 99;
    let alpha = 2;
    let data: Vec<T1> = (0..n as u8).into_iter().map(Into::into).collect();
    let x_array = Array::from(data);
    let y_true = array_scaled_cast(&x_array, alpha.into());
    for device in Device::list() {
        let x = Tensor::from_array(&device, x_array.view())?;
        let y = x.scale_into::<T2>((alpha as u8).into())?;
        let y_array = smol::block_on(y.to_array()?)?;
        assert_eq!(y_array, y_true);
    }
    Ok(())
}

#[test]
fn scaled_cast_u8_bf16() -> Result<()> {
    scaled_cast::<u8, bf16>()
}

#[test]
fn scaled_cast_u8_u32() -> Result<()> {
    scaled_cast::<u8, u32>()
}

#[test]
fn scaled_cast_u8_i32() -> Result<()> {
    scaled_cast::<u8, i32>()
}

#[test]
fn scaled_cast_u8_f32() -> Result<()> {
    scaled_cast::<u8, f32>()
}

#[test]
fn scaled_cast_u16_bf16() -> Result<()> {
    scaled_cast::<u16, bf16>()
}

#[test]
fn scaled_cast_u16_u32() -> Result<()> {
    scaled_cast::<u16, u32>()
}

#[test]
fn scaled_cast_u16_i32() -> Result<()> {
    scaled_cast::<u16, i32>()
}

#[test]
fn scaled_cast_u16_f32() -> Result<()> {
    scaled_cast::<u16, f32>()
}

#[test]
fn scaled_cast_bf16_bf16() -> Result<()> {
    scaled_cast::<bf16, bf16>()
}

#[test]
fn scaled_cast_bf16_u32() -> Result<()> {
    scaled_cast::<bf16, u32>()
}

#[test]
fn scaled_cast_bf16_i32() -> Result<()> {
    scaled_cast::<bf16, i32>()
}

#[test]
fn scaled_cast_bf16_f32() -> Result<()> {
    scaled_cast::<bf16, f32>()
}

#[test]
fn scaled_cast_u32_bf16() -> Result<()> {
    scaled_cast::<u32, bf16>()
}

#[test]
fn scaled_cast_u32_u32() -> Result<()> {
    scaled_cast::<u32, u32>()
}

#[test]
fn scaled_cast_u32_i32() -> Result<()> {
    scaled_cast::<u32, i32>()
}

#[test]
fn scaled_cast_u32_f32() -> Result<()> {
    scaled_cast::<u32, f32>()
}

#[test]
fn scaled_cast_i32_bf16() -> Result<()> {
    scaled_cast::<i32, bf16>()
}

#[test]
fn scaled_cast_i32_u32() -> Result<()> {
    scaled_cast::<i32, u32>()
}

#[test]
fn scaled_cast_i32_i32() -> Result<()> {
    scaled_cast::<i32, i32>()
}

#[test]
fn scaled_cast_i32_f32() -> Result<()> {
    scaled_cast::<i32, f32>()
}

fn to_one_hot<U: Copy + Into<u64>, T: Num>(x: &Array1<U>, nclasses: usize) -> Array2<T> {
    let mut y = Array::from_elem([x.len(), nclasses], T::zero());
    for (mut y, x) in y.outer_iter_mut().zip(x.iter().copied()) {
        y[x.into() as usize] = T::one();
    }
    y
}

fn one_hot<U: Unsigned + Copy + Into<u64> + From<u8>, T: Num>() -> Result<()> {
    let batch_size = 67;
    let nclasses = 9;
    let data: Vec<U> = (0..nclasses as u8)
        .into_iter()
        .cycle()
        .take(batch_size)
        .map(Into::into)
        .collect();
    let x_array = Array::from(data.clone());
    let y_true = to_one_hot(&x_array, nclasses);
    for device in Device::list() {
        let x = Tensor::from_array(&device, x_array.view())?;
        let y = x.to_one_hot::<f32>(nclasses)?;
        let y_array = smol::block_on(y.to_array()?)?;
        assert_eq!(y_array, y_true);
    }
    Ok(())
}

#[test]
fn one_hot_u8_bf16() -> Result<()> {
    one_hot::<u8, bf16>()
}

#[test]
fn one_hot_u16_bf16() -> Result<()> {
    one_hot::<u16, bf16>()
}

#[test]
fn one_hot_u32_bf16() -> Result<()> {
    one_hot::<u32, bf16>()
}

#[test]
fn one_hot_u8_u32() -> Result<()> {
    one_hot::<u8, u32>()
}

#[test]
fn one_hot_u16_u32() -> Result<()> {
    one_hot::<u16, u32>()
}

#[test]
fn one_hot_u32_u32() -> Result<()> {
    one_hot::<u32, u32>()
}

#[test]
fn one_hot_u8_i32() -> Result<()> {
    one_hot::<u8, i32>()
}

#[test]
fn one_hot_u16_i32() -> Result<()> {
    one_hot::<u16, i32>()
}

#[test]
fn one_hot_u32_i32() -> Result<()> {
    one_hot::<u32, i32>()
}

#[test]
fn one_hot_u8_f32() -> Result<()> {
    one_hot::<u8, f32>()
}

#[test]
fn one_hot_u16_f32() -> Result<()> {
    one_hot::<u16, f32>()
}

#[test]
fn one_hot_u32_f32() -> Result<()> {
    one_hot::<u32, f32>()
}
