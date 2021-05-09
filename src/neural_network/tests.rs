use super::*;
use ndarray::{Array, ArrayViewMut1, ArrayView2};
use num_traits::FromPrimitive;

fn array_bias_backward<T: Copy + Into<f32> + FromPrimitive>(db: &mut ArrayViewMut1<T>, dy: &ArrayView2<T>) {
    for (db, dy) in db.iter_mut().zip(dy.axis_iter(Axis(1))) {
        let acc: f32 = dy.iter()
            .copied()
            .map(Into::into)
            .sum();
        *db = T::from_f32((*db).into() + acc).unwrap();
    }
}

fn test_bias_backward<T: Float + From<u8> + Into<f32> + FromPrimitive>() -> Result<()> {
    let batch_size = 3;
    let units = 7;
    let data = (0 .. batch_size * units).into_iter()
        .map(|x| (x as u8).into())
        .collect();
    let dy_array = Array::from_shape_vec([batch_size, units], data)?;
    let mut db_true = Array::from_elem(units, T::one());
    array_bias_backward(&mut db_true.view_mut(), &dy_array.view());
    for device in Device::list() {
        let dy = Tensor::from_array(&device, dy_array.view())?;
        let mut db = Tensor::ones(&device, units)?;
        dbg!(smol::block_on(db.to_array()?)?);
        bias_backward(&mut db.view_mut(), &dy.view())?;
        let db_array = smol::block_on(db.to_array()?)?;
        assert_eq!(db_array, db_true);
    }
    Ok(())
}

#[test]
fn bias_backward_bf16() -> Result<()> {
    test_bias_backward::<bf16>()
}

#[test]
fn bias_backward_f32() -> Result<()> {
    test_bias_backward::<f32>()
}

fn array_cross_entropy_loss(x: &ArrayView2<f32>, t: &ArrayView2<f32>) -> Array2<f32> {
    todo!()
}

fn test_cross_entropy_loss<T: Float>() -> Result<()> {
    let n = 3;
    let c = 4;
    let x_data: Vec<T> = (0 .. n * c).into_iter()
        .map(|x| (x as u8).into())
        .collect();
    let t_data: Vec<T> = x_data.iter()
        .copied()
        .rev()
        .collect();
    let x_array = Array::from_shape_vec([n, c], x_data);
    let t_array = Array::from_shape_vec([n, c], t_data);
    let y_true = array_cross_entropy_loss(&x_array.view(), &t_array.view());
    for device in Device::list() {
        let x = Tensor::from_array(&device, x_array.view());
        
    }
}
