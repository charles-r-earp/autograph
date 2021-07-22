use autograph::{backend::Device, cluster::kmeans::KMeans, Result};
use ndarray::Array;

#[test]
fn kmeans_new_random_f32() -> Result<()> {
    for device in Device::list() {
        let mut kmeans = KMeans::<f32>::new(&device, 4)?;
        let data = Array::<f32, _>::zeros([100, 4]);
        smol::block_on(kmeans.init_random(&data))?;
    }
    Ok(())
}
