use autograph::{backend::Device, cluster::kmeans::KMeans, dataset::SimpleDataset, Result};
use ndarray::Array;

#[test]
fn kmeans_new_random() -> Result<()> {
    for device in Device::list() {
        let mut kmeans = KMeans::<f32>::new(&device, 4)?;
        let data = SimpleDataset::from_input(Array::zeros([100, 4]));
        kmeans.init_random(&data)?;
    }
    Ok(())
}
