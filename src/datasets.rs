use super::Tensor;
use std::{env, fs, fs::DirBuilder, io, marker::PhantomData, iter::FromIterator};
use rand::seq::index::IndexVecIntoIter;

pub struct Mnist {
  train_images: Vec<u8>,
  train_labels: Vec<u8>,
  test_images: Vec<u8>,
  test_labels: Vec<u8>
}

impl Mnist {
  pub fn new() -> Self {
    use io::Read;
    use byteorder::{ReadBytesExt, BigEndian};
    let current_dir = env::current_dir().unwrap();
    let mnist_dir = current_dir.join("datasets").join("mnist");
    DirBuilder::new()
      .recursive(true)
      .create(&mnist_dir)
      .unwrap();
    println!("mnist_dir: {:?}", &mnist_dir);
    let train_images = {
      let mut train_images = Vec::new();
      let train_images_path = mnist_dir.clone().join("train-images-idx3-ubyte.gz");
      if !train_images_path.is_file() {
        println!("downloading train_images: {:?}", &train_images_path);
        reqwest::get("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
          .unwrap()
          .copy_to(&mut fs::File::create(&train_images_path).unwrap())
          .unwrap();
      }
      println!("loading 60,000 train_images: {:?}", &train_images_path);
      let mut decoder = flate2::read::GzDecoder::new(fs::File::open(&train_images_path).unwrap());
      assert_eq!(decoder.read_i32::<BigEndian>().unwrap(), 2_051);
      assert_eq!(decoder.read_i32::<BigEndian>().unwrap(), 60_000);
      assert_eq!(decoder.read_i32::<BigEndian>().unwrap(), 28);
      assert_eq!(decoder.read_i32::<BigEndian>().unwrap(), 28);
      decoder.read_to_end(&mut train_images)
        .unwrap();
      assert_eq!(train_images.len(), 60_000 * 28 * 28);
      train_images
    };
    let train_labels = {
      let mut train_labels = Vec::new();
      let train_labels_path = mnist_dir.clone().join("train-labels-idx1-ubyte.gz");
      if !train_labels_path.is_file() {
        println!("downloading train_labels: {:?}", &train_labels_path);
        reqwest::get("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")
          .unwrap()
          .copy_to(&mut fs::File::create(&train_labels_path).unwrap())
          .unwrap();
      }
      println!("loading 60,000 train_labels: {:?}", &train_labels_path);
      let mut decoder = flate2::read::GzDecoder::new(fs::File::open(&train_labels_path).unwrap());
      assert_eq!(decoder.read_i32::<BigEndian>().unwrap(), 2_049);
      assert_eq!(decoder.read_i32::<BigEndian>().unwrap(), 60_000);
      decoder.read_to_end(&mut train_labels)
        .unwrap();
      assert_eq!(train_labels.len(), 60_000);
      train_labels
    };
    let test_images = {
      let mut test_images = Vec::new();
      let test_images_path = mnist_dir.clone().join("t10k-images-idx3-ubyte.gz");
      if !test_images_path.is_file() {
        println!("downloading test_images: {:?}", &test_images_path);
        reqwest::get("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")
          .unwrap()
          .copy_to(&mut fs::File::create(&test_images_path).unwrap())
          .unwrap();
      }
      println!("loading 10,000 test_images: {:?}", &test_images_path);
      let mut decoder = flate2::read::GzDecoder::new(fs::File::open(&test_images_path).unwrap());
      assert_eq!(decoder.read_i32::<BigEndian>().unwrap(), 2_051);
      assert_eq!(decoder.read_i32::<BigEndian>().unwrap(), 10_000);
      assert_eq!(decoder.read_i32::<BigEndian>().unwrap(), 28);
      assert_eq!(decoder.read_i32::<BigEndian>().unwrap(), 28);
      decoder.read_to_end(&mut test_images)
        .unwrap();
      assert_eq!(test_images.len(), 10_000 * 28 * 28);
      test_images
    };
    let test_labels = {
      let mut test_labels = Vec::new();
      let test_labels_path = mnist_dir.clone().join("t10k-labels-idx1-ubyte.gz");
      if !test_labels_path.is_file() {
        println!("downloading test_labels: {:?}", &test_labels_path);
        reqwest::get("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")
          .unwrap()
          .copy_to(&mut fs::File::create(&test_labels_path).unwrap())
          .unwrap();
      }
      println!("loading 10,000 test_labels: {:?}", &test_labels_path);
      let mut decoder = flate2::read::GzDecoder::new(fs::File::open(&test_labels_path).unwrap());
      assert_eq!(decoder.read_i32::<BigEndian>().unwrap(), 2_049);
      assert_eq!(decoder.read_i32::<BigEndian>().unwrap(), 10_000);
      decoder.read_to_end(&mut test_labels)
        .unwrap();
      assert_eq!(test_labels.len(), 10_000);
      test_labels
    };
    Self{train_images, train_labels, test_images, test_labels}
  }
  pub fn train<T: num_traits::Float + 'static>(&self, batch_size: usize) -> impl Iterator<Item=(Tensor<T>, Tensor<u8>)> + '_ {
    let index_vec = rand::seq::index::sample(&mut rand::thread_rng(), 60_000, 60_000)
      .into_iter();
    MnistTrainIter{mnist: &self, index_vec, batch_size, _m: <_>::default()}
      .take(60_000 / batch_size)
  }
  pub fn test<T: num_traits::Float>(&self, batch_size: usize) -> impl Iterator<Item=(Tensor<T>, Tensor<u8>)> + '_ {
    self.test_images.chunks(batch_size*28*28)
      .zip(self.test_labels.chunks(batch_size))
      .map(|(u_chunk, t_chunk)| {
      let x = Tensor::from_iter(u_chunk.iter()
        .map(|&u| T::from(u).unwrap() / T::from(255).unwrap()))
        .into_shape([t_chunk.len(), 1, 28, 28]);
      let t = Tensor::from_iter(t_chunk.iter().copied());
      (x, t)
    })
  }
}

struct MnistTrainIter<'a, T> {
  mnist: &'a Mnist,
  index_vec: IndexVecIntoIter,
  batch_size: usize,
  _m: PhantomData<T>
}

impl<'a, T: num_traits::Float> Iterator for MnistTrainIter<'a, T> {
  type Item = (Tensor<T>, Tensor<u8>);
  fn next(&mut self) -> Option<Self::Item> {
    let mut x = unsafe { Tensor::<T>::uninitialized([self.batch_size, 1, 28, 28]) };
    let mut t = unsafe { Tensor::<u8>::uninitialized([self.batch_size]) };
    x.as_mut_slice()
      .chunks_exact_mut(28*28)
      .zip(t.as_mut_slice().iter_mut())
      .for_each(move |(x, t)| {
      let i = self.index_vec.next()
        .unwrap();
      x.iter_mut()
        .zip(self.mnist.train_images[i*28*28..(i+1)*28*28].iter().copied())
        .for_each(|(x, u)| *x = T::from(u).unwrap() / T::from(255).unwrap());
      *t = self.mnist.train_labels[i];
    });
    Some((x, t))
  }
}


