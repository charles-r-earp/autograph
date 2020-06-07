use std::{env, fs::{File, DirBuilder}, io::Read, iter::Iterator};
use byteorder::{ReadBytesExt, BigEndian};
use flate2::read::GzDecoder;
use ndarray::{ArrayView, ArrayView4, ArrayView1};

pub struct Mnist {
  train_images: Vec<u8>,
  train_labels: Vec<u8>,
  test_images: Vec<u8>,
  test_labels: Vec<u8>
}

impl Mnist {
  pub fn new() -> Self {
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
          .copy_to(&mut File::create(&train_images_path).unwrap())
          .unwrap();
      }
      println!("loading 60,000 train_images: {:?}", &train_images_path);
      let mut decoder = GzDecoder::new(File::open(&train_images_path).unwrap());
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
          .copy_to(&mut File::create(&train_labels_path).unwrap())
          .unwrap();
      }
      println!("loading 60,000 train_labels: {:?}", &train_labels_path);
      let mut decoder = GzDecoder::new(File::open(&train_labels_path).unwrap());
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
          .copy_to(&mut File::create(&test_images_path).unwrap())
          .unwrap();
      }
      println!("loading 10,000 test_images: {:?}", &test_images_path);
      let mut decoder = GzDecoder::new(File::open(&test_images_path).unwrap());
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
          .copy_to(&mut File::create(&test_labels_path).unwrap())
          .unwrap();
      }
      println!("loading 10,000 test_labels: {:?}", &test_labels_path);
      let mut decoder = GzDecoder::new(File::open(&test_labels_path).unwrap());
      assert_eq!(decoder.read_i32::<BigEndian>().unwrap(), 2_049);
      assert_eq!(decoder.read_i32::<BigEndian>().unwrap(), 10_000);
      decoder.read_to_end(&mut test_labels)
        .unwrap();
      assert_eq!(test_labels.len(), 10_000);
      test_labels
    };
    Self{train_images, train_labels, test_images, test_labels}
  }
  pub fn train<'a>(&'a self, batch_size: usize) -> impl Iterator<Item=(ArrayView4<'a, u8>, ArrayView1<'a, u8>)> + 'a {
    self.train_images.as_slice()
      .chunks_exact(batch_size*28*28)
      .map(move |x| ArrayView::from_shape([batch_size, 1, 28, 28], x).unwrap())
      .zip(self.train_labels.as_slice()
        .chunks_exact(batch_size)
        .map(move |t| ArrayView::from_shape([batch_size], t).unwrap()))
  }
  pub fn eval<'a>(&'a self, batch_size: usize) -> impl Iterator<Item=(ArrayView4<'a, u8>, ArrayView1<'a, u8>)> + 'a {
    self.test_images.as_slice()
      .chunks_exact(batch_size*28*28)
      .map(move |x| ArrayView::from_shape([batch_size, 1, 28, 28], x).unwrap())
      .zip(self.test_labels.as_slice()
        .chunks_exact(batch_size)
        .map(move |t| ArrayView::from_shape([batch_size], t).unwrap()))
  }
}

