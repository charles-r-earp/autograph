use anyhow::{bail, ensure, Error, Result};
use byteorder::{BigEndian, ReadBytesExt};
use downloader::{Download, Downloader};
use flate2::read::GzDecoder;
use http::StatusCode;
use ndarray::{Array, Array1, Array4};
use std::{
    fs::{self, File},
    io::Read,
    path::Path,
};

/// The kind of Mnist.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MnistKind {
    /// [MNIST](<http://yann.lecun.com/exdb/mnist/>)
    Digits,
    /// [FashionMNIST](<https://github.com/zalandoresearch/fashion-mnist>)
    Fashion,
}

/// Mnist builder.
pub mod builders {
    use super::{Mnist, MnistKind, Result};
    use std::path::Path;

    /// Mnist builder.
    #[derive(Debug)]
    pub struct MnistBuilder<'a> {
        pub(super) path: Option<&'a Path>,
        pub(super) kind: MnistKind,
        pub(super) download: bool,
        pub(super) verbose: bool,
    }

    impl Default for MnistBuilder<'_> {
        fn default() -> Self {
            Self {
                path: None,
                kind: MnistKind::Digits,
                download: false,
                verbose: false,
            }
        }
    }

    impl MnistBuilder<'_> {
        /// The path to load the dataset from.
        ///
        /// This is the folder the files will be downloaded to / loaded from. If not specified, uses the OS specific "Downloads" directory or the "Temp" directory.
        pub fn path(self, path: &Path) -> MnistBuilder {
            MnistBuilder {
                path: Some(path),
                kind: self.kind,
                download: self.download,
                verbose: self.verbose,
            }
        }
        /// The kind of Mnist to use. Defaults to [`MnistKind::Digits`] (ie the original MNIST dataset).
        pub fn kind(self, kind: MnistKind) -> Self {
            Self { kind, ..self }
        }
        /// Whether to download the data. Defaults to false.
        pub fn download(self, download: bool) -> Self {
            Self { download, ..self }
        }
        /// Print messages to stderr. Defaults to false.
        pub fn verbose(self, verbose: bool) -> Self {
            Self { verbose, ..self }
        }
        /// Builds the dataset.
        ///
        /// **Errors**
        /// - The download failed.
        /// - The files were not found.
        /// - Decompressing / loading the data failed.
        pub fn build(&self) -> Result<Mnist> {
            Mnist::build(self)
        }
    }
}
use builders::MnistBuilder;

/// The MNIST dataset.
pub struct Mnist {
    /// The kind of MNIST.
    pub kind: MnistKind,
    /// The train images.
    ///
    /// Shape = \[60_000, 1, 28, 28\].
    pub train_images: Array4<u8>,
    /// The train classes.
    ///
    /// Shape = \[60_000\].
    ///
    /// The classes range from 0 to 9 inclusive.
    pub train_classes: Array1<u8>,
    /// The train images.
    ///
    /// Shape = \[10_000, 1, 28, 28\].
    pub test_images: Array4<u8>,
    /// The test classes.
    ///
    /// Shape = \[10_000\].
    ///
    /// The classes range from 0 to 9 inclusive.
    pub test_classes: Array1<u8>,
}

impl Mnist {
    /// Returns an [`MnistBuilder`] used to specify options.
    /*
    ```
    # use autograph::{
    #    result::Result,
    #    dataset::mnist::{Mnist, MnistKind},
    # };
    # fn main() -> Result<()> {
        let mnist = Mnist::builder()
            .path("data")
            .kind(MnistKind::Fashion)
            .download(true)
            .build()?;
        # Ok(())
    # }
    */
    pub fn builder() -> MnistBuilder<'static> {
        MnistBuilder::default()
    }
    fn build(builder: &MnistBuilder) -> Result<Self> {
        let mnist_name = match builder.kind {
            MnistKind::Digits => "mnist",
            MnistKind::Fashion => "fashion-mnist",
        };
        let mnist_path = builder
            .path
            .map(Path::to_owned)
            .unwrap_or_else(|| dirs::download_dir().unwrap_or_else(std::env::temp_dir))
            .join(mnist_name);

        if builder.download {
            fs::create_dir_all(&mnist_path)?;
            let names: Vec<_> = NAMES
                .iter()
                .filter(|name| !mnist_path.join(name).with_extension("gz").exists())
                .copied()
                .collect();
            if !names.is_empty() {
                if builder.verbose {
                    eprintln!("Downloading mnist {:?} to {mnist_path:?}...", builder.kind);
                }
                download(builder.kind, &mnist_path, &names)?;
                if builder.verbose {
                    eprintln!("Done!");
                }
            }
        } else if !mnist_path.exists() {
            bail!("mnist not found at {mnist_path:?}!");
        }
        let mut data = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for (name, data) in NAMES.into_iter().zip(data.iter_mut()) {
            *data = unzip(&mnist_path, name)?;
        }
        let [train_images, train_classes, test_images, test_classes] = data;
        let train_images =
            Array::from_shape_vec([60_000, 1, 28, 28], train_images).map_err(Error::msg)?;
        let train_classes = Array::from_shape_vec([60_000], train_classes).map_err(Error::msg)?;
        let test_images =
            Array::from_shape_vec([10_000, 1, 28, 28], test_images).map_err(Error::msg)?;
        let test_classes = Array::from_shape_vec([10_000], test_classes).map_err(Error::msg)?;
        Ok(Self {
            kind: builder.kind,
            train_images,
            train_classes,
            test_images,
            test_classes,
        })
    }
}

static NAMES: [&str; 4] = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
];

fn download(kind: MnistKind, mnist_path: &Path, names: &[&str]) -> Result<()> {
    let downloads: Vec<_> = names
        .iter()
        .map(|name| {
            let path = mnist_path.join(name).with_extension("gz");
            let url = match kind {
                MnistKind::Digits => {
                    format!("http://yann.lecun.com/exdb/mnist/{}.gz", name)
                }
                MnistKind::Fashion => format!(
                    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/{}.gz",
                    name
                ),
            };
            Download::new(&url).file_name(&path)
        })
        .collect();
    let mut downloader = Downloader::builder()
        .download_folder(&mnist_path)
        .retries(10)
        .build()?;
    let summaries = downloader.download(&downloads)?;
    for summary in summaries {
        match summary {
            Ok(_) => (),
            Err(downloader::Error::Download(summary)) => {
                if let Some((_, status)) = summary.status.last() {
                    StatusCode::from_u16(*status)?;
                }
            }
            _ => {
                summary?;
            }
        }
    }
    Ok(())
}
fn unzip(mnist_path: &Path, name: &str) -> Result<Vec<u8>> {
    let train = name.contains("train");
    let image = name.contains("images");
    let magic = if image { 2_051 } else { 2_049 };
    let n = if train { 60_000 } else { 10_000 };
    let gz_path = mnist_path.join(name).with_extension("gz");
    let mut data = Vec::new();
    let mut decoder = GzDecoder::new(File::open(&gz_path)?);
    ensure!(decoder.read_i32::<BigEndian>().unwrap() == magic);
    ensure!(decoder.read_i32::<BigEndian>().unwrap() == n as i32);
    if image {
        ensure!(decoder.read_i32::<BigEndian>().unwrap() == 28);
        ensure!(decoder.read_i32::<BigEndian>().unwrap() == 28);
    }
    decoder.read_to_end(&mut data)?;
    if image {
        ensure!(data.len() == n * 28 * 28);
    } else {
        ensure!(data.len() == n);
    }
    Ok(data)
}
