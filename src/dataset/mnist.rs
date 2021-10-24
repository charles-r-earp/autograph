use crate::result::Result;
use anyhow::ensure;
use byteorder::{BigEndian, ReadBytesExt};
use downloader::{Download, Downloader};
use flate2::read::GzDecoder;
use http::StatusCode;
use ndarray::{ArcArray, ArcArray1, Ix4};
use std::{
    fs::{self, File},
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
    }

    impl Default for MnistBuilder<'_> {
        fn default() -> Self {
            Self {
                path: None,
                kind: MnistKind::Digits,
                download: false,
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
            }
        }
        /// The kind of Mnist to use. Defaults to [`MnistKind::Digits`] (ie the original MNIST dataset).
        ///
        /// Note: FashionMNIST not yet implemented.
        pub fn kind(self, kind: MnistKind) -> Self {
            Self { kind, ..self }
        }
        /// Whether to download the data. Defaults to false.
        pub fn download(self, download: bool) -> Self {
            Self { download, ..self }
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
#[derive(Clone)]
pub struct Mnist {
    kind: MnistKind,
    images: ArcArray<u8, Ix4>,
    classes: ArcArray1<u8>,
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
        use std::io::Read;
        let mnist_name = match builder.kind {
            MnistKind::Digits => "mnist",
            MnistKind::Fashion => "fashion-mnist",
        };
        let mnist_path = builder
            .path
            .map(Path::to_owned)
            .unwrap_or_else(|| dirs::download_dir().unwrap_or_else(std::env::temp_dir))
            .join(mnist_name);
        let names = &[
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
        ];
        if builder.download {
            fs::create_dir_all(&mnist_path)?;
            let names: Vec<_> = names
                .iter()
                .filter(|name| !mnist_path.join(name).with_extension("gz").exists())
                .collect();
            if !names.is_empty() {
                let downloads: Vec<_> = names
                    .iter()
                    .map(|name| {
                        let path = mnist_path.join(name).with_extension("gz");
                        let url = match builder.kind {
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
                let summaries = std::thread::spawn(move || downloader.download(&downloads))
                    .join()
                    .unwrap()?;
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
            }
        }
        let mut images = Vec::new();
        let mut labels = Vec::new();
        {
            // unzip
            for &name in names.iter() {
                let (train, image) = match name {
                    "train-images-idx3-ubyte" => (true, true),
                    "train-labels-idx1-ubyte" => (true, false),
                    "t10k-images-idx3-ubyte" => (false, true),
                    "t10k-labels-idx1-ubyte" => (false, false),
                    _ => unreachable!(),
                };
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
                    images.extend(&data);
                } else {
                    ensure!(data.len() == n);
                    labels.extend(&data);
                }
            }
        }
        let images = ArcArray::from_shape_vec([70_000, 1, 28, 28], images)?;
        let classes = ArcArray::from_shape_vec([70_000], labels)?;
        Ok(Self {
            kind: builder.kind,
            images,
            classes,
        })
    }
    /// The images.
    ///
    /// Shape = \[70_000, 1, 28, 28\].
    pub fn images(&self) -> &ArcArray<u8, Ix4> {
        &self.images
    }
    /// The classes.
    ///
    /// Shape = \[70_000\].
    ///
    /// The classes range from 0 to 9 inclusive.
    pub fn classes(&self) -> &ArcArray1<u8> {
        &self.classes
    }
}

// TODO: tests?
