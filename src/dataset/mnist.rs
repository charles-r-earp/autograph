use anyhow::{bail, ensure, Result};
use byteorder::{BigEndian, ReadBytesExt};
use curl::easy::Easy;
use flate2::read::GzDecoder;
use indicatif::{MultiProgress, ProgressBar, ProgressFinish, ProgressStyle};
use ndarray::{Array, Array1, Array4};
use rayon::prelude::*;
use std::{
    fs::{self, File},
    io::Read,
    io::Write,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

/// The kind of Mnist.
#[derive(Clone, Copy, Debug, Eq, PartialEq, derive_more::Display)]
pub enum MnistKind {
    /// [mnist](<http://yann.lecun.org/exdb/mnist>)
    #[display(fmt = "mnist")]
    Mnist,
    /// [fashion-mnist](<https://github.com/zalandoresearch/fashion-mnist>)
    #[display(fmt = "fashion-mnist")]
    Fashion,
}

/// Mnist builder.
pub mod builders {
    use super::{Mnist, MnistKind, Result};
    use std::path::PathBuf;

    /// Mnist builder.
    #[derive(Debug)]
    pub struct MnistBuilder {
        pub(super) path: Option<PathBuf>,
        pub(super) kind: MnistKind,
        pub(super) download: bool,
        pub(super) verbose: bool,
    }

    impl Default for MnistBuilder {
        fn default() -> Self {
            Self {
                path: None,
                kind: MnistKind::Mnist,
                download: false,
                verbose: false,
            }
        }
    }

    impl MnistBuilder {
        /// The path to load the dataset from.
        ///
        /// This is the folder the files will be downloaded to / loaded from. If not specified, uses the OS specific "Downloads" directory or the "Temp" directory.
        pub fn path(self, path: impl Into<PathBuf>) -> Self {
            Self {
                path: Some(path.into()),
                ..self
            }
        }
        /// The kind of Mnist to use. Defaults to [`MnistKind::Mnist`].
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
        /// # Errors
        /// - The download failed.
        /// - The files were not found.
        /// - Decompressing / loading the data failed.
        pub fn build(self) -> Result<Mnist> {
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
    /**
    ```no_run
    # use autograph::{
    #    anyhow::Result,
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
    pub fn builder() -> MnistBuilder {
        MnistBuilder::default()
    }
    fn build(builder: MnistBuilder) -> Result<Self> {
        let kind = builder.kind;
        let mnist_path = builder
            .path
            .unwrap_or_else(|| dirs::download_dir().unwrap_or_else(std::env::temp_dir))
            .join(kind.to_string());
        let names = [
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
        ];
        let sizes = match kind {
            MnistKind::Mnist => [9_912_422, 28_881, 1_648_877, 4_542],
            MnistKind::Fashion => [26_421_880, 29_515, 4_422_102, 5_148],
        };
        if !mnist_path.exists() {
            if builder.download {
                fs::create_dir_all(&mnist_path)?;
                download(builder.kind, &mnist_path, names, sizes, builder.verbose)
                    .map_err(|e| e.context(format!("Downloading {kind} failed!")))?;
            } else {
                bail!("{kind} not found at {mnist_path:?}!");
            }
        }
        let [train_images, train_classes, test_images, test_classes] =
            unzip(&mnist_path, names, sizes)
                .map_err(|e| e.context(format!("Decompressing {mnist_path:?} failed!")))?;
        let train_images = Array::from_shape_vec([60_000, 1, 28, 28], train_images).unwrap();
        let train_classes = Array::from_shape_vec([60_000], train_classes).unwrap();
        let test_images = Array::from_shape_vec([10_000, 1, 28, 28], test_images).unwrap();
        let test_classes = Array::from_shape_vec([10_000], test_classes).unwrap();
        Ok(Self {
            kind: builder.kind,
            train_images,
            train_classes,
            test_images,
            test_classes,
        })
    }
}

struct AbortGuard<'a> {
    inner: Option<(&'a AtomicBool, &'a ProgressBar)>,
}

impl<'a> AbortGuard<'a> {
    fn new(done: &'a AtomicBool, bar: &'a ProgressBar) -> Self {
        Self {
            inner: Some((done, bar)),
        }
    }
    fn finish(mut self) {
        self.inner.take();
    }
}

impl Drop for AbortGuard<'_> {
    fn drop(&mut self) {
        if let Some((done, bar)) = self.inner.as_ref() {
            done.store(true, Ordering::Relaxed);
            bar.finish_and_clear();
        }
    }
}

fn download(
    kind: MnistKind,
    mnist_path: &Path,
    names: [&str; 4],
    sizes: [usize; 4],
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!("Downloading {kind} to {mnist_path:?}...");
    }
    let style = ProgressStyle::with_template(
        "[{elapsed}] eta {eta} [{bar:40}] {bytes:>7} / {total_bytes:7}: {msg}",
    )
    .unwrap()
    .progress_chars("=> ");
    let multi_bar = MultiProgress::new();
    let bars = std::array::from_fn::<_, 4, _>(|i| {
        let name = names[i];
        let size = sizes[i];
        if verbose {
            let bar = ProgressBar::new(size as u64)
                .with_style(style.clone())
                .with_message(format!("{name}.gz"))
                .with_finish(ProgressFinish::AndClear);
            multi_bar.add(bar)
        } else {
            ProgressBar::hidden()
        }
    });
    let done = AtomicBool::new(false);
    let result = names.into_par_iter().zip(bars).try_for_each(|(name, bar)| {
        let guard = AbortGuard::new(&done, &bar);
        let url = match kind {
            MnistKind::Mnist => {
                format!("http://yann.lecun.org/exdb/mnist/{}.gz", name)
            }
            MnistKind::Fashion => format!(
                "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/{}.gz",
                name
            ),
        };
        let gz_path = mnist_path.join(name).with_extension("gz");
        let file = std::fs::File::create(gz_path)?;
        let mut writer = std::io::BufWriter::new(file);
        let mut write_result = Ok(());
        {
            let mut easy = Easy::new();
            easy.url(&url)?;
            let mut transfer = easy.transfer();
            transfer.write_function(|bytes| {
                if done.load(Ordering::Relaxed) {
                    return Ok(0);
                }
                write_result = writer.write_all(bytes);
                if write_result.is_err() {
                    bar.finish_and_clear();
                    return Ok(0);
                }
                bar.inc(bytes.len() as u64);
                Ok(bytes.len())
            })?;
            transfer.perform()?;
        }
        bar.finish_and_clear();
        write_result?;
        writer.flush()?;
        guard.finish();
        Ok(())
    });
    multi_bar.clear()?;
    result
}

fn unzip(mnist_path: &Path, names: [&str; 4], sizes: [usize; 4]) -> Result<[Vec<u8>; 4]> {
    let mut data = <[Vec<u8>; 4]>::default();
    data.par_iter_mut()
        .zip(names.into_par_iter().zip(sizes))
        .try_for_each(|(data, (name, size))| {
            let gz_path = mnist_path.join(name).with_extension("gz");
            let file = File::open(gz_path)?;
            {
                let file_len = file.metadata()?.len();
                let size_u64 = u64::try_from(size).unwrap();
                if file_len != size_u64 {
                    bail!("Expected {name}.gz to be {size_u64} bytes, found {file_len} bytes!");
                }
            }
            let train = name.contains("train");
            let image = name.contains("images");
            let magic = if image { 2_051 } else { 2_049 };
            let n = if train { 60_000 } else { 10_000 };
            let len = if image { n * 28 * 28 } else { n };
            let mut decoder = GzDecoder::new(file);
            ensure!(decoder.read_i32::<BigEndian>()? == magic);
            ensure!(decoder.read_i32::<BigEndian>()? == n as i32);
            if image {
                ensure!(decoder.read_i32::<BigEndian>()? == 28);
                ensure!(decoder.read_i32::<BigEndian>()? == 28);
            }
            *data = Vec::with_capacity(len);
            decoder.read_to_end(data)?;
            ensure!(data.len() == len);
            Ok(())
        })?;
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mnist() {
        let dir = tempfile::tempdir().unwrap();
        Mnist::builder()
            .kind(MnistKind::Mnist)
            .download(true)
            .path(dir.path())
            .verbose(false)
            .build()
            .unwrap();
        dir.close().unwrap();
    }

    #[test]
    fn fashion() {
        let dir = tempfile::tempdir().unwrap();
        Mnist::builder()
            .kind(MnistKind::Mnist)
            .download(true)
            .path(dir.path())
            .verbose(false)
            .build()
            .unwrap();
        dir.close().unwrap();
    }
}
