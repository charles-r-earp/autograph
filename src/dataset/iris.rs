//!
//! From the [UCI Machine Learning Repository](<http://archive.ics.uci.edu/ml/machine-learning-databases/iris/>):

/*!
1. Title: Iris Plants Database
    Updated Sept 21 by C.Blake - Added discrepency information

2. Sources:
     (a) Creator: R.A. Fisher
     (b) Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
     (c) Date: July, 1988

3. Past Usage:
   - Publications: too many to mention!!!  Here are a few.
   1. Fisher,R.A. "The use of multiple measurements in taxonomic problems"
      Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions
      to Mathematical Statistics" (John Wiley, NY, 1950).
   2. Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis.
      (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
   3. Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
      Structure and Classification Rule for Recognition in Partially Exposed
      Environments".  IEEE Transactions on Pattern Analysis and Machine
      Intelligence, Vol. PAMI-2, No. 1, 67-71.
      -- Results:
         -- very low misclassification rates (0% for the setosa class)
   4. Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE
      Transactions on Information Theory, May 1972, 431-433.
      -- Results:
         -- very low misclassification rates again
   5. See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al's AUTOCLASS II
      conceptual clustering system finds 3 classes in the data.

4. Relevant Information:
   --- This is perhaps the best known database to be found in the pattern
       recognition literature.  Fisher's paper is a classic in the field
       and is referenced frequently to this day.  (See Duda & Hart, for
       example.)  The data set contains 3 classes of 50 instances each,
       where each class refers to a type of iris plant.  One class is
       linearly separable from the other 2; the latter are NOT linearly
       separable from each other.
   --- Predicted attribute: class of iris plant.
   --- This is an exceedingly simple domain.
   --- This data differs from the data presented in Fishers article
    (identified by Steve Chadwick,  spchadwick@espeedaz.net )
    The 35th sample should be: 4.9,3.1,1.5,0.2,"Iris-setosa"
    where the error is in the fourth feature.
    The 38th sample: 4.9,3.6,1.4,0.1,"Iris-setosa"
    where the errors are in the second and third features.

5. Number of Instances: 150 (50 in each of three classes)

6. Number of Attributes: 4 numeric, predictive attributes and the class

7. Attribute Information:
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class:
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica

8. Missing Attribute Values: None

Summary Statistics:
             Min  Max   Mean    SD   Class Correlation
   sepal length: 4.3  7.9   5.84  0.83    0.7826
    sepal width: 2.0  4.4   3.05  0.43   -0.4194
   petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
    petal width: 0.1  2.5   1.20  0.76    0.9565  (high!)

9. Class Distribution: 33.3% for each of 3 classes.
*/

use ndarray::{ArcArray, ArcArray1, ArcArray2};

// Data from http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

const DIMENSIONS: [[f32; 4]; 150] = [
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2],
    [5.0, 3.6, 1.4, 0.2],
    [5.4, 3.9, 1.7, 0.4],
    [4.6, 3.4, 1.4, 0.3],
    [5.0, 3.4, 1.5, 0.2],
    [4.4, 2.9, 1.4, 0.2],
    [4.9, 3.1, 1.5, 0.1],
    [5.4, 3.7, 1.5, 0.2],
    [4.8, 3.4, 1.6, 0.2],
    [4.8, 3.0, 1.4, 0.1],
    [4.3, 3.0, 1.1, 0.1],
    [5.8, 4.0, 1.2, 0.2],
    [5.7, 4.4, 1.5, 0.4],
    [5.4, 3.9, 1.3, 0.4],
    [5.1, 3.5, 1.4, 0.3],
    [5.7, 3.8, 1.7, 0.3],
    [5.1, 3.8, 1.5, 0.3],
    [5.4, 3.4, 1.7, 0.2],
    [5.1, 3.7, 1.5, 0.4],
    [4.6, 3.6, 1.0, 0.2],
    [5.1, 3.3, 1.7, 0.5],
    [4.8, 3.4, 1.9, 0.2],
    [5.0, 3.0, 1.6, 0.2],
    [5.0, 3.4, 1.6, 0.4],
    [5.2, 3.5, 1.5, 0.2],
    [5.2, 3.4, 1.4, 0.2],
    [4.7, 3.2, 1.6, 0.2],
    [4.8, 3.1, 1.6, 0.2],
    [5.4, 3.4, 1.5, 0.4],
    [5.2, 4.1, 1.5, 0.1],
    [5.5, 4.2, 1.4, 0.2],
    [4.9, 3.1, 1.5, 0.1],
    [5.0, 3.2, 1.2, 0.2],
    [5.5, 3.5, 1.3, 0.2],
    [4.9, 3.1, 1.5, 0.1],
    [4.4, 3.0, 1.3, 0.2],
    [5.1, 3.4, 1.5, 0.2],
    [5.0, 3.5, 1.3, 0.3],
    [4.5, 2.3, 1.3, 0.3],
    [4.4, 3.2, 1.3, 0.2],
    [5.0, 3.5, 1.6, 0.6],
    [5.1, 3.8, 1.9, 0.4],
    [4.8, 3.0, 1.4, 0.3],
    [5.1, 3.8, 1.6, 0.2],
    [4.6, 3.2, 1.4, 0.2],
    [5.3, 3.7, 1.5, 0.2],
    [5.0, 3.3, 1.4, 0.2],
    [7.0, 3.2, 4.7, 1.4],
    [6.4, 3.2, 4.5, 1.5],
    [6.9, 3.1, 4.9, 1.5],
    [5.5, 2.3, 4.0, 1.3],
    [6.5, 2.8, 4.6, 1.5],
    [5.7, 2.8, 4.5, 1.3],
    [6.3, 3.3, 4.7, 1.6],
    [4.9, 2.4, 3.3, 1.0],
    [6.6, 2.9, 4.6, 1.3],
    [5.2, 2.7, 3.9, 1.4],
    [5.0, 2.0, 3.5, 1.0],
    [5.9, 3.0, 4.2, 1.5],
    [6.0, 2.2, 4.0, 1.0],
    [6.1, 2.9, 4.7, 1.4],
    [5.6, 2.9, 3.6, 1.3],
    [6.7, 3.1, 4.4, 1.4],
    [5.6, 3.0, 4.5, 1.5],
    [5.8, 2.7, 4.1, 1.0],
    [6.2, 2.2, 4.5, 1.5],
    [5.6, 2.5, 3.9, 1.1],
    [5.9, 3.2, 4.8, 1.8],
    [6.1, 2.8, 4.0, 1.3],
    [6.3, 2.5, 4.9, 1.5],
    [6.1, 2.8, 4.7, 1.2],
    [6.4, 2.9, 4.3, 1.3],
    [6.6, 3.0, 4.4, 1.4],
    [6.8, 2.8, 4.8, 1.4],
    [6.7, 3.0, 5.0, 1.7],
    [6.0, 2.9, 4.5, 1.5],
    [5.7, 2.6, 3.5, 1.0],
    [5.5, 2.4, 3.8, 1.1],
    [5.5, 2.4, 3.7, 1.0],
    [5.8, 2.7, 3.9, 1.2],
    [6.0, 2.7, 5.1, 1.6],
    [5.4, 3.0, 4.5, 1.5],
    [6.0, 3.4, 4.5, 1.6],
    [6.7, 3.1, 4.7, 1.5],
    [6.3, 2.3, 4.4, 1.3],
    [5.6, 3.0, 4.1, 1.3],
    [5.5, 2.5, 4.0, 1.3],
    [5.5, 2.6, 4.4, 1.2],
    [6.1, 3.0, 4.6, 1.4],
    [5.8, 2.6, 4.0, 1.2],
    [5.0, 2.3, 3.3, 1.0],
    [5.6, 2.7, 4.2, 1.3],
    [5.7, 3.0, 4.2, 1.2],
    [5.7, 2.9, 4.2, 1.3],
    [6.2, 2.9, 4.3, 1.3],
    [5.1, 2.5, 3.0, 1.1],
    [5.7, 2.8, 4.1, 1.3],
    [6.3, 3.3, 6.0, 2.5],
    [5.8, 2.7, 5.1, 1.9],
    [7.1, 3.0, 5.9, 2.1],
    [6.3, 2.9, 5.6, 1.8],
    [6.5, 3.0, 5.8, 2.2],
    [7.6, 3.0, 6.6, 2.1],
    [4.9, 2.5, 4.5, 1.7],
    [7.3, 2.9, 6.3, 1.8],
    [6.7, 2.5, 5.8, 1.8],
    [7.2, 3.6, 6.1, 2.5],
    [6.5, 3.2, 5.1, 2.0],
    [6.4, 2.7, 5.3, 1.9],
    [6.8, 3.0, 5.5, 2.1],
    [5.7, 2.5, 5.0, 2.0],
    [5.8, 2.8, 5.1, 2.4],
    [6.4, 3.2, 5.3, 2.3],
    [6.5, 3.0, 5.5, 1.8],
    [7.7, 3.8, 6.7, 2.2],
    [7.7, 2.6, 6.9, 2.3],
    [6.0, 2.2, 5.0, 1.5],
    [6.9, 3.2, 5.7, 2.3],
    [5.6, 2.8, 4.9, 2.0],
    [7.7, 2.8, 6.7, 2.0],
    [6.3, 2.7, 4.9, 1.8],
    [6.7, 3.3, 5.7, 2.1],
    [7.2, 3.2, 6.0, 1.8],
    [6.2, 2.8, 4.8, 1.8],
    [6.1, 3.0, 4.9, 1.8],
    [6.4, 2.8, 5.6, 2.1],
    [7.2, 3.0, 5.8, 1.6],
    [7.4, 2.8, 6.1, 1.9],
    [7.9, 3.8, 6.4, 2.0],
    [6.4, 2.8, 5.6, 2.2],
    [6.3, 2.8, 5.1, 1.5],
    [6.1, 2.6, 5.6, 1.4],
    [7.7, 3.0, 6.1, 2.3],
    [6.3, 3.4, 5.6, 2.4],
    [6.4, 3.1, 5.5, 1.8],
    [6.0, 3.0, 4.8, 1.8],
    [6.9, 3.1, 5.4, 2.1],
    [6.7, 3.1, 5.6, 2.4],
    [6.9, 3.1, 5.1, 2.3],
    [5.8, 2.7, 5.1, 1.9],
    [6.8, 3.2, 5.9, 2.3],
    [6.7, 3.3, 5.7, 2.5],
    [6.7, 3.0, 5.2, 2.3],
    [6.3, 2.5, 5.0, 1.9],
    [6.5, 3.0, 5.2, 2.0],
    [6.2, 3.4, 5.4, 2.3],
    [5.9, 3.0, 5.1, 1.8],
];

const CLASSES: [u8; 150] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
];

/// The Iris dataset.
#[derive(Clone, Debug)]
pub struct Iris {
    dimensions: ArcArray2<f32>,
    classes: ArcArray1<u8>,
}

impl Iris {
    /// Constructs the dataset.
    pub fn new() -> Self {
        let dimensions =
            ArcArray::from_shape_vec([150, 4], DIMENSIONS.iter().flatten().copied().collect())
                .unwrap();
        let classes = ArcArray::from(CLASSES.to_vec());
        Self {
            dimensions,
            classes,
        }
    }
    /*
    fn load(path: impl AsRef<Path>) -> Result<Self> {
        use anyhow::{bail, ensure};
        use std::{fs, path::Path, str::FromStr};
        let csv = fs::read_to_string(path)?;
        let mut dimensions = vec![-1.; 150 * 4];
        let mut classes = vec![150; 150];
        for (line, (x, t)) in csv
            .lines()
            .zip(dimensions.chunks_exact_mut(4).zip(classes.iter_mut()))
        {
            let mut items = line.split(',');
            for (x, s) in x.iter_mut().zip(&mut items) {
                *x = f32::from_str(s)?;
            }
            *t = match items.next() {
                Some("Iris-setosa") => 0,
                Some("Iris-versicolor") => 1,
                Some("Iris-virginica") => 2,
                Some(s) => bail!("Unable to parse iris class {:?}", s),
                None => bail!("Unable to parse iris.data: expected class, found None!"),
            };
        }
        ensure!(dimensions.iter().all(|x| *x > 0.));
        ensure!(classes.iter().all(|x| *x <= 2));
        let dimensions = ArcArray::from_shape_vec([150, 4], dimensions)?;
        let classes = ArcArray::from(classes);
        Ok(Self {
            dimensions,
            classes,
        })
    }*/
    /// Returns the dimensions of each flower.
    ///
    /// Dimensions:
    /// - sepal length
    /// - sepal width,
    /// - petal length
    /// - petal width
    pub fn dimensions(&self) -> &ArcArray2<f32> {
        &self.dimensions
    }
    /// Returns the names of the dimensions.
    ///
    /// Dimensions:
    /// - sepal length
    /// - sepal width,
    /// - petal length
    /// - petal width
    pub fn dimension_names(&self) -> [&'static str; 4] {
        ["sepal length", "sepal width", "petal length", "petal width"]
    }
    /// Returns the class of each flower.
    ///
    /// Classes:
    /// - Iris Setosa
    /// - Iris Versicolour
    /// - Iris Virginica
    pub fn classes(&self) -> &ArcArray1<u8> {
        &self.classes
    }
    /// Returns the class names.
    ///
    /// Classes:
    /// - Iris Setosa
    /// - Iris Versicolour
    /// - Iris Virginica
    pub fn class_names(&self) -> [&'static str; 3] {
        ["Iris Setosa", "Iris Versicolour", "Iris Virginica"]
    }
}

impl Default for Iris {
    fn default() -> Self {
        Self::new()
    }
}
