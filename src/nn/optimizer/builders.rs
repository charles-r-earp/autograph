use super::Sgd;

pub struct SgdBuilder {
    pub(super) learning_rate: f32,
    pub(super) momentum: f32
}

impl Default for SgdBuilder {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            momentum: 0.
        }
    }
}

impl SgdBuilder {
    pub fn learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }
    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }
    pub fn build(self) -> Sgd {
        self.into()
    }
}
