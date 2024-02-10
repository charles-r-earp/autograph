#[cfg(feature = "neural-network")]
use krnl::scalar::Scalar;
use ndarray::{ArrayViewMut, Dimension, RawArrayViewMut};
#[cfg(feature = "neural-network")]
use ndarray::{Axis, Ix4, Ix5, RemoveAxis};
use std::marker::PhantomData;

#[cfg(feature = "neural-network")]
pub(crate) fn array_par_outer_iter_mut_for_each<T: Scalar, D: RemoveAxis, F>(
    array: ArrayViewMut<T, D>,
    f: F,
) where
    F: Fn(usize, ArrayViewMut<T, D::Smaller>) + Send + Sync,
{
    let items = array.shape().first().copied().unwrap_or(1);
    let sync_array = SyncRawArrayViewMut::try_from(array).unwrap();
    rayon::scope(|scope| {
        scope.spawn_broadcast(move |_scope, context| {
            let _ = &sync_array;
            let item_id = context.index();
            let threads = context.num_threads();
            (item_id..items).step_by(threads).for_each(|item_id| {
                let item = sync_array.inner.clone().index_axis_move(Axis(0), item_id);
                let item = unsafe { item.deref_into_view_mut() };
                f(item_id, item);
            });
        });
    });
}

#[derive(Clone)]
pub(crate) struct SyncRawArrayViewMut<'a, T, D: Dimension> {
    #[allow(unused)]
    inner: RawArrayViewMut<T, D>,
    _m: PhantomData<&'a T>,
}

#[cfg(feature = "neural-network")]
pub(crate) type SyncRawArrayViewMut4<'a, T> = SyncRawArrayViewMut<'a, T, Ix4>;
#[cfg(feature = "neural-network")]
pub(crate) type SyncRawArrayViewMut5<'a, T> = SyncRawArrayViewMut<'a, T, Ix5>;

impl<'a, T, D: Dimension> TryFrom<ArrayViewMut<'a, T, D>> for SyncRawArrayViewMut<'a, T, D> {
    type Error = ();
    fn try_from(mut array: ArrayViewMut<T, D>) -> Result<Self, ()> {
        if array.is_standard_layout() {
            Ok(Self {
                inner: unsafe {
                    RawArrayViewMut::from_shape_ptr(array.raw_dim(), array.as_mut_ptr())
                },
                _m: PhantomData,
            })
        } else {
            Err(())
        }
    }
}

#[cfg(feature = "neural-network")]
impl<'a, T, D: Dimension> SyncRawArrayViewMut<'a, T, D> {
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.inner.as_mut_ptr()
    }
    pub(crate) fn dim(&self) -> D::Pattern {
        self.inner.dim()
    }
    /*
    pub(crate) unsafe fn uget<I>(&self, index: I) -> &'a T
    where
        I: NdIndex<D> + AsRef<[usize]> + Clone,
    {
        #[cfg(debug_assertions)]
        {
            if self.inner.get_ptr(index.clone()).is_none() {
                panic!(
                    "index {:?} out of bounds for array with shape {:?}!",
                    index.as_ref(),
                    self.inner.shape()
                );
            }
        }
        let offset = index
            .as_ref()
            .iter()
            .copied()
            .zip(self.inner.strides().iter().copied())
            .map(|(i, s)| i * s as usize)
            .sum();
        unsafe { &*self.inner.as_ptr().add(offset) }
    }
    pub(crate) unsafe fn uget_mut<I>(&mut self, index: I) -> &'a mut T
    where
        I: NdIndex<D> + AsRef<[usize]> + Clone,
    {
        #[cfg(debug_assertions)]
        {
            if self.inner.get_mut_ptr(index.clone()).is_none() {
                panic!(
                    "index {:?} out of bounds for array with shape {:?}!",
                    index.as_ref(),
                    self.inner.shape()
                );
            }
        }
        let offset = index
            .as_ref()
            .iter()
            .copied()
            .zip(self.inner.strides().iter().copied())
            .map(|(i, s)| i * s as usize)
            .sum();
        unsafe { &mut *self.inner.as_mut_ptr().add(offset) }
    }*/
}

unsafe impl<T: Send + Sync + 'static, D: Dimension> Send for SyncRawArrayViewMut<'_, T, D> {}
unsafe impl<T: Send + Sync + 'static, D: Dimension> Sync for SyncRawArrayViewMut<'_, T, D> {}

#[cfg(feature = "neural-network")]
pub(crate) fn broadcast(threads: Option<usize>, f: impl Fn(usize, usize) + Send + Sync) {
    let threads = threads
        .unwrap_or(usize::MAX)
        .min(rayon::current_num_threads());
    if threads == 1 {
        f(0, 1);
    } else {
        rayon::in_place_scope(|scope| {
            scope.spawn_broadcast(|_scope, context| {
                let thread_id = context.index();
                debug_assert!(threads <= context.num_threads());
                if thread_id < threads {
                    f(thread_id, threads);
                }
            });
        });
    }
}
