use crate::Result;
use async_std::future::Future;
use bytemuck::Pod;
use std::borrow::Cow;
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{DerefMut, RangeBounds};
use std::sync::{Arc, Mutex};

pub mod local;

/// This is a virtual slice in device memory
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Mem(u64);

#[doc(hidden)]
pub struct NodeBase {
    dyn_node: DynNode,
    mem_sets: Vec<Mutex<MemSet>>,
}

#[derive(Clone)]
pub struct Node(Arc<NodeBase>);

impl Node {
    fn new(dyn_node: DynNode) -> Self {
        let mem_sets = (0..dyn_node.num_devices())
            .into_iter()
            .map(|_| Mutex::new(MemSet::new()))
            .collect();
        Self(Arc::new(NodeBase { dyn_node, mem_sets }))
    }
    pub fn devices(&self) -> Vec<Device> {
        (0..self.0.dyn_node.num_devices() as u32)
            .into_iter()
            .map(|id| Device {
                node: self.clone(),
                id,
            })
            .collect()
    }
    fn mem_set(&self, device: u32) -> Result<impl DerefMut<Target = MemSet> + '_> {
        self.0
            .mem_sets
            .get(device as usize)
            .ok_or_else(|| {
                format!(
                    "Node: Device {} out of range ({} mem sets)!",
                    device,
                    self.0.mem_sets.len()
                )
            })?
            .lock()
            .map_err(|_| "Node: Unable to lock Mutex<MetSet>!".into())
    }
    fn dyn_node(&self) -> &DynNode {
        &self.0.dyn_node
    }
    fn alloc<'a>(&self, device: u32, size: usize, data: Option<Cow<'a, [u8]>>) -> Result<Mem> {
        let mem = self.mem_set(device)?.get();
        self.dyn_node().alloc(device, mem, size, data)?;
        Ok(mem)
    }
    fn dealloc(&self, device: u32, mem: Mem) -> Result<()> {
        self.mem_set(device)?.remove(&mem);
        self.dyn_node().dealloc(device, mem)
    }
    fn read<'a>(
        &self,
        device: u32,
        mem: Mem,
        offset: usize,
        data: &'a mut [u8],
    ) -> Result<impl Future<Output = Result<()>> + 'a> {
        self.dyn_node().read(device, mem, offset, data)
    }
}

impl Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.dyn_node.fmt(f)
    }
}

#[doc(hidden)]
pub mod mem_set {
    use super::Mem;
    use std::alloc::{alloc, dealloc, Layout};
    use std::collections::HashSet;

    pub struct MemSet(HashSet<Mem>);

    impl MemSet {
        pub(super) fn new() -> Self {
            Self(HashSet::new())
        }
        pub(super) fn get(&mut self) -> Mem {
            let ptr = unsafe { alloc(Layout::new::<u8>()) };
            let mem = Mem(ptr as _);
            self.0.insert(mem);
            mem
        }
        pub(super) fn remove(&mut self, mem: &Mem) {
            if let Some(mem) = self.0.take(mem) {
                unsafe {
                    let ptr = mem.0 as usize as *mut u8;
                    dealloc(ptr as usize as *mut u8, Layout::new::<u8>());
                }
            }
        }
    }
}
use mem_set::MemSet;

#[doc(hidden)]
#[proxy_enum::proxy(DynNode)]
pub mod dyn_node_proxy {
    use super::*;

    #[derive(Debug)]
    pub enum DynNode {
        Local(local::Node),
    }

    impl DynNode {
        #[implement]
        pub(super) fn num_devices(&self) -> usize {}
        #[implement]
        pub(super) fn alloc<'a>(
            &self,
            device: u32,
            mem: Mem,
            size: usize,
            data: Option<Cow<'a, [u8]>>,
        ) -> Result<()> {
        }
        #[implement]
        pub(super) fn dealloc(&self, device: u32, mem: Mem) -> Result<()> {}
        #[implement]
        pub(super) fn read<'a>(
            &self,
            device: u32,
            mem: Mem,
            offset: usize,
            data: &'a mut [u8],
        ) -> Result<impl Future<Output = Result<()>> + 'a> {
        }
    }
}
use dyn_node_proxy::DynNode;

#[derive(Clone, Debug)]
pub struct Device {
    node: Node,
    id: u32,
}

impl Device {
    fn alloc<'a>(&self, size: usize, data: Option<Cow<'a, [u8]>>) -> Result<Mem> {
        self.node.alloc(self.id, size, data)
    }
    fn dealloc(&self, mem: Mem) -> Result<()> {
        self.node.dealloc(self.id, mem)
    }
    fn read<'a>(
        &self,
        mem: Mem,
        offset: usize,
        data: &'a mut [u8],
    ) -> Result<impl Future<Output = Result<()>> + 'a> {
        self.node.read(self.id, mem, offset, data)
    }
}

mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

trait RangeBoundsExt: RangeBounds<usize> {
    fn to_offset_len(&self) -> (usize, Option<usize>);
}

impl<B: RangeBounds<usize>> RangeBoundsExt for B {
    fn to_offset_len(&self) -> (usize, Option<usize>) {
        use std::ops::Bound::*;
        let offset = match self.start_bound() {
            Included(&b) => b,
            Excluded(&b) => b + 1,
            Unbounded => 0,
        };
        let len = match self.end_bound() {
            Included(&b) => Some(b),
            Excluded(&b) => Some(b - 1),
            Unbounded => None,
        };
        (offset, len)
    }
}

pub trait AsSlice: Sealed {
    type Elem;
    fn as_slice(&self) -> Slice<Self::Elem>;
    fn slice(&self, bounds: impl RangeBounds<usize>) -> Option<Slice<Self::Elem>>;
}

pub trait AsSliceMut: AsSlice {
    fn as_slice_mut(&mut self) -> SliceMut<Self::Elem>;
    fn slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> Option<SliceMut<Self::Elem>>;
}

pub struct Buffer<T> {
    device: Device,
    mem: Mem,
    len: usize,
    _m: PhantomData<T>,
}

impl<T> Buffer<T> {
    /// # Safety
    ///   
    /// The caller should ensure that the data is overwritten prior to being read.
    pub unsafe fn uninitialized(device: &Device, len: usize) -> Result<Self> {
        let device = device.clone();
        let mem = device.alloc(len * std::mem::size_of::<T>(), None)?;
        Ok(Self {
            device,
            mem,
            len,
            _m: PhantomData::default(),
        })
    }
    pub fn from_vec<'a>(device: &Device, data: impl Into<Cow<'a, [T]>>) -> Result<Self>
    where
        T: Pod,
    {
        let device = device.clone();
        let data = data.into();
        let len = data.len();
        let mem = device.alloc(
            len * std::mem::size_of::<T>(),
            Some(bytemuck::cast_slice(&data).into()),
        )?;
        Ok(Self {
            device,
            mem,
            len,
            _m: PhantomData::default(),
        })
    }
    fn offset(&self) -> usize {
        0
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        self.device.dealloc(self.mem).unwrap();
    }
}

impl<T> Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct(&format!("Buffer<{}>", std::any::type_name::<T>()))
            .field("device", &self.device)
            .field("mem", &self.mem)
            .field("len", &self.len)
            .finish()
    }
}

#[allow(unused)]
pub struct Slice<'a, T> {
    device: Device,
    mem: Mem,
    offset: usize,
    len: usize,
    _m: PhantomData<&'a T>,
}

impl<T> Slice<'_, T> {
    // Needed for impl_buffer_methods!
    fn offset(&self) -> usize {
        self.offset
    }
}

impl<T> Debug for Slice<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct(&format!("Slice<{}>", std::any::type_name::<T>()))
            .field("device", &self.device)
            .field("mem", &self.mem)
            .field("offset", &self.offset)
            .field("len", &self.len)
            .finish()
    }
}

#[allow(unused)]
pub struct SliceMut<'a, T> {
    device: Device,
    mem: Mem,
    offset: usize,
    len: usize,
    _m: PhantomData<&'a mut T>,
}

impl<T> SliceMut<'_, T> {
    fn offset(&self) -> usize {
        self.offset
    }
}

impl<T> Debug for SliceMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct(&format!("SliceMut<{}>", std::any::type_name::<T>()))
            .field("device", &self.device)
            .field("mem", &self.mem)
            .field("offset", &self.offset)
            .field("len", &self.len)
            .finish()
    }
}

// We can't use Deref / AsRef, and can't use a trait with futures, and there are lifetime issues,
// so we have to implement these for each via a macro
macro_rules! impl_buffer_methods {
    ($t:ty) => {
        impl<T> $t {
            pub fn len(&self) -> usize {
                self.len
            }
            pub fn as_slice(&self) -> Slice<T> {
                Slice {
                    device: self.device.clone(),
                    mem: self.mem,
                    offset: self.offset(),
                    len: self.len,
                    _m: PhantomData::default(),
                }
            }
            pub fn slice(&self, bounds: impl RangeBounds<usize>) -> Option<Slice<T>> {
                let (offset, len) = bounds.to_offset_len();
                let len = len.unwrap_or(self.len() - offset);
                let offset = self.offset() + offset;
                if offset + len < self.offset() + self.len() {
                    Some(Slice {
                        device: self.device.clone(),
                        mem: self.mem,
                        offset,
                        len,
                        _m: PhantomData::default(),
                    })
                } else {
                    None
                }
            }
        }

        impl<T> Sealed for $t {}

        impl<T> AsSlice for $t {
            type Elem = T;
            fn as_slice(&self) -> Slice<T> {
                Self::as_slice(self)
            }
            fn slice(&self, bounds: impl RangeBounds<usize>) -> Option<Slice<T>> {
                Self::slice(self, bounds)
            }
        }

        impl<T: Pod + Sync> $t {
            pub fn read<'a>(
                &self,
                data: &'a mut [T],
            ) -> Result<impl Future<Output = Result<()>> + 'a> {
                let data: &mut [u8] = bytemuck::cast_slice_mut(data);
                self.device.read(self.mem, self.offset(), data)
            }
            pub fn to_vec(&self) -> Result<impl Future<Output = Result<Vec<T>>>> {
                use std::cell::UnsafeCell;
                let vec = Arc::new(UnsafeCell::new(vec![T::zeroed(); self.len]));
                let read_future = {
                    let slice = unsafe { &mut *vec.get() };
                    self.read(slice)?
                };
                Ok(async move {
                    read_future.await?;
                    let vec = Arc::try_unwrap(vec)
                        .map_err(|_| "Unable to unwrap!")?
                        .into_inner();
                    Ok(vec)
                })
            }
        }
    };
    (mut $t:ty) => {
        impl_buffer_methods!($t);

        impl<T> $t {
            pub fn as_slice_mut(&mut self) -> SliceMut<T> {
                SliceMut {
                    device: self.device.clone(),
                    mem: self.mem,
                    offset: self.offset(),
                    len: self.len,
                    _m: PhantomData::default(),
                }
            }
            pub fn slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> Option<SliceMut<T>> {
                let (offset, len) = bounds.to_offset_len();
                let len = len.unwrap_or(self.len() - offset);
                let offset = self.offset() + offset;
                if offset + len < self.offset() + self.len() {
                    Some(SliceMut {
                        device: self.device.clone(),
                        mem: self.mem,
                        offset,
                        len,
                        _m: PhantomData::default(),
                    })
                } else {
                    None
                }
            }
        }

        impl<T> AsSliceMut for $t {
            fn as_slice_mut(&mut self) -> SliceMut<T> {
                Self::as_slice_mut(self)
            }
            fn slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> Option<SliceMut<T>> {
                Self::slice_mut(self, bounds)
            }
        }
    };
}

impl_buffer_methods!(mut Buffer<T>);
impl_buffer_methods!(Slice<'_, T>);
impl_buffer_methods!(mut SliceMut<'_, T>);
