use crate::Result;
use async_std::future::Future;
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::hash::Hash;
use std::borrow::Cow;
use bytemuck::Pod;
use std::ops::DerefMut;

pub mod local;
pub use local::Node as LocalNode;

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
        let mem_sets = (0 .. dyn_node.num_devices()).into_iter()
            .map(|_| Mutex::new(MemSet::new()))
            .collect();
        Self(Arc::new(NodeBase {
            dyn_node,
            mem_sets
        }))
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
    fn mem_set(&self, device: u32) -> Result<impl DerefMut<Target=MemSet> + '_> {
        self.0.mem_sets.get(device as usize).ok_or_else(|| {
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
        &'a self,
        device: u32,
        mem: Mem,
        offset: usize,
        data: &'a mut [u8]
    ) -> Result<impl Future<Output=Result<()>> + 'a> {
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
    use std::collections::HashSet;
    use std::alloc::{alloc, dealloc, Layout};
    
    pub struct MemSet(HashSet<Mem>);
    
    impl MemSet {
        pub(super) fn new() -> Self {
            Self(HashSet::new())
        }    
        pub(super) fn get(&mut self) -> Mem {
            let ptr = unsafe {
                alloc(Layout::new::<u8>())
            };
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
        Local(LocalNode),
    }

    impl DynNode {
        #[implement]
        pub(super) fn num_devices(&self) -> usize {}
        #[implement]
        pub(super) fn alloc<'a>(&self, device: u32, mem: Mem, size: usize, data: Option<Cow<'a, [u8]>>) -> Result<()> {}
        #[implement]
        pub(super) fn dealloc(&self, device: u32, mem: Mem) -> Result<()> {}
        #[implement]
        pub(super) fn read<'a>(
            &'a self,
            device: u32,
            mem: Mem,
            offset: usize,
            data: &'a mut [u8]
        ) -> Result<impl Future<Output=Result<()>> + 'a> {
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
    fn read<'a>(&'a self, mem: Mem, offset: usize, data: &'a mut [u8]) -> Result<impl Future<Output=Result<()>> + 'a> {
        self.node.read(self.id, mem, offset, data)
    }
}

pub struct Buffer<T> {
    device: Device,
    mem: Mem,
    len: usize,
    _m: PhantomData<T>,
}

impl<T: Pod> Buffer<T> {
    /// # Safety
    ///   
    /// The caller should ensure that the data is not read, or that T is safe to read uninitialized. 
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
    pub fn from_vec<'a>(device: &Device, data: impl Into<Cow<'a, [T]>>) -> Result<Self> {
        let device = device.clone();
        let data = data.into();
        let len = data.len();
        let mem = device.alloc(len * std::mem::size_of::<T>(), Some(bytemuck::cast_slice(&data).into()))?;
        Ok(Self {
            device,
            mem,
            len,
            _m: PhantomData::default(),
        })
    } 
    pub fn read<'a>(&'a self, data: &'a mut [T]) -> Result<impl Future<Output=Result<()>> + 'a> {
        let data: &mut [u8] = bytemuck::cast_slice_mut(data);
        self.device.read(self.mem, 0, data)
    }
    pub fn to_vec(&self) -> Result<impl Future<Output=Result<Vec<T>>> + '_>
        where T: Pod + Sync {
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

#[allow(unused)]
pub struct SliceMut<'a, T> {
    device: Device,
    mem: Mem,
    offset: usize,
    len: usize,
    _m: PhantomData<&'a mut T>,
}
