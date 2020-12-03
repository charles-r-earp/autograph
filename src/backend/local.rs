use super::Mem;
use crate::Result;
use async_std::future::Future;
use std::fmt::Debug;
use std::borrow::Cow;

#[doc(hidden)]
pub mod gpu;
#[doc(inline)]
pub use gpu::Gpu;
use gpu::GpuBuilder;

#[proxy_enum::proxy(Device)]
pub mod device_proxy {
    use super::*;

    #[derive(Debug)]
    pub enum Device {
        Gpu(Gpu),
    }

    impl Device {
        #[implement]
        pub(super) fn alloc<'a>(&self, mem: Mem, size: usize, data: Option<Cow<'a, [u8]>>) -> Result<()> {}
        #[implement]
        pub(super) fn dealloc(&self, mem: Mem) -> Result<()> {}
        #[implement]
        pub(super) fn read<'a>(
            &'a self,
            mem: Mem,
            offset: usize,
            data: &'a mut [u8]
        ) -> Result<impl Future<Output=Result<()>> + 'a> {
        }
    }
}
pub use device_proxy::Device;

pub mod builder {
    use super::*;

    #[proxy_enum::proxy(DeviceBuilder)]
    pub mod device_builder_proxy {
        use super::*;

        pub enum DeviceBuilder {
            Gpu(GpuBuilder),
        }

        impl Default for DeviceBuilder {
            fn default() -> Self {
                Gpu::builder().into()
            }
        }

        impl DeviceBuilder {
            #[implement]
            pub(super) fn build(self) -> impl Future<Output = Result<Device>> {}
        }
    }
    #[doc(inline)]
    pub use device_builder_proxy::DeviceBuilder;

    #[derive(Default)]
    pub struct NodeBuilder {
        devices: Vec<DeviceBuilder>,
    }

    impl NodeBuilder {
        pub async fn build(mut self) -> Result<super::super::Node> {
            let mut devices = Vec::new();
            if self.devices.is_empty() {
                self.devices.push(DeviceBuilder::default());
            }
            for device in self.devices {
                devices.push(device.build().await?);
            }
            let node = Node { devices };
            Ok(super::super::Node::new(node.into()))
        }
    }
}
use builder::NodeBuilder;

#[derive(Debug)]
pub struct Node {
    devices: Vec<Device>,
}

impl Node {
    pub fn builder() -> NodeBuilder {
        NodeBuilder::default()
    }
    pub(super) fn num_devices(&self) -> usize {
        self.devices.len()
    }
    fn device(&self, device: u32) -> Result<&Device> {
        self.devices.get(device as usize).ok_or_else(|| {
            format!(
                "Device {} out of range ({} devices)!",
                device,
                self.devices.len()
            )
            .into()
        })
    }
    pub(super) fn alloc<'a>(&self, device: u32, mem: Mem, size: usize, data: Option<Cow<'a, [u8]>>) -> Result<()> {
        self.device(device)?.alloc(mem, size, data)
    }
    pub(super) fn dealloc(&self, device: u32, mem: Mem) -> Result<()> {
        self.device(device)?.dealloc(mem)
    }
    pub(super) fn read<'a>(
        &'a self,
        device: u32,
        mem: Mem,
        offset: usize,
        data: &'a mut [u8]
    ) -> Result<impl Future<Output=Result<()>> + 'a> {
        self.device(device)?
            .read(mem, offset, data)
    }
}
