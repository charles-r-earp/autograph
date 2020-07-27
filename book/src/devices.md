# Devices

A Device represents either a cpu or gpu. Generally the gpu has its own memory. Transferring data from cpu to gpu memory is relatively expensive compared to executing operations. Generally operations can only be performed on data that is in a particular device's memory, at least multi device operations are slower than single device operations. 

In autograph, a Device is an enum:
```
#[derive(Clone, Debug)]
pub enum Device {
    Cpu(Arc<Cpu>),
    #[cfg(feature = "cuda")]
    Cuda(Arc<CudaGpu>),
}
```
This represents either a Cpu or CudaGpu (potentially more in the future). A Device can be created from a Cpu or CudaGpu using the From trait:
```
let cpu = Device::from(Cpu::new());
let gpu = Device::from(CudaGpu::new(0)); // Get the first gpu
```
Device also implements Default, which can be used to get a device, potentially the first gpu if available:
```
let device = Device::default();
```
Device can be cloned, which clones the internal Arc (a threadsafe shared pointer). Devices cloned this way refer to the same object. The Cpu and CudaGpu structs store handles used to execute operations on with the backends (oneDNN and cuDNN for example).
