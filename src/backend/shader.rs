use crate::{
    Result,
    future::BlockingFuture,
};
use std::{
    mem::size_of,
    borrow::Cow,
    sync::{Arc, atomic::{AtomicUsize, Ordering}},
    alloc::Layout,
    collections::{HashMap, HashSet},
};
use anyhow::{anyhow, bail, ensure};
use rspirv::{
    binary::Parser,
    dr::{Loader, Operand},
    spirv::{Decoration, ExecutionMode, ExecutionModel, Op, StorageClass, Word},
};
use smol::lock::Mutex;
use hibitset::BitSet;
use lazy_static::lazy_static;
use tinyvec::ArrayVec;

pub(crate) static MAX_PUSH_CONSTANT_SIZE: usize = 64;

lazy_static! {
    static ref MODULE_IDS: Mutex<BitSet> = Mutex::new(BitSet::new());
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct ModuleId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct EntryId(u32);

impl ModuleId {
    async fn create() -> Result<Self> {
        let mut ids = MODULE_IDS.lock().await;
        todo!()
    }
}


pub struct Module {
    spirv: Cow<'static, [u8]>,
    descriptor: ModuleDescriptor,
    id: ModuleId,
}

impl Module {
    pub fn from_spirv(spirv: impl Into<Cow<'static, [u8]>>) -> Result<Self> {
        let spirv = spirv.into();
        let descriptor = ModuleDescriptor::parse(&spirv)?;
        let id = ModuleId::create().block()?;
        Ok(Self {
            spirv,
            descriptor,
            id,
        })
    }
}

#[derive(Default, Debug)]
struct ModuleDescriptor {
    entries: HashMap<String, EntryDescriptor>,
}

impl ModuleDescriptor {
    fn parse(spirv: &[u8]) -> Result<Self> {
        parse_spirv(spirv)
    }
}

#[derive(Clone, Copy, Debug)]
struct EntryDescriptor {
    id: EntryId,
    local_size: [u32; 3],
    buffers: Vec<bool>,
    push_constants: u8,
}

impl EntryDescriptor {
    pub(crate) fn mutable_buffers(&self) -> usize {
        self.mutable_buffers as usize
    }
    pub(crate) fn immutable_buffers(&self) -> usize {
        self.immutable_buffers as usize
    }
    pub(crate) fn push_constants(&self) -> usize {
        self.push_constants as usize
    }
}

#[derive(Clone, Debug)]
struct EntryPoint {
    name: String,
    local_size: [u32; 3],
}

#[derive(Clone, Copy, Debug)]
struct BufferBinding {
    descriptor_set: u32,
    binding: u32,
    mutable: bool,
}

fn parse_spirv(spirv: &[u8]) -> Result<ModuleDescriptor> {
    let mut loader = Loader::new();
    Parser::new(&spirv, &mut loader)
        .parse()
        .map_err(|e| anyhow!("{:?}", e))?;
    let module = loader.module();

    let mut entry_points = HashMap::<Word, EntryPoint>::new();

    for (i, inst) in module.entry_points.iter().enumerate() {
        if inst.class.opcode == Op::EntryPoint {
            match inst.operands.get(0) {
                Some(Operand::ExecutionModel(ExecutionModel::GLCompute)) => (),
                _ => continue,
            };
            let fn_id = match inst.operands.get(1) {
                Some(&Operand::IdRef(fn_id)) => fn_id,
                _ => bail!(
                    "entry_points[{}] operands[1] invalid entry id:\n{:#?}",
                    i,
                    &inst
                ),
            };
            let name = match inst.operands.get(2) {
                Some(Operand::LiteralString(name)) => name.to_string(),
                _ => bail!(
                    "entry_points[{}] operands[1] invalid entry name:\n{:#?}",
                    i,
                    &inst
                ),
            };
            entry_points.insert(
                fn_id,
                EntryPoint {
                    name,
                    local_size: [1, 1, 1],
                },
            );
        }
    }

    for (i, inst) in module.execution_modes.iter().enumerate() {
        if inst.class.opcode == Op::ExecutionMode {
            match inst.operands.get(1) {
                Some(Operand::ExecutionMode(ExecutionMode::LocalSize)) => (),
                _ => continue,
            }
            let fn_id = match inst.operands.get(0) {
                Some(&Operand::IdRef(fn_id)) => fn_id,
                _ => bail!(
                    "execution_modes[{}] operands[0] invalid entry id:\n{:#?}",
                    i,
                    &inst
                ),
            };
            let local_size = match inst.operands.get(2..=4) {
                Some(
                    &[Operand::LiteralInt32(x), Operand::LiteralInt32(y), Operand::LiteralInt32(z)],
                ) => [x, y, z],
                _ => bail!(
                    "execution_modes[{}] operands[2] invalid local_size:\n{:#?}",
                    i,
                    &inst
                ),
            };
            if let Some(entry_point) = entry_points.get_mut(&fn_id) {
                entry_point.local_size = local_size;
            }
        }
    }

    let mut descriptor_sets = HashMap::<Word, u32>::new();
    let mut bindings = HashMap::<Word, u32>::new();
    let mut nonwritable = HashSet::<Word>::new();
    let mut field_offsets = HashMap::<(Word, u32), u32>::new();

    for (i, inst) in module.annotations.iter().enumerate() {
        match inst.class.opcode {
            Op::Decorate => {
                let id = match inst.operands.get(0) {
                    Some(&Operand::IdRef(id)) => Ok(id),
                    _ => Err(anyhow!(
                        "annotations[{}] operands[0] invalid id:\n{:#?}",
                        i,
                        &inst
                    )),
                };
                let x = match inst.operands.get(2) {
                    Some(&Operand::LiteralInt32(x)) => Ok(x),
                    _ => Err(anyhow!(
                        "annotations[{}] operands[2] invalid set / binding:\n{:#?}",
                        i,
                        &inst
                    )),
                };
                if let Some(&Operand::Decoration(decoration)) = inst.operands.get(1) {
                    match decoration {
                        Decoration::DescriptorSet => {
                            descriptor_sets.insert(id?, x?);
                        }
                        Decoration::Binding => {
                            bindings.insert(id?, x?);
                        }
                        _ => (),
                    }
                }
            }
            Op::MemberDecorate => {
                let id = match inst.operands.get(0) {
                    Some(&Operand::IdRef(id)) => Ok(id),
                    _ => Err(anyhow!(
                        "annotations[{}] operands[0] invalid id:\n{:#?}",
                        i,
                        &inst
                    )),
                };
                match inst.operands.get(2) {
                    Some(&Operand::Decoration(Decoration::Offset)) => {
                        let index = match inst.operands.get(1) {
                            Some(&Operand::LiteralInt32(index)) => index,
                            _ => bail!(
                                "annotations[{}] operands[1] invalid field offset index:\n{:#?}",
                                i,
                                &inst
                            ),
                        };
                        let offset = match inst.operands.get(3) {
                            Some(&Operand::LiteralInt32(offset)) => offset,
                            _ => bail!(
                                "annotations[{}] operands[3] invalid field offset:\n{:#?}",
                                i,
                                &inst
                            ),
                        };
                        field_offsets.insert((id?, index), offset);
                    }
                    Some(&Operand::Decoration(Decoration::NonWritable)) => {
                        nonwritable.insert(id?);
                    }
                    _ => (),
                }
            }
            _ => (),
        }
    }

    let mut layouts = HashMap::<Word, Layout>::new();
    let mut pointers = HashMap::<Word, (StorageClass, Word)>::new();
    let mut variables = HashMap::<Word, (StorageClass, Word)>::new();

    for (i, inst) in module.types_global_values.iter().enumerate() {
        let result_id = inst.result_id.ok_or_else(|| {
            anyhow!(
                "types_global_values[{}].result_id found None:\n{:#?}",
                i,
                &inst
            )
        })?;
        match inst.class.opcode {
            Op::TypeBool => {
                let layout = Layout::new::<bool>();
                layouts.insert(result_id, layout);
            }
            Op::TypeInt | Op::TypeFloat => {
                let layout = match inst.operands.get(0) {
                    Some(&Operand::LiteralInt32(bits)) => match bits {
                        8 => Layout::new::<u8>(),
                        16 => Layout::new::<u16>(),
                        32 => Layout::new::<u32>(),
                        64 => Layout::new::<u64>(),
                        _ => bail!("types_global_values[{}] operands[0] unsupported Int / Float width:\n{:#?}\n\nSupported widths are 8, 16, 32, 64.", i, &inst),
                    },
                    _ => bail!("types_global_values[{}] operands[0] invalid Int / Float width:\n{:#?}", i, &inst),
                };
                layouts.insert(result_id, layout);
            }
            Op::TypeStruct => {
                let mut layout = Some(Layout::from_size_align(0, 1).unwrap());
                for (o, operand) in inst.operands.iter().enumerate() {
                    if let Operand::IdRef(field) = *operand {
                        if let Some(&field_layout) = layouts.get(&field) {
                            let (next_layout, offset) =
                                layout.unwrap().extend(field_layout).map_err(|_| {
                                    anyhow!(
                                        "types_global_values[{}] unable to get layout for struct:\n{:#?}\nlayout = {:?}\nfield_layout[{}] = {:?}",
                                        i, &inst, &layout, field, &field_layout
                                    )
                                })?;
                            ensure!(
                                field_offsets.get(&(result_id, o as u32)) == Some(&(offset as u32)),
                                "types_global_values[{}] operands[{}] field offsets do not match:\n{:#?}", i, o, &inst
                            );
                            layout.replace(next_layout);
                        } else {
                            layout.take();
                            break;
                        }
                    } else {
                        bail!(
                            "types_global_values[{}] operands[{}] invalid struct field:\n{:#?}",
                            i,
                            o,
                            &inst
                        );
                    }
                }
                if let Some(layout) = layout {
                    layouts.insert(result_id, layout);
                }
            }
            Op::TypeVector => {
                let elem = match inst.operands.get(0) {
                    Some(&Operand::IdRef(elem)) => elem,
                    _ => bail!(
                        "types_global_values[{}] operands[0] invalid vector elem:\n{:#?}",
                        i,
                        &inst
                    ),
                };
                let n = match inst.operands.get(1) {
                    Some(&Operand::LiteralInt32(n)) => n,
                    _ => bail!(
                        "types_global_values[{}] operands[0] invalid vector len:\n{:#?}",
                        i,
                        &inst
                    ),
                };
                if let Some(&elem_layout) = layouts.get(&elem) {
                    let mut layout = Layout::from_size_align(0, 1).unwrap();
                    for _ in 0..n {
                        let (next_layout, _) = layout.extend(elem_layout).map_err(|_| {
                            anyhow!(
                                "types_global_values[{}] unable to get layout for vector:\n{:#?}\nlayout = {:?}\nelem_layout = {:?}",
                                i, &inst, &layout, &elem_layout
                            )
                        })?;
                        layout = next_layout;
                    }
                    layouts.insert(result_id, layout);
                }
            }
            Op::TypePointer => {
                let storage_class = match inst.operands.get(0) {
                    Some(&Operand::StorageClass(storage_class)) => {
                        if !(storage_class == StorageClass::StorageBuffer
                            || storage_class == StorageClass::Uniform
                            || storage_class == StorageClass::PushConstant)
                        {
                            continue;
                        } else {
                            storage_class
                        }
                    }
                    _ => continue,
                };
                let pointee = match inst.operands.get(1) {
                    Some(&Operand::IdRef(pointee)) => pointee,
                    _ => bail!(
                        "types_global_values[{}] operands[1] invalid pointee:\n{:#?}",
                        i,
                        &inst
                    ),
                };
                pointers.insert(result_id, (storage_class, pointee));
            }
            Op::Variable => {
                let storage_class = match inst.operands.get(0) {
                    Some(&Operand::StorageClass(storage_class)) => {
                        if !(storage_class == StorageClass::StorageBuffer
                            || storage_class == StorageClass::Uniform
                            || storage_class == StorageClass::PushConstant)
                        {
                            continue;
                        } else {
                            storage_class
                        }
                    }
                    _ => continue,
                };
                let result_type = inst.result_type.ok_or_else(|| {
                    anyhow!(
                        "types_global_values[{}].result_type found None:\n{:#?}",
                        i,
                        &inst
                    )
                })?;
                variables.insert(result_id, (storage_class, result_type));
            }
            _ => (),
        }
    }

    let mut buffer_bindings = HashMap::<Word, BufferBinding>::new();
    let mut push_constants = HashMap::<Word, u8>::new();

    for (&variable, &(storage_class, pointer)) in variables.iter() {
        let &(storage_class2, pointee) = pointers.get(&pointer).ok_or_else(|| {
            anyhow!(
                "(internal error) pointer not found {}:\npointers = {:?}",
                pointer,
                &pointers
            )
        })?;

        ensure!(
            storage_class == storage_class2,
            "invalid spirv: variable {} storage classs {:?} does not match pointer {} storage class {:?}",
            variable, storage_class, pointer, storage_class2
        );
        match storage_class {
            StorageClass::StorageBuffer | StorageClass::Uniform => {
                let &descriptor_set = descriptor_sets.get(&variable)
                    .ok_or_else(|| {
                    anyhow!(
                        "(internal error) descriptor set not found for variable {}:\ndescriptor_sets = {:?}",
                        variable, &descriptor_sets
                    )
                })?;
                let &binding = bindings.get(&variable).ok_or_else(|| {
                    anyhow!(
                        "(internal error) binding not found for variable {}:\nbindings = {:?}",
                        variable,
                        &bindings
                    )
                })?;
                let mutable = !nonwritable.contains(&pointee);
                buffer_bindings.insert(
                    variable,
                    BufferBinding {
                        descriptor_set,
                        binding,
                        mutable,
                    },
                );
            }
            StorageClass::PushConstant => {
                let layout = layouts.get(&pointee).ok_or_else(|| {
                    anyhow!(
                        "(internal error) layout not found for push constant:\npointee = {}\nlayouts = {:#?}",
                        pointee, &layouts,
                    )
                })?;
                let layout_size = layout.size();
                let push_consts = if layout_size > MAX_PUSH_CONSTANT_SIZE {
                    bail!("Push constant range must be less than {}!", MAX_PUSH_CONSTANT_SIZE);
                } else {
                    layout_size as u8
                };
                push_constants.insert(variable, push_consts);
            }
            _ => unreachable!(),
        }
    }

    let mut entry_id = 0;
    let mut module_descriptor = ModuleDescriptor::default();

    for (f, function) in module.functions.iter().enumerate() {
        let fn_id = if let Some(def) = function.def.as_ref() {
            def.result_id.ok_or_else(|| {
                anyhow!(
                    "functions[{}].def.result_id found None:\n{:#?}",
                    f,
                    &function
                )
            })?
        } else {
            continue;
        };
        if let Some(entry_point) = entry_points.get(&fn_id) {
            let mut parameters = HashSet::<Word>::new();
            for (b, block) in function.blocks.iter().enumerate() {
                for (i, inst) in block.instructions.iter().enumerate() {
                    match inst.class.opcode {
                        Op::Load
                        | Op::Store
                        | Op::AccessChain
                        | Op::InBoundsAccessChain
                        | Op::PtrAccessChain => {
                            let variable = match inst.operands.get(0) {
                                Some(&Operand::IdRef(variable)) => variable,
                                _ => bail!("entry_point: {} functions[{}].blocks[{}].instructions[{}].operands[0] invalid variable:\n{:#?}", &entry_point.name, f, b, i, &function),
                            };
                            parameters.insert(variable);
                        }
                        _ => (),
                    }
                }
            }
            let mut buffers = Vec::<BufferBinding>::new();
            let mut push_constant_range = 0;
            for &variable in parameters.iter() {
                if let Some(&buffer) = buffer_bindings.get(&variable) {
                    ensure!(
                        buffer.descriptor_set == 0,
                        "entry_point: {} functions[{}] descriptor set must be 0\n{:#?}",
                        &entry_point.name,
                        f,
                        &function,
                    );
                    buffers.push(buffer);
                } else if let Some(&push_consts) = push_constants.get(&variable) {
                    if push_constant_range > 0 {
                        bail!("entry_point: {} functions[{}] only 1 push constant block is allowed:\n{:#?}", &entry_point.name, f, &function);
                    }
                    push_constant_range = push_consts;
                }
            }
            buffers.sort_by_key(|b| b.binding);
            let mut mutable_buffers = 0;
            let mut immutable_buffers = 0;
            for buffer in buffers.iter() {
                if buffer.mutable {
                    if immutable_buffers > 0 {
                        bail!("entry_point: {} functions[{}] mutable buffers must be before immutable_buffers:\n{:#?}", &entry_point.name, f, &function);
                    }
                    mutable_buffers += 1;
                } else {
                    immutable_buffers += 1;
                }
            }
            module_descriptor.entries.insert(entry_point.name.clone(), EntryDescriptor {
                id: EntryId({
                    let id = entry_id;
                    entry_id += 1;
                    id
                }),
                local_size: entry_point.local_size,
                mutable_buffers,
                immutable_buffers,
                push_constants: push_constant_range,
            });
        }
    }

    Ok(module_descriptor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_module_from_spirv() -> Result<()> {
        ShaderModule::from_spirv(include_shader!("rust/shader.spv"))?;
        Ok(())
    }
}
