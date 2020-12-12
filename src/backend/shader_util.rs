use super::{BufferDescriptor, EntryDescriptor, PushConstantDescriptor, PushConstantRange};
use crate::error::ShaderModuleError::InvalidSpirv;
use crate::Result;
use rspirv::binary::Parser;
use rspirv::dr::{Loader, Operand};
use rspirv::spirv::{Decoration, ExecutionMode, ExecutionModel, Op, StorageClass, Word};
use std::alloc::Layout;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

pub fn compile_glsl(src: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use std::io::Read;
    let mut spirv = Vec::new();
    glsl_to_spirv::compile(src, glsl_to_spirv::ShaderType::Compute)?.read_to_end(&mut spirv)?;
    Ok(spirv)
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

pub(super) fn entry_descriptors_from_spirv(spirv: &[u8]) -> Result<Vec<EntryDescriptor>> {
    let mut loader = Loader::new();
    Parser::new(&spirv, &mut loader)
        .parse()
        .or(Err(InvalidSpirv))?;
    let module = loader.module();

    let mut entry_points = HashMap::<Word, EntryPoint>::new();

    for inst in module.entry_points.iter() {
        if inst.class.opcode == Op::EntryPoint {
            match inst.operands.get(0) {
                Some(Operand::ExecutionModel(ExecutionModel::GLCompute)) => (),
                _ => continue,
            };
            let fn_id = match inst.operands.get(1) {
                Some(&Operand::IdRef(fn_id)) => fn_id,
                _ => Err(InvalidSpirv)?,
            };
            let name = match inst.operands.get(2) {
                Some(Operand::LiteralString(name)) => name.to_string(),
                _ => Err(InvalidSpirv)?,
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

    for inst in module.execution_modes.iter() {
        if inst.class.opcode == Op::ExecutionMode {
            match inst.operands.get(1) {
                Some(Operand::ExecutionMode(ExecutionMode::LocalSize)) => (),
                _ => continue,
            }
            let fn_id = match inst.operands.get(0) {
                Some(&Operand::IdRef(fn_id)) => fn_id,
                _ => Err(InvalidSpirv)?,
            };
            let local_size = match inst.operands.get(2..=4) {
                Some(
                    &[Operand::LiteralInt32(x), Operand::LiteralInt32(y), Operand::LiteralInt32(z)],
                ) => [x, y, z],
                _ => Err(InvalidSpirv)?,
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

    for inst in module.annotations.iter() {
        match inst.class.opcode {
            Op::Decorate => {
                let id = match inst.operands.get(0) {
                    Some(&Operand::IdRef(id)) => Ok(id),
                    _ => Err(InvalidSpirv),
                };
                let x = match inst.operands.get(2) {
                    Some(&Operand::LiteralInt32(x)) => Ok(x),
                    _ => Err(InvalidSpirv),
                };
                match inst.operands.get(1) {
                    Some(&Operand::Decoration(decoration)) => match decoration {
                        Decoration::DescriptorSet => {
                            descriptor_sets.insert(id?, x?);
                        }
                        Decoration::Binding => {
                            bindings.insert(id?, x?);
                        }
                        _ => (),
                    },
                    _ => (),
                }
            }
            Op::MemberDecorate => {
                let id = match inst.operands.get(0) {
                    Some(&Operand::IdRef(id)) => Ok(id),
                    _ => Err(InvalidSpirv),
                };
                match inst.operands.get(2) {
                    Some(&Operand::Decoration(Decoration::Offset)) => {
                        let index = match inst.operands.get(1) {
                            Some(&Operand::LiteralInt32(index)) => index,
                            _ => Err(InvalidSpirv)?,
                        };
                        let offset = match inst.operands.get(3) {
                            Some(&Operand::LiteralInt32(index)) => index,
                            _ => Err(InvalidSpirv)?,
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

    for inst in module.types_global_values.iter() {
        match inst.class.opcode {
            Op::TypeBool => {
                let result_id = inst.result_id.ok_or(InvalidSpirv)?;
                let layout = Layout::new::<bool>();
                layouts.insert(result_id, layout);
            }
            Op::TypeInt | Op::TypeFloat => {
                let result_id = inst.result_id.ok_or(InvalidSpirv)?;
                let layout = match inst.operands.get(0) {
                    Some(&Operand::LiteralInt32(bits)) => match bits {
                        8 => Layout::new::<u8>(),
                        16 => Layout::new::<u16>(),
                        32 => Layout::new::<u32>(),
                        64 => Layout::new::<u64>(),
                        _ => Err(InvalidSpirv)?,
                    },
                    _ => Err(InvalidSpirv)?,
                };
                layouts.insert(result_id, layout);
            }
            Op::TypeStruct => {
                let result_id = inst.result_id.ok_or(InvalidSpirv)?;
                let mut layout = Some(Layout::from_size_align(0, 1).unwrap());
                for (o, operand) in inst.operands.iter().enumerate() {
                    match operand {
                        &Operand::IdRef(field) => {
                            if let Some(&field_layout) = layouts.get(&field) {
                                let (next_layout, offset) =
                                    layout.unwrap().extend(field_layout).or(Err(InvalidSpirv))?;
                                if field_offsets.get(&(result_id, o as u32))
                                    != Some(&(offset as u32))
                                {
                                    Err(InvalidSpirv)?;
                                }
                                layout.replace(next_layout);
                            } else {
                                layout.take();
                                break;
                            }
                        }
                        _ => Err(InvalidSpirv)?,
                    }
                }
                if let Some(layout) = layout {
                    layouts.insert(result_id, layout);
                }
            }
            Op::TypeVector => {
                let result_id = inst.result_id.ok_or(InvalidSpirv)?;
                let elem = match inst.operands.get(0) {
                    Some(&Operand::IdRef(elem)) => elem,
                    _ => Err(InvalidSpirv)?,
                };
                let n = match inst.operands.get(1) {
                    Some(&Operand::LiteralInt32(n)) => n,
                    _ => Err(InvalidSpirv)?,
                };
                if let Some(&elem_layout) = layouts.get(&elem) {
                    let mut layout = Layout::from_size_align(0, 1).unwrap();
                    for _ in 0..n {
                        let (next_layout, _) = layout.extend(elem_layout).or(Err(InvalidSpirv))?;
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
                    _ => Err(InvalidSpirv)?,
                };
                let result_id = inst.result_id.ok_or(InvalidSpirv)?;
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
                let result_type = inst.result_type.ok_or(InvalidSpirv)?;
                let result_id = inst.result_id.ok_or(InvalidSpirv)?;
                variables.insert(result_id, (storage_class, result_type));
            }
            _ => (),
        }
    }

    let mut buffer_bindings = HashMap::<Word, BufferBinding>::new();
    let mut push_constants = HashMap::<Word, PushConstantDescriptor>::new();
    let mut push_constant_offset = 0;

    for (&variable, &(storage_class, pointer)) in variables.iter() {
        let &(storage_class2, pointee) = pointers.get(&pointer).ok_or(InvalidSpirv)?;

        if storage_class != storage_class2 {
            Err(InvalidSpirv)?;
        }

        match storage_class {
            StorageClass::StorageBuffer | StorageClass::Uniform => {
                let &descriptor_set = descriptor_sets.get(&variable).ok_or(InvalidSpirv)?;
                let &binding = bindings.get(&variable).ok_or(InvalidSpirv)?;
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
                let layout = layouts.get(&pointee).ok_or(InvalidSpirv)?;
                let range = PushConstantRange {
                    start: push_constant_offset as u32,
                    end: (push_constant_offset + layout.size()) as u32,
                };
                push_constant_offset += layout.size();
                push_constants.insert(variable, PushConstantDescriptor { range });
            }
            _ => unreachable!(),
        }
    }

    let mut entry_descriptors = Vec::<EntryDescriptor>::new();

    for function in module.functions.iter() {
        let fn_id = if let Some(def) = function.def.as_ref() {
            def.result_id.ok_or(InvalidSpirv)?
        } else {
            continue;
        };
        if let Some(entry_point) = entry_points.get(&fn_id) {
            let mut parameters = HashSet::<Word>::new();
            for block in function.blocks.iter() {
                for inst in block.instructions.iter() {
                    match inst.class.opcode {
                        Op::Load
                        | Op::Store
                        | Op::AccessChain
                        | Op::InBoundsAccessChain
                        | Op::PtrAccessChain => {
                            let variable = match inst.operands.get(0) {
                                Some(&Operand::IdRef(variable)) => variable,
                                _ => Err(InvalidSpirv)?,
                            };
                            parameters.insert(variable);
                        }
                        _ => (),
                    }
                }
            }
            let mut buffer_descriptors = Vec::<BufferDescriptor>::new();
            let mut push_constant_descriptor: Option<PushConstantDescriptor> = None;
            for &variable in parameters.iter() {
                if let Some(&buffer) = buffer_bindings.get(&variable) {
                    if buffer.descriptor_set != 0 {
                        Err(InvalidSpirv)?;
                    }
                    buffer_descriptors.push(BufferDescriptor {
                        binding: buffer.binding,
                        mutable: buffer.mutable,
                    });
                } else if let Some(&push_constant) = push_constants.get(&variable) {
                    if push_constant_descriptor.is_some() {
                        Err(InvalidSpirv)?;
                    }
                    push_constant_descriptor.replace(push_constant);
                }
            }
            buffer_descriptors.sort_by_key(|b| b.binding);
            entry_descriptors.push(EntryDescriptor {
                name: entry_point.name.clone(),
                local_size: entry_point.local_size,
                buffer_descriptors,
                push_constant_descriptor,
            });
        }
    }

    Ok(entry_descriptors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let spirv = compile_glsl(
            r#"
        #version 450
        
        layout(local_size_x=64) in;
        
        layout(binding=0) readonly buffer X1 {
            float x1[];    
        };
        
        layout(binding=1) readonly buffer X2 {
            float x2[];
        };
        
        layout(binding=2) buffer Y {
            float y[];
        };
        
        layout(push_constant) uniform PushConsts {
            uint n;
        };
        
        void main() {
            uint gid = gl_GlobalInvocationID.x;
            if (gid < n) {
                y[gid] = x1[gid] + x2[gid];
            }
        }
        
        "#,
        )
        .unwrap();

        let entry_descriptors = entry_descriptors_from_spirv(&spirv).unwrap();

        let target = vec![EntryDescriptor {
            name: String::from("main"),
            local_size: [64, 1, 1],
            buffer_descriptors: vec![
                BufferDescriptor {
                    binding: 0,
                    mutable: false,
                },
                BufferDescriptor {
                    binding: 1,
                    mutable: false,
                },
                BufferDescriptor {
                    binding: 2,
                    mutable: true,
                },
            ],
            push_constant_descriptor: Some(PushConstantDescriptor {
                range: PushConstantRange { start: 0, end: 4 },
            }),
        }];

        assert_eq!(
            &entry_descriptors, &target,
            "output:\n{:#?}\n!=\ntarget\n{:#?}",
            entry_descriptors, target
        );
    }
}
