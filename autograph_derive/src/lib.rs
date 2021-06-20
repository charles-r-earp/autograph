//! # Usage
//! You can derive Network and Forward for structs and tuple structs, (enums not yet implemented).
//!```
//! use autograph::{
//!     Result,
//!     neural_network::{
//!         Network, Forward, Dense, Identity,
//!         autograd::{VariableD, Parameter2, Parameter1, ParameterMutD},
//!     },
//! };
//!
//! // Derive Network for custom layers.
//! // Network will call parameters_count / collect_paramters_mut / layers_mut on each field in
//! // order.
//! #[derive(Network)]
//! struct Layer {
//!    #[autograph(parameter)]
//!    weight: Parameter2,
//!    #[autograph(optional_parameter)]
//!    bias: Option<Parameter1>,
//!    #[autograph(layer)]
//!    dense1: Dense,
//!    #[autograph(optional_layer)]
//!    dense2: Option<Dense<Identity>>,
//!    s: String, // untagged fields are ignored
//! }
//!
//! impl Forward for Layer {
//!     fn forward(&self, input: VariableD) -> Result<VariableD> {
//!         todo!() // custom impl
//!     }
//!     // implement forward_mut because Layer has parameters
//!     // this can be used for lazy initialization
//!     fn forward_mut(&mut self, input: VariableD) -> Result<VariableD> {
//!         todo!() // custom impl
//!     }
//! }
//!
//! // Derive Network and Forward for Sequential Networks
//! // Forward will call forward / forward mut on each field in order.
//! #[derive(Network, Forward)]
//! struct Sequential {
//!    #[autograph(layer)]
//!    dense1: Dense,
//!    #[autograph(optional_layer)]
//!    dense2: Option<Dense<Identity>>,
//!    s: String, // untagged fields are ignored
//! }
//!```

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use syn::{Attribute, Data, DataStruct, DeriveInput, Fields, Index, Meta, NestedMeta};

fn autograph_path(attributes: &[Attribute]) -> TokenStream2 {
    for attribute in attributes {
        if let Ok(Meta::List(meta_list)) = attribute.parse_meta() {
            if &meta_list.path.to_token_stream().to_string() == "autograph" {
                if let Some(NestedMeta::Meta(Meta::Path(path))) = meta_list.nested.first() {
                    if path.to_token_stream().to_string() == "crate" {
                        return quote! {
                            crate
                        };
                    }
                }
            }
        }
    }
    quote! {
        ::autograph
    }
}

enum FieldKind {
    Parameter,
    OptionalParameter,
    Layer,
    OptionalLayer,
}

impl FieldKind {
    fn parse(attrs: &[Attribute]) -> Option<Self> {
        for attr in attrs {
            if let Ok(Meta::List(meta_list)) = attr.parse_meta() {
                if &meta_list.path.to_token_stream().to_string() == "autograph" {
                    if let Some(NestedMeta::Meta(Meta::Path(path))) = meta_list.nested.first() {
                        let s = path.to_token_stream().to_string();
                        match s.as_str() {
                            "parameter" => { return Some(Self::Parameter); },
                            "optional_parameter" => { return Some(Self::OptionalParameter); },
                            "layer" => { return Some(Self::Layer); },
                            "optional_layer" => { return Some(Self::OptionalLayer); },
                            _ => (),
                        }
                    }
                }
            }
        }
        None
    }
}

fn get_struct_fields(data_struct: &DataStruct) -> Vec<(TokenStream2, FieldKind)> {
    let fields = match &data_struct.fields {
        Fields::Named(fields) => &fields.named,
        Fields::Unnamed(fields) => &fields.unnamed,
        Fields::Unit => {
            return Vec::new();
        }
    };
    fields
        .iter()
        .enumerate()
        .filter_map(|(i, field)| {
            let ident = if let Some(ident) = &field.ident {
                ident.to_token_stream()
            } else {
                let index = Index::from(i);
                quote! { #index }
            };
            Some((ident, FieldKind::parse(&field.attrs)?))
        })
        .collect()
}

fn derive_network_struct(input: &DeriveInput, data_struct: &DataStruct) -> TokenStream {
    let autograph_path = autograph_path(&input.attrs);
    let fields = get_struct_fields(data_struct);
    let mut parameters_count_impl = TokenStream2::new();
    let mut collect_parameters_mut_impl = TokenStream2::new();
    let mut layers_mut_impl = TokenStream2::new();
    if !fields.is_empty() {
        let mut parameters_count_inner = Vec::new();
        let mut collect_parameters_mut_inner = TokenStream2::new();
        let mut layers_mut_inner = Vec::new();
        for (ident, kind) in fields.iter() {
            match kind {
                FieldKind::Parameter => {
                    parameters_count_inner.push(quote! {
                        1
                    });
                    collect_parameters_mut_inner.extend(quote! {
                        parameters.push(self. #ident .as_mut().into_dyn());
                    });
                }
                FieldKind::OptionalParameter => {
                    parameters_count_inner.push(quote! {
                        if self. #ident .is_some() { 1 } else { 0 }
                    });
                    collect_parameters_mut_inner.extend(quote! {
                        if let Some(parameter) = self. #ident .as_mut() {
                            parameters.push(parameter.as_mut().into_dyn());
                        }
                    })
                }
                FieldKind::Layer => {
                    parameters_count_inner.push(quote! {
                        self. #ident .parameters_count()
                    });
                    collect_parameters_mut_inner.extend(quote! {
                        self. #ident .collect_parameters_mut(parameters);
                    });
                    layers_mut_inner.push(quote! {
                        ::core::iter::once(&mut self. #ident as &mut dyn Network)
                    });
                }
                FieldKind::OptionalLayer => {
                    parameters_count_inner.push(quote! {
                        self. #ident .as_ref().map_or(0, Network::parameters_count)
                    });
                    collect_parameters_mut_inner.extend(quote! {
                        if let Some(layer) = self. #ident .as_mut() {
                            layer.collect_parameters_mut(parameters);
                        }
                    });
                    layers_mut_inner.push(quote! {
                        self. #ident .as_mut().map(|layer| layer as &mut dyn Network)
                    });
                }
            }
        }
        parameters_count_impl = quote! {
            fn parameters_count(&self) -> usize {
                #(#parameters_count_inner)+*
            }
        };
        collect_parameters_mut_impl = quote! {
            fn collect_parameters_mut<'a>(
                &'a mut self,
                parameters: &mut Vec<#autograph_path::neural_network::autograd::ParameterMutD<'a>>,
            ) {
                #collect_parameters_mut_inner
            }
        };
        layers_mut_impl = quote! {
            fn layers_mut(&mut self) -> Vec<&mut dyn Network> {
                ::core::iter::empty::<&mut dyn Network>()
                    #(.chain(#layers_mut_inner))*
                    .collect()
            }
        };
    }
    let ident = &input.ident;
    TokenStream::from(quote! {
        #[automatically_derived]
        impl Network for #ident {
            #parameters_count_impl
            #collect_parameters_mut_impl
            #layers_mut_impl
        }
    })
}

#[proc_macro_derive(Network, attributes(autograph))]
pub fn derive_network(input: TokenStream) -> TokenStream {
    let input: DeriveInput = syn::parse(input).unwrap();

    match &input.data {
        Data::Struct(data_struct) => derive_network_struct(&input, data_struct),
        Data::Enum(_data_enum) => TokenStream::from(quote! {
            compile_error!("Not yet implemented!")
        }),
        Data::Union(_) => TokenStream::from(quote! {
            compile_error!("Unions unsupported!")
        }),
    }
}

fn derive_forward_struct(input: &DeriveInput, data_struct: &DataStruct) -> TokenStream {
    let fields = get_struct_fields(data_struct);
    let mut forward_inner = Vec::new();
    let mut forward_mut_inner = Vec::new();
    for (ident, kind) in fields.iter() {
        match kind {
            FieldKind::Layer => {
                forward_inner.push(quote! {
                    x = self. #ident .forward(x)?;
                });
                forward_mut_inner.push(quote! {
                    x = self. #ident .forward_mut(x)?;
                });
            }
            FieldKind::OptionalLayer => {
                forward_inner.push(quote! {
                    if let Some(layer) = self. #ident .as_ref() {
                        x = layer.forward(x)?;
                    }
                });
                forward_mut_inner.push(quote! {
                    if let Some(layer) = self. #ident .as_mut() {
                        x = layer.forward_mut(x)?;
                    }
                });
            }
            FieldKind::Parameter | FieldKind::OptionalParameter => {
                forward_inner.push(quote! {
                    compile_error!("Cannot derive Forward with Parameters!");
                });
            }
        }
    }
    let autograph_path = autograph_path(&input.attrs);
    let ident = &input.ident;
    TokenStream::from(quote! {
        #[automatically_derived]
        impl Forward for #ident {
            fn forward(&self, input: #autograph_path::neural_network::autograd::VariableD) -> #autograph_path::Result<#autograph_path::neural_network::autograd::VariableD> {
                let mut x = input;
                #(#forward_inner)*
                Ok(x)
            }
            fn forward_mut(&mut self, input: #autograph_path::neural_network::autograd::VariableD) -> #autograph_path::Result<#autograph_path::neural_network::autograd::VariableD> {
                let mut x = input;
                #(#forward_mut_inner)*
                Ok(x)
            }
        }
    })
}

#[proc_macro_derive(Forward, attributes(autograph))]
pub fn derive_forward(input: TokenStream) -> TokenStream {
    let input: DeriveInput = syn::parse(input).unwrap();

    match &input.data {
        Data::Struct(data_struct) => derive_forward_struct(&input, data_struct),
        Data::Enum(_data_enum) => TokenStream::from(quote! {
            compile_error!("Not yet implemented!")
        }),
        Data::Union(_) => TokenStream::from(quote! {
            compile_error!("Unions unsupported!")
        }),
    }
}
