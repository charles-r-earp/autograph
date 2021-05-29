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

fn is_ignore(attributes: &[Attribute]) -> bool {
    let ignore_string = quote! {
        #[autograph(ignore)]
    }
    .to_string();
    for attribute in attributes {
        if attribute.to_token_stream().to_string() == ignore_string {
            return true;
        }
    }
    false
}

fn get_layers_struct(data_struct: &DataStruct) -> Vec<TokenStream2> {
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
            if is_ignore(&field.attrs) {
                None
            } else if let Some(ident) = &field.ident {
                Some(ident.to_token_stream())
            } else {
                let index = Index::from(i);
                Some(quote! { #index })
            }
        })
        .collect()
}

fn derive_network_struct(input: &DeriveInput, data_struct: &DataStruct) -> TokenStream {
    let autograph_path = autograph_path(&input.attrs);
    let layers = get_layers_struct(data_struct);
    let collect_paramters_mut_impl = match layers.as_slice() {
        &[] => TokenStream2::new(),
        layers => {
            quote! {
                fn collect_paramters_mut<'a>(
                    &'a mut self,
                    parameters: &mut Vec<#autograph_path::neural_network::autograd::ParameterViewMutD<'a>>,
                ) -> #autograph_path::Result<()> {
                    #(self. #layers.collect_paramters_mut(parameters)?;)*
                    Ok(())
                }
            }
        }
    };
    let layers_mut_impl = match layers.as_slice() {
        &[] => TokenStream2::new(),
        layers => {
            quote! {
                fn layers_mut(&mut self) -> Vec<&mut dyn Network> {
                    vec![#(&mut self. #layers),*]
                }
            }
        }
    };
    let ident = &input.ident;
    TokenStream::from(quote! {
        #[automatically_derived]
        impl Network for #ident {
            #collect_paramters_mut_impl
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
    let layers = get_layers_struct(data_struct);
    let forward_body = match layers.as_slice() {
        &[] => quote! {
            Ok(input)
        },
        layers => quote! {
            input #(.forward(&self. #layers))?*
        },
    };
    let forward_mut_body = match layers.as_slice() {
        &[] => quote! {
            Ok(input)
        },
        layers => quote! {
            input #(.forward_mut(&mut self. #layers))?*
        },
    };
    let autograph_path = autograph_path(&input.attrs);
    let ident = &input.ident;
    TokenStream::from(quote! {
        #[automatically_derived]
        impl Forward for #ident {
            fn forward(&self, input: #autograph_path::neural_network::autograd::VariableD) -> #autograph_path::Result<#autograph_path::neural_network::autograd::VariableD> {
                #forward_body
            }
            fn forward_mut(&mut self, input: #autograph_path::neural_network::autograd::VariableD) -> #autograph_path::Result<#autograph_path::neural_network::autograd::VariableD> {
                #forward_mut_body
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
