//! # Derive Layer and implement Forward
//! Layer can be derived for a struct composed of other layers\
//! #[impl_forward(D, Dy)] generates a sequential implementation for Forward<D, OutputDim=Dy>\
//! use #[autograph(skip)] to skip fields\
//!```
//! use autograph::autograd::{Variable, ParameterD};
//! use autograph::layer::{
//!    Layer, 
//!    Forward,
//!    Dense,
//!    Conv2d,
//!    MaxPool2d,
//!    Relu,
//!    Flatten
//! }; 
//! use ndarray::{Ix2, Ix4};
//!
//! #[impl_forward(Ix4, Ix2)] 
//! #[derive(Layer)]
//! struct Lenet5 (
//!     Conv2d,
//!     Relu,
//!     MaxPool2d,
//!     Conv2d,
//!     Relu,
//!     MaxPool2d,
//!     Flatten,
//!     Dense,
//!     Relu,
//!     Dense,
//!     Relu,
//!     Dense
//! );
//!```
//! Generates:\
//! ```
//! impl Layer for Lenet5 {
//!     fn parameters(&self) -> Vec<ParameterD> {
//!         self.0
//!             .parameters()
//!             .into_iter()
//!             .chain(self.1.parameters())
//!             .chain(self.2.parameters())
//!             .chain(self.3.parameters())
//!             .chain(self.4.parameters())
//!             .chain(self.5.parameters())
//!             .chain(self.6.parameters())
//!             .chain(self.7.parameters())
//!             .chain(self.8.parameters())
//!             .chain(self.9.parameters())
//!             .chain(self.10.parameters())
//!             .chain(self.11.parameters())
//!             .collect()
//!     }
//!     fn set_training(&mut self, training: bool) {
//!         self.0.set_training(training);
//!         self.1.set_training(training);
//!         self.2.set_training(training);
//!         self.3.set_training(training);
//!         self.4.set_training(training);
//!         self.5.set_training(training);
//!         self.6.set_training(training);
//!         self.7.set_training(training);
//!         self.8.set_training(training);
//!         self.9.set_training(training);
//!         self.10.set_training(training);
//!         self.11.set_training(training);
//!    }
//! }
//! impl Forward<Ix4> for Lenet5 {
//!     type OutputDim = Ix2;
//!     fn forward(&self, input: &Variable<Ix4>) -> Variable<Ix2> {
//!         input
//!             .forward(&self.0)
//!             .forward(&self.1)
//!             .forward(&self.2)
//!             .forward(&self.3)
//!             .forward(&self.4)
//!             .forward(&self.5)
//!             .forward(&self.6)
//!             .forward(&self.7)
//!             .forward(&self.8)
//!             .forward(&self.9)
//!             .forward(&self.10)
//!             .forward(&self.11)
//!     }
//! }
//!```

use proc_macro::TokenStream as BaseTokenStream;
use proc_macro2::TokenStream;
use syn::{
    DeriveInput,
    Data,
    Fields,
    Index,
    ItemStruct,
    Attribute,
    Meta,
    NestedMeta
};
use quote::{ToTokens, quote};

fn is_autograph_skip(attributes: &[Attribute]) -> bool {
    let skip = quote! {
        #[autograph(skip)]    
    };
    for attribute in attributes {
        if attribute.to_token_stream().to_string() == skip.to_string() {
            return true;
        }
    }
    return false;
}

#[proc_macro_derive(Layer, attributes(autograph))]
pub fn derive_layer(input: BaseTokenStream) -> BaseTokenStream {
    let input: DeriveInput = syn::parse(input).unwrap();
    let fields = match &input.data {
        Data::Struct(data_struct) => {
            match &data_struct.fields {
                Fields::Named(fields) => Some(&fields.named),
                Fields::Unnamed(fields) => Some(&fields.unnamed),
                Fields::Unit => None
            }
        },
        _ => unimplemented!()
    };
    let ident = &input.ident;
    let default_impl = quote! {
        impl Layer for #ident {}
    };
    if let Some(fields) = fields {
        if !fields.is_empty() {
            let len = fields.len();
            let mut parameters_impl = TokenStream::new();
            let mut set_training_impl = TokenStream::new();
            for (i, field) in fields.into_iter().enumerate() {
                if is_autograph_skip(&field.attrs) {
                    continue;
                }
                if let Some(ident) = &field.ident {
                    if len == 1 {
                        parameters_impl.extend(quote! { self. #ident .parameters() });
                    }
                    else if i == 0 {
                        parameters_impl.extend(quote! { self. #ident .parameters().into_iter() });
                    }
                    else {
                        parameters_impl.extend(quote! { .chain(self. #ident .parameters()) });
                    }
                    set_training_impl.extend(quote! { self. #ident .set_training(training); });
                }
                else {
                    let index = Index::from(i);
                    if len == 1 {
                        parameters_impl.extend(quote! { self. #index .parameters() });
                    }
                    else if i == 0 {
                        parameters_impl.extend(quote! { self. #index .parameters().into_iter() });
                    }
                    else {
                        parameters_impl.extend(quote! { .chain(self. #index .parameters()) });
                    }
                    set_training_impl.extend(quote! { self. #index .set_training(training); });
                } 
            }
            if len > 1 {
                parameters_impl.extend(quote! { .collect() });
            }
            let ident = &input.ident;
            let gen = quote! {
                impl Layer for #ident {
                    fn parameters(&self) -> Vec<ParameterD> {
                        #parameters_impl
                    }
                    fn set_training(&mut self, training: bool) {
                        #set_training_impl
                    }
                }   
            };
            BaseTokenStream::from(gen)
        }
        else {
            BaseTokenStream::from(default_impl)
        }
    }
    else {
        BaseTokenStream::from(default_impl)
    }
}



#[proc_macro_attribute]
pub fn impl_forward(attr: BaseTokenStream, item: BaseTokenStream) -> BaseTokenStream {
    let mut output = item.clone();
    let nested = syn::parse_macro_input!(attr as Vec<NestedMeta>);
    let input_dim = match nested.first().unwrap() {
        NestedMeta::Meta(Meta::Path(input_dim)) => input_dim, 
        _ => unimplemented!()
    };
    let output_dim = match nested.last().unwrap() {
        NestedMeta::Meta(Meta::Path(output_dim)) => output_dim,
        _ => unimplemented!()
    };
    
    let input: ItemStruct = syn::parse(item.clone()).unwrap();
    let fields = match &input.fields {
        Fields::Named(fields) => Some(&fields.named),
        Fields::Unnamed(fields) => Some(&fields.unnamed),
        Fields::Unit => None
    };
    let ident = &input.ident;
    let identity_impl = quote! {
        impl Forward<#output_dim> for #ident {
            type OutputDim = #output_dim;
            fn forward(&self, input: &Variable<#output_dim>) -> Variable<#output_dim> {
                input.clone()
            }
        }
    };
    if let Some(fields) = fields {
        if !fields.is_empty() {
            let forward_impl: TokenStream = fields.into_iter()
                .enumerate()
                .filter(|(_, field)| !is_autograph_skip(&field.attrs))
                .map(|(i, field)| {
                    if let Some(ident) = &field.ident {
                        quote! { .forward(&self. #ident) }
                    }
                    else {
                        let index = Index::from(i);
                        quote! { .forward(&self. #index) }
                    } 
                 })
                 .collect();
            let ident = &input.ident;
            let gen = quote! {
                impl Forward<#input_dim> for #ident {
                    type OutputDim = #output_dim;
                    fn forward(&self, input: &Variable<#input_dim>) -> Variable<#output_dim> {
                        input #forward_impl
                    }       
                }   
            };
            output.extend(BaseTokenStream::from(gen));
        }
    }
    else {
        output.extend(BaseTokenStream::from(identity_impl));
    }
    output
}

