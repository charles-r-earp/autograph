use proc_macro::TokenStream as BaseTokenStream;
use proc_macro2::TokenStream;
use syn::{
    DeriveInput,
    Data,
    Fields,
    Index,
    ItemStruct,
    AttributeArgs,
    Meta,
    NestedMeta
};
use quote::quote;

#[proc_macro_derive(Layer)]
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
    output
}

