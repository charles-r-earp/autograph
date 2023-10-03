/*!
# Usage
You can derive Layer and Forward for structs and enums:
```no_run
use autograph::{
    anyhow::Result,
    learn::neural_network::{
        autograd::{Variable4, Variable2},
        layer::{Layer, Forward, Flatten, Conv2, Relu, MaxPool2, Dense},
    },
};

// Layer and Forward can be derived for structs composed of layers.
#[derive(Layer, Forward)]
#[autograph(forward(Variable4, Output=Variable2))]
struct Network {
    #[layer]
    conv: Conv2<Relu>,
    #[layer]
    flatten: Flatten,
    #[layer]
    dense: Dense,
}

// Can also be applied to enums.
#[derive(Layer, Forward)]
#[autograph(forward(Variable4, Output=Variable4))]
enum Dynamic {
    Conv(Conv2),
    Pool(MaxPool2),
}
*/

use derive_syn_parse::Parse;
use proc_macro::TokenStream;
use proc_macro2::{Span as Span2, TokenStream as TokenStream2};
use quote::{quote, ToTokens};
use syn::{
    parse_quote,
    punctuated::Punctuated,
    token::{Comma, Eq as SynEq, Paren},
    Attribute, Data, DeriveInput, Error, Field, Fields, Ident, Index, Path, Result, Type, Variant,
};

#[derive(Parse)]
struct AutographArgs {
    #[allow(unused)]
    #[paren]
    paren: Paren,
    #[inside(paren)]
    #[call(Punctuated::parse_terminated)]
    args: Punctuated<AutographArg, Comma>,
}

#[derive(Parse)]
struct AutographArg {
    #[allow(unused)]
    crate_token: Option<syn::token::Crate>,
    #[allow(unused)]
    #[parse_if(crate_token.is_some())]
    eq: Option<SynEq>,
    #[parse_if(crate_token.is_some())]
    autograph_crate: Option<Path>,
    #[allow(unused)]
    #[parse_if(crate_token.is_none())]
    ident: Option<Ident>,
    #[parse_if(ident.as_ref().map(|x| x == "forward").unwrap_or_default())]
    forward_args: Option<ForwardArgs>,
}

#[derive(Parse)]
struct ForwardArgs {
    #[allow(unused)]
    #[paren]
    paren: Paren,
    #[inside(paren)]
    input: Type,
    #[allow(unused)]
    #[inside(paren)]
    comma: Comma,
    #[inside(paren)]
    output_ident: Ident,
    #[allow(unused)]
    #[inside(paren)]
    eq: SynEq,
    #[inside(paren)]
    output: Type,
}

impl ForwardArgs {
    fn from_attributes(attrs: &[Attribute]) -> Result<Vec<Self>> {
        let mut forward_args = Vec::new();
        for attr in attrs {
            if attr.path.to_token_stream().to_string() == "autograph" {
                let args = syn::parse2::<AutographArgs>(attr.tokens.to_token_stream())?;
                for arg in args.args {
                    if let Some(args) = arg.forward_args {
                        if args.output_ident != "Output" {
                            return Err(Error::new_spanned(
                                &args.output_ident,
                                "expected `Output`",
                            ));
                        }
                        forward_args.push(args);
                    }
                }
            }
        }
        Ok(forward_args)
    }
}

fn autograph_crate(attrs: &[Attribute]) -> Result<Path> {
    for attr in attrs {
        if attr.path.to_token_stream().to_string() == "autograph" {
            let args = syn::parse2::<AutographArgs>(attr.tokens.to_token_stream())?;
            for arg in args.args {
                if let Some(autograph_crate) = arg.autograph_crate {
                    return Ok(autograph_crate);
                }
            }
        }
    }
    Ok(parse_quote! {
        ::autograph
    })
}

enum Layers {
    Struct(Vec<Layer>),
    Enum(Vec<Layer>),
}

impl Layers {
    fn parse(data: &Data) -> Result<Self> {
        match data {
            Data::Struct(data) => Ok(Self::Struct(
                data.fields
                    .iter()
                    .enumerate()
                    .filter_map(|(index, field)| Layer::parse_field(field, index))
                    .collect(),
            )),
            Data::Enum(data) => {
                let mut layers = Vec::with_capacity(data.variants.len());
                for variant in data.variants.iter() {
                    layers.push(Layer::parse_variant(variant)?);
                }
                Ok(Self::Enum(layers))
            }
            Data::Union(_) => Err(Error::new(Span2::call_site(), "unions not supported")),
        }
    }
}

enum Layer {
    Ident(Ident),
    Index(Index),
}

impl Layer {
    fn parse_field(field: &Field, index: usize) -> Option<Self> {
        if !field.attrs.iter().any(layer_attribute) {
            return None;
        }
        if let Some(ident) = field.ident.clone() {
            Some(Self::Ident(ident))
        } else {
            Some(Self::Index(index.into()))
        }
    }
    fn parse_variant(variant: &Variant) -> Result<Self> {
        if let Fields::Unnamed(fields) = &variant.fields {
            if fields.unnamed.len() != 1 {
                return Err(Error::new_spanned(
                    fields,
                    "expected variant with 1 unnamed field",
                ));
            }
        } else {
            return Err(Error::new_spanned(
                &variant.fields,
                "expected variant with 1 unnamed field",
            ));
        };
        Ok(Self::Ident(variant.ident.clone()))
    }
}

impl ToTokens for Layer {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        match self {
            Self::Ident(ident) => ident.to_tokens(tokens),
            Self::Index(index) => index.to_tokens(tokens),
        }
    }
}

fn layer_attribute(attr: &Attribute) -> bool {
    attr.to_token_stream().to_string() == "#[layer]"
}

fn layer_impl(input: TokenStream2) -> Result<TokenStream2> {
    let input: DeriveInput = syn::parse2(input)?;
    let layers = Layers::parse(&input.data)?;
    let autograph = autograph_crate(&input.attrs)?;
    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let set_training = match &layers {
        Layers::Struct(layers) => layers
            .iter()
            .flat_map(|layer| {
                quote! {
                    self.#layer.set_training(training)?;
                }
            })
            .chain(quote! {
                Ok(())
            })
            .collect(),
        Layers::Enum(layers) => {
            let set_training = layers.iter().flat_map(|layer| {
                quote! {
                    Self::#layer(layer) => layer.set_training(training),
                }
            });
            quote! {
                match self {
                    #(#set_training)*
                }
            }
        }
    };
    let parameters_mut: TokenStream2 = match &layers {
        Layers::Struct(layers) => {
            let parameters_mut = layers.iter().map(|layer| {
                quote! {
                    self.#layer.parameters_mut()?
                }
            });
            quote! {
                Ok(
                    std::iter::empty()
                    #(.chain(#parameters_mut))*
                    .collect()
                )
            }
        }
        Layers::Enum(layers) => {
            let parameters_mut = layers.iter().flat_map(|layer| {
                quote! {
                    Self::#layer(layer) => layer.parameters_mut(),
                }
            });
            quote! {
                match self {
                    #(#parameters_mut)*
                }
            }
        }
    };
    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics Layer for #ident #ty_generics #where_clause {
            fn set_training(&mut self, training: bool) -> #autograph::anyhow::Result<()> {
                #set_training
            }
            fn parameters_mut(&mut self) -> #autograph::anyhow::Result<Vec<#autograph::learn::neural_network::autograd::ParameterViewMutD>> {
                #parameters_mut
            }
        }
    })
}

/// Derive for Layer.
///
/// See [`autograph_derive`](crate).
#[proc_macro_derive(Layer, attributes(autograph, layer))]
pub fn layer(input: TokenStream) -> TokenStream {
    match layer_impl(input.into()) {
        Ok(output) => output.into(),
        Err(err) => err.into_compile_error().into(),
    }
}

fn forward_impl(input: TokenStream2) -> Result<TokenStream2> {
    let input: DeriveInput = syn::parse2(input)?;
    let layers = Layers::parse(&input.data)?;
    let autograph = autograph_crate(&input.attrs)?;
    let forward_args = ForwardArgs::from_attributes(&input.attrs)?;
    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let forward = match layers {
        Layers::Struct(layers) => {
            let forward = layers.iter().map(|layer| {
                quote! {
                    .forward(&self.#layer)?
                }
            });
            quote! {
                Ok(input #(#forward)*)
            }
        }
        Layers::Enum(layers) => {
            let forward = layers.iter().flat_map(|layer| {
                quote! {
                    Self::#layer(layer) => layer.forward(input),
                }
            });
            quote! {
                match self {
                    #(#forward)*
                }
            }
        }
    };
    Ok(forward_args
        .into_iter()
        .flat_map(|forward_args| {
            let ForwardArgs { input, output, .. } = forward_args;
            quote! {
                #[automatically_derived]
                impl #impl_generics Forward<#input> for #ident #ty_generics #where_clause {
                    type Output = #output;
                    fn forward(&self, input: #input) -> #autograph::anyhow::Result<#output> {
                        #forward
                    }
                }
            }
        })
        .collect())
}

/// Derive for Forward.
///
/// See [`autograph_derive`](crate).
#[proc_macro_derive(Forward, attributes(autograph, layer))]
pub fn forward(input: TokenStream) -> TokenStream {
    match forward_impl(input.into()) {
        Ok(output) => output.into(),
        Err(err) => err.into_compile_error().into(),
    }
}
