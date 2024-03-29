//! Derive macros for [autograph](https://docs.rs/autograph).
#![forbid(unsafe_code)]

use derive_syn_parse::Parse;
use proc_macro::TokenStream;
use proc_macro2::{Span as Span2, TokenStream as TokenStream2};
use quote::{format_ident, quote, ToTokens};
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
            if attr
                .path
                .get_ident()
                .map_or(false, |path| path == "autograph")
            {
                let args = syn::parse2::<AutographArgs>(attr.tokens.clone())?;
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
        if attr
            .path
            .get_ident()
            .map_or(false, |path| path == "autograph")
        {
            let args = syn::parse2::<AutographArgs>(attr.tokens.clone())?;
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

fn autograph_skip(attrs: &[Attribute]) -> Result<bool> {
    for attr in attrs {
        if attr
            .path
            .get_ident()
            .map_or(false, |path| path == "autograph")
        {
            let args = syn::parse2::<AutographArgs>(attr.tokens.clone())?;
            for arg in args.args {
                if arg.forward_args.is_some() {
                    continue;
                }
                if let Some(ident) = arg.ident.as_ref() {
                    if ident == "skip" {
                        return Ok(true);
                    } else {
                        return Err(Error::new(
                            ident.span(),
                            "expected `crate`, `forward` or  `skip`",
                        ));
                    }
                }
            }
        }
    }
    Ok(false)
}

fn autograph_skip_field(attrs: &[Attribute]) -> Result<bool> {
    for attr in attrs {
        if attr
            .path
            .get_ident()
            .map_or(false, |path| path == "autograph")
        {
            let ident: Ident = attr.parse_args()?;
            if ident == "skip" {
                return Ok(true);
            }
            return Err(Error::new(ident.span(), "expected `skip`"));
        }
    }
    Ok(false)
}

enum Layers {
    Struct(Vec<Layer>),
    Enum(Vec<Layer>),
}

impl Layers {
    fn parse(input: &DeriveInput) -> Result<Self> {
        if autograph_skip(&input.attrs)? {
            return Ok(Self::Struct(Vec::new()));
        }
        match &input.data {
            Data::Struct(data) => {
                let mut layers = Vec::with_capacity(data.fields.len());
                for (index, field) in data.fields.iter().enumerate() {
                    layers.extend(Layer::parse_field(field, index)?);
                }
                Ok(Self::Struct(layers))
            }
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
    fn try_for_each(&self, method: Ident, arg: TokenStream2) -> TokenStream2 {
        match self {
            Self::Struct(layers) => {
                quote! {
                    #(self.#layers.#method(#arg)?;)*
                    Ok(())
                }
            }
            Self::Enum(layers) => {
                quote! {
                    match self {
                        #(
                            Self::#layers(layer) => layer.#method(#arg),
                        )*
                    }
                }
            }
        }
    }
    /*
    fn try_map(&self, method: Ident, arg: TokenStream2) -> TokenStream2 {
        match self {
            Self::Struct(layers) => {
                quote! {
                    Ok(Self {
                        #(
                            #layers: self.#layers.#method(#arg)?,
                        )*
                    })
                }
            }
            Self::Enum(layers) => {
                quote! {
                    match self {
                        #(
                            Self::#layers(layer) => Ok(Self::#layers(layer.#method(#arg)?)),
                        )*
                    }
                }
            }
        }
    }
    */
}

enum Layer {
    Ident(Ident),
    Index(Index),
}

impl Layer {
    fn parse_field(field: &Field, index: usize) -> Result<Option<Self>> {
        if autograph_skip_field(&field.attrs)? {
            Ok(None)
        } else if let Some(ident) = field.ident.clone() {
            Ok(Some(Self::Ident(ident)))
        } else {
            Ok(Some(Self::Index(index.into())))
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

fn layer_impl(input: TokenStream2) -> Result<TokenStream2> {
    let input: DeriveInput = syn::parse2(input)?;
    let autograph = autograph_crate(&input.attrs)?;
    let layers = Layers::parse(&input)?;
    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let try_for_each_parameter =
        layers.try_for_each(format_ident!("try_for_each_parameter"), quote! { &mut f });
    let try_for_each_parameter_view_mut = layers.try_for_each(
        format_ident!("try_for_each_parameter_view_mut"),
        quote! { &mut f },
    );
    let cast_mut = layers.try_for_each(format_ident!("cast_mut"), quote!(scalar_type));
    let to_device_mut = layers.try_for_each(format_ident!("to_device_mut"), quote!(device.clone()));
    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics Layer for #ident #ty_generics #where_clause {
            fn try_for_each_parameter<F, E>(&self, mut f: F) -> #autograph::anyhow::Result<(), E>
                where
                    F: FnMut(#autograph::learn::neural_network::autograd::ParameterD) -> #autograph::anyhow::Result<(), E>,
                {
                #try_for_each_parameter
            }
            fn try_for_each_parameter_view_mut<F, E>(&mut self, mut f: F) -> #autograph::anyhow::Result<()>
               where
                   F: FnMut(#autograph::learn::neural_network::autograd::ParameterViewMutD) -> #autograph::anyhow::Result<(), E>,
                   #autograph::anyhow::Error: From<E>,
            {
                #try_for_each_parameter_view_mut
            }
            fn cast_mut(&mut self, scalar_type: #autograph::krnl::scalar::ScalarType) -> #autograph::anyhow::Result<()> {
                #cast_mut
            }
            fn to_device_mut(&mut self, device: #autograph::krnl::device::Device) -> #autograph::anyhow::Result<()> {
                #to_device_mut
            }
        }
    })
}

/// Derive for Layer.
#[proc_macro_derive(Layer, attributes(autograph))]
pub fn layer(input: TokenStream) -> TokenStream {
    match layer_impl(input.into()) {
        Ok(output) => output.into(),
        Err(err) => err.into_compile_error().into(),
    }
}

fn forward_impl(input: TokenStream2) -> Result<TokenStream2> {
    let input: DeriveInput = syn::parse2(input)?;
    let autograph = autograph_crate(&input.attrs)?;
    let forward_args = ForwardArgs::from_attributes(&input.attrs)?;
    let layers = Layers::parse(&input)?;
    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let forward = match layers {
        Layers::Struct(layers) => {
            if let Some((last, layers)) = layers.split_last() {
                quote! {
                    input #(.forward(&self.#layers)?)*
                        .forward(&self.#last)
                }
            } else {
                quote! {
                    Ok(input)
                }
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
#[proc_macro_derive(Forward, attributes(autograph, layer))]
pub fn forward(input: TokenStream) -> TokenStream {
    match forward_impl(input.into()) {
        Ok(output) => output.into(),
        Err(err) => err.into_compile_error().into(),
    }
}
