#![forbid(unsafe_code)]

/*!
Derive macros for [**autograph**](https://docs.rs/autograph).
*/

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
    fn iter(&self, method: Ident) -> TokenStream2 {
        match self {
            Self::Struct(layers) => {
                quote! {
                    ::std::iter::empty()
                    #(.chain(self.#layers.#method()))*
                }
            }
            Self::Enum(layers) => {
                quote! {
                    ::std::iter::empty()
                    #(
                        .chain((if let Self::#layers(layer) = self {
                            Some(layer.#method())
                        } else {
                            None
                        }).into_iter().flatten())
                    )*
                }
            }
        }
    }
    fn try_iter_mut(&self, method: Ident) -> TokenStream2 {
        match self {
            Self::Struct(layers) => {
                quote! {
                    Ok(
                        ::std::iter::empty()
                        #(.chain(self.#layers.#method()?))*
                    )
                }
            }
            Self::Enum(layers) => {
                let some_layer = quote! { Some(layer) };
                let none = quote! { None };
                let match_arms = layers.iter().enumerate().map(|(i, layer)| {
                    let fields =
                        (0..layers.len()).map(|u| if i == u { &some_layer } else { &none });
                    quote! {
                        Self::#layer(layer) => (#(#fields),*)
                    }
                });
                let iters = (0 .. layers.len()).map(|u| {
                    let index = Index::from(u);
                    quote! {
                        layers.#index.map(|layer| layer.#method()).transpose()?.into_iter().flatten()
                    }
                });
                quote! {
                    let layers = match self {
                        #(#match_arms),*
                    };
                    Ok(
                        ::std::iter::empty()
                        #(
                            .chain(#iters)
                        )*
                    )
                }
            }
        }
    }
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
}

enum Layer {
    Ident(Ident),
    Index(Index),
}

impl Layer {
    fn parse_field(field: &Field, index: usize) -> Option<Self> {
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

fn layer_impl(input: TokenStream2) -> Result<TokenStream2> {
    let input: DeriveInput = syn::parse2(input)?;
    let layers = Layers::parse(&input.data)?;
    let autograph = autograph_crate(&input.attrs)?;
    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let parameters = layers.iter(format_ident!("parameters"));
    let make_parameters_mut = layers.try_iter_mut(format_ident!("make_parameters_mut"));
    let set_training = layers.try_for_each(format_ident!("set_training"), quote! { training });
    let cast_mut = layers.try_for_each(format_ident!("cast_mut"), quote!(scalar_type));
    let to_device_mut = layers.try_for_each(format_ident!("to_device_mut"), quote!(device.clone()));
    let into_device = layers.try_map(format_ident!("into_device"), quote! { device.clone() });
    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics Layer for #ident #ty_generics #where_clause {
            fn parameters(&self) -> impl ::std::iter::Iterator<Item=#autograph::learn::neural_network::autograd::ParameterD> + '_ {
                #parameters
            }
            fn make_parameters_mut(&mut self) -> #autograph::anyhow::Result<impl ::std::iter::Iterator<Item= #autograph::learn::neural_network::autograd::ParameterViewMutD> + '_> {
                #make_parameters_mut
            }
            fn set_training(&mut self, training: bool) -> #autograph::anyhow::Result<()> {
                #set_training
            }
            fn cast_mut(&mut self, scalar_type: #autograph::krnl::scalar::ScalarType) -> #autograph::anyhow::Result<()> {
                #cast_mut
            }
            fn to_device_mut(&mut self, device: #autograph::krnl::device::Device) -> #autograph::anyhow::Result<()> {
                #to_device_mut
            }
            fn into_device(self, device: #autograph::krnl::device::Device) -> #autograph::anyhow::Result<Self>
            where Self: Sized {
                #into_device
            }
        }
    })
}

/// Derive for Layer.
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
            quote! {
                Ok(input #(.forward(&self.#layers)?)*)
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
