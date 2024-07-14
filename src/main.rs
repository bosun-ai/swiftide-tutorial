use std::{path::PathBuf, str::FromStr};

use anyhow::Result;
use clap::Parser;
use swiftide::{
    indexing::Pipeline,
    integrations::{openai::OpenAI, qdrant::Qdrant, treesitter::SupportedLanguages},
    loaders::FileLoader,
    transformers::{ChunkCode, ChunkMarkdown, Embed, MetadataQACode, MetadataQAText},
};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    language: String,

    #[arg(short, long, default_value = "./")]
    path: PathBuf,

    query: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    let openai = OpenAI::builder()
        .default_embed_model("text-embedding-3-small")
        .default_prompt_model("gpt-3.5-turbo")
        .build()?;

    let qdrant = Qdrant::builder()
        .vector_size(1536)
        .collection_name("swiftide-tutorial")
        .build()?;

    index_markdown(&args.path, &openai, &qdrant).await?;
    index_code(&args.language, &args.path, &openai, &qdrant).await?;

    Ok(())
}

async fn index_markdown(path: &PathBuf, openai: &OpenAI, qdrant: &Qdrant) -> Result<()> {
    tracing::info!(path=?path, "Indexing markdown");

    // Loads all markdown files into the pipeline
    Pipeline::from_loader(FileLoader::new(path).with_extensions(&["md"]))
        // The range ensures that chunks smaller than 50 characters are dropped, with a maximum size up to 1024 characters
        .then_chunk(ChunkMarkdown::from_chunk_range(50..1024))
        // Generate questions and answers and them to the metadata of the node
        .then(MetadataQAText::new(openai.clone()))
        // Embed chunks in batches of 100. By default the metadata in the node is included in the
        // embedding
        .then_in_batch(100, Embed::new(openai.clone()))
        // Finally store the embeddings into Qdrant
        .then_store_with(qdrant.clone())
        .run()
        .await
}

async fn index_code(
    language: &str,
    path: &PathBuf,
    openai: &OpenAI,
    qdrant: &Qdrant,
) -> Result<()> {
    tracing::info!(path=?path, language, "Indexing code");

    // Parse the language to a supported one by the tree-sitter integration. Clap could also do
    // this. It's also possible to directly pass the language as string to the transformer, but we
    // save some time by getting the extensions.
    let language = SupportedLanguages::from_str(language)?;

    Pipeline::from_loader(FileLoader::new(path).with_extensions(language.file_extensions()))
        // Uses tree-sitter to extract best effort blocks of code. We still keep the minimum
        // fairly high and double the chunk size
        .then_chunk(ChunkCode::try_for_language_and_chunk_size(
            language,
            50..1024,
        )?)
        .then(MetadataQACode::new(openai.clone()))
        .then_in_batch(100, Embed::new(openai.clone()))
        .then_store_with(qdrant.clone())
        .run()
        .await
}
