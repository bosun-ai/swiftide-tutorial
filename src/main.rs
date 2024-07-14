use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use swiftide::{
    indexing::Pipeline,
    integrations::openai::OpenAI,
    loaders::FileLoader,
    transformers::{ChunkMarkdown, MetadataQAText},
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
        .default_prompt_model("gpt-3.5")
        .build()?;

    index_markdown(&args.path, &openai).await?;

    Ok(())
}

async fn index_markdown(path: &PathBuf, openai: &OpenAI) -> Result<()> {
    tracing::info!(path=?path, "Indexing markdown");

    // Loads all markdown files into the pipeline
    Pipeline::from_loader(FileLoader::new(path).with_extensions(&[".md"]))
        // The range ensures that chunks smaller than 50 characters are dropped, with a maximum size up to 1024 characters
        .then_chunk(ChunkMarkdown::from_chunk_range(50..1024))
        // Generate questions and answers and them to the metadata of the node
        .then(MetadataQAText::new(openai.clone()))
        .run()
        .await
}
