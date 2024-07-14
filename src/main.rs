use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use swiftide::{indexing::Pipeline, loaders::FileLoader, transformers::ChunkMarkdown};

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

    index_markdown(&args.path).await?;

    Ok(())
}

async fn index_markdown(path: &PathBuf) -> Result<()> {
    tracing::info!(path=?path, "Indexing markdown");

    // Loads all markdown files into the pipeline
    Pipeline::from_loader(FileLoader::new(path).with_extensions(&[".md"]))
        .then_chunk(ChunkMarkdown::from_chunk_range(50..1024))
        .run()
        .await
}
