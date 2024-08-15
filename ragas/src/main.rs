use swiftide::{
    indexing::{
        loaders::FileLoader,
        transformers::{ChunkCode, ChunkMarkdown, Embed, MetadataQACode, MetadataQAText},
    },
    query::{
        self,
        answers::Simple,
        evaluators,
        query_transformers::{self, GenerateSubquestions},
        states, Query,
    },
};

use std::{path::PathBuf, str::FromStr};

use anyhow::{Context as _, Result};
use clap::Parser;
use indoc::formatdoc;
use qdrant_client::qdrant::SearchPointsBuilder;
use swiftide::{
    indexing::Pipeline,
    integrations::{openai::OpenAI, qdrant::Qdrant, redis::Redis, treesitter::SupportedLanguages},
};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    language: String,

    #[arg(short, long, default_value = "./")]
    path: PathBuf,

    queries: Vec<String>,
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
        .collection_name("swiftide-ragas")
        .build()?;

    index_all(&args.language, &args.path, &openai, &qdrant).await?;

    let openai = OpenAI::builder()
        .default_embed_model("text-embedding-3-small")
        .default_prompt_model("gpt-4o")
        .build()?;

    let response = query(&openai, &qdrant, &args.queries).await?;
    println!(
        "{}",
        response
            .into_iter()
            .map(|q| q.answer().to_string())
            .collect::<Vec<_>>()
            .join("\n")
    );

    Ok(())
}

async fn index_all(language: &str, path: &PathBuf, openai: &OpenAI, qdrant: &Qdrant) -> Result<()> {
    tracing::info!(path=?path, language, "Indexing code");

    let language = SupportedLanguages::from_str(language)?;
    let mut extensions = language.file_extensions().to_owned();
    extensions.push("md");

    let (mut markdown, mut code) =
        Pipeline::from_loader(FileLoader::new(path).with_extensions(&extensions))
            .with_concurrency(50)
            .filter_cached(Redis::try_from_url(
                "redis://localhost:6379",
                "swiftide-tutorial",
            )?)
            .split_by(|node| {
                // Any errors at this point we just pass to 'markdown'
                let Ok(node) = node else { return true };

                // On true we go 'markdown', on false we go 'code'.
                node.path.extension().map_or(true, |ext| ext == "md")
            });

    code = code
        // Uses tree-sitter to extract best effort blocks of code. We still keep the minimum
        // fairly high and double the chunk size
        .then_chunk(ChunkCode::try_for_language_and_chunk_size(
            language,
            50..1024,
        )?)
        .then(MetadataQACode::new(openai.clone()));

    markdown = markdown
        .then_chunk(ChunkMarkdown::from_chunk_range(50..1024))
        // Generate questions and answers and them to the metadata of the node
        .then(MetadataQAText::new(openai.clone()));

    code.merge(markdown)
        .then_in_batch(50, Embed::new(openai.clone()))
        .then_store_with(qdrant.clone())
        .run()
        .await
}

async fn query(
    openai: &OpenAI,
    qdrant: &Qdrant,
    questions: &[String],
) -> Result<Vec<Query<states::Answered>>> {
    let ragas = evaluators::ragas::Ragas::from_prepared_questions(questions);

    let pipeline = query::Pipeline::default()
        .evaluate_with(ragas.clone())
        .then_transform_query(GenerateSubquestions::from_client(openai.clone()))
        .then_transform_query(query_transformers::Embed::from_client(openai.clone()))
        .then_retrieve(qdrant.clone())
        .then_answer(Simple::from_client(openai.clone()));

    let answers = pipeline.query_all(ragas.questions().await).await;

    let json = ragas.to_json().await;
    std::fs::write("ragas.json", json).context("Failed to write ragas.json")?;

    answers
}
