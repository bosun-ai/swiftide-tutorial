use swiftide::{
    indexing::{
        loaders::FileLoader,
        transformers::{ChunkCode, ChunkMarkdown, Embed, MetadataQACode, MetadataQAText},
    },
    query::{
        self,
        answers::Simple,
        evaluators::{self, ragas::EvaluationDataSet},
        query_transformers::{self, GenerateSubquestions},
    },
};

use std::{path::PathBuf, str::FromStr};

use anyhow::{Context as _, Result};
use clap::Parser;
use swiftide::{
    indexing::Pipeline,
    integrations::{openai::OpenAI, qdrant::Qdrant, redis::Redis, treesitter::SupportedLanguages},
};

const COLLECTION_NAME: &str = "swiftide-ragas";

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    /// Language of the code to index
    language: String,

    #[arg(short, long, default_value = "./")]
    /// Path to the code to index
    path: PathBuf,

    #[command(flatten)]
    dataset: DatasetArg,

    #[arg(short, long, default_value = "false")]
    /// Records answers as ground truth
    record_ground_truth: bool,

    #[arg(short, long)]
    /// Output file to write the evaluation results to
    output: PathBuf,
}

#[derive(clap::Args, Debug, Clone)]
#[group(required = true, multiple = false)]
struct DatasetArg {
    /// Dataset json file to load questions and ground truths from
    #[arg(short, long, conflicts_with = "questions")]
    file: Option<PathBuf>,
    /// List of questions to use for evaluation
    questions: Option<Vec<String>>,
}

struct Context {
    openai: OpenAI,
    qdrant: Qdrant,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    let openai = OpenAI::builder()
        .default_embed_model("text-embedding-3-small")
        .default_prompt_model("gpt-4o-mini")
        .build()?;

    let qdrant = Qdrant::builder()
        .vector_size(1536)
        .collection_name(COLLECTION_NAME)
        .build()?;

    let context = Context { openai, qdrant };

    force_delete_qdrant_collection(&context).await?;

    index_all(&args.language, &args.path, &context).await?;

    let dataset: EvaluationDataSet = if let Some(path) = args.dataset.file {
        std::fs::read_to_string(path)?.parse()?
    } else {
        args.dataset
            .questions
            .ok_or(anyhow::anyhow!("Expected questions"))?
            .into()
    };
    let evaluation = query(dataset, args.record_ground_truth, &context).await?;

    let json = evaluation.to_json().await;
    std::fs::write(args.output, json).context("Failed to write ragas.json")?;

    Ok(())
}

async fn index_all(language: &str, path: &PathBuf, context: &Context) -> Result<()> {
    tracing::info!(path=?path, language, "Indexing code");

    let language = SupportedLanguages::from_str(language)?;
    let mut extensions = language.file_extensions().to_owned();
    extensions.push("md");

    let (mut markdown, mut code) =
        Pipeline::from_loader(FileLoader::new(path).with_extensions(&extensions))
            .with_concurrency(50)
            .split_by(|node| {
                // Any errors at this point we just pass to 'markdown'
                let Ok(node) = node else { return true };

                // On true we go 'markdown', on false we go 'code'.
                node.path.extension().map_or(true, |ext| ext == "md")
            });

    if cfg!(feature = "chunk") {
        code = code
            // Uses tree-sitter to extract best effort blocks of code. We still keep the minimum
            // fairly high and double the chunk size
            .then_chunk(ChunkCode::try_for_language_and_chunk_size(
                language,
                50..1024,
            )?);
    }

    if cfg!(feature = "metadata") {
        code = code.then(MetadataQACode::new(context.openai.clone()));
    }

    if cfg!(feature = "chunk") {
        markdown = markdown.then_chunk(ChunkMarkdown::from_chunk_range(50..1024));
        // Generate questions and answers and them to the metadata of the node
    }

    if cfg!(feature = "metadata") {
        markdown = markdown.then(MetadataQAText::new(context.openai.clone()));
    }

    code.merge(markdown)
        .then_in_batch(50, Embed::new(context.openai.clone()))
        .then_store_with(context.qdrant.clone())
        .run()
        .await
}

async fn query(
    questions: EvaluationDataSet,
    record_ground_truth: bool,
    context: &Context,
) -> Result<evaluators::ragas::Ragas> {
    let ragas = evaluators::ragas::Ragas::from_prepared_questions(questions);

    let pipeline = query::Pipeline::default()
        .evaluate_with(ragas.clone())
        .then_transform_query(GenerateSubquestions::from_client(context.openai.clone()))
        .then_transform_query(query_transformers::Embed::from_client(
            context.openai.clone(),
        ))
        .then_retrieve(context.qdrant.clone())
        .then_answer(Simple::from_client(context.openai.clone()));

    pipeline.query_all(ragas.questions().await).await?;

    if record_ground_truth {
        ragas.record_answers_as_ground_truth().await;
    }

    Ok(ragas)
}

async fn force_delete_qdrant_collection(context: &Context) -> Result<()> {
    let _ = context
        .qdrant
        .client()
        .delete_collection(COLLECTION_NAME)
        .await;

    Ok(())
}
