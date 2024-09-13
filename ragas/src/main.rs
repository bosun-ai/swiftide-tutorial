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

    // Initialize the OpenAI client
    let openai = OpenAI::builder()
        .default_embed_model("text-embedding-3-small")
        .default_prompt_model("gpt-4o-mini")
        .build()?;

    // Initialize the Qdrant client
    let qdrant = Qdrant::builder()
        .vector_size(1536)
        .collection_name(COLLECTION_NAME)
        .build()?;

    let context = Context { openai, qdrant };

    // Delete the collection if it already exists
    force_delete_qdrant_collection(&context).await?;

    // Index the code
    index_all(&args.language, &args.path, &context).await?;

    // Either load the dataset from a file or use the questions provided
    // Then create the evaluation dataset to be used
    let dataset: EvaluationDataSet = if let Some(path) = args.dataset.file {
        std::fs::read_to_string(path)?.parse()?
    } else {
        args.dataset
            .questions
            .ok_or(anyhow::anyhow!("Expected questions"))?
            .into()
    };

    // Query the indexed dataset and return the evaluation
    let evaluation = query(dataset, args.record_ground_truth, &context).await?;

    // Write the evaluation to a json file so it can be used in the python notebook
    let json = evaluation.to_json().await;
    std::fs::write(args.output, json).context("Failed to write ragas.json")?;

    Ok(())
}

async fn index_all(language: &str, path: &PathBuf, context: &Context) -> Result<()> {
    tracing::info!(path=?path, language, "Indexing code");

    let language = SupportedLanguages::from_str(language)?;
    let mut extensions = language.file_extensions().to_owned();
    extensions.push("md");

    // Index all code and markdown files in the provided directory
    let (mut markdown, mut code) = Pipeline::from_loader(
        FileLoader::new(path).with_extensions(&extensions),
    )
    .split_by(|node| {
        // Any errors at this point we just pass to 'markdown'
        let Ok(node) = node else { return true };

        // On true we go 'markdown', on false we go 'code'.
        node.path.extension().map_or(true, |ext| ext == "md")
    });

    // For each feature that we want to test, enable them conditionally

    if cfg!(feature = "chunk") {
        code = code
            // Uses tree-sitter to extract best effort blocks of code. We still keep the minimum
            // fairly high and double the chunk size
            .then_chunk(ChunkCode::try_for_language_and_chunk_size(
                language,
                50..1024,
            )?);

        markdown = markdown.then_chunk(ChunkMarkdown::from_chunk_range(50..1024));
    }

    if cfg!(feature = "metadata") {
        code = code.then(MetadataQACode::new(context.openai.clone()));
        markdown = markdown.then(MetadataQAText::new(context.openai.clone()));
    }

    // Merge both pipelines and generate embeddings
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
    // Create a new evaluator with prepared questions, either from the input file or the provided
    // questions
    let ragas = evaluators::ragas::Ragas::from_prepared_questions(questions);

    // Run a query pipeline that answers all provided questions
    let pipeline = query::Pipeline::default()
        .evaluate_with(ragas.clone())
        .then_transform_query(GenerateSubquestions::from_client(context.openai.clone()))
        .then_transform_query(query_transformers::Embed::from_client(
            context.openai.clone(),
        ))
        .then_retrieve(context.qdrant.clone())
        .then_answer(Simple::from_client(context.openai.clone()));

    pipeline.query_all(ragas.questions().await).await?;

    // If the flag is set, record the answers as ground truth.
    // Ragas needs to know the correct answers to evaluate certain metrics.
    //
    // Can also be set manually or have RAGAS handle it. There are pros and cons to each.
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
