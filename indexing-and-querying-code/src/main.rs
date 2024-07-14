use std::{path::PathBuf, str::FromStr};

use anyhow::{Context as _, Result};
use clap::Parser;
use indoc::formatdoc;
use qdrant_client::qdrant::SearchPointsBuilder;
use swiftide::{
    indexing::Pipeline,
    integrations::{openai::OpenAI, qdrant::Qdrant, redis::Redis, treesitter::SupportedLanguages},
    loaders::FileLoader,
    transformers::{ChunkCode, ChunkMarkdown, Embed, MetadataQACode, MetadataQAText},
    EmbeddingModel, SimplePrompt,
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

    index_all(&args.language, &args.path, &openai, &qdrant).await?;

    let openai = OpenAI::builder()
        .default_embed_model("text-embedding-3-small")
        .default_prompt_model("gpt-4o")
        .build()?;

    let response = query(&openai, &args.query).await?;
    println!("{}", response);

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

async fn query(openai: &OpenAI, question: &str) -> Result<String> {
    let qdrant_url =
        std::env::var("QDRANT_URL").unwrap_or_else(|_err| "http://localhost:6334".to_string());

    // Build a manual client as Swiftide does not support querying yet
    let qdrant_client = qdrant_client::Qdrant::from_url(&qdrant_url).build()?;

    // Use Swiftide's openai to rewrite the prompt to a set of questions
    let transformed_question = openai.prompt(formatdoc!(r"
        Your job is to help a code query tool finding the right context.

        Given the following question:
        {question}

        Please think of 5 additional questions that can help answering the original question. The code is written in {lang}.

        Especially consider what might be relevant to answer the question, like dependencies, usage and structure of the code.

        Please respond with the original question and the additional questions only.

        ## Example

        - {question}
        - Additional question 1
        - Additional question 2
        - Additional question 3
        - Additional question 4
        - Additional question 5
        ", question = question, lang = "rust"
    ).into()).await?;

    // Embed the full rewrite for querying
    let embedded_question = openai
        .embed(vec![transformed_question.clone()])
        .await?
        .pop()
        .context("Expected embedding")?;

    // Search for matches
    let answer_context_points = qdrant_client
        .search_points(
            SearchPointsBuilder::new("swiftide-tutorial", embedded_question, 20).with_payload(true),
        )
        .await?;

    // Concatenate all the found chunks
    let answer_context = answer_context_points
        .result
        .into_iter()
        .map(|v| v.payload.get("content").unwrap().to_string())
        .collect::<Vec<_>>()
        .join("\n\n");

    // A prompt for answering the initial question with the found context
    let prompt = formatdoc!(
        r#"
        Answer the following question(s):
        {question}

        ## Constraints
        * Only answer based on the provided context below
        * Always reference files by the full path if it is relevant to the question
        * Answer the question fully and remember to be concise
        * Only answer based on the given context. If you cannot answer the question based on the
            context, say so.
        * Do not make up anything, especially code, that is not included in the provided context

        ## Context:
        {answer_context}
        "#,
    );

    let answer = openai.prompt(prompt.into()).await?;

    Ok(answer)
}
