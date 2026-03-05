use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("json error in {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("safetensors: {0}")]
    Safetensors(String),

    #[error("tensor not found: {0}")]
    TensorNotFound(String),

    #[error("shape mismatch: {0}")]
    Shape(String),

    #[error("unsupported: {0}")]
    Unsupported(String),

    #[error("config: {0}")]
    Config(String),

    #[error("cuda: {0}")]
    Cuda(String),
}

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io { path: path.into(), source }
    }

    pub fn json(path: impl Into<PathBuf>, source: serde_json::Error) -> Self {
        Error::Json { path: path.into(), source }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
