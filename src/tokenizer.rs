use std::path::Path;

/// Convenience wrapper around HuggingFace tokenizers for Qwen3.5.
///
/// A thin codec wrapper. The model owns special-token semantics (BOS/EOS).
pub struct Qwen35Tokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl Qwen35Tokenizer {
    /// Load the tokenizer from a `tokenizer.json` file.
    pub fn new(tokenizer_path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
        Ok(Self { tokenizer })
    }

    /// Return a reference to the inner `tokenizers::Tokenizer`.
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer
            .encode(text, false)
            .expect("Failed to encode text")
            .get_ids()
            .to_vec()
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        self.tokenizer
            .decode(tokens, true)
            .expect("Failed to decode tokens")
    }

    /// Format a user message using Qwen3.5's chat template (ChatML format, same as Qwen3).
    pub fn apply_chat_template(&self, system_prompt: &str, user_message: &str) -> String {
        format!(
            "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        )
    }
}

impl std::ops::Deref for Qwen35Tokenizer {
    type Target = tokenizers::Tokenizer;

    fn deref(&self) -> &Self::Target {
        &self.tokenizer
    }
}
