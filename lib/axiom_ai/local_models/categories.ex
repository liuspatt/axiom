defmodule AxiomAi.LocalModels.Categories do
  @moduledoc """
  Defines model categories for organizing different types of AI models.
  """

  @categories [
    :text_generation,    # General text generation models (Qwen, Llama, etc.)
    :code_generation,    # Code-specific models (CodeLlama, etc.)
    :vision_language,    # Multi-modal vision+language models
    :ocr,               # Optical Character Recognition models
    :embedding,         # Text embedding models
    :http_endpoints     # HTTP/API-based models (Ollama, vLLM, etc.)
  ]

  @doc """
  Returns all available model categories.
  """
  @spec list_all() :: [atom()]
  def list_all, do: @categories

  @doc """
  Gets the description for a category.
  """
  @spec get_description(atom()) :: String.t()
  def get_description(:text_generation), do: "General text generation and chat models"
  def get_description(:code_generation), do: "Code generation and programming assistance models"
  def get_description(:vision_language), do: "Multi-modal models that can process images and text"
  def get_description(:ocr), do: "Optical Character Recognition models for text extraction"
  def get_description(:embedding), do: "Text embedding models for semantic similarity"
  def get_description(:http_endpoints), do: "Models served via HTTP APIs (Ollama, vLLM, etc.)"
  def get_description(_), do: "Unknown category"

  @doc """
  Validates if a category exists.
  """
  @spec valid_category?(atom()) :: boolean()
  def valid_category?(category), do: category in @categories
end