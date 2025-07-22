defmodule AxiomAi.LocalModels.Registry do
  @moduledoc """
  Central registry for managing model configurations organized by categories.
  """

  alias AxiomAi.LocalModels.{Categories, Templates}

  @doc """
  Gets all model configurations.
  """
  @spec get_all_models() :: %{String.t() => map()}
  def get_all_models do
    Map.merge(get_builtin_models(), get_runtime_models())
  end

  @doc """
  Gets a specific model configuration by key.
  """
  @spec get_model(String.t()) :: {:ok, map()} | {:error, :not_found}
  def get_model(model_key) do
    case Map.get(get_all_models(), model_key) do
      nil -> {:error, :not_found}
      config -> {:ok, config}
    end
  end

  @doc """
  Lists all available model keys.
  """
  @spec list_all_models() :: [String.t()]
  def list_all_models do
    get_all_models()
    |> Map.keys()
    |> Enum.sort()
  end

  @doc """
  Lists models by category.
  """
  @spec list_models_by_category(atom()) :: [String.t()]
  def list_models_by_category(category) do
    get_all_models()
    |> Enum.filter(fn {_key, config} -> Map.get(config, :category) == category end)
    |> Enum.map(fn {key, _config} -> key end)
    |> Enum.sort()
  end

  @doc """
  Adds a new model configuration at runtime.
  """
  @spec add_model(String.t(), atom(), map()) :: :ok | {:error, term()}
  def add_model(model_key, category, config) do
    if Categories.valid_category?(category) do
      enhanced_config = Map.put(config, :category, category)
      :persistent_term.put({__MODULE__, model_key}, enhanced_config)
      :ok
    else
      {:error, {:invalid_category, category}}
    end
  end

  @doc """
  Removes a runtime model configuration.
  """
  @spec remove_model(String.t()) :: :ok
  def remove_model(model_key) do
    :persistent_term.erase({__MODULE__, model_key})
    :ok
  end

  # Private functions

  # Built-in model configurations
  defp get_builtin_models do
    # Cache templates to avoid redundant calls
    text_template = Templates.create_from_template(:python_interface_text)
    speech_template = Templates.create_from_template(:python_interface_speech)

    %{
      # Text Generation Models
      "qwen2.5-0.5b" => %{
        name: "Qwen2.5 0.5B",
        category: :text_generation,
        type: :python_interface,
        model_path: "Qwen/Qwen2.5-0.5B-Instruct",
        python_deps: text_template.python_deps,
        python_code: text_template.python_code,
        context_length: 32768,
        description: "Qwen2.5 0.5B - Small but capable model for general tasks"
      },
      "qwen2.5-1.5b" => %{
        name: "Qwen2.5 1.5B",
        category: :text_generation,
        type: :python_interface,
        model_path: "Qwen/Qwen2.5-1.5B-Instruct",
        python_deps: text_template.python_deps,
        python_code: text_template.python_code,
        context_length: 32768,
        description: "Qwen2.5 1.5B - Balance of performance and efficiency"
      },
      "qwen2.5-3b" => %{
        name: "Qwen2.5 3B",
        category: :text_generation,
        type: :python_interface,
        model_path: "Qwen/Qwen2.5-3B-Instruct",
        python_deps: text_template.python_deps,
        python_code: text_template.python_code,
        context_length: 32768,
        description: "Qwen2.5 3B - Good performance for most tasks"
      },

      # Code Generation Models
      "codellama-7b" => %{
        name: "Code Llama 7B",
        category: :code_generation,
        type: :http,
        endpoint: "http://localhost:8000",
        model: "codellama/CodeLlama-7b-Instruct-hf",
        api_format: :openai,
        description: "Meta's Code Llama 7B for code generation"
      },

      # HTTP Endpoint Models
      "ollama-qwen" => %{
        name: "Ollama Qwen",
        category: :http_endpoints,
        type: :http,
        endpoint: "http://localhost:11434",
        model: "qwen2.5:latest",
        api_format: :ollama,
        description: "Qwen model served via Ollama"
      },
      "vllm-qwen" => %{
        name: "vLLM Qwen",
        category: :http_endpoints,
        type: :http,
        endpoint: "http://localhost:8000",
        model: "Qwen/Qwen2.5-3B-Instruct",
        api_format: :openai,
        description: "Qwen model served via vLLM"
      },
      "llama3-8b" => %{
        name: "Llama 3 8B",
        category: :text_generation,
        type: :http,
        endpoint: "http://localhost:8000",
        model: "meta-llama/Meta-Llama-3-8B-Instruct",
        api_format: :openai,
        description: "Meta's Llama 3 8B model"
      },
      "mistral-7b" => %{
        name: "Mistral 7B",
        category: :text_generation,
        type: :http,
        endpoint: "http://localhost:8000",
        model: "mistralai/Mistral-7B-Instruct-v0.3",
        api_format: :openai,
        description: "Mistral's 7B instruction-tuned model"
      },

      # Speech Models
      "whisper-large-v3" => %{
        name: "Whisper Large v3",
        category: :speech,
        type: :python_interface,
        model_path: "openai/whisper-large-v3",
        python_deps: speech_template.python_deps,
        python_code: speech_template.python_code,
        context_length: 30,
        description: "OpenAI Whisper Large v3 - High-quality speech-to-text model"
      },
      "whisper-large-v3-turbo" => %{
        name: "Whisper Large v3 Turbo",
        category: :speech,
        type: :python_interface,
        model_path: "openai/whisper-large-v3-turbo",
        python_deps: speech_template.python_deps,
        python_code: speech_template.python_code,
        context_length: 30,
        description: "OpenAI Whisper Large v3 Turbo - Fast speech-to-text model"
      }
    }
  end

  # Runtime models stored in persistent_term
  defp get_runtime_models do
    :persistent_term.get()
    |> Enum.filter(fn
      {{module, _key}, _value} when module == __MODULE__ -> true
      _ -> false
    end)
    |> Enum.into(%{}, fn {{_module, key}, value} -> {key, value} end)
  rescue
    # If persistent_term.get() fails or returns unexpected format, return empty map
    _ -> %{}
  end
end
