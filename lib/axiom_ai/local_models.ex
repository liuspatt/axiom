defmodule AxiomAi.LocalModels do
  @moduledoc """
  Pre-defined configurations for local AI models including Qwen and other popular models.
  Uses Pythonx for Python ML environment management.
  """

  @doc """
  Returns the predefined model configurations.
  """
  @spec get_predefined_models() :: %{atom() => map()}
  def get_predefined_models do
    %{
      # Qwen Models (Latest small versions)
      "qwen2.5-0.5b" => %{
        name: "Qwen2.5 0.5B",
        type: :pythonx,
        model_path: "Qwen/Qwen2.5-0.5B-Instruct",
        python_deps: qwen_dependencies(),
        python_code: qwen_inference_code(),
        context_length: 32768,
        description: "Qwen2.5 0.5B - Small but capable model for general tasks"
      },
      "qwen2.5-1.5b" => %{
        name: "Qwen2.5 1.5B",
        type: :pythonx,
        model_path: "Qwen/Qwen2.5-1.5B-Instruct",
        python_deps: qwen_dependencies(),
        python_code: qwen_inference_code(),
        context_length: 32768,
        description: "Qwen2.5 1.5B - Balance of performance and efficiency"
      },
      "qwen2.5-3b" => %{
        name: "Qwen2.5 3B",
        type: :pythonx,
        model_path: "Qwen/Qwen2.5-3B-Instruct",
        python_deps: qwen_dependencies(),
        python_code: qwen_inference_code(),
        context_length: 32768,
        description: "Qwen2.5 3B - Good performance for most tasks"
      },

      # OpenAI-compatible endpoints
      "ollama-qwen" => %{
        name: "Ollama Qwen",
        type: :http,
        endpoint: "http://localhost:11434",
        model: "qwen2.5:latest",
        api_format: :ollama,
        description: "Qwen model served via Ollama"
      },
      "vllm-qwen" => %{
        name: "vLLM Qwen",
        type: :http,
        endpoint: "http://localhost:8000",
        model: "Qwen/Qwen2.5-3B-Instruct",
        api_format: :openai,
        description: "Qwen model served via vLLM"
      },

      # Other popular local models
      "llama3-8b" => %{
        name: "Llama 3 8B",
        type: :http,
        endpoint: "http://localhost:8000",
        model: "meta-llama/Meta-Llama-3-8B-Instruct",
        api_format: :openai,
        description: "Meta's Llama 3 8B model"
      },
      "mistral-7b" => %{
        name: "Mistral 7B",
        type: :http,
        endpoint: "http://localhost:8000",
        model: "mistralai/Mistral-7B-Instruct-v0.3",
        api_format: :openai,
        description: "Mistral's 7B instruction-tuned model"
      },
      "codellama-7b" => %{
        name: "Code Llama 7B",
        type: :http,
        endpoint: "http://localhost:8000",
        model: "codellama/CodeLlama-7b-Instruct-hf",
        api_format: :openai,
        description: "Meta's Code Llama 7B for code generation"
      }
    }
  end

  @doc """
  Gets a specific model configuration by key.
  """
  @spec get_model_config(String.t()) :: {:ok, map()} | {:error, :not_found}
  def get_model_config(model_key) do
    case Map.get(get_predefined_models(), model_key) do
      nil -> {:error, :not_found}
      config -> {:ok, config}
    end
  end

  @doc """
  Lists all available predefined model keys.
  """
  @spec list_models() :: [String.t()]
  def list_models do
    get_predefined_models()
    |> Map.keys()
    |> Enum.sort()
  end

  @doc """
  Creates a custom model configuration.
  """
  @spec create_custom_config(map()) :: map()
  def create_custom_config(config) do
    default_config = %{
      name: "Custom Model",
      type: :http,
      endpoint: "http://localhost:8000",
      api_format: :openai,
      description: "Custom AI model configuration"
    }

    Map.merge(default_config, config)
  end

  # Private function to define Python dependencies for Qwen models
  defp qwen_dependencies do
    """
    [project]
    name = "qwen_inference"
    version = "0.1.0"
    requires-python = "==3.10.*"
    dependencies = [
      "torch >= 2.0.0",
      "transformers >= 4.35.0", 
      "accelerate >= 0.20.0",
      "tokenizers >= 0.14.0",
      "numpy >= 1.24.0"
    ]
    """
  end

  # Private function to define Python inference code for Qwen models
  defp qwen_inference_code do
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Global variables to cache model and tokenizer
    _model = None
    _tokenizer = None
    _current_model_path = None

    def load_model(model_path):
        global _model, _tokenizer, _current_model_path
        
        if _current_model_path != model_path:
            _tokenizer = AutoTokenizer.from_pretrained(model_path)
            _model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            _current_model_path = model_path
        
        return _tokenizer, _model

    def generate_response(model_path, prompt, max_tokens=1024, temperature=0.7):
        tokenizer, model = load_model(model_path)
        
        messages = [{"role": "user", "content": prompt}]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    # Export the function for Elixir to call
    generate_response
    """
  end
end
