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
    %{
      # Text Generation Models
      "qwen2.5-0.5b" => %{
        name: "Qwen2.5 0.5B",
        category: :text_generation,
        type: :pythonx,
        model_path: "Qwen/Qwen2.5-0.5B-Instruct",
        python_deps: Templates.create_from_template(:pythonx_text).python_deps,
        python_code: Templates.create_from_template(:pythonx_text).python_code,
        context_length: 32768,
        description: "Qwen2.5 0.5B - Small but capable model for general tasks"
      },
      "qwen2.5-1.5b" => %{
        name: "Qwen2.5 1.5B",
        category: :text_generation,
        type: :pythonx,
        model_path: "Qwen/Qwen2.5-1.5B-Instruct",
        python_deps: Templates.create_from_template(:pythonx_text).python_deps,
        python_code: Templates.create_from_template(:pythonx_text).python_code,
        context_length: 32768,
        description: "Qwen2.5 1.5B - Balance of performance and efficiency"
      },
      "qwen2.5-3b" => %{
        name: "Qwen2.5 3B",
        category: :text_generation,
        type: :pythonx,
        model_path: "Qwen/Qwen2.5-3B-Instruct",
        python_deps: Templates.create_from_template(:pythonx_text).python_deps,
        python_code: Templates.create_from_template(:pythonx_text).python_code,
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

      # OCR Models
      "nanonets-ocr-s" => %{
        name: "Nanonets OCR Small",
        category: :ocr,
        type: :pythonx,
        model_path: "nanonets/Nanonets-OCR-s",
        python_deps: ocr_dependencies(),
        python_code: ocr_inference_code(),
        context_length: 8192,
        description: "Nanonets OCR Small - Optimized for optical character recognition tasks"
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

  # OCR-specific dependencies and code (keeping the existing implementation)
  defp ocr_dependencies do
    """
    [project]
    name = "ocr_inference"
    version = "0.1.0"
    requires-python = "==3.10.*"
    dependencies = [
      "torch >= 2.0.0",
      "torchvision >= 0.15.0",
      "transformers >= 4.35.0", 
      "accelerate >= 0.20.0",
      "tokenizers >= 0.14.0",
      "numpy >= 1.24.0",
      "pillow >= 9.0.0",
      "PyMuPDF >= 1.23.0"
    ]
    """
  end

  defp ocr_inference_code do
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
    from PIL import Image
    import base64
    import io
    import fitz  # PyMuPDF
    import os

    # Global variables to cache model and tokenizer
    _model = None
    _tokenizer = None
    _processor = None
    _current_model_path = None

    def load_model(model_path):
        global _model, _tokenizer, _processor, _current_model_path
        
        if _current_model_path != model_path:
            try:
                _tokenizer = AutoTokenizer.from_pretrained(model_path)
                _processor = AutoProcessor.from_pretrained(model_path)
                _model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,  # Use float32 for better stability
                    device_map="cpu",          # Force CPU to avoid GPU memory issues
                    low_cpu_mem_usage=True     # Optimize memory usage
                )
                _current_model_path = model_path
            except Exception as e:
                print(f"Error loading model: {e}")
                raise e
        
        return _tokenizer, _processor, _model

    def process_image(image_data):
        if isinstance(image_data, str):
            # Assume base64 encoded image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            # Assume PIL Image or file path
            if isinstance(image_data, str):
                image = Image.open(image_data)
            else:
                image = image_data
        
        return image.convert('RGB')

    def process_pdf(pdf_path, page_number=0):
        \"\"\"Convert PDF page to PIL Image\"\"\"
        try:
            doc = fitz.open(pdf_path)
            if page_number >= len(doc):
                page_number = 0  # Default to first page if page number is out of range
            
            page = doc.load_page(page_number)
            # Render page to pixmap with good resolution
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            doc.close()
            
            return image.convert('RGB')
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise e

    def generate_response(model_path, prompt, image_path=None, max_tokens=512, temperature=0.7):
        try:
            tokenizer, processor, model = load_model(model_path)
            
            # Check if prompt contains an image path (format: "image_path|actual_prompt")
            if "|" in prompt and image_path is None:
                parts = prompt.split("|", 1)
                if len(parts) == 2:
                    image_path = parts[0].strip()
                    prompt = parts[1].strip()
            
            if image_path and image_path != "":
                try:
                    # Check if it's a PDF file
                    if image_path.lower().endswith('.pdf'):
                        image = process_pdf(image_path, 0)  # Process first page
                    else:
                        image = Image.open(image_path).convert('RGB')
                    
                    # Resize image to reduce memory usage
                    image = image.resize((512, 512), Image.LANCZOS)
                    inputs = processor(text=prompt, images=image, return_tensors="pt")
                    use_image = True
                except Exception as e:
                    # If image/PDF loading fails, fall back to text-only
                    print(f"Warning: Failed to load file {image_path}: {e}")
                    inputs = tokenizer(prompt, return_tensors="pt")
                    use_image = False
            else:
                inputs = tokenizer(prompt, return_tensors="pt")
                use_image = False
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 256),  # Limit tokens to avoid memory issues
                    temperature=temperature,
                    do_sample=False,  # Use greedy decoding for stability
                    pad_token_id=tokenizer.eos_token_id,
                    num_beams=1       # Reduce beam size for memory efficiency
                )
            
            if use_image:
                # For vision models, decode from input length
                generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
            else:
                # For text-only, decode from input length
                generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
            
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
            
        except Exception as e:
            print(f"Error in generate_response: {e}")
            return f"Error: {str(e)}"

    # Export the function for Elixir to call
    generate_response
    """
  end
end