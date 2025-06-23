defmodule AxiomAi.LocalModels do
  @moduledoc """
  Pre-defined configurations for local AI models including Qwen and other popular models.
  Uses Pythonx for Python ML environment management.
  """

  @doc """
  Returns the predefined model configurations.
  """
  @spec get_predefined_models() :: %{String.t() => map()}
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
      },

      # OCR Models
      "nanonets-ocr-s" => %{
        name: "Nanonets OCR Small",
        type: :pythonx,
        model_path: "nanonets/Nanonets-OCR-s",
        python_deps: ocr_dependencies(),
        python_code: ocr_inference_code(),
        context_length: 8192,
        description: "Nanonets OCR Small - Optimized for optical character recognition tasks"
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

  # Private function to define Python dependencies for OCR models
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

  # Private function to define Python inference code for OCR models
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
