defmodule AxiomAi.LocalModels.Templates do
  @moduledoc """
  Template system for creating model configurations with common patterns.
  """

  @doc """
  Creates a model configuration from a template.
  """
  @spec create_from_template(atom(), map()) :: map()
  def create_from_template(template_type, overrides \\ %{}) do
    base_template = get_template(template_type)
    Map.merge(base_template, overrides)
  end

  @doc """
  Gets available template types.
  """
  @spec list_templates() :: [atom()]
  def list_templates do
    [:pythonx_text, :pythonx_vision, :http_openai, :http_ollama, :custom]
  end

  # Private template definitions
  defp get_template(:pythonx_text) do
    %{
      type: :pythonx,
      context_length: 32768,
      python_deps: default_text_dependencies(),
      python_code: default_text_inference_code(),
      description: "Python-based text generation model"
    }
  end

  defp get_template(:pythonx_vision) do
    %{
      type: :pythonx,
      context_length: 8192,
      python_deps: default_vision_dependencies(),
      python_code: default_vision_inference_code(),
      description: "Python-based vision-language model"
    }
  end

  defp get_template(:http_openai) do
    %{
      type: :http,
      endpoint: "http://localhost:8000",
      api_format: :openai,
      description: "OpenAI-compatible HTTP endpoint"
    }
  end

  defp get_template(:http_ollama) do
    %{
      type: :http,
      endpoint: "http://localhost:11434",
      api_format: :ollama,
      description: "Ollama HTTP endpoint"
    }
  end

  defp get_template(:custom) do
    %{
      name: "Custom Model",
      type: :http,
      endpoint: "http://localhost:8000",
      api_format: :openai,
      description: "Custom AI model configuration"
    }
  end

  defp get_template(_), do: get_template(:custom)

  # Default Python dependencies for text models
  defp default_text_dependencies do
    """
    [project]
    name = "text_inference"
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

  # Default Python dependencies for vision models
  defp default_vision_dependencies do
    """
    [project]
    name = "vision_inference"
    version = "0.1.0"
    requires-python = "==3.10.*"
    dependencies = [
      "torch >= 2.0.0",
      "torchvision >= 0.15.0",
      "transformers >= 4.35.0", 
      "accelerate >= 0.20.0",
      "tokenizers >= 0.14.0",
      "numpy >= 1.24.0",
      "pillow >= 9.0.0"
    ]
    """
  end

  # Default Python inference code for text models
  defp default_text_inference_code do
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

  # Default Python inference code for vision models
  defp default_vision_inference_code do
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
    from PIL import Image
    import base64
    import io
    import os

    # Global variables to cache model and tokenizer
    _model = None
    _tokenizer = None
    _processor = None
    _current_model_path = None

    def load_model(model_path):
        global _model, _tokenizer, _processor, _current_model_path
        
        if _current_model_path != model_path:
            _tokenizer = AutoTokenizer.from_pretrained(model_path)
            _processor = AutoProcessor.from_pretrained(model_path)
            _model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            _current_model_path = model_path
        
        return _tokenizer, _processor, _model

    def process_image(image_path):
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        return image

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
                    image = process_image(image_path)
                    image = image.resize((512, 512), Image.LANCZOS)
                    inputs = processor(text=prompt, images=image, return_tensors="pt")
                except Exception as e:
                    print(f"Warning: Failed to load image {image_path}: {e}")
                    inputs = tokenizer(prompt, return_tensors="pt")
            else:
                inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 256),
                    temperature=temperature,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
            
        except Exception as e:
            return f"Error: {str(e)}"

    # Export the function for Elixir to call
    generate_response
    """
  end
end
