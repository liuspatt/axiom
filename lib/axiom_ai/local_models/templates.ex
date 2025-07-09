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
    [:pythonx_text, :pythonx_vision, :pythonx_speech, :http_openai, :http_ollama, :custom]
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

  defp get_template(:pythonx_speech) do
    %{
      type: :pythonx,
      context_length: 30,
      python_deps: default_speech_dependencies(),
      python_code: default_speech_inference_code(),
      description: "Python-based speech-to-text model"
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

  # Default Python dependencies for speech models
  defp default_speech_dependencies do
    """
    [project]
    name = "speech_inference"
    version = "0.1.0"
    requires-python = "==3.10.*"
    dependencies = [
      "torch >= 2.0.0",
      "transformers >= 4.35.0",
      "accelerate >= 0.20.0",
      "tokenizers >= 0.14.0",
      "numpy >= 1.24.0",
      "librosa >= 0.10.0",
      "pydub >= 0.25.0",
      "psutil >= 5.9.0"
    ]
    """
  end

  # Default Python inference code for speech models
  defp default_speech_inference_code do
    """
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import librosa
    import numpy as np
    import os
    import tempfile
    from pydub import AudioSegment
    import math

    # Global variables to cache model and processor
    _model = None
    _processor = None
    _pipeline = None
    _current_model_path = None

    def get_best_device():
        if torch.cuda.is_available():
            return "cuda", torch.float16
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps", torch.float32  # MPS works better with float32
        else:
            return "cpu", torch.float32

    def cleanup_model():
        # Clean up model from memory to prevent OOM
        global _model, _processor, _pipeline
        if _model is not None:
            del _model
            _model = None
        if _processor is not None:
            del _processor
            _processor = None
        if _pipeline is not None:
            del _pipeline
            _pipeline = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear PyTorch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def check_memory_usage():
        # Basic memory check to prevent OOM
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        # If less than 2GB available, try to free memory
        if available_gb < 2.0:
            print(f"Low memory detected: {available_gb:.1f}GB available")
            cleanup_model()
            return False
        return True

    def load_model(model_path):
        global _model, _processor, _pipeline, _current_model_path

        if _current_model_path != model_path:
            # Check memory before loading
            if not check_memory_usage():
                print("Insufficient memory, cleaning up previous models...")
                cleanup_model()

            try:
                print(f"Loading model: {model_path}")
                _processor = AutoProcessor.from_pretrained(model_path)

                # Determine best device and dtype
                device, dtype = get_best_device()
                print(f"Using device: {device}, dtype: {dtype}")

                # Enhanced model loading with aggressive memory optimizations
                model_kwargs = {
                    "torch_dtype": dtype,
                    "low_cpu_mem_usage": True,
                }

                # Device mapping strategy with memory optimization
                if device == "cuda":
                    model_kwargs["device_map"] = "auto"
                elif device == "mps":
                    # MPS-specific optimizations - force CPU loading first
                    model_kwargs["device_map"] = "cpu"
                else:
                    model_kwargs["device_map"] = "cpu"

                # Load model with memory constraints
                _model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, **model_kwargs)

                # Move model to device after loading if MPS
                if device == "mps":
                    print("Moving model to MPS...")
                    _model = _model.to("mps")

                # Skip pipeline creation to save memory - use direct model inference
                _pipeline = None

                _current_model_path = model_path
                print("Model loaded successfully")

            except Exception as e:
                print(f"Error loading model: {e}")
                cleanup_model()
                # Fallback to CPU with minimal settings
                try:
                    _processor = AutoProcessor.from_pretrained(model_path)
                    _model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        model_path,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                        low_cpu_mem_usage=True
                    )
                    _current_model_path = model_path
                    print("Fallback to CPU successful")
                except Exception as e2:
                    print(f"Fallback loading failed: {e2}")
                    raise e2

        return _processor, _model, _pipeline

    def process_audio(audio_path):
        # Handle different audio formats
        file_ext = os.path.splitext(str(audio_path))[1].lower()

        if file_ext in ['.mp4', '.m4a', '.mp3', '.flv', '.avi']:
            # Use pydub to convert to wav format first
            try:
                # Ensure audio_path is a string
                audio_path_str = str(audio_path)
                audio_segment = AudioSegment.from_file(audio_path_str)

                # Convert to mono and set sample rate
                audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)

                # Create temporary wav file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                    temp_wav_path = temp_wav.name

                # Export to temporary file
                audio_segment.export(temp_wav_path, format='wav')

                # Load with librosa
                audio, sr = librosa.load(temp_wav_path, sr=16000)

                # Clean up temporary file
                try:
                    os.unlink(temp_wav_path)
                except:
                    pass  # Ignore cleanup errors

                return audio

            except Exception as e:
                print(f"Error converting audio format: {e}")
                # Fall back to direct librosa loading
                try:
                    audio, sr = librosa.load(str(audio_path), sr=16000)
                    return audio
                except Exception as e2:
                    raise Exception(f"Failed to load audio with both pydub and librosa: {e}, {e2}")
        else:
            # Direct loading for wav, flac, etc.
            audio, sr = librosa.load(str(audio_path), sr=16000)
            return audio

    def chunk_audio(audio, chunk_length_s=30, overlap_s=5, sample_rate=16000):
        chunk_length_samples = int(chunk_length_s * sample_rate)
        overlap_samples = int(overlap_s * sample_rate)

        if len(audio) <= chunk_length_samples:
            return [audio]

        chunks = []
        start = 0
        while start < len(audio):
            end = min(start + chunk_length_samples, len(audio))
            chunk = audio[start:end]
            chunks.append(chunk)

            # Move start position with overlap
            start = end - overlap_samples
            if start >= len(audio):
                break

        return chunks

    def transcribe_with_pipeline(audio_path, pipeline_obj):
        # Pipeline is disabled to save memory, use chunked processing instead
        if pipeline_obj is None:
            print("Pipeline disabled for memory optimization, using chunked processing")
            return None
        
        try:
            # Pipeline handles chunking automatically if available
            result = pipeline_obj(audio_path,
                                batch_size=4,  # Reduced batch size for memory
                                return_timestamps=True)

            if isinstance(result, dict) and 'text' in result:
                return result['text']
            elif isinstance(result, list) and len(result) > 0:
                # Concatenate chunks if returned as list
                return ' '.join([chunk['text'] for chunk in result if 'text' in chunk])
            else:
                return str(result)
        except Exception as e:
            print(f"Pipeline transcription failed: {e}")
            return None

    def generate_response(model_path, prompt, max_tokens=512, temperature=0.7):
        try:
            processor, model, pipeline_obj = load_model(model_path)

            # Speech models expect audio path in the prompt (format: "audio_path|actual_prompt")
            audio_path = None
            use_chunking = False

            if "|" in prompt:
                parts = prompt.split("|", 1)
                if len(parts) == 2:
                    audio_path = parts[0].strip()
                    prompt_text = parts[1].strip()
                    # Check if chunking is requested
                    use_chunking = "chunk" in prompt_text.lower() or "large" in prompt_text.lower()

            if audio_path and audio_path != "":
                try:
                    # For large files or when chunking is requested, use pipeline
                    if use_chunking:
                        transcription = transcribe_with_pipeline(audio_path, pipeline_obj)
                        if transcription:
                            return transcription

                    # Standard processing for normal files
                    audio = process_audio(audio_path)

                    # Check if audio is too long (> 30 seconds) and use chunking
                    if len(audio) > 30 * 16000:  # 30 seconds at 16kHz
                        print("Large audio detected, using chunked processing...")
                        chunks = chunk_audio(audio, chunk_length_s=30, overlap_s=2)
                        transcriptions = []

                        for i, chunk in enumerate(chunks):
                            try:
                                inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")

                                # Move inputs to same device as model
                                device, _ = get_best_device()
                                if device != "cpu":
                                    inputs = {k: v.to(device) for k, v in inputs.items()}

                                # Whisper models have a max_target_positions limit of 448
                                # Use the full limit for better transcription quality
                                safe_max_tokens = min(max_tokens, 448)

                                with torch.no_grad():
                                    predicted_ids = model.generate(
                                        inputs["input_features"],
                                        max_new_tokens=safe_max_tokens,
                                        temperature=temperature,
                                        do_sample=False
                                    )

                                chunk_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                                transcriptions.append(chunk_transcription)

                            except Exception as e:
                                print(f"Error processing chunk {i} - {str(e)}")
                                continue

                        # Combine transcriptions with basic overlap handling
                        if transcriptions:
                            full_transcription = transcriptions[0]
                            for i in range(1, len(transcriptions)):
                                # Simple overlap handling - remove common ending/beginning words
                                current = transcriptions[i].strip()
                                if current and not current.lower().startswith(full_transcription.lower().split()[-3:]):
                                    full_transcription += " " + current
                            return full_transcription
                        else:
                            return "Error: Failed to transcribe any chunks"

                    else:
                        # Standard processing for shorter files
                        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

                        # Move inputs to same device as model
                        device, _ = get_best_device()
                        if device != "cpu":
                            inputs = {k: v.to(device) for k, v in inputs.items()}

                        # Whisper models have a max_target_positions limit of 448
                        # Use the full limit for better transcription quality
                        safe_max_tokens = min(max_tokens, 448)

                        with torch.no_grad():
                            predicted_ids = model.generate(
                                inputs["input_features"],
                                max_new_tokens=safe_max_tokens,
                                temperature=temperature,
                                do_sample=False
                            )

                        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                        return transcription

                except Exception as e:
                    return f"Error processing audio: {str(e)}"
            else:
                return "Error: No audio file provided"

        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            # Clean up temporary variables and force garbage collection
            import gc
            gc.collect()

    # Export the function for Elixir to call
    generate_response
    """
  end
end
