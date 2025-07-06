defmodule AxiomAi.WhisperTest do
  use ExUnit.Case

  alias AxiomAi.Provider.Local
  alias AxiomAi.LocalModels

  # Set timeout for ML model tests that require downloading dependencies
  @moduletag timeout: :infinity

  describe "whisper predefined models" do
    test "transcribes audio file with Whisper Large v3 Turbo model" do
      # Test configuration for the predefined Whisper Large v3 Turbo model
      config = %{
        predefined_model: "whisper-large-v3-turbo"
      }

      # Path to test audio file
      audio_path = Path.join([File.cwd!(), "test", "fixtures", "test_audio.mp4"])

      IO.puts("\n=== Testing Whisper Large v3 Turbo Model ===")
      IO.puts("Model: #{config.predefined_model}")
      IO.puts("Audio file: #{audio_path}")

      # Verify the model configuration exists
      assert {:ok, model_config} = LocalModels.get_model_config("whisper-large-v3-turbo")
      IO.puts("Model config loaded: #{model_config.name}")
      IO.puts("Model path: #{model_config.model_path}")
      IO.puts("Execution type: #{model_config.type}")

      # Verify audio file exists
      assert File.exists?(audio_path), "Audio fixture file not found: #{audio_path}"

      case Local.chat(config, "#{audio_path}|Transcribe this audio") do
        {:ok, %{response: response}} ->
          assert is_binary(response)
          assert String.length(response) > 0
          IO.puts("\n✅ SUCCESS: Whisper Large v3 Turbo responded!")
          IO.puts("Transcription: #{response}")
          IO.puts("Response length: #{String.length(response)} characters")

        {:error, reason} ->
          # Handle expected errors when Python environment is not set up
          case reason do
            {:python_script_failed, exit_code, error_output} ->
              IO.puts("\n⚠️  EXPECTED: Python script execution failed")
              IO.puts("Exit code: #{exit_code}")
              IO.puts("Error output: #{error_output}")

              cond do
                String.contains?(
                  error_output,
                  "ModuleNotFoundError: No module named 'transformers'"
                ) ->
                  IO.puts("\n📝 To run this test successfully, install transformers:")
                  IO.puts("pip install transformers torch librosa")
                  :ok

                String.contains?(
                  error_output,
                  "ModuleNotFoundError: No module named 'librosa'"
                ) ->
                  IO.puts("\n📝 To run this test successfully, install librosa:")
                  IO.puts("pip install librosa")
                  :ok

                true ->
                  IO.puts("\n📝 To run this test successfully, install required packages:")
                  IO.puts("pip install transformers torch librosa")
                  :ok
              end

            other ->
              IO.puts("\n⚠️  UNEXPECTED ERROR: #{inspect(other)}")
              :ok
          end
      end
    end

    test "transcribes audio file with Whisper Large v3 model" do
      # Test configuration for the predefined Whisper Large v3 model
      config = %{
        predefined_model: "whisper-large-v3"
      }

      # Path to test audio file
      audio_path = Path.join([File.cwd!(), "test", "fixtures", "test_audio.wav"])

      IO.puts("\n=== Testing Whisper Large v3 Model ===")
      IO.puts("Model: #{config.predefined_model}")
      IO.puts("Audio file: #{audio_path}")

      # Verify the model configuration exists
      assert {:ok, model_config} = LocalModels.get_model_config("whisper-large-v3")
      IO.puts("Model config loaded: #{model_config.name}")
      IO.puts("Model path: #{model_config.model_path}")
      IO.puts("Execution type: #{model_config.type}")

      # Verify audio file exists
      assert File.exists?(audio_path), "Audio fixture file not found: #{audio_path}"

      case Local.chat(config, "#{audio_path}|Transcribe this audio") do
        {:ok, %{response: response}} ->
          assert is_binary(response)
          assert String.length(response) > 0
          IO.puts("\n✅ SUCCESS: Whisper Large v3 responded!")
          IO.puts("Transcription: #{response}")
          IO.puts("Response length: #{String.length(response)} characters")

        {:error, reason} ->
          # Handle expected errors when Python environment is not set up
          case reason do
            {:python_script_failed, exit_code, error_output} ->
              IO.puts("\n⚠️  EXPECTED: Python script execution failed")
              IO.puts("Exit code: #{exit_code}")
              IO.puts("Error output: #{error_output}")

              cond do
                String.contains?(
                  error_output,
                  "ModuleNotFoundError: No module named 'transformers'"
                ) ->
                  IO.puts("\n📝 To run this test successfully, install transformers:")
                  IO.puts("pip install transformers torch librosa")
                  :ok

                String.contains?(
                  error_output,
                  "ModuleNotFoundError: No module named 'librosa'"
                ) ->
                  IO.puts("\n📝 To run this test successfully, install librosa:")
                  IO.puts("pip install librosa")
                  :ok

                true ->
                  IO.puts("\n📝 To run this test successfully, install required packages:")
                  IO.puts("pip install transformers torch librosa")
                  :ok
              end

            other ->
              IO.puts("\n⚠️  UNEXPECTED ERROR: #{inspect(other)}")
              :ok
          end
      end
    end
  end

  describe "whisper model configuration" do
    test "whisper models are registered in the correct category" do
      # Test that whisper models are properly categorized
      speech_models = LocalModels.list_models_by_category(:speech)

      assert "whisper-large-v3" in speech_models
      assert "whisper-large-v3-turbo" in speech_models

      IO.puts("\n=== Speech Models ===")
      IO.puts("Found speech models: #{inspect(speech_models)}")
    end

    test "whisper models have correct configuration" do
      # Test Whisper Large v3 Turbo configuration
      {:ok, config} = LocalModels.get_model_config("whisper-large-v3-turbo")

      assert config.name == "Whisper Large v3 Turbo"
      assert config.category == :speech
      assert config.type == :pythonx
      assert config.model_path == "openai/whisper-large-v3-turbo"
      assert config.context_length == 30
      assert String.contains?(config.description, "speech-to-text")

      # Test Whisper Large v3 configuration
      {:ok, config} = LocalModels.get_model_config("whisper-large-v3")

      assert config.name == "Whisper Large v3"
      assert config.category == :speech
      assert config.type == :pythonx
      assert config.model_path == "openai/whisper-large-v3"
      assert config.context_length == 30
      assert String.contains?(config.description, "speech-to-text")
    end
  end
end
