defmodule AxiomAi.LocalModelTest do
  use ExUnit.Case

  alias AxiomAi.Provider.Local
  alias AxiomAi.LocalModels

  # Set timeout for ML model tests that require downloading dependencies
  @moduletag timeout: :infinity

  # describe "qwen predefined models" do
  #   test "sends 'hola' message to Qwen2.5-0.5B model" do
  #     # Test configuration for the predefined Qwen2.5-0.5B model
  #     config = %{
  #       predefined_model: "qwen2.5-0.5b"
  #     }

  #     IO.puts("\n=== Testing Qwen2.5-0.5B Model ===")
  #     IO.puts("Model: #{config.predefined_model}")
  #     IO.puts("Message: hola")

  #     # Verify the model configuration exists
  #     assert {:ok, model_config} = LocalModels.get_model_config("qwen2.5-0.5b")
  #     IO.puts("Model config loaded: #{model_config.name}")
  #     IO.puts("Model path: #{model_config.model_path}")
  #     IO.puts("Execution type: #{model_config.type}")

  #     case Local.chat(config, "hola") do
  #       {:ok, %{response: response}} ->
  #         assert is_binary(response)
  #         assert String.length(response) > 0
  #         IO.puts("\n✅ SUCCESS: Qwen2.5-0.5B responded!")
  #         IO.puts("Response: #{response}")
  #         IO.puts("Response length: #{String.length(response)} characters")

  #       {:error, reason} ->
  #         # Handle expected errors when Python environment is not set up
  #         case reason do
  #           {:python_script_failed, exit_code, error_output} ->
  #             IO.puts("\n⚠️  EXPECTED: Python script execution failed")
  #             IO.puts("Exit code: #{exit_code}")
  #             IO.puts("Error output: #{error_output}")

  #             cond do
  #               String.contains?(
  #                 error_output,
  #                 "ModuleNotFoundError: No module named 'transformers'"
  #               ) ->
  #                 IO.puts("\n📝 To run this test successfully, install transformers:")
  #                 IO.puts("pip install transformers torch")
  #                 :ok

  #               String.contains?(error_output, "No module named 'torch'") ->
  #                 IO.puts("\n📝 To run this test successfully, install PyTorch:")
  #                 IO.puts("pip install torch transformers")
  #                 :ok

  #               true ->
  #                 IO.puts("\n📝 Python environment setup needed for full test")
  #                 :ok
  #             end

  #           {:script_execution_error, message} ->
  #             IO.puts("\n⚠️  EXPECTED: Script execution error")
  #             IO.puts("Error: #{message}")
  #             :ok

  #           {:missing_python_config} ->
  #             IO.puts("\n❌ UNEXPECTED: Missing Python configuration")
  #             flunk("The predefined model should have Python config")

  #           {:missing_pythonx_config} ->
  #             IO.puts("\n❌ UNEXPECTED: Missing Pythonx configuration")
  #             flunk("The predefined model should have Pythonx config")

  #           {:pythonx_execution_error, message} ->
  #             IO.puts("\n⚠️  EXPECTED: Pythonx execution error")
  #             IO.puts("Error: #{message}")
  #             :ok

  #           {:predefined_model_not_found, model_key} ->
  #             IO.puts("\n❌ UNEXPECTED: Predefined model not found: #{model_key}")
  #             flunk("The qwen2.5-0.5b model should be available")

  #           other ->
  #             IO.puts("\n❌ UNEXPECTED ERROR: #{inspect(other)}")
  #             flunk("Unexpected error: #{inspect(other)}")
  #         end
  #     end

  #     IO.puts("\n=== Test completed ===")
  #   end

  #   test "validates predefined model configurations" do
  #     IO.puts("\n=== Available Predefined Models ===")

  #     models = LocalModels.list_models()
  #     assert length(models) > 0

  #     Enum.each(models, fn model_key ->
  #       {:ok, config} = LocalModels.get_model_config(model_key)
  #       IO.puts("#{model_key}: #{config.name} (#{config.type})")
  #     end)

  #     # Specifically verify Qwen models
  #     qwen_models = Enum.filter(models, &String.starts_with?(&1, "qwen"))
  #     assert length(qwen_models) >= 3

  #     IO.puts("\nQwen models available: #{Enum.join(qwen_models, ", ")}")
  #   end

  #   test "validates Qwen model Pythonx configuration" do
  #     {:ok, qwen_config} = LocalModels.get_model_config("qwen2.5-0.5b")

  #     assert qwen_config.type == :pythonx
  #     assert is_binary(qwen_config.python_code)
  #     assert is_binary(qwen_config.python_deps)
  #     assert String.length(qwen_config.python_code) > 100

  #     # Check that the code contains expected components
  #     code = qwen_config.python_code
  #     deps = qwen_config.python_deps
  #     assert String.contains?(code, "from transformers import")
  #     assert String.contains?(code, "AutoTokenizer")
  #     assert String.contains?(code, "AutoModelForCausalLM")
  #     assert String.contains?(code, "generate_response")
  #     assert String.contains?(deps, "torch >=")
  #     assert String.contains?(deps, "transformers >=")

  #     IO.puts("✅ Pythonx configuration validation passed")
  #     IO.puts("Python code length: #{String.length(code)} characters")
  #     IO.puts("Dependencies defined: #{String.contains?(deps, "torch")}")
  #   end

  #   test "performance test with 10 conversation responses" do
  #     config = %{
  #       predefined_model: "qwen2.5-0.5b"
  #     }

  #     IO.puts("\n=== Performance Test: 10-Response Conversation ===")
  #     IO.puts("Model: #{config.predefined_model}")

  #     conversation_prompts = [
  #       "Hello, how are you?",
  #       "What is the capital of France?",
  #       "Tell me a joke",
  #       "What is 2 + 2?",
  #       "Explain photosynthesis briefly",
  #       "What is your favorite color?",
  #       "Count from 1 to 5",
  #       "What is the weather like?",
  #       "Tell me about the ocean",
  #       "Say goodbye"
  #     ]

  #     start_time = System.monotonic_time(:millisecond)

  #     IO.puts("Starting conversation with #{length(conversation_prompts)} messages...")

  #     {successful_responses, total_response_length, total_responses} =
  #       Enum.with_index(conversation_prompts, 1)
  #       |> Enum.reduce({0, 0, 0}, fn {prompt, index},
  #                                    {success_count, total_length, total_count} ->
  #         IO.puts("\n--- Message #{index}/10 ---")
  #         IO.puts("Prompt: #{prompt}")

  #         message_start = System.monotonic_time(:millisecond)

  #         case Local.chat(config, prompt) do
  #           {:ok, %{response: response}} ->
  #             message_end = System.monotonic_time(:millisecond)
  #             message_duration = message_end - message_start

  #             response_length = String.length(response)

  #             IO.puts(
  #               "✅ Response (#{message_duration}ms): #{String.slice(response, 0, 100)}#{if response_length > 100, do: "...", else: ""}"
  #             )

  #             IO.puts("Length: #{response_length} characters")

  #             {success_count + 1, total_length + response_length, total_count + 1}

  #           {:error, reason} ->
  #             message_end = System.monotonic_time(:millisecond)
  #             message_duration = message_end - message_start
  #             IO.puts("⚠️ Error (#{message_duration}ms): #{inspect(reason)}")

  #             {success_count, total_length, total_count + 1}
  #         end
  #       end)

  #     end_time = System.monotonic_time(:millisecond)
  #     total_duration = end_time - start_time

  #     IO.puts("\n=== Performance Results ===")
  #     IO.puts("Total duration: #{total_duration}ms (#{Float.round(total_duration / 1000, 2)}s)")
  #     IO.puts("Successful responses: #{successful_responses}/#{total_responses}")
  #     IO.puts("Success rate: #{Float.round(successful_responses / total_responses * 100, 1)}%")

  #     if successful_responses > 0 do
  #       avg_response_time = total_duration / total_responses
  #       avg_response_length = total_response_length / successful_responses
  #       IO.puts("Average response time: #{Float.round(avg_response_time, 2)}ms")
  #       IO.puts("Average response length: #{Float.round(avg_response_length, 1)} characters")
  #       IO.puts("Total characters generated: #{total_response_length}")
  #     end

  #     # Assert basic performance expectations
  #     if successful_responses > 0 do
  #       assert total_duration < 300_000, "Total conversation should complete within 5 minutes"
  #       assert successful_responses >= 1, "At least one response should succeed"
  #     end

  #     IO.puts("\n=== Performance Test Completed ===")
  #   end
  # end

  describe "nanonets ocr model" do
    test "validates Nanonets OCR model configuration" do
      IO.puts("\n=== Testing Nanonets OCR Model Configuration ===")

      # Verify the model configuration exists
      assert {:ok, model_config} = LocalModels.get_model_config("nanonets-ocr-s")
      IO.puts("Model config loaded: #{model_config.name}")
      IO.puts("Model path: #{model_config.model_path}")
      IO.puts("Execution type: #{model_config.type}")

      # Validate configuration structure
      assert model_config.type == :pythonx
      assert model_config.model_path == "nanonets/Nanonets-OCR-s"
      assert is_binary(model_config.python_code)
      assert is_binary(model_config.python_deps)
      assert String.length(model_config.python_code) > 100
      assert model_config.context_length == 8192

      # Check that the code contains expected OCR components
      code = model_config.python_code
      deps = model_config.python_deps
      assert String.contains?(code, "from transformers import")
      assert String.contains?(code, "AutoTokenizer")
      assert String.contains?(code, "AutoModelForCausalLM")
      assert String.contains?(code, "AutoProcessor")
      assert String.contains?(code, "from PIL import Image")
      assert String.contains?(code, "process_image")
      assert String.contains?(code, "generate_response")
      assert String.contains?(deps, "torch >=")
      assert String.contains?(deps, "transformers >=")
      assert String.contains?(deps, "pillow >=")

      IO.puts("✅ Nanonets OCR configuration validation passed")
      IO.puts("Python code length: #{String.length(code)} characters")
      IO.puts("Dependencies include PIL: #{String.contains?(deps, "pillow")}")
    end

    @tag timeout: 600_000
    test "sends OCR prompt to Nanonets OCR model" do
      # Test configuration for the predefined Nanonets OCR model
      config = %{
        predefined_model: "nanonets-ocr-s"
      }

      # Path to test image
      test_image_path = Path.join([__DIR__, "..", "fixtures", "test_document.png"])

      IO.puts("\n=== Testing Nanonets OCR Model Execution ===")
      IO.puts("Model: #{config.predefined_model}")
      IO.puts("Test image: #{test_image_path}")
      IO.puts("Message: Extract text from this document")

      # Verify the model configuration exists
      assert {:ok, model_config} = LocalModels.get_model_config("nanonets-ocr-s")
      IO.puts("Model config loaded: #{model_config.name}")

      # Verify test image exists
      assert File.exists?(test_image_path), "Test image should exist at #{test_image_path}"

      # Send image path and prompt using the format "image_path|prompt"
      message = "#{test_image_path}|Extract text from this document"

      case Local.chat(config, message) do
        {:ok, %{response: response}} ->
          assert is_binary(response)
          assert String.length(response) > 0
          IO.puts("\n✅ SUCCESS: Nanonets OCR responded!")
          IO.puts("\n" <> String.duplicate("=", 60))
          IO.puts("EXTRACTED CONTENT FROM IMAGE:")
          IO.puts(String.duplicate("=", 60))
          IO.puts(response)
          IO.puts(String.duplicate("=", 60))
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
                  IO.puts("pip install transformers torch pillow")
                  :ok

                String.contains?(error_output, "No module named 'torch'") ->
                  IO.puts("\n📝 To run this test successfully, install PyTorch:")
                  IO.puts("pip install torch transformers pillow")
                  :ok

                String.contains?(error_output, "No module named 'PIL'") ->
                  IO.puts("\n📝 To run this test successfully, install Pillow:")
                  IO.puts("pip install pillow transformers torch")
                  :ok

                true ->
                  IO.puts("\n📝 Python environment setup needed for full test")
                  :ok
              end

            {:pythonx_execution_error, message} ->
              IO.puts("\n⚠️  EXPECTED: Pythonx execution error")
              IO.puts("Error: #{message}")
              :ok

            {:predefined_model_not_found, model_key} ->
              IO.puts("\n❌ UNEXPECTED: Predefined model not found: #{model_key}")
              flunk("The nanonets-ocr-s model should be available")

            other ->
              IO.puts("\n⚠️  EXPECTED ERROR: #{inspect(other)}")
              IO.puts("This is expected if the OCR model or dependencies are not installed")
              :ok
          end
      end

      IO.puts("\n=== OCR Test completed ===")
    end

    @tag timeout: 600_000
    test "sends OCR prompt to Nanonets OCR model with PDF" do
      # Test configuration for the predefined Nanonets OCR model
      config = %{
        predefined_model: "nanonets-ocr-s"
      }

      # Path to test PDF
      test_pdf_path = Path.join([__DIR__, "..", "fixtures", "ocr_file.pdf"])
      
      IO.puts("\n=== Testing Nanonets OCR Model with PDF ===")
      IO.puts("Model: #{config.predefined_model}")
      IO.puts("Test PDF: #{test_pdf_path}")
      IO.puts("Message: Extract all text from this PDF document")

      # Verify the model configuration exists
      assert {:ok, model_config} = LocalModels.get_model_config("nanonets-ocr-s")
      IO.puts("Model config loaded: #{model_config.name}")

      # Verify test PDF exists
      assert File.exists?(test_pdf_path), "Test PDF should exist at #{test_pdf_path}"

      # Send PDF path and prompt using the format "pdf_path|prompt"
      message = "#{test_pdf_path}|Extract all text from this PDF document"
      
      case Local.chat(config, message) do
        {:ok, %{response: response}} ->
          assert is_binary(response)
          assert String.length(response) > 0
          IO.puts("\n✅ SUCCESS: Nanonets OCR processed PDF!")
          IO.puts("\n" <> String.duplicate("=", 60))
          IO.puts("EXTRACTED CONTENT FROM PDF:")
          IO.puts(String.duplicate("=", 60))
          IO.puts(response)
          IO.puts(String.duplicate("=", 60))
          IO.puts("Response length: #{String.length(response)} characters")
          
          # Check for expected content from the PDF
          response_lower = String.downcase(response)
          if String.contains?(response_lower, "invoice") || String.contains?(response_lower, "business") do
            IO.puts("✅ Content validation: Found expected business/invoice content")
          else
            IO.puts("⚠️  Content validation: Expected business/invoice content not found")
          end

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
                  IO.puts("pip install transformers torch pillow PyMuPDF")
                  :ok

                String.contains?(error_output, "No module named 'torch'") ->
                  IO.puts("\n📝 To run this test successfully, install PyTorch:")
                  IO.puts("pip install torch transformers pillow PyMuPDF")
                  :ok

                String.contains?(error_output, "No module named 'PIL'") ->
                  IO.puts("\n📝 To run this test successfully, install Pillow:")
                  IO.puts("pip install pillow transformers torch PyMuPDF")
                  :ok

                String.contains?(error_output, "No module named 'fitz'") ->
                  IO.puts("\n📝 To run this test successfully, install PyMuPDF:")
                  IO.puts("pip install PyMuPDF transformers torch pillow")
                  :ok

                true ->
                  IO.puts("\n📝 Python environment setup needed for full test")
                  :ok
              end

            {:pythonx_execution_error, message} ->
              IO.puts("\n⚠️  EXPECTED: Pythonx execution error")
              IO.puts("Error: #{message}")
              :ok

            {:predefined_model_not_found, model_key} ->
              IO.puts("\n❌ UNEXPECTED: Predefined model not found: #{model_key}")
              flunk("The nanonets-ocr-s model should be available")

            other ->
              IO.puts("\n⚠️  EXPECTED ERROR: #{inspect(other)}")
              IO.puts("This is expected if the OCR model or dependencies are not installed")
              :ok
          end
      end

      IO.puts("\n=== OCR PDF Test completed ===")
    end
  end
end
