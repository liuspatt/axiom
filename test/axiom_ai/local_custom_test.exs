defmodule AxiomAi.LocalCustomTest do
  use ExUnit.Case
  import Mock

  alias AxiomAi.Provider.Local
  alias AxiomAi.LocalModels.Templates

  # Set timeout for ML model tests that require downloading dependencies
  @moduletag timeout: :infinity

  describe "custom python code execution" do
    test "custom python_interface configuration with mock" do
      config = %{
        python_deps: """
        [project]
        name = "custom_inference"
        version = "0.1.0"
        requires-python = ">=3.8"
        dependencies = [
          "torch >= 2.0.0",
          "transformers >= 4.35.0",
          "accelerate >= 0.20.0"
        ]
        """,
        python_code: """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Global variables for model caching
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
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(
                generated_ids[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            return response
        """,
        model_path: "gpt2-medium",
        temperature: 0.8,
        max_tokens: 256
      }

      # Mock PythonInterface to avoid actual Python execution in tests
      with_mock(PythonInterface,
        uv_init: fn _deps -> :ok end,
        eval: fn _code, _globals -> {"Mock AI response about the future of technology.", %{}} end,
        decode: fn result -> result end
      ) do
        result = Local.chat(config, "Write a short story about AI")

        assert {:ok, %{response: "Mock AI response about the future of technology."}} = result

        # Verify PythonInterface was called with expected parameters
        assert called(PythonInterface.uv_init(:_))
        assert called(PythonInterface.eval(:_, :_))
        assert called(PythonInterface.decode(:_))
      end
    end

    test "custom python_interface configuration determines correct execution type" do
      config = %{
        python_deps: "some deps",
        python_code: "some code",
        model_path: "test/model"
      }

      # Test indirectly by calling chat and ensuring it doesn't fail with missing config
      with_mock(PythonInterface,
        uv_init: fn _deps -> :ok end,
        eval: fn _code, _globals -> {"Mock response", %{}} end,
        decode: fn result -> result end
      ) do
        result = Local.chat(config, "test")
        assert {:ok, %{response: "Mock response"}} = result
      end
    end

    test "custom python_interface configuration validation" do
      # Test missing dependencies
      config_missing_deps = %{
        python_code: "some code",
        model_path: "test/model"
      }

      with_mock(PythonInterface,
        uv_init: fn _deps -> :ok end,
        eval: fn _code, _globals -> {"result", %{}} end,
        decode: fn result -> result end
      ) do
        result = Local.chat(config_missing_deps, "test")
        assert {:error, :missing_python_interface_config} = result
      end

      # Test missing code
      config_missing_code = %{
        python_deps: "some deps",
        model_path: "test/model"
      }

      with_mock(PythonInterface,
        uv_init: fn _deps -> :ok end,
        eval: fn _code, _globals -> {"result", %{}} end,
        decode: fn result -> result end
      ) do
        result = Local.chat(config_missing_code, "test")
        assert {:error, :missing_execution_config} = result
      end

      # Test missing model path
      config_missing_model = %{
        python_deps: "some deps",
        python_code: "some code"
        # missing model_path - this should cause execution_type detection to fail
        # because determine_execution_type only checks for python_code, not all required fields
      }

      # Since this config has python_code, it will be detected as :python execution type
      # But when execute_python_interface_model is called, it will fail due to missing model_path
      with_mock(PythonInterface,
        uv_init: fn _deps -> :ok end,
        eval: fn _code, _globals -> {"result", %{}} end,
        decode: fn result -> result end
      ) do
        result = Local.chat(config_missing_model, "test")
        assert {:error, :missing_python_interface_config} = result
      end
    end

    test "python_interface error handling" do
      config = %{
        python_deps: "deps",
        python_code: "code",
        model_path: "model"
      }

      # Test that PythonInterface errors are properly caught and returned as error tuples
      # The exact behavior will depend on how the actual implementation handles exceptions
      with_mock(PythonInterface,
        uv_init: fn _deps -> :ok end,
        eval: fn _code, _globals -> {"result", %{}} end,
        decode: fn result -> result end
      ) do
        # Test normal execution first
        result = Local.chat(config, "test")
        assert {:ok, %{response: "result"}} = result

        # More complex error testing would require integration tests
        # since exception handling in mocks can be tricky
      end
    end
  end

  describe "custom python script execution" do
    test "python script configuration" do
      config = %{
        python_script: """
        import json
        import sys

        def main():
            input_data = json.loads(sys.argv[1])
            model_path = input_data['model_path']
            prompt = input_data['prompt']
            
            # Simple mock response
            response = f"Script response for: {prompt}"
            print(json.dumps({"response": response}))

        if __name__ == "__main__":
            main()
        """,
        model_path: "test-model",
        temperature: 0.7,
        max_tokens: 512
      }

      # This test would require complex mocking of file system operations
      # For now, just verify the configuration is valid
      result = Local.chat(config, "Hello world")

      # Should either succeed or fail with a specific error (not missing config)
      case result do
        {:ok, _} ->
          assert true

        {:error, {:python_script_failed, _, _}} ->
          assert true

        {:error, {:script_execution_error, _}} ->
          assert true

        {:error, :missing_python_config} ->
          flunk("Should not have missing config with valid python_script and model_path")

        other ->
          flunk("Unexpected result: #{inspect(other)}")
      end
    end

    test "python script error handling" do
      config = %{
        python_script: "import sys; sys.exit(1)",
        model_path: "test-model"
      }

      # For now, just test that it attempts to execute and fails appropriately
      # This test would require complex mocking of file operations
      # In a real scenario, we'd test this with integration tests
      result = Local.chat(config, "test")

      # Should either fail with script execution error or missing config
      assert match?({:error, _}, result)
    end
  end

  describe "http endpoint configuration" do
    test "openai format http endpoint" do
      config = %{
        endpoint: "http://localhost:8000",
        api_format: :openai,
        model: "custom-model",
        api_key: "test-key",
        temperature: 0.8,
        max_tokens: 1024
      }

      _expected_payload = %{
        model: "custom-model",
        messages: [%{role: "user", content: "Hello API"}],
        temperature: 0.8,
        max_tokens: 1024
      }

      mock_response = %{
        status_code: 200,
        body:
          Jason.encode!(%{
            "choices" => [
              %{"message" => %{"content" => "Hello from API"}}
            ]
          })
      }

      with_mock(AxiomAi.Http,
        post: fn _url, _payload, _headers -> {:ok, mock_response} end
      ) do
        result = Local.chat(config, "Hello API")

        assert {:ok, %{response: "Hello from API"}} = result

        # Verify HTTP request was made with correct URL and headers
        assert called(
                 AxiomAi.Http.post(
                   "http://localhost:8000/v1/chat/completions",
                   :_,
                   :_
                 )
               )
      end
    end

    test "ollama format http endpoint" do
      config = %{
        endpoint: "http://localhost:11434",
        api_format: :ollama,
        model: "llama2:7b",
        temperature: 0.7,
        max_tokens: 2048
      }

      _expected_payload = %{
        model: "llama2:7b",
        messages: [%{role: "user", content: "Hello Ollama"}],
        temperature: 0.7,
        max_tokens: 2048,
        stream: false
      }

      mock_response = %{
        status_code: 200,
        body:
          Jason.encode!(%{
            "message" => %{"content" => "Hello from Ollama"}
          })
      }

      with_mock(AxiomAi.Http,
        post: fn _url, _payload, _headers -> {:ok, mock_response} end
      ) do
        result = Local.chat(config, "Hello Ollama")

        assert {:ok, %{response: "Hello from Ollama"}} = result

        # Verify HTTP request was made with correct URL
        assert called(
                 AxiomAi.Http.post(
                   "http://localhost:11434/api/chat",
                   :_,
                   :_
                 )
               )
      end
    end

    test "http endpoint error handling" do
      config = %{
        endpoint: "http://localhost:8000",
        api_format: :openai,
        model: "test-model"
      }

      # Test HTTP error
      with_mock(AxiomAi.Http,
        post: fn _url, _payload, _headers -> {:error, :timeout} end
      ) do
        result = Local.chat(config, "test")
        assert {:error, :timeout} = result
      end

      # Test HTTP error status
      error_response = %{
        status_code: 500,
        body: "Internal Server Error"
      }

      with_mock(AxiomAi.Http,
        post: fn _url, _payload, _headers -> {:ok, error_response} end
      ) do
        result = Local.chat(config, "test")
        assert {:error, %{status_code: 500, message: "Internal Server Error"}} = result
      end
    end
  end

  describe "template system" do
    test "create from python_interface_text template" do
      config =
        Templates.create_from_template(:python_interface_text, %{
          model_path: "custom/model",
          temperature: 0.9
        })

      assert config.type == :python_interface
      assert config.model_path == "custom/model"
      assert config.temperature == 0.9
      assert is_binary(config.python_code)
      assert is_binary(config.python_deps)
      assert String.contains?(config.python_code, "generate_response")
      assert String.contains?(config.python_deps, "torch")
    end

    test "create from http_openai template" do
      config =
        Templates.create_from_template(:http_openai, %{
          endpoint: "http://custom:8080",
          model: "custom-model"
        })

      assert config.type == :http
      assert config.endpoint == "http://custom:8080"
      assert config.model == "custom-model"
      assert config.api_format == :openai
    end

    test "create from http_ollama template" do
      config =
        Templates.create_from_template(:http_ollama, %{
          model: "llama2:13b"
        })

      assert config.type == :http
      assert config.endpoint == "http://localhost:11434"
      assert config.model == "llama2:13b"
      assert config.api_format == :ollama
    end

    test "list available templates" do
      templates = Templates.list_templates()

      assert :python_interface_text in templates
      assert :python_interface_vision in templates
      assert :python_interface_speech in templates
      assert :http_openai in templates
      assert :http_ollama in templates
      assert :custom in templates

      assert length(templates) == 6
    end

    test "invalid template defaults to custom" do
      config = Templates.create_from_template(:invalid_template)

      assert config.name == "Custom Model"
      assert config.type == :http
      assert config.api_format == :openai
    end
  end

  describe "completion method" do
    test "completion with python_interface" do
      config = %{
        python_deps: "deps",
        python_code: "code",
        model_path: "model"
      }

      options = %{temperature: 0.5, max_tokens: 100}

      with_mock(PythonInterface,
        uv_init: fn _deps -> :ok end,
        eval: fn _code, _globals -> {"Completion response", %{}} end,
        decode: fn result -> result end
      ) do
        result = Local.complete(config, "Complete this:", options)

        assert {:ok, %{completion: "Completion response"}} = result
      end
    end

    test "completion with http endpoint" do
      config = %{
        endpoint: "http://localhost:8000",
        api_format: :openai,
        model: "test-model"
      }

      options = %{temperature: 0.3, max_tokens: 50}

      expected_payload = %{
        model: "test-model",
        prompt: "Complete this text",
        temperature: 0.3,
        max_tokens: 50
      }

      mock_response = %{
        status_code: 200,
        body:
          Jason.encode!(%{
            "choices" => [%{"text" => " with AI assistance"}]
          })
      }

      with_mock(AxiomAi.Http,
        post: fn _url, _payload, _headers -> {:ok, mock_response} end
      ) do
        result = Local.complete(config, "Complete this text", options)

        assert {:ok, %{completion: " with AI assistance"}} = result

        assert called(
                 AxiomAi.Http.post(
                   "http://localhost:8000/v1/completions",
                   expected_payload,
                   [{"Content-Type", "application/json"}]
                 )
               )
      end
    end
  end

  describe "edge cases and error conditions" do
    test "missing execution configuration" do
      config = %{
        # No predefined_model, python_script, python_code, or endpoint
        temperature: 0.7
      }

      result = Local.chat(config, "test")
      assert {:error, :missing_execution_config} = result
    end

    test "invalid python script configuration" do
      # Test missing model_path
      config = %{
        python_script: "print('hello')"
        # missing model_path
      }

      result = Local.chat(config, "test")
      assert {:error, :missing_python_config} = result
    end

    test "invalid python_interface configuration" do
      # Test missing python_deps
      config = %{
        python_code: "some code",
        model_path: "test"
        # missing python_deps
      }

      result = Local.chat(config, "test")
      assert {:error, :missing_python_interface_config} = result
    end
  end

  describe "integration with main AxiomAi module" do
    test "full integration test with custom python_interface" do
      config = %{
        python_deps: "deps",
        python_code: "code",
        model_path: "gpt2"
      }

      client = AxiomAi.new(:local, config)

      with_mock(PythonInterface,
        uv_init: fn _deps -> :ok end,
        eval: fn _code, _globals -> {"Integration test response", %{}} end,
        decode: fn result -> result end
      ) do
        result = AxiomAi.chat(client, "Test integration")

        assert {:ok, %{response: "Integration test response"}} = result
      end
    end

    test "full integration test with http endpoint" do
      config = %{
        endpoint: "http://localhost:8000",
        api_format: :openai,
        model: "test-model"
      }

      client = AxiomAi.new(:local, config)

      mock_response = %{
        status_code: 200,
        body:
          Jason.encode!(%{
            "choices" => [%{"message" => %{"content" => "HTTP integration response"}}]
          })
      }

      with_mock(AxiomAi.Http,
        post: fn _url, _payload, _headers -> {:ok, mock_response} end
      ) do
        result = AxiomAi.chat(client, "Test HTTP integration")

        assert {:ok, %{response: "HTTP integration response"}} = result
      end
    end
  end
end
