defmodule AxiomAi.StreamTest do
  use ExUnit.Case
  import Mock

  alias AxiomAi.Provider.VertexAi
  alias AxiomAi.Http

  describe "streaming functionality" do
    test "stream/2 calls streamGenerateContent endpoint" do
      config = %{
        project_id: "test-project",
        region: "us-central1",
        model: "gemini-1.5-pro",
        access_token: "test-token"
      }

      # Mock the HTTP stream response
      with_mock(Http, [:passthrough], [
        post_stream: fn _url, _payload, _headers ->
          {:ok, mock_stream()}
        end
      ]) do
        result = VertexAi.stream(config, "Hello world")
        
        assert {:ok, _stream} = result
        
        # Verify the correct endpoint was called
        assert called(Http.post_stream(
          "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-pro:streamGenerateContent",
          %{
            contents: [%{role: "user", parts: [%{text: "Hello world"}]}],
            generationConfig: %{
              temperature: 0.7,
              maxOutputTokens: 65536,
              topK: 40,
              topP: 0.95
            }
          },
          [
            {"Authorization", "Bearer test-token"},
            {"Content-Type", "application/json"}
          ]
        ))
      end
    end

    test "stream/4 calls streamGenerateContent endpoint with history" do
      config = %{
        project_id: "test-project",
        region: "us-central1",
        model: "gemini-1.5-pro",
        access_token: "test-token"
      }

      history = [
        %{role: "user", content: "What is 2+2?"},
        %{role: "assistant", content: "4"}
      ]

      # Mock the HTTP stream response
      with_mock(Http, [:passthrough], [
        post_stream: fn _url, _payload, _headers ->
          {:ok, mock_stream()}
        end
      ]) do
        result = VertexAi.stream(config, "You are a helpful assistant", history, "What is 3+3?")
        
        assert {:ok, _stream} = result
        
        # Verify the correct endpoint was called
        assert called(Http.post_stream(
          "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-pro:streamGenerateContent",
          %{
            contents: [
              %{role: "user", parts: [%{text: "You are a helpful assistant"}]},
              %{role: "model", parts: [%{text: "I understand. I'll follow your instructions."}]},
              %{role: "user", parts: [%{text: "What is 2+2?"}]},
              %{role: "model", parts: [%{text: "4"}]},
              %{role: "user", parts: [%{text: "What is 3+3?"}]}
            ],
            generationConfig: %{
              temperature: 0.7,
              maxOutputTokens: 65536,
              topK: 40,
              topP: 0.95
            }
          },
          [
            {"Authorization", "Bearer test-token"},
            {"Content-Type", "application/json"}
          ]
        ))
      end
    end

    test "stream respects custom configuration options" do
      config = %{
        project_id: "test-project",
        region: "europe-west1",
        model: "gemini-1.5-flash",
        access_token: "test-token",
        temperature: 0.5,
        max_tokens: 1000,
        top_k: 20,
        top_p: 0.8
      }

      # Mock the HTTP stream response
      with_mock(Http, [:passthrough], [
        post_stream: fn _url, _payload, _headers ->
          {:ok, mock_stream()}
        end
      ]) do
        result = VertexAi.stream(config, "Test message")
        
        assert {:ok, _stream} = result
        
        # Verify the correct endpoint and config were used
        assert called(Http.post_stream(
          "https://europe-west1-aiplatform.googleapis.com/v1/projects/test-project/locations/europe-west1/publishers/google/models/gemini-1.5-flash:streamGenerateContent",
          %{
            contents: [%{role: "user", parts: [%{text: "Test message"}]}],
            generationConfig: %{
              temperature: 0.5,
              maxOutputTokens: 1000,
              topK: 20,
              topP: 0.8
            }
          },
          [
            {"Authorization", "Bearer test-token"},
            {"Content-Type", "application/json"}
          ]
        ))
      end
    end

    test "stream handles HTTP errors" do
      config = %{
        project_id: "test-project",
        region: "us-central1",
        model: "gemini-1.5-pro",
        access_token: "test-token"
      }

      # Mock HTTP error
      with_mock(Http, [:passthrough], [
        post_stream: fn _url, _payload, _headers ->
          {:error, :timeout}
        end
      ]) do
        result = VertexAi.stream(config, "Hello world")
        
        assert {:error, :timeout} = result
      end
    end
  end

  describe "HTTP streaming" do
    test "post_stream returns stream from HTTPoison async response" do
      # Mock HTTPoison async response
      with_mock(HTTPoison, [:passthrough], [
        post: fn _url, _payload, _headers, _options ->
          {:ok, %HTTPoison.AsyncResponse{id: :test_id}}
        end
      ]) do
        result = Http.post_stream("https://example.com", %{test: "data"}, [])
        
        assert {:ok, stream} = result
        assert is_function(stream)
        
        # Verify HTTPoison was called with stream_to option
        assert called(HTTPoison.post(
          "https://example.com",
          "{\"test\":\"data\"}",
          [
            {"Content-Type", "application/json"},
            {"User-Agent", "AxiomAI/0.1.0"}
          ],
          stream_to: :_,
          timeout: 30_000,
          recv_timeout: 30_000
        ))
      end
    end

    test "post_stream handles HTTPoison errors" do
      # Mock HTTPoison error
      with_mock(HTTPoison, [:passthrough], [
        post: fn _url, _payload, _headers, _options ->
          {:error, %HTTPoison.Error{reason: :timeout}}
        end
      ]) do
        result = Http.post_stream("https://example.com", %{test: "data"}, [])
        
        assert {:error, :timeout} = result
      end
    end
  end

  describe "AxiomAi main module streaming" do
    test "stream/2 delegates to client stream" do
      client = AxiomAi.new(:vertex_ai, %{project_id: "test-project"})
      
      # Mock the provider stream function
      with_mock(AxiomAi.Provider.VertexAi, [:passthrough], [
        stream: fn _config, _message ->
          {:ok, mock_stream()}
        end
      ]) do
        result = AxiomAi.stream(client, "Hello world")
        
        assert {:ok, _stream} = result
        assert called(AxiomAi.Provider.VertexAi.stream(:_, "Hello world"))
      end
    end

    test "stream/4 delegates to client stream with history" do
      client = AxiomAi.new(:vertex_ai, %{project_id: "test-project"})
      
      # Mock the provider stream function
      with_mock(AxiomAi.Provider.VertexAi, [:passthrough], [
        stream: fn _config, _system_prompt, _history, _prompt ->
          {:ok, mock_stream()}
        end
      ]) do
        result = AxiomAi.stream(client, "You are helpful", [], "Hello!")
        
        assert {:ok, _stream} = result
        assert called(AxiomAi.Provider.VertexAi.stream(:_, "You are helpful", [], "Hello!"))
      end
    end
  end

  describe "other providers streaming" do
    test "OpenAI stream returns not implemented" do
      config = %{api_key: "test-key"}
      
      result = AxiomAi.Provider.OpenAi.stream(config, "Hello")
      assert {:error, :not_implemented} = result
      
      result = AxiomAi.Provider.OpenAi.stream(config, "System", [], "Hello")
      assert {:error, :not_implemented} = result
    end

    test "Anthropic stream returns not implemented" do
      config = %{api_key: "test-key"}
      
      result = AxiomAi.Provider.Anthropic.stream(config, "Hello")
      assert {:error, :not_implemented} = result
      
      result = AxiomAi.Provider.Anthropic.stream(config, "System", [], "Hello")
      assert {:error, :not_implemented} = result
    end

    test "DeepSeek stream returns not implemented" do
      config = %{api_key: "test-key"}
      
      result = AxiomAi.Provider.DeepSeek.stream(config, "Hello")
      assert {:error, :not_implemented} = result
      
      result = AxiomAi.Provider.DeepSeek.stream(config, "System", [], "Hello")
      assert {:error, :not_implemented} = result
    end

    test "Bedrock stream returns not implemented" do
      config = %{model: "test-model"}
      
      result = AxiomAi.Provider.Bedrock.stream(config, "Hello")
      assert {:error, :not_implemented} = result
      
      result = AxiomAi.Provider.Bedrock.stream(config, "System", [], "Hello")
      assert {:error, :not_implemented} = result
    end

    test "Local stream returns not implemented" do
      config = %{predefined_model: "test-model"}
      
      result = AxiomAi.Provider.Local.stream(config, "Hello")
      assert {:error, :not_implemented} = result
      
      result = AxiomAi.Provider.Local.stream(config, "System", [], "Hello")
      assert {:error, :not_implemented} = result
    end
  end

  # Helper function to create a mock stream
  defp mock_stream do
    Stream.cycle([
      {:status, 200},
      {:headers, []},
      {:chunk, "data: {\"text\": \"Hello\"}"},
      {:chunk, "data: {\"text\": \" world\"}"}
    ])
  end
end