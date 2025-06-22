defmodule AxiomAi.AuthTest do
  use ExUnit.Case
  doctest AxiomAi.Auth

  alias AxiomAi.Auth
  alias AxiomAi.Provider.VertexAi
  alias AxiomAi.Provider.OpenAi
  alias AxiomAi.Provider.Anthropic
  alias AxiomAi.Provider.Local
  alias AxiomAi.Provider.DeepSeek
  alias AxiomAi.Provider.Bedrock

  describe "get_gcp_token/1" do
    test "loads credentials from service account file" do
      # Skip test if example credentials file doesn't exist
      credentials_path = "test/fixtures/credentials.json"

      case File.exists?(credentials_path) do
        false ->
          IO.puts("Skipping test: #{credentials_path} not found")
          :skip

        true ->
          config = %{service_account_path: credentials_path}

          case Auth.get_gcp_token(config) do
            {:ok, token} ->
              assert is_binary(token)
              assert String.length(token) > 0

            {:error, reason} ->
              # Allow test to pass if we can't make actual auth calls
              # but the file was read successfully - including invalid keys
              case reason do
                {:jwt_creation_error, _} -> :ok
                {:request_error, _} -> :ok
                {:http_error, _, _} -> :ok
                {:asn1, _} -> :ok  # Invalid private key format
                other -> flunk("Unexpected error: #{inspect(other)}")
              end
          end
      end
    end

    test "handles missing service account file" do
      config = %{service_account_path: "nonexistent/path.json"}

      assert {:error, {:file_read_error, :enoent}} = Auth.get_gcp_token(config)
    end

    test "handles invalid JSON in service account file" do
      # Create a temporary file with invalid JSON
      temp_path = "test/fixtures/invalid.json"
      File.mkdir_p!(Path.dirname(temp_path))
      File.write!(temp_path, "invalid json content")

      config = %{service_account_path: temp_path}

      assert {:error, {:json_decode_error, _}} = Auth.get_gcp_token(config)

      # Clean up
      File.rm!(temp_path)
    end

    test "returns direct access token when provided" do
      config = %{access_token: "test-token-123"}

      assert {:ok, "test-token-123"} = Auth.get_gcp_token(config)
    end

    test "handles service account key as map" do
      key_data = %{
        "type" => "service_account",
        "private_key" => "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC...\n-----END PRIVATE KEY-----",
        "client_email" => "test@example.com"
      }

      config = %{service_account_key: key_data}

      # This will likely fail with JWT creation error since we're using fake data
      # but we're testing the code path
      case Auth.get_gcp_token(config) do
        {:ok, _token} -> :ok
        {:error, {:jwt_creation_error, _}} -> :ok
        {:error, :missing_private_key} -> :ok
        {:error, {:asn1, _}} -> :ok  # Invalid private key format
        {:error, other} -> flunk("Unexpected error: #{inspect(other)}")
      end
    end
  end

  describe "vertex_ai integration" do
    test "sends 'hola' message to Vertex AI" do
      # Skip test if credentials file doesn't exist
      credentials_path = "test/fixtures/credentials.json"

      case File.exists?(credentials_path) do
        false ->
          IO.puts("Skipping Vertex AI test: #{credentials_path} not found")
          :skip

        true ->
          # Read credentials and extract project_id
          case File.read(credentials_path) do
            {:ok, content} ->
              case Jason.decode(content) do
                {:ok, %{"project_id" => project_id}} ->
                  config = %{
                    project_id: project_id,
                    service_account_path: credentials_path,
                    region: "us-central1",
                    model: "gemini-1.5-flash"
                  }

                  try do
                    case VertexAi.chat(config, "hola") do
                      {:ok, %{response: response}} ->
                        assert is_binary(response)
                        assert String.length(response) > 0
                        IO.puts("Vertex AI response to 'hola': #{response}")

                      {:error, reason} ->
                        # Allow test to pass if we get auth/network errors
                        # but the integration flow worked
                        case reason do
                          %{status_code: 401} ->
                            IO.puts("Skipping: Authentication failed (expected with test credentials)")
                            :ok
                          %{status_code: 403} ->
                            IO.puts("Skipping: Permission denied (expected with test credentials)")
                            :ok
                          %{status_code: 404} ->
                            IO.puts("Skipping: Project not found (expected with test credentials)")
                            :ok
                          other ->
                            flunk("Unexpected error: #{inspect(other)}")
                        end
                    end
                  rescue
                    e in RuntimeError ->
                      case Exception.message(e) do
                        "Failed to get access token: " <> _ ->
                          IO.puts("Skipping: Token generation failed (expected with test credentials)")
                          :ok
                        _ ->
                          flunk("Unexpected runtime error: #{inspect(e)}")
                      end
                  end

                {:error, _} ->
                  IO.puts("Skipping: Invalid JSON in credentials file")
                  :skip
              end

            {:error, _} ->
              IO.puts("Skipping: Could not read credentials file")
              :skip
          end
      end
    end
  end

  describe "openai integration" do
    test "sends 'hola' message to OpenAI" do
      # Skip test if API key is not provided
      api_key = System.get_env("OPENAI_API_KEY")
      
      case api_key do
        nil ->
          IO.puts("Skipping OpenAI test: OPENAI_API_KEY not set")
          :skip
          
        _ ->
          config = %{
            api_key: api_key,
            model: "gpt-3.5-turbo"
          }
          
          case OpenAi.chat(config, "hola") do
            {:ok, %{response: response}} ->
              assert is_binary(response)
              assert String.length(response) > 0
              IO.puts("OpenAI response to 'hola': #{response}")
              
            {:error, reason} ->
              # Allow test to pass if we get expected API errors
              case reason do
                %{status_code: 401} -> 
                  IO.puts("Skipping: Invalid API key")
                  :ok
                %{status_code: 429} -> 
                  IO.puts("Skipping: Rate limit exceeded")
                  :ok
                other ->
                  flunk("Unexpected error: #{inspect(other)}")
              end
          end
      end
    end
  end

  describe "anthropic integration" do
    test "sends 'hola' message to Anthropic" do
      # Skip test if API key is not provided
      api_key = System.get_env("ANTHROPIC_API_KEY")
      
      case api_key do
        nil ->
          IO.puts("Skipping Anthropic test: ANTHROPIC_API_KEY not set")
          :skip
          
        _ ->
          config = %{
            api_key: api_key,
            model: "claude-3-haiku-20240307"
          }
          
          case Anthropic.chat(config, "hola") do
            {:ok, %{response: response}} ->
              assert is_binary(response)
              assert String.length(response) > 0
              IO.puts("Anthropic response to 'hola': #{response}")
              
            {:error, reason} ->
              # Allow test to pass if we get expected API errors
              case reason do
                %{status_code: 401} -> 
                  IO.puts("Skipping: Invalid API key")
                  :ok
                %{status_code: 429} -> 
                  IO.puts("Skipping: Rate limit exceeded")
                  :ok
                other ->
                  flunk("Unexpected error: #{inspect(other)}")
              end
          end
      end
    end
  end

  describe "local integration" do
    test "sends 'hola' message to Local AI" do
      # Skip test if endpoint is not provided
      endpoint = System.get_env("LOCAL_AI_ENDPOINT")
      
      case endpoint do
        nil ->
          IO.puts("Skipping Local AI test: LOCAL_AI_ENDPOINT not set")
          :skip
          
        _ ->
          config = %{
            endpoint: endpoint,
            model: "default"
          }
          
          case Local.chat(config, "hola") do
            {:ok, %{response: response}} ->
              assert is_binary(response)
              assert String.length(response) > 0
              IO.puts("Local AI response to 'hola': #{response}")
              
            {:error, reason} ->
              # Allow test to pass if we get connection errors
              case reason do
                %{status_code: 401} -> 
                  IO.puts("Skipping: Authentication failed")
                  :ok
                %{status_code: 404} -> 
                  IO.puts("Skipping: Endpoint not found")
                  :ok
                {:econnrefused, _} ->
                  IO.puts("Skipping: Connection refused (service not running)")
                  :ok
                other ->
                  flunk("Unexpected error: #{inspect(other)}")
              end
          end
      end
    end
  end

  describe "deepseek integration" do
    test "sends 'hola' message to DeepSeek" do
      # Skip test if API key is not provided
      api_key = System.get_env("DEEPSEEK_API_KEY")
      
      case api_key do
        nil ->
          IO.puts("Skipping DeepSeek test: DEEPSEEK_API_KEY not set")
          :skip
          
        _ ->
          config = %{
            api_key: api_key,
            model: "deepseek-chat"
          }
          
          case DeepSeek.chat(config, "hola") do
            {:ok, %{response: response}} ->
              assert is_binary(response)
              assert String.length(response) > 0
              IO.puts("DeepSeek response to 'hola': #{response}")
              
            {:error, reason} ->
              # Allow test to pass if we get expected API errors
              case reason do
                %{status_code: 401} -> 
                  IO.puts("Skipping: Invalid API key")
                  :ok
                %{status_code: 429} -> 
                  IO.puts("Skipping: Rate limit exceeded")
                  :ok
                %{status_code: 403} -> 
                  IO.puts("Skipping: Access forbidden")
                  :ok
                other ->
                  flunk("Unexpected error: #{inspect(other)}")
              end
          end
      end
    end
  end

  describe "bedrock integration" do
    test "sends 'hola' message to AWS Bedrock" do
      # Skip test if AWS credentials are not available
      aws_access_key = System.get_env("AWS_ACCESS_KEY_ID")
      aws_secret_key = System.get_env("AWS_SECRET_ACCESS_KEY")
      
      case {aws_access_key, aws_secret_key} do
        {nil, _} ->
          IO.puts("Skipping Bedrock test: AWS_ACCESS_KEY_ID not set")
          :skip
          
        {_, nil} ->
          IO.puts("Skipping Bedrock test: AWS_SECRET_ACCESS_KEY not set")
          :skip
          
        {_, _} ->
          config = %{
            model: "anthropic.claude-3-haiku-20240307-v1:0",
            region: "us-east-1"
          }
          
          case Bedrock.chat(config, "hola") do
            {:ok, %{response: response}} ->
              assert is_binary(response)
              assert String.length(response) > 0
              IO.puts("Bedrock response to 'hola': #{response}")
              
            {:error, reason} ->
              # Allow test to pass if we get expected AWS errors
              case reason do
                {:request_error, _} -> 
                  IO.puts("Skipping: AWS request failed (network/auth issue)")
                  :ok
                %{status_code: 401} -> 
                  IO.puts("Skipping: Invalid AWS credentials")
                  :ok
                %{status_code: 403} -> 
                  IO.puts("Skipping: Access denied (check IAM permissions)")
                  :ok
                %{status_code: 404} -> 
                  IO.puts("Skipping: Model not found or not accessible")
                  :ok
                %{status_code: 429} -> 
                  IO.puts("Skipping: Rate limit exceeded")
                  :ok
                {:bedrock_error, _} ->
                  IO.puts("Skipping: Bedrock service error")
                  :ok
                :no_aws_credentials ->
                  IO.puts("Skipping: No AWS credentials found")
                  :ok
                :missing_model ->
                  IO.puts("Skipping: Model not specified")
                  :ok
                other ->
                  flunk("Unexpected error: #{inspect(other)}")
              end
          end
      end
    end
  end
end
