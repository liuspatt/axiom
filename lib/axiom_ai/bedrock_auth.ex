defmodule AxiomAi.BedrockAuth do
  @moduledoc """
  AWS Bedrock authentication and request signing utilities.
  """

  @doc """
  Gets AWS credentials from configuration or environment.
  
  Supports multiple authentication methods:
  - Direct access_key and secret_key in config
  - AWS environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
  - AWS profiles (~/.aws/credentials)
  - IAM roles (for EC2/ECS/Lambda)
  """
  @spec get_aws_config(map()) :: map()
  def get_aws_config(config) do
    cond do
      Map.has_key?(config, :access_key) and Map.has_key?(config, :secret_key) ->
        %{
          access_key_id: config.access_key,
          secret_access_key: config.secret_key,
          region: Map.get(config, :region, "us-east-1")
        }

      System.get_env("AWS_ACCESS_KEY_ID") and System.get_env("AWS_SECRET_ACCESS_KEY") ->
        %{
          access_key_id: System.get_env("AWS_ACCESS_KEY_ID"),
          secret_access_key: System.get_env("AWS_SECRET_ACCESS_KEY"),
          region: Map.get(config, :region, System.get_env("AWS_DEFAULT_REGION", "us-east-1"))
        }

      true ->
        # Fall back to AWS credential chain (profiles, IAM roles, etc.)
        case ExAws.Config.new(:bedrock_runtime, region: Map.get(config, :region, "us-east-1")) do
          %{access_key_id: nil} ->
            {:error, :no_aws_credentials}

          aws_config ->
            aws_config
        end
    end
  end

  @doc """
  Creates an AWS request for Bedrock Runtime InvokeModel API.
  """
  @spec create_invoke_model_request(String.t(), String.t(), map(), map()) :: map()
  def create_invoke_model_request(model_id, _region, payload, aws_config) do
    %ExAws.Operation.JSON{
      service: :bedrock_runtime,
      http_method: :post,
      path: "/model/#{model_id}/invoke",
      data: payload,
      headers: [
        {"content-type", "application/json"},
        {"accept", "application/json"}
      ]
    }
    |> ExAws.Config.new(aws_config)
  end

  @doc """
  Formats model-specific payload for different Bedrock models.
  """
  @spec format_model_payload(String.t(), String.t(), map()) :: map()
  def format_model_payload(model_id, message, config) do
    cond do
      String.contains?(model_id, "anthropic.claude") ->
        format_claude_payload(message, config)

      String.contains?(model_id, "amazon.titan") ->
        format_titan_payload(message, config)

      String.contains?(model_id, "meta.llama") ->
        format_llama_payload(message, config)

      String.contains?(model_id, "ai21.j2") ->
        format_ai21_payload(message, config)

      true ->
        # Default OpenAI-compatible format
        %{
          messages: [%{role: "user", content: message}],
          max_tokens: Map.get(config, :max_tokens, 1024),
          temperature: Map.get(config, :temperature, 0.7)
        }
    end
  end

  @doc """
  Parses model-specific response from different Bedrock models.
  """
  @spec parse_model_response(String.t(), map()) :: {:ok, String.t()} | {:error, any()}
  def parse_model_response(model_id, response) do
    cond do
      String.contains?(model_id, "anthropic.claude") ->
        parse_claude_response(response)

      String.contains?(model_id, "amazon.titan") ->
        parse_titan_response(response)

      String.contains?(model_id, "meta.llama") ->
        parse_llama_response(response)

      String.contains?(model_id, "ai21.j2") ->
        parse_ai21_response(response)

      true ->
        # Default parsing
        case response do
          %{"choices" => [%{"message" => %{"content" => content}} | _]} ->
            {:ok, content}
          %{"content" => [%{"text" => text} | _]} ->
            {:ok, text}
          %{"outputs" => [%{"text" => text} | _]} ->
            {:ok, text}
          _ ->
            {:error, {:unexpected_response_format, response}}
        end
    end
  end

  # Private functions for model-specific formatting

  defp format_claude_payload(message, config) do
    %{
      prompt: "\\n\\nHuman: #{message}\\n\\nAssistant:",
      max_tokens_to_sample: Map.get(config, :max_tokens, 1024),
      temperature: Map.get(config, :temperature, 0.7),
      top_p: Map.get(config, :top_p, 0.9)
    }
  end

  defp format_titan_payload(message, config) do
    %{
      inputText: message,
      textGenerationConfig: %{
        maxTokenCount: Map.get(config, :max_tokens, 1024),
        temperature: Map.get(config, :temperature, 0.7),
        topP: Map.get(config, :top_p, 0.9)
      }
    }
  end

  defp format_llama_payload(message, config) do
    %{
      prompt: message,
      max_gen_len: Map.get(config, :max_tokens, 1024),
      temperature: Map.get(config, :temperature, 0.7),
      top_p: Map.get(config, :top_p, 0.9)
    }
  end

  defp format_ai21_payload(message, config) do
    %{
      prompt: message,
      maxTokens: Map.get(config, :max_tokens, 1024),
      temperature: Map.get(config, :temperature, 0.7),
      topP: Map.get(config, :top_p, 0.9)
    }
  end

  # Private functions for model-specific response parsing

  defp parse_claude_response(response) do
    case response do
      %{"completion" => completion} -> {:ok, completion}
      %{"error" => error} -> {:error, error}
      _ -> {:error, {:unexpected_claude_response, response}}
    end
  end

  defp parse_titan_response(response) do
    case response do
      %{"results" => [%{"outputText" => text} | _]} -> {:ok, text}
      %{"error" => error} -> {:error, error}
      _ -> {:error, {:unexpected_titan_response, response}}
    end
  end

  defp parse_llama_response(response) do
    case response do
      %{"generation" => text} -> {:ok, text}
      %{"error" => error} -> {:error, error}
      _ -> {:error, {:unexpected_llama_response, response}}
    end
  end

  defp parse_ai21_response(response) do
    case response do
      %{"completions" => [%{"data" => %{"text" => text}} | _]} -> {:ok, text}
      %{"error" => error} -> {:error, error}
      _ -> {:error, {:unexpected_ai21_response, response}}
    end
  end
end