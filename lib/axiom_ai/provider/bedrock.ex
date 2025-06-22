defmodule AxiomAi.Provider.Bedrock do
  @moduledoc """
  AWS Bedrock provider implementation.
  
  Supports multiple model families:
  - Anthropic Claude (anthropic.claude-*)
  - Amazon Titan (amazon.titan-*)
  - Meta Llama (meta.llama*)
  - AI21 Jurassic (ai21.j2-*)
  """

  @behaviour AxiomAi.Provider

  alias AxiomAi.BedrockAuth

  @impl true
  def chat(config, message) do
    with {:ok, aws_config} <- get_aws_config(config),
         {:ok, model_id} <- get_model_id(config),
         {:ok, region} <- get_region(config),
         {:ok, response} <- invoke_model(model_id, message, config, aws_config, region) do
      {:ok, %{response: response}}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  @impl true
  def complete(config, prompt, options) do
    # Merge options into config for complete method
    merged_config = Map.merge(config, options)
    
    with {:ok, aws_config} <- get_aws_config(merged_config),
         {:ok, model_id} <- get_model_id(merged_config),
         {:ok, region} <- get_region(merged_config),
         {:ok, response} <- invoke_model(model_id, prompt, merged_config, aws_config, region) do
      {:ok, %{completion: response}}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  defp get_aws_config(config) do
    case BedrockAuth.get_aws_config(config) do
      {:error, reason} -> {:error, reason}
      aws_config -> {:ok, aws_config}
    end
  end

  defp get_model_id(config) do
    case Map.get(config, :model) do
      nil -> {:error, :missing_model}
      model -> {:ok, model}
    end
  end

  defp get_region(config) do
    region = Map.get(config, :region, "us-east-1")
    {:ok, region}
  end

  defp invoke_model(model_id, message, config, aws_config, region) do
    try do
      # Format payload based on model type
      payload = BedrockAuth.format_model_payload(model_id, message, config)
      
      # Create and execute the request
      request = BedrockAuth.create_invoke_model_request(model_id, region, payload, aws_config)
      
      case ExAws.request(request) do
        {:ok, %{status_code: 200, body: body}} ->
          case Jason.decode(body) do
            {:ok, response_data} ->
              BedrockAuth.parse_model_response(model_id, response_data)
            
            {:error, reason} ->
              {:error, {:json_decode_error, reason}}
          end

        {:ok, %{status_code: status_code, body: body}} ->
          case Jason.decode(body) do
            {:ok, %{"message" => message}} ->
              {:error, %{status_code: status_code, message: message}}
            
            {:ok, error_data} ->
              {:error, %{status_code: status_code, message: error_data}}
            
            {:error, _} ->
              {:error, %{status_code: status_code, message: body}}
          end

        {:error, reason} ->
          {:error, {:request_error, reason}}
      end
    rescue
      e ->
        {:error, {:bedrock_error, Exception.message(e)}}
    end
  end
end