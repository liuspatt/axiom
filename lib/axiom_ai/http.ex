defmodule AxiomAi.Http do
  @moduledoc """
  HTTP client utilities for AxiomAI.
  """

  @doc """
  Performs a POST request with JSON payload.
  """
  @spec post(String.t(), map(), list()) :: {:ok, map()} | {:error, any()}
  def post(url, payload, headers \\ []) do
    json_payload = Jason.encode!(payload)

    default_headers = [
      {"Content-Type", "application/json"},
      {"User-Agent", "AxiomAI/0.1.0"}
    ]

    final_headers = default_headers ++ headers

    case HTTPoison.post(url, json_payload, final_headers, timeout: 30_000, recv_timeout: 30_000) do
      {:ok, response} ->
        {:ok,
         %{status_code: response.status_code, body: response.body, headers: response.headers}}

      {:error, %HTTPoison.Error{reason: reason}} ->
        {:error, reason}
    end
  end

  @doc """
  Performs a GET request.
  """
  @spec get(String.t(), list()) :: {:ok, map()} | {:error, any()}
  def get(url, headers \\ []) do
    default_headers = [
      {"User-Agent", "AxiomAI/0.1.0"}
    ]

    final_headers = default_headers ++ headers

    case HTTPoison.get(url, final_headers, timeout: 30_000, recv_timeout: 30_000) do
      {:ok, response} ->
        {:ok,
         %{status_code: response.status_code, body: response.body, headers: response.headers}}

      {:error, %HTTPoison.Error{reason: reason}} ->
        {:error, reason}
    end
  end
end
