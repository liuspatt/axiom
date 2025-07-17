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

  @doc """
  Performs a POST request with JSON payload and returns a stream for processing chunked responses.
  """
  @spec post_stream(String.t(), map(), list()) :: {:ok, Enumerable.t()} | {:error, any()}
  def post_stream(url, payload, headers \\ []) do
    json_payload = Jason.encode!(payload)

    default_headers = [
      {"Content-Type", "application/json"},
      {"User-Agent", "AxiomAI/0.1.0"}
    ]

    final_headers = default_headers ++ headers

    case HTTPoison.post(url, json_payload, final_headers,
           stream_to: self(),
           timeout: 30_000,
           recv_timeout: 30_000
         ) do
      {:ok, %HTTPoison.AsyncResponse{id: id}} ->
        {:ok, build_stream(id)}

      {:error, %HTTPoison.Error{reason: reason}} ->
        {:error, reason}
    end
  end

  defp build_stream(request_id) do
    Stream.resource(
      fn -> request_id end,
      fn id ->
        receive do
          %HTTPoison.AsyncStatus{id: ^id, code: code} ->
            {[{:status, code}], id}

          %HTTPoison.AsyncHeaders{id: ^id, headers: headers} ->
            {[{:headers, headers}], id}

          %HTTPoison.AsyncChunk{id: ^id, chunk: chunk} ->
            {[{:chunk, chunk}], id}

          %HTTPoison.AsyncEnd{id: ^id} ->
            {:halt, id}

          msg ->
            # Handle unexpected messages or errors
            {[{:error, {:unexpected_message, msg}}], id}
        after
          30_000 ->
            {:halt, id}
        end
      end,
      fn _id -> :ok end
    )
  end
end
