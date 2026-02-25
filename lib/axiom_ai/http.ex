defmodule AxiomAi.Http do
  @moduledoc """
  HTTP client utilities for AxiomAI.
  """

  require Logger

  @doc """
  Performs a POST request with JSON payload.
  """
  @spec post(String.t(), map(), list(), keyword()) :: {:ok, map()} | {:error, any()}
  def post(url, payload, headers \\ [], opts \\ []) do
    Logger.info("🌐 [AXIOM HTTP] POST to: #{url}")
    json_payload = Jason.encode!(payload)
    Logger.info("🌐 [AXIOM HTTP] Payload size: #{byte_size(json_payload)} bytes")

    default_headers = [
      {"Content-Type", "application/json"},
      {"User-Agent", "AxiomAI/0.1.0"}
    ]

    final_headers = default_headers ++ headers

    # Extract timeout options with defaults
    timeout = Keyword.get(opts, :timeout, 30_000)
    recv_timeout = Keyword.get(opts, :recv_timeout, 30_000)

    case HTTPoison.post(url, json_payload, final_headers,
           timeout: timeout,
           recv_timeout: recv_timeout
         ) do
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
  @spec get(String.t(), list(), keyword()) :: {:ok, map()} | {:error, any()}
  def get(url, headers \\ [], opts \\ []) do
    default_headers = [
      {"User-Agent", "AxiomAI/0.1.0"}
    ]

    final_headers = default_headers ++ headers

    # Extract timeout options with defaults
    timeout = Keyword.get(opts, :timeout, 30_000)
    recv_timeout = Keyword.get(opts, :recv_timeout, 30_000)

    case HTTPoison.get(url, final_headers, timeout: timeout, recv_timeout: recv_timeout) do
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
  @spec post_stream(String.t(), map(), list(), keyword()) ::
          {:ok, Enumerable.t()} | {:error, any()}
  def post_stream(url, payload, headers \\ [], opts \\ []) do
    Logger.info("🌊 [AXIOM HTTP STREAM] POST to: #{url}")
    json_payload = Jason.encode!(payload)
    Logger.info("🌊 [AXIOM HTTP STREAM] Payload size: #{byte_size(json_payload)} bytes")

    default_headers = [
      {"Content-Type", "application/json"},
      {"User-Agent", "AxiomAI/0.1.0"}
    ]

    final_headers = default_headers ++ headers

    # Extract timeout options with defaults
    timeout = Keyword.get(opts, :timeout, 30_000)
    recv_timeout = Keyword.get(opts, :recv_timeout, 30_000)
    Logger.info("🌊 [AXIOM HTTP STREAM] timeout: #{timeout}ms, recv_timeout: #{recv_timeout}ms")

    Logger.info("🌊 [AXIOM HTTP STREAM] Making async POST request...")
    case HTTPoison.post(url, json_payload, final_headers,
           stream_to: self(),
           timeout: timeout,
           recv_timeout: recv_timeout
         ) do
      {:ok, %HTTPoison.AsyncResponse{id: id}} ->
        Logger.info("🌊 [AXIOM HTTP STREAM] ✅ Async response started, ID: #{inspect(id)}")
        {:ok, build_stream(id, recv_timeout)}

      {:error, %HTTPoison.Error{reason: reason}} ->
        Logger.error("🌊 [AXIOM HTTP STREAM] ❌ HTTP Error: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp build_stream(request_id, recv_timeout) do
    Logger.info("🔄 [AXIOM STREAM] Building stream for request #{inspect(request_id)}")

    Stream.resource(
      fn ->
        Logger.info("🔄 [AXIOM STREAM] Stream initialized")
        request_id
      end,
      fn id ->
        receive do
          %HTTPoison.AsyncStatus{id: ^id, code: code} ->
            Logger.info("📡 [AXIOM STREAM] Status: #{code}")
            {[{:status, code}], id}

          %HTTPoison.AsyncHeaders{id: ^id, headers: headers} ->
            Logger.debug("📋 [AXIOM STREAM] Headers received")
            {[{:headers, headers}], id}

          %HTTPoison.AsyncChunk{id: ^id, chunk: chunk} ->
            Logger.debug("📦 [AXIOM STREAM] Chunk: #{byte_size(chunk)} bytes")
            {[{:chunk, chunk}], id}

          %HTTPoison.AsyncEnd{id: ^id} ->
            Logger.info("🏁 [AXIOM STREAM] Stream ended")
            {:halt, id}

          msg ->
            Logger.warning("❓ [AXIOM STREAM] Unexpected message: #{inspect(msg)}")
            {[{:error, {:unexpected_message, msg}}], id}
        after
          recv_timeout ->
            Logger.warning("⏰ [AXIOM STREAM] Timeout after #{recv_timeout}ms")
            {:halt, id}
        end
      end,
      fn _id ->
        Logger.info("🧹 [AXIOM STREAM] Stream cleanup")
        :ok
      end
    )
  end
end
