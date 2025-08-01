defmodule AxiomAi.Auth do
  @moduledoc """
  Authentication utilities for various AI providers.
  """

  @doc """
  Gets an access token for Google Cloud Platform authentication.

  Supports multiple authentication methods:
  - Direct access token (explicit token provided)
  - Service account key data (JSON map)
  - Service account key file (path to JSON file)
  - Application Default Credentials (ADC):
    * Cloud Run/GCE metadata service (production)
    * gcloud CLI (local development)
  """
  @spec get_gcp_token(map()) :: {:ok, String.t()} | {:error, any()}
  def get_gcp_token(config) do
    cond do
      Map.has_key?(config, :access_token) ->
        {:ok, config.access_token}

      Map.has_key?(config, :service_account_key) ->
        get_service_account_token(config.service_account_key)

      Map.has_key?(config, :service_account_path) ->
        case File.read(config.service_account_path) do
          {:ok, content} ->
            case Jason.decode(content) do
              {:ok, key_data} -> get_service_account_token(key_data)
              {:error, reason} -> {:error, {:json_decode_error, reason}}
            end

          {:error, reason} ->
            {:error, {:file_read_error, reason}}
        end

      true ->
        get_application_default_credentials()
    end
  end

  defp get_service_account_token(key_data) when is_map(key_data) do
    with {:ok, private_key} <- extract_private_key(key_data),
         {:ok, jwt} <- create_jwt(key_data, private_key),
         {:ok, token} <- exchange_jwt_for_token(jwt) do
      {:ok, token}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  defp get_service_account_token(_), do: {:error, :invalid_service_account_key}

  defp extract_private_key(%{"private_key" => private_key}) do
    {:ok, private_key}
  end

  defp extract_private_key(_), do: {:error, :missing_private_key}

  defp create_jwt(key_data, private_key) do
    now = System.system_time(:second)

    claims = %{
      "iss" => key_data["client_email"],
      "scope" => "https://www.googleapis.com/auth/cloud-platform",
      "aud" => "https://oauth2.googleapis.com/token",
      "exp" => now + 3600,
      "iat" => now
    }

    try do
      signer = create_signer(private_key)

      case Joken.encode_and_sign(claims, signer) do
        {:ok, jwt, _claims} -> {:ok, jwt}
        {:error, reason} -> {:error, {:jwt_creation_error, reason}}
      end
    rescue
      e -> {:error, {:jwt_creation_error, e}}
    end
  end

  defp create_signer(private_key) do
    Joken.Signer.create("RS256", %{"pem" => private_key})
  end

  defp exchange_jwt_for_token(jwt) do
    url = "https://oauth2.googleapis.com/token"

    payload = %{
      "grant_type" => "urn:ietf:params:oauth:grant-type:jwt-bearer",
      "assertion" => jwt
    }

    headers = [{"Content-Type", "application/x-www-form-urlencoded"}]
    body = URI.encode_query(payload)

    case HTTPoison.post(url, body, headers) do
      {:ok, %{status_code: 200, body: response_body}} ->
        case Jason.decode(response_body) do
          {:ok, %{"access_token" => token}} -> {:ok, token}
          {:ok, response} -> {:error, {:unexpected_response, response}}
          {:error, reason} -> {:error, {:json_decode_error, reason}}
        end

      {:ok, %{status_code: status_code, body: body}} ->
        {:error, {:http_error, status_code, body}}

      {:error, reason} ->
        {:error, {:request_error, reason}}
    end
  end

  defp get_application_default_credentials do
    # Try Cloud Run/GCE metadata service first (for production environments)
    case get_metadata_service_token() do
      {:ok, token} ->
        {:ok, token}

      {:error, _} ->
        # Fallback to gcloud CLI (for local development)
        get_gcloud_token()
    end
  end

  defp get_metadata_service_token do
    # Google Cloud metadata service endpoint
    url =
      "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"

    headers = [{"Metadata-Flavor", "Google"}]

    case HTTPoison.get(url, headers, timeout: 5000, recv_timeout: 5000) do
      {:ok, %{status_code: 200, body: body}} ->
        case Jason.decode(body) do
          {:ok, %{"access_token" => token}} -> {:ok, token}
          {:ok, response} -> {:error, {:unexpected_metadata_response, response}}
          {:error, reason} -> {:error, {:metadata_json_decode_error, reason}}
        end

      {:ok, %{status_code: status_code, body: body}} ->
        {:error, {:metadata_http_error, status_code, body}}

      {:error, %HTTPoison.Error{reason: :timeout}} ->
        {:error, :metadata_timeout}

      {:error, %HTTPoison.Error{reason: :nxdomain}} ->
        {:error, :metadata_not_available}

      {:error, reason} ->
        {:error, {:metadata_request_error, reason}}
    end
  end

  defp get_gcloud_token do
    case System.cmd("gcloud", ["auth", "application-default", "print-access-token"]) do
      {token, 0} ->
        {:ok, String.trim(token)}

      {error, _} ->
        {:error, {:gcloud_adc_error, error}}
    end
  end
end
