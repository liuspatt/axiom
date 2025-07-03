defmodule AxiomAi.IntegrationTest do
  use ExUnit.Case
  import Mock

  describe "AxiomAi integration with Vertex AI" do
    test "chat workflow with access token" do
      config = %{
        project_id: "test-project",
        access_token: "test-token"
      }

      # Mock successful Vertex AI API response
      with_mock HTTPoison, [:passthrough],
        post: fn _url, _body, _headers, _opts ->
          {:ok,
           %{
             status_code: 200,
             body:
               Jason.encode!(%{
                 "candidates" => [
                   %{
                     "content" => %{
                       "parts" => [%{"text" => "Hello! I'm a helpful AI assistant."}]
                     }
                   }
                 ]
               }),
             headers: []
           }}
        end do
        client = AxiomAi.new(:vertex_ai, config)

        assert {:ok, response} = AxiomAi.chat(client, "Hello, how are you?")
        assert response.response == "Hello! I'm a helpful AI assistant."
      end
    end

    test "completion workflow with access token" do
      config = %{
        project_id: "test-project",
        access_token: "test-token"
      }

      # Mock successful Vertex AI API response
      with_mock HTTPoison, [:passthrough],
        post: fn _url, body, _headers, _opts ->
          # Verify the request contains generation config
          decoded_body = Jason.decode!(body)
          assert Map.has_key?(decoded_body, "generationConfig")

          {:ok,
           %{
             status_code: 200,
             body:
               Jason.encode!(%{
                 "candidates" => [
                   %{
                     "content" => %{
                       "parts" => [%{"text" => "blue and beautiful today."}]
                     }
                   }
                 ]
               }),
             headers: []
           }}
        end do
        client = AxiomAi.new(:vertex_ai, config)

        assert {:ok, response} = AxiomAi.complete(client, "The sky is", %{max_tokens: 50})
        assert response.completion == "blue and beautiful today."
      end
    end

    test "handles API error responses" do
      config = %{
        project_id: "test-project",
        access_token: "test-token"
      }

      # Mock API error response
      with_mock HTTPoison, [:passthrough],
        post: fn _url, _body, _headers, _opts ->
          {:ok,
           %{
             status_code: 400,
             body:
               Jason.encode!(%{
                 "error" => %{
                   "code" => 400,
                   "message" => "Invalid request format"
                 }
               }),
             headers: []
           }}
        end do
        client = AxiomAi.new(:vertex_ai, config)

        assert {:error, error_response} = AxiomAi.chat(client, "Hello")
        assert error_response.status_code == 400
      end
    end

    test "handles network errors" do
      config = %{
        project_id: "test-project",
        access_token: "test-token"
      }

      # Mock network error
      with_mock HTTPoison, [:passthrough],
        post: fn _url, _body, _headers, _opts ->
          {:error, %HTTPoison.Error{reason: :timeout}}
        end do
        client = AxiomAi.new(:vertex_ai, config)

        assert {:error, :timeout} = AxiomAi.chat(client, "Hello")
      end
    end

    test "chat with ADC authentication" do
      config = %{
        project_id: "test-project"
        # No explicit auth - should use ADC
      }

      # Mock gcloud ADC command
      with_mock System, [:passthrough],
        cmd: fn "gcloud", ["auth", "application-default", "print-access-token"] ->
          {"adc-token-12345\n", 0}
        end do
        # Mock successful API response
        with_mock HTTPoison, [:passthrough],
          post: fn _url, _body, headers, _opts ->
            # Verify ADC token is used
            auth_header = Enum.find(headers, fn {key, _} -> key == "Authorization" end)
            assert {"Authorization", "Bearer adc-token-12345"} = auth_header

            {:ok,
             %{
               status_code: 200,
               body:
                 Jason.encode!(%{
                   "candidates" => [
                     %{
                       "content" => %{
                         "parts" => [%{"text" => "Response using ADC auth"}]
                       }
                     }
                   ]
                 }),
               headers: []
             }}
          end do
          client = AxiomAi.new(:vertex_ai, config)

          assert {:ok, response} = AxiomAi.chat(client, "Hello with ADC")
          assert response.response == "Response using ADC auth"
        end
      end
    end

    test "service account authentication flow" do
      config = %{
        project_id: "test-project",
        # Simplified - using direct token
        access_token: "service-account-token"
      }

      # Mock Vertex AI API response
      with_mock HTTPoison, [:passthrough],
        post: fn _url, _body, headers, _opts ->
          # Verify service account token is used
          auth_header = Enum.find(headers, fn {key, _} -> key == "Authorization" end)
          assert {"Authorization", "Bearer service-account-token"} = auth_header

          {:ok,
           %{
             status_code: 200,
             body:
               Jason.encode!(%{
                 "candidates" => [
                   %{
                     "content" => %{
                       "parts" => [%{"text" => "Response using service account"}]
                     }
                   }
                 ]
               }),
             headers: []
           }}
        end do
        client = AxiomAi.new(:vertex_ai, config)

        assert {:ok, response} = AxiomAi.chat(client, "Hello with service account")
        assert response.response == "Response using service account"
      end
    end

    test "validates endpoint URL construction" do
      config = %{
        project_id: "my-test-project",
        region: "europe-west1",
        model: "gemini-1.0-pro",
        access_token: "test-token"
      }

      # Mock and verify the endpoint URL
      with_mock HTTPoison, [:passthrough],
        post: fn url, _body, _headers, _opts ->
          # Verify the URL is constructed correctly
          expected_url =
            "https://europe-west1-aiplatform.googleapis.com/v1/projects/my-test-project/locations/europe-west1/publishers/google/models/gemini-1.0-pro:generateContent"

          assert url == expected_url

          {:ok,
           %{
             status_code: 200,
             body:
               Jason.encode!(%{
                 "candidates" => [
                   %{
                     "content" => %{
                       "parts" => [%{"text" => "Custom region response"}]
                     }
                   }
                 ]
               }),
             headers: []
           }}
        end do
        client = AxiomAi.new(:vertex_ai, config)

        assert {:ok, response} = AxiomAi.chat(client, "Hello")
        assert response.response == "Custom region response"
      end
    end

    test "chat with service account credentials from fixtures" do
      # Load service account credentials from fixtures
      credentials_path = Path.join([__DIR__, "..", "fixtures", "credentials.json"])
      {:ok, credentials_json} = File.read(credentials_path)
      credentials = Jason.decode!(credentials_json)

      config = %{
        project_id: credentials["project_id"],
        service_account_path: credentials_path,
        region: "us-central1",
        model: "gemini-2.5-flash",
        temperature: 0.5,
        max_tokens: 65536
      }

      client = AxiomAi.new(:vertex_ai, config)

      assert {:ok, response} = AxiomAi.chat(client, "Hello with fixture credentials")
      assert response.response != ""
    end
  end
end
