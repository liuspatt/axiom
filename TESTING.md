# Testing Guide

## Authentication Tests

### Running Credential Tests

The `test/axiom_ai/auth_test.exs` file contains tests for loading credentials from files and AI provider integrations. These tests are designed to:

1. **Skip gracefully** if credential files don't exist
2. **Handle various error conditions** gracefully
3. **Test file loading functionality** without requiring real credentials
4. **Test provider integrations** with "hola" messages to:
   - Vertex AI (Google Cloud)
   - OpenAI
   - Anthropic Claude
   - DeepSeek AI
   - AWS Bedrock
   - Local AI endpoints

### Test Files

The tests use example credential files in `test/fixtures/`:

- `credentials.json` - Example Google Cloud service account key file

### Running Tests

```bash
# Run all auth tests
mix test test/axiom_ai/auth_test.exs

# Run specific test
mix test test/axiom_ai/auth_test.exs:8
```

### Test Behavior

The tests will:
- **Skip** if credential files are missing (with informative message)
- **Pass** if files exist but contain invalid credentials (expected for test environment)
- **Verify** file reading and parsing functionality
- **Handle** various error conditions (missing files, invalid JSON, invalid keys)
- **Test provider integrations** by sending "hola" messages and handling auth/connection failures gracefully

### Example Output

```
Running ExUnit with seed: 433194, max_cases: 24

Skipping Local AI test: LOCAL_AI_ENDPOINT not set
.Skipping OpenAI test: OPENAI_API_KEY not set
....Skipping DeepSeek test: DEEPSEEK_API_KEY not set
.Skipping Anthropic test: ANTHROPIC_API_KEY not set
.Skipping: Token generation failed (expected with test credentials)
...Skipping Bedrock test: AWS_ACCESS_KEY_ID not set
.
Finished in 0.08 seconds (0.00s async, 0.08s sync)
11 tests, 0 failures
```

The tests validate authentication flows and provider integrations without requiring actual API credentials, making them safe to run in any environment.

### Provider Integration Tests

Each provider integration test sends a "hola" message and handles authentication gracefully:

#### Vertex AI Integration
- **Skip** if `test/fixtures/credentials.json` doesn't exist
- **Skip** if authentication fails (expected with test credentials)
- **Pass** and show response if real GCP credentials are provided

#### OpenAI Integration  
- **Skip** if `OPENAI_API_KEY` environment variable is not set
- **Skip** if API key is invalid or rate limit exceeded
- **Pass** and show response if valid API key is provided

#### Anthropic Integration
- **Skip** if `ANTHROPIC_API_KEY` environment variable is not set  
- **Skip** if API key is invalid or rate limit exceeded
- **Pass** and show response if valid API key is provided

#### DeepSeek Integration
- **Skip** if `DEEPSEEK_API_KEY` environment variable is not set
- **Skip** if API key is invalid or rate limit exceeded  
- **Pass** and show response if valid API key is provided

#### AWS Bedrock Integration
- **Skip** if `AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY` not set
- **Skip** if AWS credentials are invalid or insufficient permissions
- **Skip** if model is not accessible or doesn't exist
- **Pass** and show response if valid AWS credentials and model access

#### Local AI Integration
- **Skip** if `LOCAL_AI_ENDPOINT` environment variable is not set
- **Skip** if endpoint is unreachable or authentication fails
- **Pass** and show response if local AI service is running and accessible

### Running with Real Credentials

To test with real API credentials, set the appropriate environment variables:

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For DeepSeek
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# For AWS Bedrock
export AWS_ACCESS_KEY_ID="your-aws-access-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
export AWS_DEFAULT_REGION="us-east-1"  # optional, defaults to us-east-1

# For Local AI
export LOCAL_AI_ENDPOINT="http://localhost:8080"

# Then run tests
mix test test/axiom_ai/auth_test.exs
```

### AWS Bedrock Models

The Bedrock integration supports multiple model families. The test uses Anthropic Claude by default, but you can test with other models:

```bash
# Example Bedrock model IDs:
# - anthropic.claude-3-haiku-20240307-v1:0 (default in test)
# - anthropic.claude-3-sonnet-20240229-v1:0
# - amazon.titan-text-express-v1
# - meta.llama2-70b-chat-v1
# - ai21.j2-ultra-v1
```

**Note:** You need appropriate IAM permissions to access Bedrock models. Ensure your AWS credentials have:
- `bedrock:InvokeModel` permission
- Access to the specific model you want to use