# Northstar Queue Client

## Connection

Create a client with `QueueClient(endpoint, token)`. The client accepts HTTPS endpoints and validates the token before sending a request.

## Retry policy

For HTTP 429 and 503 responses, the client retries a request up to **2 times** after the initial attempt. The delay starts at **200 milliseconds** and doubles after each retry, up to a maximum delay of **2 seconds**. Other HTTP errors are returned without an automatic retry.

## Batch processing

Submit at most **100 messages** in one batch. The client preserves the order of messages within a batch. A message is acknowledged only after its handler completes successfully.

## Local development

Set `NORTHSTAR_ENDPOINT` and `NORTHSTAR_TOKEN` in the local environment. Keep the token out of source control. The client reads these values when the application creates its configuration.
