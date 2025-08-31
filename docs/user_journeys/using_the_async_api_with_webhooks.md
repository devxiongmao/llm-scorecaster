# User Journey: Using the Asynchronous API with Webhooks

## Persona

**Name:** Sarah Chen  
**Role:** Asynchronous API user with webhook integration  
**Goal:** Wants to calculate metrics asynchronously for batch analysis while receiving automatic notifications when results are ready, eliminating the need for polling.

---

## Journey Overview

Sarah has evolved her content generation system to be more event-driven. She still needs to evaluate hundreds of LLM responses daily, but now wants her system to automatically react when metrics are ready rather than constantly checking job status. By providing a webhook URL, her application can continue other work and respond immediately when evaluation results become available, creating a more efficient and responsive analytics pipeline.

---

## Steps

1. **Entry Point**

   - Sarah has accumulated a batch of 500+ text pairs from the day's content generation
   - She wants to calculate comprehensive metrics for trend analysis and reporting
   - Chooses async API with webhook to create an event-driven evaluation workflow

2. **Webhook Endpoint Setup**

   - Configures her application to expose a webhook endpoint: `https://sarah-app.com/api/webhooks/metrics-complete`
   - Implements endpoint to handle incoming POST requests with job results
   - Sets up proper authentication/validation to ensure requests are from the metrics service
   - Tests endpoint is accessible and properly configured

3. **Batch Job Submission with Webhook**

   - Constructs POST request to `/api/v1/async/evaluate`
   - Includes Authorization header with organization's API key: `Bearer org-analytics-key`
   - Formats request body with text pairs, multiple metrics: `["bert_score", "bleu", "rouge", ...]`
   - **Adds webhook configuration:** `"webhook_url": "https://sarah-app.com/api/webhooks/metrics-complete"`

4. **Job Initialization**

   - Sends asynchronous HTTP POST request to the running metrics service
   - Receives immediate response with job ID and status: `{"job_id": "abc-123", "status": "pending", "webhook_configured": true}`
   - Stores job ID for reference and continues with other application tasks
   - No longer needs to implement polling logic

5. **Automatic Results Delivery**

   - Metrics service processes the job in the background
   - Upon completion, service automatically sends POST request to Sarah's webhook URL
   - Webhook payload contains: `{"job_id": "abc-123", "status": "completed", "results": {...}, "timestamp": "2025-08-31T10:30:00Z"}`
   - Sarah's application receives notification immediately when results are ready

6. **Webhook Processing**

   - Sarah's webhook endpoint receives and validates the incoming request
   - Extracts job results from the webhook payload
   - Triggers downstream processing: data analysis, visualization, and trend tracking

---

## Emotions

- 🟢 Delighted with the event-driven workflow that eliminates manual checking
- 🟢 Relieved that her application can focus on other tasks without polling overhead
- 🟢 Confident in the automatic delivery of results when they're ready
- 🟢 Excited about building more responsive, real-time analytics workflows

---

## Pain Points

- Initial complexity of setting up and securing webhook endpoints
- Need to handle webhook delivery failures or retries gracefully
- Ensuring webhook endpoint is always available and properly authenticated
- Managing webhook payload validation and potential security concerns
- Debugging webhook delivery issues when notifications don't arrive

---

## Opportunities

- Provide webhook delivery retry logic with exponential backoff
- Include webhook signature validation for enhanced security
- Offer webhook payload customization (e.g., include/exclude specific fields)
- Add webhook delivery status tracking and failure notifications
- Support multiple webhook URLs for different event types or environments
- Provide webhook testing tools to validate endpoint configuration

---

## Outcome

Sarah has successfully implemented an event-driven LLM evaluation system using webhooks. Her application now automatically processes evaluation results as soon as they're available, creating a more responsive analytics pipeline. The elimination of polling reduces system overhead and allows her to build real-time dashboards and alerts. Sarah can now scale her content analysis system more efficiently, with immediate insights into LLM performance enabling faster model improvements and more timely executive reporting on content quality metrics.