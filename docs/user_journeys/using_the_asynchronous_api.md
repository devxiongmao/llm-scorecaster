# User Journey: Using the Asynchronous API

## Persona
**Name:** Sarah Chen
**Role:** Asynchronous API user  
**Goal:** Wants to calculate metrics asynchronously for batch analysis and tracking LLM performance over time.

---

## Journey Overview
The user has an application that generates large volumes of LLM responses and wants to analyze performance trends over time. Sarah works on a content generation system that produces hundreds of articles daily using LLMs. She needs to evaluate these responses in batches to track quality metrics like BERT score and ROUGE over time for reporting and model improvement insights. The asynchronous API allows her to submit large evaluation jobs without blocking her application.

---

## Steps

1. **Entry Point**
   - Sarah has accumulated a batch of 500+ text pairs from the day's content generation
   - She wants to calculate comprehensive metrics for trend analysis and reporting
   - Chooses async API to avoid long wait times and potential timeouts

2. **Batch Job Submission**
   - Constructs POST request to `/api/v1/metrics/evaluate-async`
   - Includes Authorization header with organization's API key: `Bearer org-analytics-key`
   - Formats large request body with text pairs and multiple metrics: `["bert_score", "bleu", "rouge_l", "align_score"]`

3. **Job Initialization**
   - Sends asynchronous HTTP POST request to the running metrics service
   - Receives immediate response with job ID and status: `{"job_id": "abc-123", "status": "pending"}`
   - Stores job ID for later retrieval and continues with other tasks

4. **Status Monitoring**
   - Periodically checks job status via GET `/api/v1/jobs/{job_id}/status`
   - Monitors progression: pending → processing → completed (or failed)
   - May implement polling every 30 seconds or webhook-based notifications

5. **Results Retrieval**
   - Once status shows "completed", calls GET `/api/v1/jobs/{job_id}/results`
   - Downloads comprehensive JSON response with all calculated metrics
   - Processes results for data analysis, visualization, and trend tracking

6. **Data Analysis**
   - Imports results into analytics pipeline or data warehouse
   - Calculates aggregate statistics, trends, and performance insights
   - Generates reports showing metric evolution over time periods

---

## Emotions
- 🟢 Relieved that large batch jobs don't block her application workflow
- 🟢 Satisfied with the ability to process hundreds of evaluations efficiently
- 🟡 Slightly impatient waiting for large jobs to complete processing
- 🟢 Excited about the comprehensive data for trend analysis and insights

---

## Pain Points
- Uncertainty about how long different sized jobs will take to process
- Need to implement polling logic to check job completion status
- Managing multiple concurrent jobs and their respective job IDs
- Handling job failures or timeouts gracefully in batch processing workflows

---

## Opportunities
- Provide job processing time estimates based on batch size and selected metrics
- Add webhook support for job completion notifications instead of polling
- Include progress indicators showing percentage completion for long-running jobs
- Offer job priority queuing for urgent vs. routine batch analyses
- Provide batch size recommendations for optimal processing efficiency

---

## Outcome
Sarah successfully processes large batches of LLM evaluations using the asynchronous API. She can now track performance trends across her content generation system, identifying patterns like declining BERT scores that indicate model drift or improving ROUGE scores after fine-tuning. The async processing allows her to maintain application performance while gathering the comprehensive metrics data needed for continuous improvement and executive reporting on content quality.