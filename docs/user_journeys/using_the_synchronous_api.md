# User Journey: Using the Synchronous API

## Persona

**Name:** John Doe  
**Role:** Synchronous API user  
**Goal:** Wants to calculate metrics synchronously for their LLM candidate responses.

---

## Journey Overview

The user has an application that uses LLMs and has already set up the llm scorecaster metrics evaluation tool for his organization. John now wants to use the synchronous API to calculate metrics for his LLM responses in real-time. The tool is already running and configured - he just needs to make API calls to evaluate text pairs and get immediate results for his chatbot evaluation system.

---

## Steps

1. **Entry Point**

   - John's chatbot generates a response that needs evaluation
   - He has the reference text (expected response) and candidate text (LLM-generated response)
   - Decides to use the synchronous API for immediate feedback

2. **API Request Preparation**

   - Constructs POST request to `/api/v1/metrics/evaluate`
   - Includes Authorization header with his organization's API key: `Bearer his-org-api-key`
   - Formats request body with text pairs and desired metrics: `["bert_score", "bleu_score", "rouge_score"]`

3. **API Call Execution**

   - Sends synchronous HTTP POST request to the running metrics service
   - Request includes one or more text pairs for evaluation
   - Waits for immediate response (typically 1-3 seconds)

4. **Response Processing**

   - Receives JSON response with calculated metrics for each text pair
   - Parses the structured results to extract specific metric scores
   - Reviews metrics like BERT score: 0.85, BLEU: 0.72, ROUGE-L: 0.78

5. **Decision Making**
   - Uses the calculated metrics to assess LLM response quality
   - Applies business logic based on metric thresholds (e.g., BERT score > 0.8 = good response)
   - Either accepts the response or triggers alternative actions based on the evaluation

---

## Emotions

- 🟢 Confident because the service is working reliably
- 🟢 Satisfied with the fast response times for real-time evaluation needs
- 🟡 Focused on interpreting the metric results correctly for decision making
- 🟢 Pleased with the structured, easy-to-parse JSON response format

---

## Pain Points

- Uncertainty about which metrics are most relevant for his specific use case
- Need to understand what constitutes "good" vs "poor" scores for each metric
- Occasional latency when evaluating longer text pairs or multiple metrics
- Deciding on appropriate score thresholds for automated decision making

---

## Opportunities

- Provide metric interpretation guides with score ranges and meanings
- Include recommended thresholds for different types of applications (chatbots, translation, etc.)
- Add response time estimates for different metrics and text lengths
- Offer batch processing recommendations when evaluating multiple pairs
- Include example use cases showing how to apply metrics for decision making

---

## Outcome

John successfully uses the synchronous API to evaluate his LLM responses in real-time. He receives immediate metric calculations that help him assess response quality and make informed decisions about whether to accept, modify, or regenerate responses. The fast, reliable API calls integrate seamlessly into his application workflow, providing the quantitative feedback he needs for automated quality control of his chatbot system.
