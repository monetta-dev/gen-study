# [Basic Specification] Math Question Generation & Verification API System



## 1. System Overview

A Python-based backend API that automatically generates and verifies Japanese university entrance exam questions and solutions by using an OpenAI-compatible LLM API.



Two independent agents (LLMs), "Generator" and "Validator", collaborate within the system to ensure the quality of the question data and return it in JSON format.



---



## 2. System Configuration (Internal API Flow)



1. **API Request (User Input)**

   * Difficulty Level (e.g., Nitto-Koma-Sen, University of Tokyo, etc.)

   * Field/Topic (e.g., Probability, Calculus, etc.)

2. **Generator Agent**

   * Generates a question, solution, and explanation using the configured LLM provider.

3. **Validator Agent**

   * Reviews the output from the Generator Agent.

   * **Evaluation Criteria**:

     * Check for mathematical flaws, errors, or any insufficiency/redundancy in the given conditions.

     * **Verify alignment with the Japanese National Curriculum Guidelines (High School Mathematics scope).**

4. **API Response (Final Output)**

   * Returns a JSON object containing the verified question data (mathematical expressions in $\LaTeX$ format).



---



## 3. Agent Definitions



### ① Generator Agent

* **Role**: Creates original entrance exam questions (written/structured formats) tailored to the specified difficulty level and topic.

* **Model**: configured by environment variables such as `OPENROUTER_MODEL`



### ② Validator Agent

* **Role**: Cross-checks and filters the generated questions and solutions.

* **Model**: configured by environment variables such as `OPENROUTER_MODEL`

* **Mandatory Checklist**:

  1. **Accuracy of Solution**: Verify that there are no errors in formulas or calculations, and that a valid solution exists.

  2. **Curriculum Compliance**: Ensure the question does not require advanced knowledge beyond the Japanese High School Curriculum Guidelines (e.g., university-level mathematics theorems) to be solved.



---



## 4. API Input/Output Data Specifications (JSON)



### Request (Request Body)

```json

{

  "level": "University of Tokyo",

  "topic": "Probability Recurrence Relations"

}



### Response (Response Body) *When Verification PASSES

{

  "status": "success",

  "problem": "Question text (including $LaTeX$)",

  "solution": "Solution and explanation (including $LaTeX$)"

}
