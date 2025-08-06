# **Activity: Researching AWS AI/ML Services for Data Pipeline Solutions**

### It’s all about researching and less familiar (or unknown) be more familiar 

**Objective:** Students will research popular AWS AI and Machine Learning services using official documentation, identify a practical business problem, and design a robust data pipeline and solution architecture to solve it.

> [!NOTE]
>
> * Before you can build a house, you need the blueprint. A great data engineer is an **architect** first—they research the terrain, understand the goal, and select the right materials. This activity is about drawing that blueprint.
> * Data engineering isn't about knowing every tool; it's about knowing how to **find the right tool to solve a specific problem**. Your most valuable skill is the ability to research a challenge and design a solution. That's our entire focus here.
>
> **Research** $\rightarrow$ **Design** $\rightarrow$ __Code__

**Instructions for Groups**

**Step 1: Research AWS AI/ML Services** For each service below, use the official AWS documentation to understand its purpose and answer the "Research Question." The goal is to learn what the service does, what data it needs (input), and what data it produces (output).

#### Vision & Document Intelligence

- **AWS Rekognition:** Analyzes images and videos for object detection, facial analysis, and text recognition.
  - **Research Question:** According to the AWS documentation for the `DetectLabels` API, what are some key fields returned in the JSON response? How would you determine an object's name and the confidence score for that detection?
- **AWS Textract:** Extracts text, handwriting, and data from scanned documents like PDFs and images.
  - **Research Question:** Consult the Textract documentation. What is the difference in the API response structure when using `AnalyzeDocument` for 'FORMS' versus 'TABLES' feature types?

#### Language, Speech, & Generative AI

- **AWS Comprehend:** Uses Natural Language Processing (NLP) to find insights like sentiment and entities in text.
  - **Research Question:** Look at the API reference for `DetectSentiment`. What information is contained in the `SentimentScore` object, and what are the possible values for the `Sentiment` field?
- **AWS Transcribe:** Converts speech to text.
  - **Research Question:** Review the documentation for Amazon Transcribe jobs. What is the standard output format of a completed transcription job, and where is it typically stored?
- **AWS Translate:** Provides language translation.
  - **Research Question:** What are the three required parameters for making a basic `TranslateText` API call in Amazon Translate?
- **AWS Polly:** Turns text into lifelike speech.
  - **Research Question:** According to the Amazon Polly documentation, what are Speech Synthesis Markup Language (SSML) tags, and how can they be used to change the speech output? Provide one example.
- **AWS Bedrock:** Provides access to a range of high-performing foundation models (FMs) for generative AI tasks.
  - **Research Question:** Read the AWS documentation for Bedrock. What is the key difference between the `InvokeModel` and `InvokeModelWithResponseStream` API actions?



#### Data Governance, Search, & Recommendations

- **AWS Macie:** A data security service that uses ML to discover and protect sensitive data in Amazon S3.
  - **Research Question:** Based on the documentation, what are "managed data identifiers" in Amazon Macie, and how do they help in discovering Personally Identifiable Information (PII)?
- **Amazon Kendra:** An intelligent enterprise search service.
  - **Research Question:** List three different types of native data source connectors available for Amazon Kendra, according to its documentation.
- **Amazon Personalize:** A service for building real-time recommendation engines.
  - **Research Question:** According to the Amazon Personalize developer guide, what are the three mandatory attributes required for an `Interactions` dataset schema?

#### Custom Machine Learning

- **Amazon SageMaker:** The core AWS platform to build, train, and deploy custom machine learning models at scale.
  - **Research Question:** Read the SageMaker documentation. Describe the primary purpose of 'SageMaker Pipelines' and how it differs from calling a pre-trained AI service API like Rekognition.

**Step 2: Identify a Business Problem & Brainstorm a Solution** As a group, identify a specific business problem that could be solved by enriching a data stream. For example:

- Automatically moderating and tagging user-submitted images (this is called content moderation).
- Analyzing call center audio to find the root cause of customer dissatisfaction.
- Redacting sensitive PII from legal documents before they are sent to an analyst.
- Creating a searchable archive of recorded company video meetings.
- **Make sure to define your business problem - you can define your own and be creative**

Brainstorm a solution that integrates one or more of the services you researched.

**Step 3: Design the Solution Architecture** Draft a high-level architecture diagram for your solution. Your diagram must show the flow of data and include the following components: **This is how you plan to solve the business problem**

- **Data Ingestion:** How does data enter your system? (e.g., Amazon S3, AWS Kinesis, AWS API Gateway)
- **Data Processing & Enrichment:** Where does transformation happen? Clearly show where the AWS AI/ML service is called. (e.g., an AWS Lambda function that triggers on an S3 upload and calls Rekognition).
- **Data Storage:** How are the raw data and the enriched results stored? (e.g., Amazon S3, Amazon DynamoDB, Amazon RDS).
- **Data Consumption:** How is the final result used? (e.g., an analyst querying with Amazon Athena, a dashboard in Amazon QuickSight, an application API).

**Step 4: Justify Your Design** Prepare a short summary to accompany your diagram that addresses the following critical data engineering considerations:

1. **Service Choice:** Why did you select your chosen AI/ML service(s)? Justify why you chose a pre-built service instead of building a custom model with SageMaker (or vice-versa).
2. **Error Handling:** How would your pipeline theoretically handle a potential failure or error from the AI service API?
3. **Scalability & Cost:** Based on the service's pricing model in the documentation, how would your architecture plan for scaling, and what are the cost implications?
4. **Data Formats:** Based on your research, what data transformations would be necessary before sending data to the AI service?
