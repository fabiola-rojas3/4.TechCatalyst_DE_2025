## Lab Activity: Choosing the Right Spark Engine

<img src="images/image-20250731165032544.png" alt="image-20250731165032544" style="zoom:50%;" />

### Context

Congratulations! Your consulting team's previous recommendation to use **Delta Lake** on S3 has been accepted by **"Streamify."** They have successfully implemented it and are seeing massive improvements in data quality and query performance.

Now, they face a new strategic decision. They've been running all their workloads on the Databricks Free Edition for prototyping, but as they move to production, they need to select their official, enterprise-grade compute platform.

The leadership is torn between two options:

1. **Databricks on AWS:** The platform their team already knows and likes.
2. **Amazon EMR:** AWS's native big data platform, which the finance department believes might be cheaper.



### Your Mission 🧑‍💻

As the lead data engineering consultant, you must now analyze these two platforms and write a formal recommendation memo to Streamify's CTO. Your recommendation should be based on a balanced analysis of technical capabilities, developer productivity, and total cost of ownership.

------



### 1. Research & Analysis Phase

Investigate the two platforms. Since you are already familiar with Databricks, focus much of your research on understanding Amazon EMR and how it differs.

#### The Candidates:

- **Platform A: Databricks on AWS**
- **Platform B: Amazon EMR**
  - **Amazon EMR** (**traditional**): The traditional EMR where you provision and manage clusters
  - **Amazon EMR Serverless**: The serverless option where you just submit your Spark job and AWS automatically provisions, manages, and scales the necessary resources for the duration of the job

#### Key Evaluation Criteria:

Your analysis must compare the platforms across the following dimensions.

- **Ease of Use & Developer Productivity:**
  - How easy is it to launch and manage a Spark cluster in each platform?
  - Compare the development environments (e.g., Databricks Notebooks vs. options for EMR like EMR Studio or connecting a local IDE).
  - How does collaboration work on each platform?
- **Performance:**
  - What performance optimizations does each platform offer out-of-the-box? (e.g., Databricks Photon engine, EMR's optimized runtime for Spark).
  - How flexible is the underlying hardware selection (CPU, memory, GPU instances)?
- **Cost Management:**
  - What is the pricing model for each? (e.g., Databricks DBUs + AWS Cloud cost vs. EMR service charge + EC2 instance cost).
  - How easy is it to leverage AWS Spot Instances on each platform to save money?
  - Which platform is likely to have a lower *sticker price*, and which might have a lower *total cost of ownership*when you factor in developer time and operational overhead?
- **Management & Operations:**
  - Who is responsible for software updates, security patching, and environment configuration? Compare the "managed" nature of Databricks to the "self-service" nature of EMR.
  - How is monitoring and debugging handled on each platform?
- **Ecosystem Integration:**
  - How seamlessly does each platform integrate with the services your students know: **S3 (for storage)** and **AWS Glue Catalog (as the metastore)**?
  - What are the differences in how they handle security and access control (e.g., Databricks Unity Catalog vs. AWS IAM and Lake Formation)?

------



### 2. Deliverable: The Recommendation Memo

Draft a one-page memo addressed to the CTO of Streamify. The memo must have the following structure:

1. **Executive Summary:** A brief paragraph stating your final recommendation and the primary reason.
2. **Comparative Analysis:** A summary of your findings, organized by the evaluation criteria above (Ease of Use, Performance, Cost, etc.). A comparison table is highly encouraged here.
3. **Final Recommendation & Justification:**
   - Clearly state which platform (Databricks or EMR) you recommend for Streamify's production workloads.
   - Justify your choice by explicitly linking it to Streamify's implied needs: a team that values productivity and speed but is also under pressure to manage costs. There is no single "right" answer; the strength of the justification is what matters.



### In-Class Discussion Points

After the memos are complete, we will use these questions to facilitate a class discussion:

- Under what circumstances would you **definitely** choose EMR, even if the team knew Databricks better? (e.g., extreme cost constraints, need for deep customization, using a framework other than Spark).
- Under what circumstances is paying the "premium" for Databricks easily justified? (e.g., speed to market is critical, small operations team, focus on data science and ML collaboration).
- How does the concept of a **"Managed Service"** (Databricks) vs. an **"Infrastructure Service"** (EMR) change the role of a Data Engineer? How about the **Serverless** option in Amazon EMR?
