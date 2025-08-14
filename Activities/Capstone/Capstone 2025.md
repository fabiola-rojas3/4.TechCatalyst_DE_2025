## **TechCatalyst Data Engineering Capstone Project – 2025**

<img src="images/route-fifty-lead-image.jpg" alt="Taxi medallion owners are still drowning under bad loans - City & State New  York" style="zoom:67%;" />

### **Project Overview**

In this capstone project, you will work with **NYC Taxi trip data (2024–2025)** and other real-world NYC datasets.
Your mission: build a complete data pipeline from ingestion to analysis, delivering actionable insights for a **use case of your choice**.

You will:

1. **Extract** data from the `Techcatalyst-RAW` S3 bucket.
2. **Clean & Transform** it into analysis-ready form.
3. **Store** the transformed data in `Techcatalyst-TRANSFORMED` S3 bucket.
4. **Load** it into Snowflake for querying and analysis.
5. (Optional) Enrich with **external datasets** or apply ML/AI for predictions.
6. **Present** your findings through dashboards, reports, or applications.

Think of yourself as a **data consultant** building a solution for a real-world client — the insights should be **clear, data-driven, and supported by visualizations**.

------

### **Available Data**

All files are stored in `Techcatalyst-RAW` S3 bucket.

**Primary: Taxi Trip Data**

- Yellow Taxi Trip Data – 12 months (Parquet)
- Green Taxi Trip Data – 12 months (Parquet)
- For Hire Vehicle Trip Data – 6 months (Parquet)
- High Volume For Hire Vehicle Trip Data – 6 months (Parquet)
- Taxi Zone Lookup (CSV)
- Taxi Zone Shape Files (Parquet) & Borough Zone Maps (JPGs)

**Additional NYC Datasets** (CSV format) – all 2024–2025

- NYC Air Quality Data
- Construction Related Incidents
- Motor Vehicle Collisions – People
- Motor Vehicle Collisions – Vehicles
- Motor Vehicle Collisions – Crashes
- TLC New Driver Application Status

**Metadata & Lookup Tables** from NYC Open Data dictionaries are also provided.

> [!NOTE]
>
> You can find more datasets from NYC Open Data (the data in S3 are sourced from there)
>
> https://opendata.cityofnewyork.us

> [!TIp]
>
> In NYC Open Data site they have a page dedicated on how people used their data. Please use this for **inspiration only**.



------

### **Core Requirements**

1. **Data Ingestion (RAW → Processing)**
   - Access the relevant files from `s3://capstone-techcatalyst-raw`.
   - Decide on your **use case** and select additional datasets to integrate.
2. **Data Transformation & Cleaning**
   - Handle missing values, duplicates, and incorrect types.
   - Add derived columns (e.g., trip duration, day-of-week, month). 
     - Based on your concept and idea, you need to determine what additional features you need to created. The concept is similar to Machine Learning Feature Engineering were you created derived columns from other columns. Be creative to cover all that you need to make an impactful analysis. 
   - Partition and store cleaned datasets in `s3://capstone-techcatalyst-transformed`.
   - Document **data quality checks**.
     - You can select any of the ETL technologies and techniques we covered including: **DuckDB**, **AWS Glue**, **AWS DataBrew**, **Databricks** (PySpark), **Pandas**, **Dask**, **Polars**, and **Modin** for handling big datasets 
3. **Integration of Datasets**
   - Join taxi data with selected additional datasets.
   - Use spatial joins with taxi zone shapefiles where relevant.
4. **Snowflake Loading**
   - Create Snowflake stages, load transformed data into tables.
   - Enrich with lookup tables for readable outputs (e.g., payment type, rate code).
   - Create **views** tailored to your use case.
5. **Analysis & Insights**
   - Write at least **10 meaningful SQL queries** in Snowflake.
   - Present findings using Tableau, and you can supplement it with Python visualizations, or using an application UI like Gradio 
6. **(Optional) ML/AI Enhancement**
   - Use AWS Bedrock, or Ollama , PyCaret for regression, clustering, or classification. It also supports time series forecasting and anomaly detection 
     - To learn more about PyCaret modules visit their main documentation here: https://pycaret.gitbook.io/docs/get-started/modules 
   - Deploy an interactive dashboard/visualization using Tableau. Optionally, you can add visualization using Python or Gradio 
     - To learn how to create plots in Gradio check out: https://www.gradio.app/guides/creating-plots 
     - To learn more about `hvPlot` a Python wrapper for many different visualization libraries visit their documentation here https://hvplot.holoviz.org/en/docs/latest/ 
     - If you want to work with Geo spatial data you can use **GeoPanads** and you can learn and read about it here https://geopandas.org/en/stable/docs/user_guide.html 
7. **Documentation & Presentation**
   - README in GitHub with setup instructions.
   - Reference architecture diagram.
   - Performance metrics (ETL time, data size).
   - Future state proposal for scaling/improvement.

> [!NOTE]
>
> ### **Storytelling & Presentation Expectations**
>
> Although this is primarily a **Data Engineering project**, your success will also be measured by your ability to **communicate your findings effectively** to a **diverse audience**: technical peers, business stakeholders, and decision-makers.
>
> Your final presentation should:
>
> - **Tell a Compelling Story** – Frame your work around a clear problem statement, approach, and insights. Help your audience understand *why* your project matters.
> - **Balance Technical and Non-Technical Communication** –
>   - For the technical audience: clearly show your architecture, pipeline design, ETL decisions, and optimization strategies.
>   - For the business audience: focus on key insights, business impact, and actionable recommendations without overwhelming them with jargon.
> - **Showcase Your Skills Holistically** – This project is your chance to demonstrate both **data engineering expertise***and* your ability to perform meaningful **analytics and interpretation**.
> - **Demonstrate Team Cohesion** – Present as a unified team. Divide speaking roles thoughtfully so that everyone is seen as a contributor and the flow feels seamless.
> - **Use Strong Visual Aids** – Leverage dashboards, charts, maps, diagrams, and animations to make your points memorable.
> - **Engage Your Audience** – Invite questions, prepare for both technical and business-oriented queries, and ensure your answers connect back to your data.

> [!TIP]
>
> ### **Presentation Deliverable Requirements**
>
> 1. **Duration:** 15–20 minutes + 5 minutes Q&A.
> 2. **Structure:**
>    1. **Introduction** – Problem, use case, and why it matters.
>    2. **Approach** – Data sources, transformations, architecture, and methodology.
>    3. **Insights** – Key findings, visualizations, and business impact.
>    4. **Technical Deep Dive** – Tools, optimizations, and performance considerations.
>    5. **Future State** – Improvements, scalability, and additional possibilities.
>    6. **Conclusion** – Final recommendations and takeaways.
> 3. **Team Participation:** Every member must speak and contribute to the Q&A.
> 4. **Supporting Materials:**
>    - Architecture diagram
>    - Key SQL queries or transformation snippets
>    - Screenshots or live demos of dashboards/models
>    - Data storytelling visuals (charts, maps, etc.)
> 5. **Clarity Over Complexity:** Your goal is not just to impress with complexity, but to **inspire confidence that your work is valuable, scalable, and insightful**.

------

### **Suggested Use Cases**

(You may choose or design your own)

- **Peak Demand & Pricing Strategy** – Identify when and where taxi demand spikes, with suggestions for driver allocation.
- **Safety Risk Analysis** – Combine trip routes with collision data to flag dangerous areas/times.
- **Environmental Insights** – Correlate air quality readings with ride volume and patterns.
- **Impact of Construction on Mobility** – Study how construction incidents influence trip times and detours.
- **Driver Behavior & Performance** – Link TLC application data with trip characteristics and performance metrics.
- **Neighborhood Profiles** – Create mobility profiles per borough or zone, integrating collision, air quality, and construction data.

----

### **Evaluation Criteria**

| Area                           | Weight | Description                                                  |
| ------------------------------ | ------ | ------------------------------------------------------------ |
| Data Pipeline                  | 25%    | Completeness, efficiency, reproducibility                    |
| Data Cleaning & Transformation | 20%    | Quality of preprocessing and enrichment                      |
| Dataset Integration            | 15%    | Effective use of additional datasets                         |
| Snowflake Queries              | 15%    | Relevance, complexity, and insightfulness                    |
| Final Insights & Deliverables  | 10%    | Clarity, creativity, and storytelling                        |
| Documentation & Presentation   | 15%    | Ability to communicate technical and business perspectives effectively |

---

## **Suggested Use Cases for NYC Taxi & Related Datasets (2025)**

> **Note:**
> You may choose one of these or design your own. Creativity is encouraged, especially if you integrate multiple datasets or external data sources.

### **1. General Transportation Insights**

- **Trip Duration & Distance Patterns** – Explore how trip duration and distance vary by borough, taxi type, and time of day.
- **Fare Optimization** – Examine fare variations by trip length, payment type, and pickup zones to identify anomalies or optimization opportunities.
- **Traffic & Congestion Patterns** – Detect peak congestion periods using trip speeds and delays.
- **Demand Hotspots** – Map demand patterns across NYC zones and compare between taxi types.

------

### **2. Cross-Dataset Analysis (New for 2025)**

- **Safety Risk Mapping** – Combine taxi trip routes with *Motor Vehicle Collision* data to highlight high-risk areas and times.
- **Construction Impact on Mobility** – Use *Construction Incident* data to measure delays or route changes in affected areas.
- **Environmental Impact** – Correlate trip volumes with *Air Quality* data, exploring emissions and traffic-related pollution patterns.
- **Driver Workforce Analysis** – Link *TLC New Driver Application* data with trip metrics to explore onboarding trends and their effect on service availability.

------

### **3. Event & External Factor Impact**

- **Tourism & Event Influence** – Identify tourism hotspots and study the effect of events on trip demand and pricing.
- **Weather & Seasonality** – Integrate external weather data to study its effect on taxi usage patterns.
- **Nightlife Economy** – Explore nighttime trip trends and their economic implications.

------

### **4. Urban Planning & Sustainability**

- **Public Transport Gaps** – Use taxi demand to infer underserved public transit zones.
- **Accessibility Analysis** – Identify boroughs or neighborhoods with limited service coverage.
- **EV Charging Location Planning** – Propose optimal EV charger placement using trip start/end clusters.

------

### **5. Business & Operations**

- **Driver Allocation Strategies** – Suggest optimal driver deployment schedules based on demand patterns.
- **Customer Segmentation** – Group customers by trip characteristics for targeted services.
- **Fraud & Anomaly Detection** – Detect suspicious trip or fare patterns.

------

### **6. Public Safety & Emergency Planning**

- **Collision Hotspot Detection** – Visualize areas with high accident frequencies from collision datasets.
- **Emergency Evacuation Efficiency** – Analyze traffic flow during incidents or high-demand emergencies.
- **Disaster Response** – Study transportation system resilience during extreme weather or infrastructure failures.

------

### **Additional Notes for Students**

- Mix and match datasets — insights often come from joining data sources.
- Incorporate **spatial analysis** with taxi zone shapefiles for geospatial insights.
- Visualize your findings — clear visual storytelling makes results far more impactful.
- Tie your findings to **actionable recommendations** — think like a consultant.