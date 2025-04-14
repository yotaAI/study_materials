**Week 1 – Lecture Part 1: Introduction to Data Engineering**  
---

### 🔍 What is Data Engineering?

Data Engineering is the backbone of any data-driven organization. It’s about **designing, building, and maintaining the systems and architecture** that allow raw data to be transformed into usable insights.

> **Analogy:** Think of data engineers as the plumbers of data — they build and maintain the pipelines that carry water (data) to households (analytics and data science teams).

---

### 📌 Key Responsibilities of a Data Engineer:

| Role | What it Means |
|------|---------------|
| Data Pipeline Builder | Ingest raw data from various sources and move it to storage |
| Data Modeler | Organize data into usable structures |
| Infrastructure Engineer | Set up tools like Kafka, Airflow, and cloud services |
| Quality Controller | Ensure clean, validated, consistent data |
| Collaborator | Work with analysts, scientists, and stakeholders |

---

### ⚙️ Where DE Fits in the Data Ecosystem

```
[ Raw Data (APIs, Logs, DBs) ] --> [ Ingestion Layer ] --> [ Data Lake/Warehouse ] --> [ BI Tools & DS Models ]
                                   ^                     ^
                            (Kafka, Airflow, etc.)   (dbt, SQL modeling)
```

---

### 🧱 Core Pillars of Data Engineering

1. **Data Ingestion**: Pulling in data from different sources
2. **Data Processing**: Cleaning, transforming, enriching (ETL/ELT)
3. **Storage**: Data Lakes, Warehouses, and Databases
4. **Orchestration**: Scheduling and managing workflows (Airflow, Prefect)
5. **Monitoring**: Ensuring pipeline health and data quality

---

### 🛠️ Common Tools in DE Stack

| Category | Tools |
|---------|--------|
| Programming | Python, SQL |
| Pipelines | Airflow, Luigi, dbt |
| Big Data | Apache Spark, Kafka |
| Storage | S3, BigQuery, Snowflake, Redshift |
| Monitoring | Prometheus, Grafana, OpenLineage |
| Infra | Docker, Terraform, GitHub Actions |

---

### ✅ Assignment for Week 1 (Let’s Get Practical)

1. **Diagram Assignment**  
   Draw the architecture of a data platform for a ride-sharing company. Include:
   - Data sources (e.g., driver app, rider app, payments)
   - Data ingestion method
   - Processing layer (ETL tool)
   - Storage (lake and warehouse)
   - BI/dashboard and ML output

   You can sketch it by hand or use a tool like [draw.io](https://app.diagrams.net), Lucidchart, or Whimsical.

2. **Reflection Prompt**  
   Write a short piece:  
   👉 “Why do I want to become a Data Engineer?”  
   (This will help shape your learning goals and future portfolio.)

---


## 🗓️ **Week 2: Python & SQL for Data Engineering**

### 🎯 Objective:
- Master Python concepts tailored to data workflows
- Get comfortable with writing clean, efficient SQL for real-world datasets

---

## 🐍 Part 1: Python for Data Engineering

### ✅ What You Need to Know (DE-specific Python Skills):

| Topic | Why it Matters |
|-------|----------------|
| Variables, Loops, Functions | Automating repetitive tasks |
| File I/O | Read/write CSV, JSON, Parquet |
| Working with APIs | Pulling in real-time data |
| Libraries: `pandas`, `json`, `os`, `glob`, `logging` | Data manipulation, file system handling |
| Error Handling | Stable pipelines won’t crash on bad data |
| Functional Programming | `map`, `filter`, `lambda` – useful in transformations |
| Object-Oriented Programming (OOP) | When building custom ETL components |
| Python and SQL Integration | Using Python to run SQL queries (via `psycopg2`, `sqlalchemy`) |

---

### 🛠️ Exercise 1 – Read & Transform

Use **Python + pandas** to:

- Read a CSV file of ride-sharing data (riders, trips, locations)
- Filter trips with distance > 5 km
- Group by `driver_id`, count trips
- Output the result to a new CSV

> I can give you a sample dataset if you'd like!

---

## 🧠 Part 2: Advanced SQL for Data Engineers

### ✅ Key Concepts

| Topic | Example Use |
|-------|-------------|
| Joins (INNER, LEFT, OUTER) | Merging riders and trip tables |
| Aggregations | Calculating revenue per region |
| Window Functions | Ranking top drivers per city |
| Common Table Expressions (CTEs) | Making complex queries readable |
| Subqueries | Filtering based on dynamic conditions |
| Case Statements | Categorizing trip durations |
| Indexing & Query Optimization | Making large queries fast |

---

### 🛠️ Exercise 2 – Query Practice

Assume you have a schema like:

```sql
CREATE TABLE trips (
    trip_id INT,
    driver_id INT,
    rider_id INT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    distance_km FLOAT,
    city TEXT
);
```

#### Try these:

- Get total trips per driver
- Find top 5 drivers by number of trips in "New York"
- Calculate average distance per city
- Use a window function to rank drivers by trips per city

👉 Practice using [Mode SQL Tutorial](https://mode.com/sql-tutorial/) or [db-fiddle](https://www.db-fiddle.com/)

---

## 🧪 Mini-Project (Optional)

**Goal**: Build a mini ETL in Python

- Extract: Read CSV with trip data
- Transform: Clean nulls, calculate duration, enrich with city zones
- Load: Save to a PostgreSQL database

Want the sample data and starter script?

---

## 🔧 Tools to Set Up This Week

- Python 3.10+
- pip install: `pandas`, `sqlalchemy`, `psycopg2`, `jupyterlab`
- PostgreSQL (locally or via ElephantSQL)
- DB visualization tool (DBeaver or pgAdmin)


---

## 🧠 **Week 3: Data Modeling – Turning Raw into Ready**

> A well-designed data model is the difference between scalable analytics and a spaghetti mess of joins.

---

### 🎯 Objective:

- Understand **how to structure data** for analytics and performance
- Learn **dimensional modeling** (used in data warehouses)
- Know when to **normalize vs denormalize**
- Build **Star and Snowflake schemas**
- Apply these principles to real-world business use cases

---

## 🧱 Part 1: Why Data Modeling?

- Data modeling is the **blueprint** for your database or warehouse
- It defines how data is stored, related, and queried
- Without it, querying becomes slow, expensive, and painful

---

## 🔁 Part 2: OLTP vs OLAP

| Type | OLTP (Online Transaction Processing) | OLAP (Online Analytical Processing) |
|------|--------------------------------------|-------------------------------------|
| Use Case | Apps like Uber, Ecommerce | Dashboards, reports, ML models |
| Data Volume | Smaller transactions | Large, aggregated queries |
| Model | Highly normalized | Denormalized |
| Tools | PostgreSQL, MySQL | Snowflake, Redshift, BigQuery |

---

## ⭐ Part 3: Dimensional Modeling

### 💡 Key Concepts

- **Fact Table** – Holds measurable, quantitative data (e.g., trip facts)
- **Dimension Table** – Descriptive context (e.g., driver, rider, city)
- **Star Schema** – Fact table at center, surrounded by dimensions
- **Snowflake Schema** – Adds more normalization (nested dimensions)

### 🔍 Example: Ride-sharing Data Warehouse

```
          +-------------+
          | dim_driver  |
          +-------------+
                 |
+-------------+  |   +-------------+
| dim_city    |--+-->| dim_rider   |
+-------------+      +-------------+
        \                  /
         \                /
           +---------------------+
           |   fact_trips        |
           +---------------------+
```

---

### 🛠️ Exercise: Design a Star Schema

Imagine you're building an analytics platform for a **food delivery app** (like Swiggy/Zomato):

1. What would your **fact table** look like? (e.g., `fact_orders`)
2. List at least 3 dimension tables and their fields (e.g., `dim_customer`, `dim_restaurant`, `dim_time`)
3. Sketch a **star schema**

Want me to give you the table definitions and let you build it?

---

## 🧰 Tools for Data Modeling

- **dbdiagram.io** – Great for visualizing schemas
- **dbt** – For managing models in code
- **pgModeler** or **Lucidchart** – If you prefer desktop modeling

---

## 🧪 Optional Mini Project: Model + Populate a DW

1. Design a dimensional model (star schema) for a streaming service (like Netflix)
2. Create the tables in PostgreSQL
3. Write SQL to populate the fact and dimension tables from sample CSVs
4. Query: "Top 5 most-watched genres by month"

---

## 📦 What You’ll Need

- PostgreSQL (or BigQuery/Snowflake if you want cloud)
- dbdiagram.io account
- SQL skills from Week 2

---


## 🏗️ **Week 4: Data Warehousing (DWH)**

> “A data warehouse is a central repository of integrated data from one or more disparate sources.”  
> — In other words, it’s where clean, query-optimized data lives for analytics & decision-making.

---

### 🎯 Objectives

- Understand what a data warehouse is and why it matters  
- Learn core DWH concepts: schema types, partitioning, and indexing  
- Compare popular DWH technologies: **Snowflake, BigQuery, Redshift**  
- Learn how to load data efficiently (ELT vs ETL)  
- Query warehouse data with advanced SQL  
- Understand costs, scaling, and performance tuning

---

## 🧠 Part 1: What is a Data Warehouse?

| Feature | Description |
|--------|-------------|
| Central Repository | Brings data from multiple sources |
| Read-Optimized | Tuned for analytical queries, not transactions |
| Historical Data | Often stores years of data for trends and analysis |
| Immutable Storage | Data is not constantly updated, but appended |
| Query Performance | Uses columnar storage, indexing, caching |

---

## 🆚 OLTP vs DWH Recap

| Feature | OLTP | DWH |
|--------|------|-----|
| Purpose | Day-to-day operations | Reporting & analytics |
| Storage | Row-based | Columnar |
| Normalization | High | Low (denormalized) |
| Example | Banking app DB | Financial dashboards |

---

## 🚀 Part 2: Key Warehousing Concepts

### 🧱 Schema Design
- **Star schema** (most common)
- **Snowflake schema** (normalized dimensions)
- **Flat tables** (for quick MVPs)

### 🧩 Partitioning
- Divides table into parts for performance (e.g., by date, region)
- Example: `PARTITION BY DATE_TRUNC('day', event_time)`

### 📚 Clustering & Indexing
- Helps query engines skip unnecessary rows/columns
- BigQuery: `CLUSTER BY`, Snowflake: automatic indexing

### 📊 Columnar Storage
- Data is stored in columns not rows — faster for aggregations

---

## 🔧 Part 3: DWH Technologies Comparison

| Feature | **Snowflake** | **BigQuery** | **Redshift** |
|--------|----------------|--------------|--------------|
| Storage | Auto-scaled | Serverless | Manual resize |
| Pricing | Per-second compute | Per-query | Per-hour |
| Partitioning | Time-travel, micro-partitions | Partition + Cluster | Needs tuning |
| Language | SQL | SQL | SQL |
| Best for | Any size org | Event-based, Google ecosystem | AWS-native shops |

---

## 🧪 Mini-Project: Set Up a DWH (Hands-On)

### Option 1: **Use BigQuery (Free Tier)**
1. Create Google Cloud account
2. Load sample trip data to a dataset
3. Write queries to:
   - Count total trips by day
   - Top 5 busiest drivers
   - Average trip distance by city

### Option 2: **Use Snowflake (Free Trial)**
1. Load CSVs into a stage
2. Use `COPY INTO` to load into warehouse
3. Write star-schema-style queries

Want sample datasets & SQL templates for this?

---

## 🛠️ Tools for This Week

- **BigQuery (GCP)**
- **Snowflake Trial**
- **Redshift (AWS)**
- **SQL IDE**: DBeaver, DataGrip, or dbt Cloud

---

### ✅ Assignment

1. Choose any DWH platform (Snowflake / BigQuery / Redshift)  
2. Load sample data (can be CSV of trips or sales)
3. Create tables using a **Star Schema**  
4. Write 3 analytical queries on the fact table using joins and aggregates

---


## 🔄 **Week 5: ETL & ELT – Data Pipelines in Action**

> Think of ETL/ELT as the **production line** in a data factory. Raw materials (data) go in one end, and finished goods (insights, reports, ML features) come out the other.

---

### 🎯 Objectives

- Understand **ETL vs ELT** and when to use each
- Learn about pipeline architecture & components
- Build an end-to-end ETL using Python or dbt
- Learn best practices: modularity, logging, error handling
- Deploy pipelines with basic scheduling

---

## 🔁 ETL vs ELT: What’s the Difference?

| Process | ETL (Extract → Transform → Load) | ELT (Extract → Load → Transform) |
|--------|----------------------------------|----------------------------------|
| Common in | Traditional systems | Modern cloud warehouses |
| Where transformation happens | In the ETL tool | Inside the data warehouse |
| Tools | Python, Airflow, Informatica | dbt, SQL |
| Use Case | Limited warehouse power | Powerful cloud warehouse (Snowflake, BigQuery) |

---

## ⚙️ Key ETL Pipeline Components

1. **Extract** – Get data from sources:
   - APIs, databases (PostgreSQL, MongoDB), files (CSV, JSON)
2. **Transform** – Clean, filter, enrich:
   - Drop nulls, cast types, apply business logic
3. **Load** – Store data into:
   - Data warehouse, database, or data lake

---

## 🔨 Tools of the Trade

| Stage | Tools |
|-------|-------|
| Extract | Python (`requests`, `psycopg2`, `pymongo`), Fivetran, Airbyte |
| Transform | pandas, Spark, dbt, SQL |
| Load | `COPY INTO` (Snowflake), `bq load`, SQLAlchemy |
| Orchestration | Airflow, Prefect, Dagster |

---

## 🛠️ Hands-On Mini ETL (Python)

Here’s a basic ETL pipeline idea:

### Use Case: Load Trip Data from CSV to PostgreSQL

1. **Extract**: Read CSV file of trip data  
2. **Transform**: 
   - Drop rows with null distance
   - Convert timestamps
   - Create new column: trip_duration = end_time - start_time  
3. **Load**: Insert data into PostgreSQL `fact_trips` table

Want me to generate the code for this pipeline and share sample CSV?

---

## 📚 ELT with dbt (Modern Style)

### dbt: Data Build Tool – an open-source tool to **transform data inside the warehouse** using SQL

Key Features:
- SQL + Jinja templates
- Data lineage tracking
- Testing and documentation
- Version control friendly

> You define models in SQL and dbt handles dependencies and builds tables/views

Want a walkthrough on setting up **dbt with BigQuery or Snowflake**?

---

## 🧪 Optional Mini Project: Build Your First Real Pipeline

### Scenario:
You work for a food delivery startup. You need to:
- Pull data from an **orders CSV**
- Clean & enrich it (e.g., extract delivery times)
- Load it into Snowflake
- Schedule the pipeline to run daily

---

### 🧰 Tools to Install This Week

- Python libraries: `pandas`, `sqlalchemy`, `psycopg2`, `schedule`, `requests`
- PostgreSQL (or use ElephantSQL)
- dbt (install via pip)
- Jupyter Notebook or VS Code

---

### ✅ Assignment:

1. Build an ETL or ELT pipeline:
   - Python (for ETL) or dbt (for ELT)
2. Document each stage
3. Bonus: Add error handling + logging

---


## 🌊 **Week 6: Data Ingestion – Feeding the Pipeline**

> If ETL is the brain, **data ingestion is the bloodstream**. You can’t process what you don’t ingest.

---

### 🎯 Objectives

- Understand **batch vs streaming ingestion**
- Ingest data from:
  - Files (CSV, JSON, Parquet)
  - APIs
  - Databases (CDC)
  - Message Queues (Kafka)
- Use ingestion tools like **Kafka, Airbyte, Debezium**
- Write your own ingestion scripts in **Python**

---

## 🧠 Part 1: Types of Ingestion

| Type | Description | Use Case |
|------|-------------|----------|
| **Batch** | Scheduled, processes large chunks | Daily sales reports |
| **Streaming** | Continuous, real-time events | Live GPS tracking |
| **CDC (Change Data Capture)** | Captures inserts/updates/deletes from DB | Syncing prod DB to warehouse |

---

## 📂 Part 2: File-Based Ingestion

#### ✅ What You Can Ingest:
- CSVs (via `pandas`, `csv`, `dask`)
- JSON logs
- Parquet from S3

#### 🛠 Sample Python Ingestion

```python
import pandas as pd

df = pd.read_csv("trips.csv")
df = df.dropna(subset=["distance"])
df["trip_duration"] = pd.to_datetime(df["end_time"]) - pd.to_datetime(df["start_time"])
df.to_csv("cleaned_trips.csv", index=False)
```

Want a full ingestion pipeline into PostgreSQL?

---

## 🌐 Part 3: API-Based Ingestion

- Use Python’s `requests` to hit REST APIs
- Handle pagination, auth (API keys, OAuth)
- Store responses into structured format (CSV/DB)

#### 🧪 Example:

```python
import requests
import pandas as pd

url = "https://api.example.com/trips"
headers = {"Authorization": "Bearer your_token"}
response = requests.get(url, headers=headers)
data = response.json()
df = pd.DataFrame(data["results"])
```

---

## 🔁 Part 4: Database Ingestion (CDC)

> Keeping your data warehouse in sync with production systems.

- **Tool:** [Debezium](https://debezium.io/) + Kafka
- **What it does:** Listens to DB changes (like insert/update/delete) and streams them to Kafka topics
- Supports PostgreSQL, MySQL, MongoDB

---

## ⚡ Part 5: Streaming Ingestion with Kafka

Apache Kafka is the **industry standard** for real-time data ingestion.

| Concept | Kafka Equivalent |
|--------|------------------|
| Producer | Data source |
| Topic | Stream of messages |
| Consumer | Pipeline or service |
| Broker | Kafka server |
| Offset | Message position in topic |

Want me to set you up with a **Kafka demo** using Python + Docker?

---

## ⚙️ Tools to Explore

| Use Case | Tools |
|----------|------|
| Low-code ingestion | Airbyte, Fivetran |
| Real-time ingestion | Kafka, Kafka Connect, Flink |
| Custom ingestion | Python, Spark |
| CDC | Debezium, Maxwell |

---

## 🧪 Optional Mini Project: Build an Ingestion Layer

Scenario:
> You work for an e-commerce platform. Your team wants to ingest product data from:
> - Vendor APIs
> - Uploaded CSVs
> - PostgreSQL change logs

Your job:
- Write Python scripts for file + API ingestion
- Simulate CDC using Debezium or mimic it manually
- Load into a staging schema in your warehouse

---

### ✅ Assignment

1. Build **at least one ingestion pipeline**:
   - CSV → PostgreSQL  
   - or API → DataFrame → Snowflake
2. Add:
   - Logging
   - Schema validation
   - Data deduplication (bonus)

---

### 📦 Optional Setups for You

- Airbyte Docker setup
- Kafka-Python starter
- CDC simulation with PostgreSQL triggers

---


## 🔥 **Week 7: Big Data & Apache Spark**

> "If your pandas script chokes on a few million rows, Spark laughs in the face of petabytes."

---

### 🎯 Objectives

- Understand **Big Data principles** and the Hadoop ecosystem
- Learn how **Apache Spark** works under the hood
- Use **PySpark** to process large datasets
- Compare Spark to other distributed systems
- Perform batch and streaming data processing

---

## 🧠 Part 1: What is Big Data?

| Big Data Trait | What It Means |
|----------------|----------------|
| **Volume** | Terabytes to petabytes of data |
| **Velocity** | High-speed streaming data |
| **Variety** | Structured + semi-structured + unstructured |
| **Veracity** | Data quality and trust |
| **Value** | Turning data into business gold 🪙 |

---

## 🏗️ Part 2: Hadoop Ecosystem Overview

Before Spark, we had Hadoop MapReduce. Spark builds on Hadoop ideas but is **10–100x faster**.

| Tool | Purpose |
|------|---------|
| HDFS | Distributed file system |
| Hive | SQL on top of Hadoop |
| YARN | Resource manager |
| Spark | In-memory distributed compute engine |

---

## ⚡ Part 3: Spark Architecture

| Component | Function |
|-----------|----------|
| **Driver** | Master process, creates SparkContext |
| **Executors** | Worker processes that execute tasks |
| **Cluster Manager** | Allocates resources (YARN, Kubernetes, Standalone) |

Flow:
```text
Driver -> Cluster Manager -> Executors -> Data RDDs/DataFrames
```

---

## 🐍 Part 4: PySpark (Spark + Python)

You’ll mostly use **Spark DataFrames**, which are like `pandas` on steroids.

### 🔨 Sample PySpark Job

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TripProcessor").getOrCreate()

df = spark.read.csv("trips.csv", header=True, inferSchema=True)
df = df.filter(df["distance_km"] > 5)
df.groupBy("driver_id").count().show()
```

---

## 🆚 Spark vs Pandas

| Feature | pandas | PySpark |
|---------|--------|---------|
| Size | In-memory | Distributed |
| Speed | Slower for large data | Optimized for big data |
| Syntax | Pythonic | SQL-like DataFrame API |
| Use Case | Local analysis | Production-scale processing |

---

## 📦 Part 5: Spark SQL & Streaming

### Spark SQL:
You can query DataFrames using SQL:
```python
df.createOrReplaceTempView("trips")
spark.sql("SELECT city, AVG(distance_km) FROM trips GROUP BY city").show()
```

### Spark Structured Streaming:
Stream data in from Kafka, sockets, etc., and process it in micro-batches.

---

## 🛠️ Tools for This Week

- Install **Apache Spark + PySpark**
- Use **Databricks Community Edition** for free cloud-based Spark
- Use **Kaggle datasets** or trip CSVs for practice

---

## 🧪 Mini-Project: Big Data ETL with Spark

Scenario:
> You’re working at a food delivery company.  
> You receive 10M+ rows of trip and order data in CSV format every day.

Your task:
1. Load large CSVs using PySpark
2. Filter out incomplete rows
3. Join with driver and customer data
4. Write output as **Parquet files** to disk

---

### ✅ Assignment

1. Set up Spark locally or use Databricks  
2. Read + clean a large CSV file (≥1 million rows)  
3. Write the clean data to disk or S3 in Parquet format  
4. Bonus: Try Spark SQL or streaming from a Kafka topic

---

### 🧰 Resources

- [Apache Spark Docs](https://spark.apache.org/docs/latest/)
- [Databricks Community Edition](https://community.cloud.databricks.com)
- [Spark & PySpark Tutorials](https://github.com/jadianes/spark-py-notebooks)

---


## 🌊 **Week 8: Data Lakes & Lakehouses**

> Think of a **data warehouse** like a clean, well-organized kitchen. A **data lake** is the raw pantry — unstructured, messy, but full of potential.  
> A **lakehouse** is the best of both worlds. 😎

---

### 🎯 Objectives

- Understand **what a data lake is** and how it differs from a warehouse
- Learn how to design and manage **scalable lake storage**
- Use **Parquet, ORC, Avro** file formats
- Explore **Delta Lake, Apache Iceberg, and Hudi**
- Learn how **Lakehouses unify batch + streaming**

---

## 🧠 Part 1: What is a Data Lake?

| Feature | Description |
|---------|-------------|
| Purpose | Store all data: raw, semi-structured, unstructured |
| Format | Files (CSV, JSON, Parquet) |
| Storage | Cheap, scalable (e.g., S3, Azure Data Lake, GCS) |
| Processing | Spark, Presto, Hive, Trino |

You can store:
- Logs
- Images
- Clickstream
- Raw Kafka data
- SQL dump files

---

## 🆚 Data Lake vs Warehouse vs Lakehouse

| Feature | Data Lake | Warehouse | Lakehouse |
|--------|-----------|-----------|------------|
| Cost | Low | High | Medium |
| Schema | Schema-on-read | Schema-on-write | Hybrid |
| Query Speed | Slower | Fast | Fast |
| Tools | Spark, Presto | Snowflake, Redshift | Delta Lake, Iceberg |
| Ideal For | Raw data, data science | BI dashboards | Both |

---

## 📦 Part 2: Data Lake File Formats

| Format | Pros | Use Case |
|--------|------|----------|
| **CSV** | Easy to use, not efficient | Small quick loads |
| **JSON** | Semi-structured, flexible | API logs |
| **Parquet** | Columnar, compressed, query-optimized | Warehousing & Spark |
| **Avro** | Row-based, schema evolution | Kafka + CDC |
| **ORC** | Similar to Parquet, better for Hive | BigQuery, Hive, Presto |

> You’ll mostly use **Parquet** for processing with Spark and data lakes.

---

## 🧊 Part 3: Lakehouse Formats

### 1. **Delta Lake** (by Databricks)
- Brings **ACID transactions** to S3
- Great for Spark
- Supports time travel, schema evolution

### 2. **Apache Iceberg**
- Open table format
- Works with **Trino, Presto, Spark, Flink**
- Very fast for analytical queries

### 3. **Apache Hudi**
- Great for incremental updates
- Real-time ingestion & stream processing
- Integrates with Hive, Spark, Flink

---

## 🔧 Part 4: Tools for Working with Data Lakes

| Tool | Purpose |
|------|---------|
| Spark | Processing |
| AWS S3 / GCS / ADLS | Lake storage |
| Databricks / EMR / Glue | Query and transform |
| Delta Lake / Iceberg | Table format for Lakehouse |
| dbt | Transformations (now supports Lakehouse) |

---

## 🛠️ Mini Project: Build a Data Lake

**Scenario:**
> You work at a video streaming company. You get raw viewership logs in JSON format via Kafka.  
> You need to store it in a Data Lake on S3 in Parquet format with Delta Lake for query optimization.

**Steps:**
1. Ingest sample JSON logs
2. Convert to DataFrame (PySpark)
3. Write to **Delta Lake** or **Parquet** partitioned by date
4. Query with Spark SQL or Trino

---

## ✅ Assignment

1. Set up a local or cloud-based data lake (S3 / MinIO / GCS)  
2. Write sample data as Parquet files using Spark  
3. Convert your data to **Delta Lake** or **Iceberg** format  
4. Try querying it with Spark SQL

---

### 🔥 Bonus: Try this on Databricks Community Edition

- Create Delta Tables  
- Use **`VACUUM`**, **`DESCRIBE HISTORY`**, **`OPTIMIZE`** commands  
- Learn about **Z-Ordering** and file compaction

---


## ☁️ **Week 9: Cloud Platforms – Deploying and Automating Data Pipelines**

> "The cloud is not just about storage, it's about making **data** and **compute** available on demand, anywhere, anytime." — So, let’s use the cloud to **unlock massive scalability**!

---

### 🎯 Objectives

- Understand the basics of **cloud infrastructure** (AWS, GCP, Azure)
- Learn how to deploy data pipelines in the cloud (S3, GCS, ADLS)
- Automate workflows using **Airflow**, **Cloud Composer**, and **Cloud Functions**
- Learn about cloud storage options and when to use them (S3, Blob Storage, etc.)
- Set up **data processing** on cloud-based compute (EMR, Dataproc, Azure Databricks)

---

## 🧠 Part 1: Cloud Infrastructure Overview

### Cloud Providers:
- **AWS** (Amazon Web Services)
  - EC2, S3, Lambda, Glue, Redshift, EMR
- **GCP** (Google Cloud Platform)
  - Compute Engine, BigQuery, Cloud Storage, Dataflow
- **Azure**
  - Virtual Machines, Blob Storage, Data Factory, Synapse Analytics

### Core Cloud Components:
1. **Compute**: Servers (VMs) where data is processed (EC2, Dataproc, Azure VMs)
2. **Storage**: Data lakes and warehouses (S3, GCS, ADLS, Redshift, BigQuery)
3. **Networking**: VPC, load balancers, private networking for secure comms
4. **Orchestration**: Managing and scheduling jobs (Airflow, Cloud Composer, Data Factory)

---

## 📦 Part 2: Cloud Storage Options

| Service | Provider | Use Case |
|---------|----------|----------|
| **S3** | AWS | Data lake storage, objects, and backups |
| **GCS** | Google | Data storage, BigQuery integration |
| **Blob Storage** | Azure | Storing unstructured data |
| **Redshift** | AWS | Columnar data warehouse |
| **BigQuery** | GCP | Serverless data warehouse |

> For **data lakes**, use object storage like **S3** or **GCS**, and for **data warehouses**, use **Redshift** or **BigQuery**.

---

## 🧰 Part 3: Deploying and Automating Pipelines

### 1. **Airflow on Cloud**
- **AWS**: Managed service **MWAA (Managed Workflows for Apache Airflow)**
- **GCP**: **Cloud Composer**
- **Azure**: **Azure Data Factory**

Airflow allows you to schedule and automate data pipeline workflows across cloud environments.

### Example Airflow Task (in Python):
```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

def ingest_data():
    print("Ingesting data from source...")

dag = DAG('data_ingestion', start_date=datetime(2025, 1, 1))

ingest_task = PythonOperator(task_id='ingest_task', python_callable=ingest_data, dag=dag)
```

---

## 💻 Part 4: Cloud Compute for Data Processing

1. **AWS**:
   - **EMR (Elastic MapReduce)**: Managed Spark/Hadoop clusters
   - **Lambda**: Serverless compute for small tasks

2. **GCP**:
   - **Dataproc**: Managed Spark/Hadoop clusters
   - **Cloud Functions**: Serverless compute

3. **Azure**:
   - **Azure Databricks**: Managed Spark service
   - **Azure Functions**: Serverless compute

---

## 🧪 Part 5: Example – Data Pipeline in AWS

### Scenario:
> You have trip data stored in S3.  
> You want to run a Spark job on **EMR** to process this data, then store the results back in S3.

**Steps:**
1. Set up an **S3 Bucket** to store your raw data.
2. Create an **EMR cluster** and configure Spark to process the data.
3. **Write a Spark job** to clean, transform, and analyze the trip data.
4. Use **Airflow** (or MWAA) to schedule and automate this pipeline.

---

## 📊 Part 6: Cloud Data Warehousing (BigQuery & Redshift)

### **BigQuery (GCP)**:
- Serverless, so you don’t need to worry about provisioning compute.
- Use **standard SQL** for querying.
- Scale automatically.

### **Redshift (AWS)**:
- Managed data warehouse, but you need to choose instance types.
- Supports **SQL queries**, but can be optimized with `Vacuum`, `Sort Keys`, etc.

---

## 🧩 Optional Mini-Project: Deploy a Cloud Data Pipeline

**Scenario**:
> Your company needs to process sales data stored in **Google Cloud Storage** (GCS). You’ll use **Cloud Dataproc** (Spark on GCP) for processing.

**Steps**:
1. Create a **GCS Bucket** for your raw data.
2. Spin up a **Dataproc cluster**.
3. Write a **PySpark job** to process the data.
4. Use **Cloud Composer** to schedule the Spark job daily.
5. Output the results to **BigQuery** for analytics.

---

### ✅ Assignment

1. **Set up a cloud data pipeline** (using AWS, GCP, or Azure) for batch processing.
2. Use **Airflow/Cloud Composer** to automate your pipeline.
3. **Scale the pipeline** using cloud compute and storage.
4. Bonus: Add **error handling**, **monitoring**, and **logging**.

---

## 💡 Resources

- **Airflow Documentation**: [Apache Airflow Docs](https://airflow.apache.org/)
- **GCP Composer**: [Cloud Composer Docs](https://cloud.google.com/composer)
- **AWS EMR**: [AWS EMR Docs](https://aws.amazon.com/emr/)
- **Azure Databricks**: [Azure Databricks Docs](https://docs.microsoft.com/en-us/azure/databricks/)
  
---


## 🧑‍💻 **Week 10: Advanced Topics & Machine Learning Pipelines**

> "Building the infrastructure for data is like laying the foundation. Building ML pipelines on top is like putting the final layer of concrete to create something scalable and intelligent."

---

### 🎯 Objectives

- Understand the **role of data engineering** in the machine learning lifecycle
- Learn how to build **ML pipelines** for data preprocessing, model training, and serving
- Integrate **ML models** into data workflows (ETL pipelines)
- Automate ML processes using **MLflow**, **Kubeflow**, and **TFX**
- Learn how to serve models in production (via REST APIs)

---

## 🧠 Part 1: Data Engineering in the ML Pipeline

A typical ML pipeline involves the following stages:
1. **Data Collection**: Ingesting data from sources (APIs, databases, files)
2. **Data Preprocessing**: Cleaning, transforming, and feature engineering
3. **Model Training**: Training machine learning models on preprocessed data
4. **Model Validation**: Checking model performance on unseen data (validation set)
5. **Model Deployment**: Serving the model for real-time predictions
6. **Monitoring & Retraining**: Monitoring model performance over time and retraining as necessary

---

## 💡 Part 2: ML Pipeline Architecture

| Component | Purpose |
|-----------|---------|
| **Data Ingestion** | Fetch raw data from APIs, databases, files |
| **Preprocessing** | Clean and transform data for model use |
| **Feature Engineering** | Create new features from existing data |
| **Model Training** | Train machine learning models |
| **Model Deployment** | Deploy model for online or batch predictions |
| **Monitoring & Feedback Loop** | Track model performance and retrain if needed |

---

## 🛠️ Part 3: Tools for Building ML Pipelines

| Tool | Description |
|------|-------------|
| **MLflow** | Manage the lifecycle of ML models (tracking, versioning, deployment) |
| **Kubeflow** | End-to-end pipeline orchestration on Kubernetes |
| **TensorFlow Extended (TFX)** | Production pipeline framework for TensorFlow models |
| **Airflow** | Use for orchestrating batch ML workflows |
| **Seldon** | Model deployment for Kubernetes |
| **DataRobot** | AutoML tool to deploy ML models quickly |

---

## 🧑‍🔬 Part 4: Example of a Machine Learning Pipeline

### Scenario:  
You are tasked with building a recommendation system for an e-commerce platform using **Python**, **Scikit-learn**, and **MLflow**.

1. **Ingest Data**: Use **pandas** to load product and user data.
2. **Preprocess Data**: Clean missing values, encode categorical variables.
3. **Feature Engineering**: Create new features like user behavior (e.g., time spent on product).
4. **Train a Model**: Use **KNN** or **Random Forest** for recommendations.
5. **Track Experiments**: Use **MLflow** to log experiments.
6. **Model Deployment**: Use **Flask** or **FastAPI** to expose the trained model via a REST API.

```python
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv('user_behavior.csv')

# Preprocess data
# Feature engineering here...

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(df[features], df[target])

# Log with MLflow
mlflow.sklearn.log_model(model, "rf_model")
```

---

## 🧪 Part 5: Automated Model Training and Deployment

### **MLflow**: Manage Model Lifecycle

- **Tracking**: Log models, parameters, and metrics.
- **Projects**: Package ML code and dependencies.
- **Registry**: Store versions of models.
- **Serving**: Serve models for predictions.

```bash
# Log a model with MLflow
mlflow models serve -m "models:/MyModel/1" --host 0.0.0.0 --port 5000
```

### **Kubeflow**: Orchestrate with Kubernetes

- **Pipeline**: Define steps for ingestion, preprocessing, training, and serving in YAML format.
- **Argo**: Workflow engine that Kubeflow uses for pipeline orchestration.
- **Serving**: Integrates with **KServe** (model deployment) to serve ML models on Kubernetes.

---

## 🔄 Part 6: Continuous Integration and Delivery for ML

1. **Versioning**: Use tools like **DVC** (Data Version Control) to version data and models.
2. **CI/CD**: Automate testing and deployment of ML models using **GitHub Actions**, **Jenkins**, or **GitLab CI**.
3. **Model Monitoring**: Set up monitoring for model drift (i.e., when the model’s performance degrades over time).
4. **Retraining**: Trigger automatic retraining of models when performance drops.

---

## 🛠️ Part 7: Serving Machine Learning Models in Production

1. **Flask** or **FastAPI**: Lightweight frameworks to serve models as REST APIs.
2. **Seldon**: Deploy models on Kubernetes for scalable production environments.
3. **AWS SageMaker**, **GCP AI Platform**, **Azure ML**: Managed services for ML model deployment.

---

## 🧪 Mini Project: Build and Deploy an ML Pipeline

### Scenario:  
You’re building a fraud detection system using historical transaction data. Your goal is to:
1. **Preprocess** transaction data (cleaning, feature engineering).
2. **Train a fraud detection model** using **Random Forest** or **Logistic Regression**.
3. **Deploy the model** using **Flask** or **FastAPI** as a REST API.
4. **Track experiments** using **MLflow**.

---

### ✅ Assignment

1. **Set up a complete ML pipeline**:
   - Data preprocessing (cleaning, encoding, feature engineering)
   - Train a model (e.g., Random Forest, XGBoost)
   - Log the model using **MLflow**
   - Deploy using **Flask** or **FastAPI** for real-time inference.
   
2. Bonus:
   - Set up **model monitoring** for drift detection.
   - Implement **CI/CD** for continuous model deployment.

---

## 💡 Resources:

- **MLflow Documentation**: [MLflow Docs](https://mlflow.org/docs/latest/index.html)
- **Kubeflow Documentation**: [Kubeflow Docs](https://www.kubeflow.org/docs/)
- **TensorFlow Extended**: [TFX Docs](https://www.tensorflow.org/tfx)
- **FastAPI**: [FastAPI Docs](https://fastapi.tiangolo.com/)

---


## 🚀 **Week 11: Data Engineering in Production**

> "The work doesn’t stop once you deploy the model. In production, it’s all about **scalability**, **reliability**, and **maintenance**. If something breaks, you’re on call." — Let’s build your pipelines to survive in the wild. 🦸‍♂️

---

### 🎯 Objectives

- Learn the principles of **scalable, resilient data pipelines** in production
- Automate the **monitoring** and **logging** of your pipelines
- Understand the use of **containerization** and **orchestration** (Docker, Kubernetes)
- Ensure **data quality** and **consistency** in production workflows
- Learn how to **version** and **deploy** models at scale

---

## 🧠 Part 1: Building Scalable and Resilient Pipelines

### Key Principles:
1. **Idempotency**: Ensuring that your pipeline can run multiple times without causing issues (i.e., rerunning a job shouldn’t cause duplicate data or errors).
2. **Fault Tolerance**: Making sure your pipeline can recover from failures (e.g., network issues, missing data).
3. **Scalability**: Ensure your pipeline can handle an increase in data volume or complexity without breaking.

---

## 🧰 Part 2: Tools for Production Pipelines

### 1. **Containerization with Docker**
- **Docker** allows you to package your entire environment, including dependencies, code, and configurations, making it easy to run anywhere.
- **Why Docker?**: It helps avoid the "works on my machine" issue when moving to production.
  
```bash
# Dockerfile example
FROM python:3.8-slim

RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app

CMD ["python", "app.py"]
```

### 2. **Orchestration with Kubernetes**
- **Kubernetes** helps you manage your containers at scale, automating deployment, scaling, and operation of application containers.
- Useful for managing distributed data pipelines, ensuring that jobs run on multiple nodes.

---

## 🔄 Part 3: Automating and Monitoring Pipelines

### 1. **Airflow in Production**
- **Airflow** can be used to schedule and monitor pipelines in production. It supports retries, alerting, and logs.
- Use Airflow’s **Web UI** to monitor DAGs (Directed Acyclic Graphs) and see task execution status.

```python
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator

def process_data():
    # Your data processing logic here
    print("Processing data...")

dag = DAG('production_pipeline', start_date=datetime(2025, 1, 1))

start_task = DummyOperator(task_id='start', dag=dag)
process_task = PythonOperator(task_id='process_data', python_callable=process_data, dag=dag)

start_task >> process_task  # Define task dependencies
```

### 2. **Cloud Monitoring & Logging** (AWS, GCP, Azure)
- **AWS CloudWatch**, **GCP Stackdriver**, and **Azure Monitor** help track the performance of your cloud resources.
- Use these to monitor pipeline execution and set up **alerts** for failures or performance degradation.
- Store logs in **S3/GCS/ADLS** for long-term retention.

---

## 📦 Part 4: Ensuring Data Quality in Production

1. **Data Validation**:
   - Check if data meets the expected quality criteria.
   - **Great Expectations**: An open-source Python package for testing, documenting, and profiling data.

2. **Data Consistency**:
   - Use **ACID** transactions (like in **Delta Lake** or **Apache Iceberg**) to ensure that your data is consistent during batch or streaming operations.
   - Ensure that data is properly **partitioned** to avoid skew in your production jobs.

3. **Version Control**:
   - **DVC (Data Version Control)**: Version your datasets just like you would with code. You can track large data files and models.
   - **MLflow**: Keep track of different versions of models, parameters, and metadata.

---

## 🧑‍💻 Part 5: Deploying and Serving Models in Production

### 1. **Model Versioning**:
   - Use **MLflow** or **DVC** to keep track of model versions, ensuring you can easily roll back if something goes wrong.

### 2. **Serving Models**:
   - Deploy models using **Flask**, **FastAPI**, or **Seldon** for scalable inference.
   - For **high traffic**, consider using **Kubernetes** with **GPU/TPU** support or managed services like **AWS SageMaker**, **Google AI Platform**, or **Azure ML**.

### Example of serving with **FastAPI**:
```python
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/predict")
def predict(data: dict):
    prediction = model.predict([data["features"]])
    return {"prediction": prediction.tolist()}
```

---

## 🛠️ Part 6: CI/CD for Data Pipelines and ML Models

### 1. **CI/CD for Code & Models**:
   - Use **GitLab CI**, **Jenkins**, or **GitHub Actions** to automate the process of:
     - **Unit tests** for your code.
     - **Data validation** tests.
     - **Model validation** before deployment.
     - **End-to-end tests** for your entire pipeline.
   
### Example GitHub Actions CI/CD Pipeline:
```yaml
name: Data Pipeline CI/CD

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run Tests
        run: |
          pytest
```

---

## 🧪 Mini Project: Build a Scalable Data Pipeline in Production

**Scenario**:  
> You work at an e-commerce platform. Your job is to **deploy and maintain** a real-time recommendation system in production.

### Steps:
1. **Containerize** the ML model using **Docker**.
2. Use **Kubernetes** to deploy the container in a scalable environment.
3. Set up **Airflow** to automate the recommendation job.
4. Use **AWS S3** or **GCS** to store the data and logs.
5. Set up **CloudWatch** or **Stackdriver** for monitoring.
6. Ensure **model versioning** using **MLflow** or **DVC**.
7. Implement **CI/CD** using **GitHub Actions**.

---

### ✅ Assignment

1. **Dockerize** your data processing pipeline.
2. Set up **Kubernetes** for orchestration and **Airflow** for scheduling.
3. Use **Cloud Monitoring** tools (e.g., **AWS CloudWatch** or **GCP Stackdriver**) for observability.
4. Set up a **CI/CD pipeline** for testing and deploying code and models.

---

## 💡 Resources

- **Docker**: [Docker Docs](https://docs.docker.com/)
- **Kubernetes**: [Kubernetes Docs](https://kubernetes.io/docs/)
- **Airflow**: [Apache Airflow Docs](https://airflow.apache.org/docs/)
- **CloudWatch / Stackdriver / Azure Monitor**: [AWS Monitoring](https://aws.amazon.com/cloudwatch/), [GCP Monitoring](https://cloud.google.com/stackdriver), [Azure Monitor](https://azure.microsoft.com/en-us/services/monitor/)
- **MLflow**: [MLflow Docs](https://mlflow.org/docs/latest/)

---


## 🚀 **Week 12: Final Project – End-to-End Data Engineering**

> "This is where the rubber meets the road. It’s time to show that you can take a **problem**, build a **data pipeline**, process and analyze the data, deploy models, and **ensure the system runs smoothly** in production."

---

### 🎯 **Project Goals**

- **End-to-end data pipeline**: Build a complete data pipeline that ingests data, processes it, and stores the results.
- **Data Engineering skills**: Ingest data from multiple sources, clean, transform, and store it.
- **Machine Learning**: Integrate an ML model that processes the data and makes predictions.
- **Deployment**: Deploy your pipeline to production using tools like Docker, Kubernetes, and Airflow.
- **Scalability**: Ensure your pipeline is scalable and can handle large datasets efficiently.
- **Monitoring**: Set up logging, monitoring, and alerts to ensure your pipeline runs smoothly.

---

### 🧠 **Step 1: Define the Problem**

Choose a real-world problem to solve. Here are some example ideas for your project:

1. **E-Commerce Recommendation System**:
   - Build a recommendation system based on customer purchase history and product data.
   - Integrate an ML model for personalized product recommendations.
   
2. **Fraud Detection System**:
   - Use transaction data to detect fraudulent activity using an ML model.
   - Automate the data pipeline to process and analyze the transactions in real-time.

3. **Weather Forecasting**:
   - Ingest historical weather data and predict future weather patterns.
   - Use machine learning to predict future temperature or rainfall based on historical data.

4. **Healthcare Analytics**:
   - Use patient records to predict the likelihood of certain diseases or conditions.
   - Process and clean healthcare data, build predictive models, and deploy them.

---

### 🛠️ **Step 2: Design Your Data Pipeline**

Think about the following stages:

1. **Data Ingestion**:
   - Fetch data from multiple sources (APIs, databases, files, etc.).
   - Use tools like **Apache Kafka** or **Airflow** to automate ingestion.

2. **Data Processing**:
   - Clean and transform data for analysis (handle missing values, encoding categorical features, etc.).
   - Use **Spark** or **Pandas** for data transformation.

3. **Feature Engineering**:
   - Extract features from the raw data that can help the model make better predictions (e.g., aggregations, time-series features).
   
4. **Model Training**:
   - Train a machine learning model using the processed data (e.g., Random Forest, XGBoost).
   - Use tools like **MLflow** for experiment tracking and model versioning.

5. **Model Deployment**:
   - Deploy your model to a server using **Flask** or **FastAPI** for real-time inference.
   - Alternatively, use **Seldon** or **AWS SageMaker** for deployment at scale.

6. **Scheduling and Orchestration**:
   - Use **Airflow** or **Kubeflow** to schedule and automate the entire pipeline.
   - Ensure your pipeline runs on a regular basis (e.g., daily, hourly).

7. **Monitoring**:
   - Set up logging and monitoring to track pipeline performance.
   - Use **AWS CloudWatch**, **Google Stackdriver**, or **Azure Monitor**.

---

### 🧩 **Step 3: Implement Your Data Pipeline**

Here’s a high-level implementation breakdown:

1. **Set up the environment**:
   - Containerize your pipeline using **Docker**.
   - Use **Kubernetes** for orchestration and scaling.

2. **Data ingestion**:
   - Use **APIs** to fetch external data, or load **CSV**/**JSON** files.
   - Use **Airflow** to automate the process.

3. **Data processing**:
   - Perform data wrangling using **Pandas**, **Dask**, or **PySpark**.
   - For large-scale processing, consider using **Spark** on **AWS EMR**, **GCP Dataproc**, or **Azure HDInsight**.

4. **Model training**:
   - Train your model using libraries like **Scikit-learn**, **XGBoost**, or **TensorFlow**.
   - Log experiments and track models using **MLflow**.

5. **Model deployment**:
   - Expose your model via **Flask** or **FastAPI** for predictions.
   - Deploy to a cloud platform like **AWS** or **GCP**, or use **Seldon** for Kubernetes-based deployment.

6. **Pipeline scheduling**:
   - Automate pipeline execution using **Airflow** or **Kubeflow**.
   - Set up retries, alerting, and logs.

7. **Monitoring and logging**:
   - Set up logging to monitor the pipeline’s performance.
   - Use **Prometheus**, **Grafana**, or **CloudWatch** for system monitoring.

---

### 📊 **Step 4: Visualize and Report Results**

1. **Pipeline Monitoring Dashboard**:
   - Build a simple dashboard using **Grafana** or **Plotly Dash** to track the pipeline’s performance.
   - Include metrics like job success rates, processing time, and any failures.

2. **Model Metrics**:
   - Track model performance metrics (e.g., accuracy, precision, recall, AUC) on the dashboard.
   - Keep logs of every model version, and track model drift over time.

3. **Final Report**:
   - Document the architecture of your pipeline.
   - Explain how data flows through the system and how the model is integrated.
   - Detail the challenges you faced and how you overcame them.
   - Discuss how your system is scalable and resilient to failures.

---

### 💡 **Step 5: Bonus Features**

- **Data Quality Validation**: Implement data validation checks using **Great Expectations**.
- **CI/CD**: Set up a **GitHub Actions** or **GitLab CI** pipeline for testing and deploying your pipeline.
- **Error Handling**: Add error handling to your pipeline, ensuring data integrity and system recovery.

---

### ✅ **Project Submission Checklist**

1. **Dockerize** the pipeline and ensure it runs in a containerized environment.
2. Set up **Airflow** or **Kubeflow** for scheduling and automation.
3. **Deploy** your model using **Flask**, **FastAPI**, or a cloud service.
4. Ensure **scalability** and **fault tolerance** in your pipeline (using **Kubernetes**, **Cloud Services**).
5. Set up **monitoring** and **logging** for pipeline performance.
6. Create a **dashboard** to visualize the performance of the pipeline.
7. Submit a **final report** explaining your architecture, model, challenges, and solutions.

---

