# Apache Superset Sample Datasets

## Dataset Overviews and Use-Case Grouping

| Dataset/Table                           | Description & Content                                        | Use-Case / Analysis Area               |
| --------------------------------------- | ------------------------------------------------------------ | -------------------------------------- |
| **FCC 2018 Survey**                     | U.S. Federal Communications Commission survey data, likely on broadband access | Survey analysis, demographics          |
| **bart_lines**                          | Bay Area Rapid Transit (BART) line information (routes, names, etc.) | Geospatial, transportation mapping     |
| **birth_finance_by_region**             | Birth and financial statistics segmented by region           | Demographic, economic analysis         |
| **birth_names**                         | U.S. baby names by year, gender, state, and count            | Time series, demographic trends        |
| **Channel_members**                     | Membership info for users in communication channels          | Social network, communication analysis |
| **channels**                            | Metadata about communication channels                        | Social network, communication analysis |
| **cleaned_sales_data**                  | Transactional sales data (products, dates, regions, amounts) | Sales analytics, business intelligence |
| **covid_vaccines**                      | COVID-19 vaccination statistics by region/country            | Public health, time series analysis    |
| **exported_stats**                      | Likely pre-aggregated or exported statistics (context-specific) | General analytics, reporting           |
| **flights**                             | Flight records (departments, costs, classes, dates, origins, destinations, distances) | Travel, expense, time series analysis  |
| **hierarchical_dataset** (virtual)      | Hierarchical data, possibly for tree or sunburst charts (virtual view) | Hierarchical, organizational analysis  |
| **long_lat**                            | Latitude and longitude data for geospatial plotting          | Geospatial mapping                     |
| **members_channels_2** (virtual)        | Virtual join of members and channels                         | Social network, communication analysis |
| **messages**                            | Communication messages (sender, receiver, timestamp, content) | Messaging, communication analysis      |
| **messages_channels**                   | Messages linked to channels                                  | Messaging, channel communication       |
| **new_members_daily** (virtual)         | Daily new user/member sign-ups (virtual, likely aggregated)  | Growth, time series analysis           |
| **sf_population_polygons**              | San Francisco population data with geographic polygons       | Geospatial, demographic mapping        |
| **threads**                             | Discussion threads (forum-like structure)                    | Social network, communication analysis |
| **unicode_test**                        | Data for testing Unicode character handling                  | Data quality, internationalization     |
| **users**                               | User metadata (profiles, join dates, etc.)                   | User analytics, social network         |
| **users_channels**                      | User-to-channel membership mapping                           | Social network, communication analysis |
| **users_channels_uzoonNNtSRO**(virtual) | Virtual join/view of users and channels (context-specific)   | Social network, communication analysis |
| **video_games_Sales**                   | Video game sales data by region, platform, year, etc.        | Sales analytics, time series, business |
| **wb_health_population**                | World Bank health and population indicators by country/year  | Public health, demographic analysis    |

## Grouped by Use-Case

**1. Business & Sales Analytics**

- cleaned_sales_data
- video_games_Sales
- exported_stats

**2. Demographic & Public Health Analysis**

- birth_names
- birth_finance_by_region
- FCC 2018 Survey
- wb_health_population
- covid_vaccines
- sf_population_polygons

**3. Geospatial & Mapping**

- bart_lines
- long_lat
- sf_population_polygons

**4. Time Series & Growth Analytics**

- birth_names
- cleaned_sales_data
- new_members_daily (virtual)
- covid_vaccines
- video_games_Sales
- wb_health_population

**5. Social Network & Communication Analysis**

- users
- channels
- Channel_members
- users_channels
- users_channels_uzoonNNtSRO (virtual)
- members_channels_2 (virtual)
- messages
- messages_channels
- threads

**6. Hierarchical & Organizational Analysis**

- hierarchical_dataset (virtual)

**7. Data Quality & Testing**

- unicode_test

**8. Travel & Expense Analysis**

- flights

## Notes

- **Physical tables** are actual tables in the database, while **virtual tables** (or datasets) are SQL views or queries defined within Superset for more complex or joined data.
- These datasets are intended to demonstrate Superset’s features, including geospatial mapping, time series, hierarchical visualizations, and social network analysis.
- Some datasets (like `unicode_test`) are primarily for testing Superset’s handling of various data types and encodings.