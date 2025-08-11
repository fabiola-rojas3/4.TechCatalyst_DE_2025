# Connecting to Snowflake Database

* From the **Settings** menu select **Database Connections**

* Click the **+ Database** button to create a new database connection

* Select Snowflake 

  <img src="images/image-20250512104449291.png" alt="image-20250512104449291" style="zoom:50%;" />

* Use the following information:

  * Database name: **DEMO**
  * Username: Your username, example **STUDENT1**
  * Password: `KHCC_training1`
  * Display Name: Keep as **Snowflake**
  * Account: `HJDBRRB-KDC68498`
  * Warehouse: `COMPUTE_WH`
  * Role: `SUPERSET`

<img src="images/image-20250512104716026.png" alt="image-20250512104716026" style="zoom:50%;" />

Click **Connect**. This may take few second. If everything works out you will see `STEP 3 of 3` section

<img src="images/image-20250512104835833.png" alt="image-20250512104835833" style="zoom:50%;" />

* Expand the **SQL Lab** section and check the following boxes:

<img src="images/image-20250512104930507.png" alt="image-20250512104930507" style="zoom:50%;" />

Finally, click **Finish**

You should see the following under **Databases**

![image-20250512105017164](images/image-20250512105017164.png)

## Create a Dataset

Under the **Datasets** menu, click **+ Dataset**

Select the Snowflake database, then the Superset schema, finally the `diabetes_dataset` table 

<img src="images/image-20250512105202029.png" alt="image-20250512105202029" style="zoom:50%;" />

Click **Create dataset and create chart**

![image-20250512105218107](images/image-20250512105218107.png)

This may take few seconds.

