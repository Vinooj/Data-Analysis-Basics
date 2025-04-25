### Exploring Data Analysis with Python

Pandas ( Data Structures and Tools )
NumPy (Arrays & matrices )
SciPy ( Integrals, Solving diff equation, Optimization )



Visualizations
Matplotlib - for plots and graphs - most well known
Seaborn - Another visualization library

Algorithmic Libraries
Scikit-learn - ( Machine Learning: Regression, classification and so on )
StatsModels ( Explore data, estimate statistical models, and perform statistical tests )


Pandas Data frame

Import pandas and pd
Url = “../file.csv”
df = pd.reads.csv(URL, header = None)
print df.head(10)
print df.tail(10)

We can add headers manually, by
Headers = [“header1”, header2”,…]
df.columns=headers
df.head(5)
df.to_csv(path_to_file) # To write the df to file. Also supports json, sql, excel etc

df.dtypes - # give the data types associated to the data frame data. Objject(string), int64 (number), float64 (decimals)
df.describe(include=all)  # Provides quick statistics ( count, min, mx, std dev etc )
df.info 

Database access
Use Python DB API -> Connections objects( connections and transactions) and Cursor Objects
From dmodule import connect
Connection = connect(“dynamo”, ‘username’, ‘pswd’)
Cursor = connection.cursor()
Cursor.execute(“select * from mytable’)
Results = cursor.fetchall()

Cursor.close()
Connections.close()


Dealing With missing Values
df.dropna(subset=[“price”], axis=0, inplace = True ) # inplace = True, makes the modification inlace in the dataset
df.dropna(subset=[“price”], axis=0, ) # does not modify the data frame
df.replace(np.nan, df[‘column-name’].mean) # replace missing value with mean of the column

Data Formatting
df[“city-mpg”] = 235/df[“city-mpg’]
df.rename(columns={“city_mpg”: “city-l/100km”}, inplace=True)
df.dtypes #Check the datatype
df.astype() # convert the data type eg: df[“price”].astype(“init”)

Data Normalization

Important: If the scale of the value in the datafarme is widely varying, then impacts analysis’s. Eg: age ranges from 1 - 100 and income will be from 2000 to 100000.  This disparity causes linear ear regression model to weigh in on income much more than age. So a better way is to normalize the data in the 0 - 1 range by applying different methods shown below.

Normalization enables a fairer comparison between the different features, making sure they have the same impact. It is also important for computational reasons. Here is another example that will help you understand why normalization is important.

￼
 

Simple Feature scaling
df[“length”]  = df[“length”] /df[“length”].max

Min-Max
df[“length”]  = (df[“length”]  - df[“length”].min())/
                          (df[“length”].max  - df[“length”].min())

Z-Score
df[“length”]  = (df[“length”]  - df[“length”].mean())/df[“length”].std()

Binning In Python ( Has impact on Prediction )
Bins = np.linspace(min(df[“price”]), max(df[“price”]), 4) # create 4 bins
group_name = [“Low”, “Medium”, “High”]
df[“price-dinned’] = pd.cut(df[“price”], bins, labels=group_names, inlcude_lowest=True)

Turning Categorical Variables into Quantitative Variable
Eg: field types = {gas, diesel, electric ..) into numerical value by create new columns for gas, electric and diesel and the using 1 or 0 to indicate the value in the appropriate columns. This is also called one hot encoding.

temp_df = pd.get_dummies(df[‘fuel’]). # This will create 2 separate columns for each file type. Eg: diesel, gas etc note this is store in a separate data frame temp_df
dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True). # Change the column type for clarity
df = pd.concat([df, temp_df], axis=1)

Data Analysis

Description statistics

df.describe()  # values count, mean, std, min, mix and quartiles. Skips NaN 

Categorical Values
drive_wheel_counts=df[“drive-wheels”].value_counts()  # methods summarizes categorical values

drive_wheel_counts.rename(columns={‘drive-wheels’:’value_counts’}, inplace=True)
drive_wheels_count.index_name=‘drive-wheels’

Helps show data as 
￼
Numeric Data - Uses Box Plots

￼
￼
Continuous Data - Use Scatter Plot

Uses scatter pLot to check for trend
# predictor/independent variable on axis 
# Target/dependent variables on y-axis

import matplotlib.pyplot as plt

plt.title(Scatter plot of Engine Size vs Price”)
plt.xlabel(“Engine Size”)
plt.ylabel(“Price”)


Group By

df.groupby
df.pivot(


Correlation
import seaborn as sns
sns.regplot(x=“highway-mpg”, y=“price”, data=df).  # Slope determines the correlation strength
plt.ylim(0,)

Pearson Correlation
Heatmaps 
provide a comprehensive visual summary of the strength and direction of correlations among multiple variables.


Chi-Square Test
The chi-square test is a statistical method used to determine if there is a significant association between two categorical variables. This test is widely used in various fields, including social sciences, marketing, and healthcare, to analyze survey data, experimental results, and observational studies.

￼
Step 4: Interpret the Result
Using a chi-square distribution table, we compare the calculated chi-square value (44.33) with the critical value at one degree of freedom and a significance level (e.g., 0.05), approximately 3.841. Since 44.33 > 3.841, we reject the null hypothesis. This indicates a significant association between smoking status and the incidence of lung disease in this sample.

Note: if calculated chi-square value < 3.841, we fail to reject the null hypothesis, meaning there is no significant association between gender and product preference in this sample.
