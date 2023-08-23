# Advertising

Importing Libraries: The script starts by importing necessary Python libraries like NumPy, Pandas, Matplotlib, and Seaborn for data manipulation, visualization, and machine learning operations.

Reading Data: The code reads a CSV file from the specified path using Pandas' read_csv function and stores it in the DataFrame named df.

Data Exploration: The script performs some initial data exploration using methods like head(), info(), and describe() to provide a glimpse of the dataset's structure and summary statistics.

Data Cleaning: The code drops a column named "Unnamed: 0" from the DataFrame using the drop() function with axis=1 to remove an unnecessary or redundant column.

Data Visualization: Several visualizations are created using Seaborn and Matplotlib:
A pair plot using Seaborn's pairplot() function to visualize pairwise relationships between numerical variables in the DataFrame.
Scatter plots to visualize the relationships between "Sales" and individual advertising mediums ("TV," "Newspaper," "Radio").
A correlation heatmap using Seaborn's heatmap() function to display the correlation matrix of the dataset.

Data Preparation for Machine Learning: The script prepares the data for machine learning by splitting the DataFrame into feature matrix X and target vector y. 
The train_test_split() function from scikit-learn is used to split the data into training and testing sets.

Linear Regression Model: The script imports the LinearRegression class from scikit-learn and initializes a linear regression model named lr. 
The model is then fitted to the training data using the fit() method.

Predictions and Metrics: The model is used to predict the target variable on the test data, and the script calculates and prints the model's coefficient and intercept. 
The r2_score() function is used to calculate the coefficient of determination (R-squared) between the predicted and actual target values.

Creating Result DataFrame: A DataFrame named result is created to store the actual and predicted outcomes from the test data, and the first few rows of this DataFrame are displayed using the head() function
