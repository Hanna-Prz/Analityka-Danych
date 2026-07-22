import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Exercise 1
#a) Load house sales data from file kc_house_data.csv, print first records.  
#b) Print the variable list. Select categorical and numerical variables.  
#c) Compute basic statistics of numerical variables. Are all of them meaningful?
#a)
df = pd.read_csv('kc_house_data.csv')
df.head()
#b)
house_categorical_df = ['id', 'date','yr_built', 'yr_renovated', 'zipcode', 'lat', 'long']
house_numerical_df = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',  'sqft_living15', 'sqft_lot15']
#c)
house_categorical = df[house_categorical_df]
house_numerical = df[house_numerical_df]
display(house_numerical.describe())
# Exercise 2
#Using scatter plot:  
#a) Show the relation of square footage and price.  
#b) Check how price is influenced by apartment's grade. Find other variables correlated with price.  
#c) Check how good apartments are distributed over the city. First, plot zipcode versus price. Then, use apartments coordinates (longitude and lattitude) in order to show where apartments of good grade are located.
#a)
plt.figure(figsize=(8,6))
plt.scatter(df['sqft_living'], df['price'], alpha=0.5)
plt.title('Relacja powierzchnia (sqft_living) vs cena')
plt.xlabel('Powierzchnia mieszkania [sqft]')
plt.ylabel('Cena [USD]')
plt.show()
#b)
grades = sorted(df['grade'].unique())
prices_per_grade = [df[df['grade']==g]['price'] for g in grades]
plt.figure(figsize=(8,6))
plt.boxplot(prices_per_grade, labels=grades)
plt.title('Cena a grade mieszkania')
plt.xlabel('Grade')
plt.ylabel('Cena [USD]')
plt.show()
numerical_df = df.select_dtypes(include='number')
correlations = numerical_df.corr()['price'].sort_values(ascending=False)
print("Najbardziej skorelowane zmienne z ceną:")
display(correlations)
#c)
zipcodes = sorted(df['zipcode'].unique())
prices_per_zip = [df[df['zipcode']==z]['price'] for z in zipcodes]
plt.figure(figsize=(12,6))
plt.boxplot(prices_per_zip, labels=zipcodes)
plt.title('Cena mieszkań w zależności od zipcode')
plt.xlabel('Zipcode')
plt.ylabel('Cena [USD]')
plt.xticks(rotation=90)
plt.show()
good_apartments = df[df['grade'] >= 8]
plt.figure(figsize=(10,8))
plt.scatter(good_apartments['long'], good_apartments['lat'],
            c=good_apartments['price'], cmap='viridis', alpha=0.6)
plt.colorbar(label='Cena [USD]')
plt.title('Lokalizacja mieszkań o wysokim grade')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
# Exercise 3
#a) Create boxplot of apartment's grade versus it's price. Compare with plot 2b).  
#b) Create mosaic plot of any two categorical variables.
#a)
grades = sorted(df['grade'].unique())
prices_per_grade = [df[df['grade'] == g]['price'] for g in grades]
plt.figure(figsize=(8,6))
plt.boxplot(prices_per_grade, labels=grades)
plt.title("Boxplot: Grade vs Price")
plt.xlabel("Grade")
plt.ylabel("Price [USD]")
plt.show()
#b)
from statsmodels.graphics.mosaicplot import mosaic
cat1 = 'grade'
cat2 = 'view'
data_for_mosaic = df.groupby([cat1, cat2]).size()
plt.figure(figsize=(10,6))
mosaic(data_for_mosaic)
plt.title(f'Mosaic plot: {cat1} vs {cat2}')
plt.show()
# Exercise 4
#"I am interested in the house with 3 bedrooms, 2 bathrooms and 2 floors"  
#a) Compute the basic statistics of apartments fulfilling above constraints.  
#b) Distinct accepted apartments on any of previously created plots.  
#c) Find such apartments with best price to footage ratio.  
#a)
filtered_df = df.loc[(df['bedrooms'] == 3) &
                     (df['bathrooms'] == 2) &
                     (df['floors'] == 2)]
print("Liczba mieszkań spełniających kryteria:", len(filtered_df))
display(filtered_df.describe())
#b)
plt.figure(figsize=(8,6))
plt.scatter(df['sqft_living'], df['price'], alpha=0.3, label='Wszystkie mieszkania')
plt.scatter(filtered_df['sqft_living'], filtered_df['price'], color='red', label='Wybrane mieszkania')
plt.title('Powierzchnia vs Cena z wyróżnieniem wybranych mieszkań')
plt.xlabel('Powierzchnia [sqft]')
plt.ylabel('Cena [USD]')
plt.legend()
plt.show()
#c)
filtered_df_copy = filtered_df.copy()
filtered_df_copy['price_to_sqft'] = filtered_df_copy['price'] / filtered_df_copy['sqft_living']
best_ratio_df = filtered_df_copy.sort_values(by='price_to_sqft')
print("Mieszkania z najlepszym stosunkiem cena/powierzchnia:")
display(best_ratio_df.head())
