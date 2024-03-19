import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression

# Scrape the apartment website
url = "https://www.riversideapts.com/floor-plans/"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# Extract the rent data
rents = []
for listing in soup.find_all("div", class_="listing"):
    rent = listing.find("span", class_="rent").text
    rents.append(float(rent))

# Prepare the data for machine learning
X = [[i] for i in range(len(rents))]
y = rents

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict the next rent
next_rent = model.predict([[len(rents)]])
print("The predicted next rent is:", next_rent[0])