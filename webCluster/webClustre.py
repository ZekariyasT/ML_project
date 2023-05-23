import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# get the URLs from the user
urls = input("Enter the URLs to cluster (separated by commas): ")
urls = urls.split(",")

# scrape the web pages
pages = []
for url in urls:
    response = requests.get(url.strip())
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text().replace('\n', '')
    pages.append(text)

# vectorize the pages
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(pages)

# perform clustering
num_clusters = int(input("Enter the number of clusters: "))
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)

# print the results
for i, url in enumerate(urls):
    print(url.strip(), "is in cluster", kmeans.labels_[i])
