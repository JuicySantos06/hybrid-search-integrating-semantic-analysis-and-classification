# Hybrid Search with Recommandation Engine

## General Information
> The following demo aims to demonstrate the ability to enhance the search experience through the joint use of both keyword-based and vector-based search technology.
> we will seek to highlight the contribution of vector search for a product recommendation requirement based on consumer feelings.
> We will be using using the following native MongoDB technologies:
* MongoDB Atlas Search
* MongoDB Atlas Vector Search


### Step 1 : Import the Cross-Market Recommendations dataset into your Atlas Database
> Download both the US => Home and Kitchen/metadata and the US => Home and Kitchen/reviews category datasets.
```
link: https://xmrec.github.io/data/us/
```
> Create the following database and collection in Atlas
```
DB_NAME = hybrid_search_recommendation_xmarket
COLLECTION_NAME = hybrid_search_products
COLLECTION_NAME = hybrid_search_reviews
```
> Extract and import the data into the aforementioned collection using Compass or any other tools you see fit.
