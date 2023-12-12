# Hybrid Search with Recommandation Engine

## General Information
> The following demo aims to demonstrate the ability to enhance the search experience through the joint use of both keyword-based and vector-based search technology.
> we will seek to highlight the contribution of vector search for a product recommendation requirement based on consumer feelings.
> We will be using using the following native MongoDB technologies:
* MongoDB Atlas Search
* MongoDB Atlas Vector Search


### Step 1: Import the Cross-Market Recommendations dataset into your Atlas Database
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

### Step 2: Embed both the home and kitchen products and reviews dataset
> Update the file params.py with the following information:
* Your MongoDB Atlas connection string
```
MONGODB_ATLAS_CONNECTION_STRING
```
* The path to the cross-market recommendation dataset that you previously downloaded
```
XMARKET_HOME_AND_KITCHEN_PRODUCT_DATA_FILE_PATH
XMARKET_HOME_AND_KITCHEN_REVIEWS_DATA_FILE_PATH
```
> Execute both the step 1 and step 2 in the python script file

### Step3: Create the Atlas Search index
> Copy and paste the following index definition into Atlas Search
> Index name = searchIndex
```
{
    "mappings": {
      "dynamic": false,
      "fields": {
        "features": [
          {
            "type": "string"
          }
        ],
        "title": [
          {
            "type": "string"
          }
        ]
      }
    }
}
```

### Step 4: Create the Atlas Vector Search index
> Copy and paste the following index definition into Atlas Vector Search
> Index name = vectorIndex
```
{
  "type": "vectorSearch",
  "fields": [
    {
      "numDimensions": 768,
      "path": "descriptionVectorEmbedding",
      "similarity": "cosine",
      "type": "vector"
    },
    {
      "path": "asin",
      "type": "filter"
    }
  ]
}
```


### Sample user queries to evaluate
```
I want a warm and soft white pillow fluffy
```
