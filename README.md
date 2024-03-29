# Hybrid search integrating semantic analysis and classification

<img width="1427" alt="Screenshot 2024-01-17 at 10 28 40" src="https://github.com/JuicySantos06/hybrid-search-integrating-semantic-analysis-and-classification/assets/84564830/82f58429-36c9-4092-b2d9-66aaa8852698">

## General Information
> The following demo aims to demonstrate the ability to enhance the search experience through the joint use of both keyword-based and vector-based search technology.
> A semantic analysis and classification mechanism will be added to the hybrid search made possible by the joint use of Atlas Search and Atlas Vector Search to enhance the search engine.
> We will be using using the following native MongoDB technologies:
* MongoDB Atlas Search
* MongoDB Atlas Vector Search


### Step 1: Import the Cross-Market Recommendations dataset into your Atlas Database
> Download the US => Home and Kitchen/metadata category dataset.
```
link: https://xmrec.github.io/data/us/
```
> Create the following database and collection in Atlas
```
DB_NAME = hybrid_search_recommendation_xmarket
COLLECTION_NAME = hybrid_search_products
```

### Step 2: Embed the home and kitchen products dataset
> Update the file hybrid_search_with_semantic_analysis_and_classification.py with the following information:
* Your MongoDB Atlas connection string
```
mongodbAtlasUri = <MONGODB_ATLAS_CONNECTION_STRING>
```
* The path to the cross-market recommendation dataset that you previously downloaded
```
xmarketHomeAndKitchenProductDataFilePath = <XMARKET_HOME_AND_KITCHEN_PRODUCT_DATA_FILE_PATH>
```
> Only execute the following lines of codes from the python script file:
```
> open_and_load_into_mdb(xmarketHomeAndKitchenProductDataFilePath, startup_db_connection(mongodbAtlasUri), mongodbAtlasCollectionForProducts)
> data_product_embedding(startup_db_connection(mongodbAtlasUri),mongodbAtlasCollectionForProducts)
> data_product_title_embedding(startup_db_connection(mongodbAtlasUri),mongodbAtlasCollectionForProducts)
```
> After completion, you can comment those steps for later use of the script.

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
> Index name = vectorIndex_2
```
{
  "fields": [
    {
      "numDimensions": 768,
      "path": "titleVectorEmbedding",
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

### Step5: Run the script
> I share the vector search score for description, title and image objects.
> You can see the impact of those data on a product's ranking.

