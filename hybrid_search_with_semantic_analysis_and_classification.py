import gzip
import pymongo
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from transformers import pipeline
import pandas
from pymongo.collection import ObjectId, OperationFailure
import numpy as np

mongodbAtlasUri = <MONGODB_ATLAS_CONNECTION_STRING>
mongodbAtlasDatabase = "hybrid_search_recommendation_xmarket"
mongodbAtlasCollectionForProducts = "hybrid_search_products"
xmarketHomeAndKitchenProductDataFilePath = <XMARKET_HOME_AND_KITCHEN_PRODUCT_DATA_FILE_PATH>

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

#init report file
def init_result_file_html(paramQuery):
    textFile = open("./hybridSearchResultWithRecommendation.html","w")
    textFile.write("<html>\n<head>\n<title> \nMongoDB Hybrid Search Results \</title>\n</head><body><h><center>User Query: "+paramQuery+"</center></h><hr></body>\n")
    textFile.close()

#insert user query results
def insert_data_result_file(paramHtmlFileResult,paramSectionTitle):
    textFile = open("./hybridSearchResultWithRecommendation.html","a")
    textFile.write("\n<body><h1><center>"+paramSectionTitle+"</center></h1>\n</body>")
    textFile.write("\n<body><i1><center>"+paramHtmlFileResult+"</center></i1>\n</body>")
    textFile.write("\n<body><j1></j1>\n<hr></body>\n")
    textFile.close()

#init mongodb connection with mongodb client
def startup_db_connection(paramMongodbAtlasConnectionString):
    try:
        mongodbClient = pymongo.MongoClient(paramMongodbAtlasConnectionString)
        mongodbClient.list_database_names()
        return mongodbClient
    except pymongo.errors.OperationFailure as err:
        print (f"Datacase Connection failed. Error: {err}")

#load xmarket datasets into mongodb collections
def open_and_load_into_mdb(paramFilePath, paramMongoClient, paramMongoCollection):
    mongodb_collection = (paramMongoClient[mongodbAtlasDatabase])[paramMongoCollection]
    example_rev_file = paramFilePath
    review_lines = []
    with gzip.open(example_rev_file, 'rt', encoding='utf8') as f:
        review_lines = f.readlines()
    for item in review_lines:  
        try:
            mongodb_collection.insert_one(eval(item))
        except:
            try:
                mongodb_collection.insert_one(eval(item)[0])
            except:
                pass

#xmarket product description embedding
def data_product_embedding(paramMongoClient, paramMongoCollection):
    mongodb_collection = (paramMongoClient[mongodbAtlasDatabase])[paramMongoCollection]
    for doc in mongodb_collection.find():
        try:
            docFilter = {"_id": doc["_id"]}
            if (len(doc["description"]) == 0):
                mongodb_collection.aggregate([
                    {
                        "$match": {
                            "_id": doc["_id"]
                        }
                    },
                    { 
                        "$addFields": {
                            "description": { 
                                "$reduce": {
                                    "input": "$features",
                                    "initialValue": "",
                                    "in": {
                                        "$cond": {
                                            "if": { "$eq": [ { "$indexOfArray": [ "$features", "$$this" ] }, 0 ] },
                                            "then": { "$concat": [ "$$value", "$$this" ] },
                                            "else": { "$concat": [ "$$value", "_", "$$this" ] }
                                        }    
                                    }
                                }        
                            }
                        }
                    },
                    {
                        "$addFields": {
                            "descriptionVectorEmbedding": (model.encode(doc["description"])).tolist(),
                        }
                    },
                    {
                        '$merge': {
                            'into': {
                                'db': 'hybrid_search_recommendation_xmarket',
                                'coll': 'hybrid_search_products'
                            },
                            'on': '_id',
                            'whenMatched': 'replace'
                        }
                    }
                ])
            else:
                descriptionVectorEmbedding = model.encode(doc["description"])
                descriptionVectorEmbeddingList = descriptionVectorEmbedding.tolist()
                newFieldAttributes = {"$set":{"descriptionVectorEmbedding":descriptionVectorEmbeddingList}}
                mongodb_collection.update_one(docFilter,newFieldAttributes)
        except Exception as ex:
            print (f"Exception encountered: {ex}")

#xmarket product full-text search
def mongodb_atlas_search_query(paramMongoClient, paramMongoCollection,paramUserQuery,paramNumOfResults):
    mongodb_collection = (paramMongoClient[mongodbAtlasDatabase])[paramMongoCollection]
    try:
        searchResult = mongodb_collection.aggregate([
        {
            '$search': {
                'index': 'searchIndex',
                'compound': {
                    'should': [{
                        'text': {
                            'path': 'features', 
                            'query': paramUserQuery,
                            'score': {
                                'boost': {
                                    'value': 2
                                }
                            }
                        },
                        'text': {
                            'path': 'title', 
                            'query': paramUserQuery,
                            'score': {
                                'boost': {
                                    'value': 3
                                }
                            }
                        }
                    }]
                }
            }
        },
        {
            "$addFields": {
                "fts_score": {
                    "$meta": "searchScore"
                }
            }
        }, 
        {
            '$project': {
                '_id': 1, 
                'title': 1,
                'fts_score': 1
            }
        },
        {
            '$sort': {
                'fts_score': -1
            }
        }, 
        {
            '$limit': paramNumOfResults * 2
        }])
        return searchResult
    except OperationFailure as err:
        print (f"Error related to mongodb_atlas_search_query function: {err}")

#xmarket product vector search
def mongodb_atlas_vector_search_query(paramMongoClient, paramMongoCollection,paramUserQueryEmbedding,paramNumOfResults):
    mongodb_collection = (paramMongoClient[mongodbAtlasDatabase])[paramMongoCollection]
    try:
        vectorSearchResult = mongodb_collection.aggregate([
        {
            "$vectorSearch": {
                'index': 'vectorIndex',
                'path': 'descriptionVectorEmbedding',
                'queryVector': paramUserQueryEmbedding,
                'numCandidates': paramNumOfResults * 40,
                "limit": paramNumOfResults * 2
            }
        },
        {
            "$addFields": {
                "vs_score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }, 
        {
            '$project': {
                '_id': 1, 
                'title': 1,
                'vs_score': 1 
            }
        }   
        ])
        return vectorSearchResult
    except OperationFailure as err:
        print (f"Error related to mongodb_atlas_vector_search_query function: {err}")


#retrieve xmarket product image
def mongodb_atlas_product_img_retrieval(paramMongoClient, paramMongoCollection,paramProductId):
    mongodb_collection = (paramMongoClient[mongodbAtlasDatabase])[paramMongoCollection]
    try:
        productImgRetrieval = mongodb_collection.aggregate([
        {
            '$match': {
                '_id': ObjectId(paramProductId)
            }
        }, {
            '$project': {
                '_id': 1, 
                'imgUrl': {
                    '$split': [
                        '$imgUrl', ',\"'
                    ]
                }
            }
        }
        ])    
        return productImgRetrieval
    except OperationFailure as err:
        print(f"Error related to mongodb_atlas_product_img_retrieval function: {err}")

def to_img_tag(path):
    return '<img src="'+ path + '"width=50">'

def correct_encoding(dictionary):
    new = {}
    for key1, val1 in dictionary.items():
        # Nested dictionaries
        if isinstance(val1, dict):
            val1 = correct_encoding(val1)

        if isinstance(val1, np.bool_):
            val1 = bool(val1)

        if isinstance(val1, np.int64):
            val1 = int(val1)

        if isinstance(val1, np.float64):
            val1 = float(val1)

        new[key1] = val1

    return new

def inspect_title_similarity(s1,s2):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    cosine_scores = util.cos_sim(model.encode(s1), model.encode(s2))
    return cosine_scores

def main():

    #step 1
    open_and_load_into_mdb(xmarketHomeAndKitchenProductDataFilePath, startup_db_connection(mongodbAtlasUri), mongodbAtlasCollectionForProducts)

    #step 2
    data_product_embedding(startup_db_connection(mongodbAtlasUri),mongodbAtlasCollectionForProducts)
    
    #step 3 - retrieve user inputs
    user_query = input("What product are you looking for? ")
    numOfResults = input("How many items should we display to you? ")
    init_result_file_html(user_query)
    user_query_embedding = model.encode(user_query)
    listOfHtmlFileResults = dict()
    pandas.set_option('mode.chained_assignment',None)
    
    #step 4 - atlas full-text search
    searchResultList = mongodb_atlas_search_query(startup_db_connection(mongodbAtlasUri),mongodbAtlasCollectionForProducts,user_query,int(numOfResults))
    searchResultDataFrame = pandas.DataFrame(list(searchResultList))
    searchResultNumpyArray = searchResultDataFrame.to_numpy()
    displaySearchResult = pandas.DataFrame(searchResultNumpyArray,columns=['ID','NAME','FTS_SCORE'])
    displaySearchResult['IMAGE'] = None
    displaySearchResult['VS_SCORE'] = 0
    dictOfProductImg = []
    for i in range(len(displaySearchResult)):
        productId = displaySearchResult['ID'].iloc[i]
        for doc in mongodb_atlas_product_img_retrieval(startup_db_connection(mongodbAtlasUri),mongodbAtlasCollectionForProducts,productId):
            dictOfProductImg.append(doc['imgUrl'][0].split('"')[1])  
    displaySearchResult['IMAGE'] = dictOfProductImg
    
    cursor_result_list = []
    mongodb_collection_cursor = ((startup_db_connection(mongodbAtlasUri))[mongodbAtlasDatabase])[mongodbAtlasCollectionForProducts]
    for searchResultDataFrameRow in searchResultDataFrame.itertuples():
        cursor_product_id = searchResultDataFrame.at[searchResultDataFrameRow.Index,'_id']
        cursor_product_asin = ""
        for doc in mongodb_collection_cursor.find({'_id':cursor_product_id},{'_id':1,'asin':1}):
            cursor_product_asin = doc['asin']
            cursor_product_vector_search_result = mongodb_collection_cursor.aggregate([
                {
                    "$vectorSearch": {
                        'index': 'vectorIndex',
                        'path': 'descriptionVectorEmbedding',
                        'filter': {'asin': {'$eq':cursor_product_asin}},
                        'queryVector': user_query_embedding.tolist(),
                        'numCandidates': 1,
                        "limit": 1
                    }
                },
                {
                    "$addFields": {
                        "vs_score": {
                            "$meta": "vectorSearchScore"
                        }
                    }
                }, 
                {
                    '$project': {
                        '_id': 1, 
                        'title': 1,
                        'vs_score': 1 
                    }
                }
            ])
        for doc in cursor_product_vector_search_result:
            cursor_result_list.append(doc)

    cursor_product_vector_search_result_dataframe = pandas.DataFrame(cursor_result_list)
    cursor_product_vector_search_result_dataframe = pandas.DataFrame(cursor_product_vector_search_result_dataframe.to_numpy(),columns=['ID','NAME','VS_SCORE'])
    displaySearchResult.update(cursor_product_vector_search_result_dataframe)
    displaySearchResult['AGG_SCORE'] = displaySearchResult['FTS_SCORE'] * displaySearchResult['VS_SCORE']
    displaySearchResult.sort_values(by='VS_SCORE',ascending=False,inplace=True)

    listOfHtmlFileResults["Keyword-led Search & Vector Search Results"] = displaySearchResult.to_html(escape=False,formatters=dict(IMAGE=to_img_tag)) 

    #step 5 - atlas vector search
    vectorSearchResultList = mongodb_atlas_vector_search_query(startup_db_connection(mongodbAtlasUri),mongodbAtlasCollectionForProducts,user_query_embedding.tolist(),int(numOfResults))
    vectorSearchResultDataFrame = pandas.DataFrame(list(vectorSearchResultList))
    vectorSearchResultNumpyArray = vectorSearchResultDataFrame.to_numpy()

    displayVectorSearchResult = pandas.DataFrame(vectorSearchResultNumpyArray,columns=['ID','NAME','VS_SCORE'])
    displayVectorSearchResult['IMAGE'] = None
    dictOfProductImg = []
    for i in range(len(displayVectorSearchResult)):
        productId = displayVectorSearchResult['ID'].iloc[i]
        for doc in mongodb_atlas_product_img_retrieval(startup_db_connection(mongodbAtlasUri),mongodbAtlasCollectionForProducts,productId):
            dictOfProductImg.append(doc['imgUrl'][0].split('"')[1])  
    displayVectorSearchResult['IMAGE'] = dictOfProductImg
    
    cursor_result_list = []
    mongodb_collection_cursor = ((startup_db_connection(mongodbAtlasUri))[mongodbAtlasDatabase])[mongodbAtlasCollectionForProducts]
    for vectorSearchResultDataFrameRow in vectorSearchResultDataFrame.itertuples():
        cursor_product_id = vectorSearchResultDataFrame.at[vectorSearchResultDataFrameRow.Index,'_id']
        cursor_product_search_result = mongodb_collection_cursor.aggregate([
            {
                '$search': {
                    'index': 'searchIndex', 
                    'compound': {
                        'should': [
                            {
                                'text': {
                                    'path': 'features', 
                                    'query': user_query, 
                                    'score': {
                                        'boost': {
                                            'value': 2
                                        }
                                    }
                                }, 
                                'text': {
                                    'path': 'title', 
                                    'query': user_query, 
                                    'score': {
                                        'boost': {
                                            'value': 3
                                        }
                                    }
                                }
                            }
                        ]
                    }
                }
            }, {
                '$addFields': {
                    'fts_score': {
                        '$meta': 'searchScore'
                    }
                }
            }, {
                '$project': {
                    '_id': 1, 
                    'title': 1, 
                    'fts_score': 1
                }
            }, {
                '$match': {
                    '_id': ObjectId(cursor_product_id)
                }
            }
        ])

        for doc in cursor_product_search_result:
            cursor_result_list.append(doc)

    cursor_product_search_result_dataframe = pandas.DataFrame(cursor_result_list)
    cursor_product_search_result_dataframe = pandas.DataFrame(cursor_product_search_result_dataframe.to_numpy(),columns=['ID_2','NAME_2','FTS_SCORE'])
    display_vector_search_result = pandas.concat([displayVectorSearchResult,cursor_product_search_result_dataframe],axis=1,join='inner')
    display_vector_search_result.drop(['ID_2','NAME_2'],axis=1,inplace=True)
    display_vector_search_result['AGG_SCORE'] = display_vector_search_result['FTS_SCORE'] * display_vector_search_result['VS_SCORE']
    display_vector_search_result.sort_values(by='FTS_SCORE',ascending=False,inplace=True)

    listOfHtmlFileResults["Vector-led Search & Keyword Search Results"] = display_vector_search_result.to_html(escape=False,formatters=dict(IMAGE=to_img_tag))

    agg_result = pandas.concat([displaySearchResult,display_vector_search_result],ignore_index=True)
    agg_result.sort_values(by='AGG_SCORE',ascending=False,inplace=True)
    agg_result = agg_result.drop(agg_result[agg_result.VS_SCORE < 0.7].index)    
    agg_result = agg_result.reindex(columns = ['ID','NAME','IMAGE','AGG_SCORE','TITLE_SCORE'])   
    agg_result['TITLE_SCORE'] = None
    
    for searchResultDataFrameRow in agg_result.itertuples(index=True):
        curr_row_title = agg_result.at[searchResultDataFrameRow.Index,'NAME']
        curr_row_title_inspection_score = inspect_title_similarity(curr_row_title,user_query)
        s = (str(curr_row_title_inspection_score)).split("[")
        agg_result.at[searchResultDataFrameRow.Index,'TITLE_SCORE'] = float((s[2].split("]"))[0])
    
    agg_result = agg_result.drop(agg_result[agg_result.TITLE_SCORE < 0.5].index) 
    agg_result = agg_result.drop(columns=['TITLE_SCORE'])
    agg_result.drop_duplicates(subset='ID',inplace=True)
    listOfHtmlFileResults["Aggregated Search Results"] = agg_result.to_html(escape=False,formatters=dict(IMAGE=to_img_tag))

    init_result_file_html(user_query)
    for key,value in listOfHtmlFileResults.items():
        insert_data_result_file(value,key)
    
if __name__ == "__main__":
    main()
