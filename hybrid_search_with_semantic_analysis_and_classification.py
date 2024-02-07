import gzip
import pymongo
from sentence_transformers import SentenceTransformer
import pandas
from pymongo.collection import ObjectId, OperationFailure
import numpy as np
import datetime
import requests
from PIL import Image
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity

mongodbAtlasUri = <MONGODB_ATLAS_CONNECTION_STRING>
mongodbAtlasDatabase = "hybrid_search_recommendation_xmarket"
mongodbAtlasCollectionForProducts = "hybrid_search_products"
xmarketHomeAndKitchenProductDataFilePath = <XMARKET_HOME_AND_KITCHEN_PRODUCT_DATA_FILE_PATH>

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def init_result_file_html(paramQuery):
    textFile = open("./hybridSearchResultWithRecommendation.html","w")
    textFile.write("<html>\n<head>\n<title> \nMongoDB Hybrid Search Results \</title>\n</head><body><h><center>User Query: "+paramQuery+"</center></h><hr></body>\n")
    textFile.close()

def insert_data_result_file(paramHtmlFileResult,paramSectionTitle):
    textFile = open("./hybridSearchResultWithRecommendation.html","a")
    textFile.write("\n<body><h1><center>"+paramSectionTitle+"</center></h1>\n</body>")
    textFile.write("\n<body><i1><center>"+paramHtmlFileResult+"</center></i1>\n</body>")
    textFile.write("\n<body><j1></j1>\n<hr></body>\n")
    textFile.close()

def startup_db_connection(paramMongodbAtlasConnectionString):
    try:
        mongodbClient = pymongo.MongoClient(paramMongodbAtlasConnectionString)
        mongodbClient.list_database_names()
        return mongodbClient
    except pymongo.errors.OperationFailure as err:
        print (f"Datacase Connection failed. Error: {err}")

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

def data_product_embedding(paramMongoClient, paramMongoCollection):
    print("starting product description embedding")

    mongodb_client = paramMongoClient
    mongodb_session = mongodb_client.start_session()
    mongodb_session_id = mongodb_session.session_id

    mongodb_collection = (paramMongoClient[mongodbAtlasDatabase])[paramMongoCollection]
    cursor = mongodb_collection.find({},no_cursor_timeout=True)
    refresh_timestamp = datetime.datetime.now()

    for doc in cursor:
        print("product description embedding in progress")
        if (datetime.datetime.now() - refresh_timestamp).total_seconds() > 300:
            print("refreshing session")
            mongodb_session.client.admin.command({"refreshSessions": [ mongodb_session_id]})
            refresh_timestamp = datetime.datetime.now()
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
    cursor.close()
    print("product description embedding completed")

def data_product_title_embedding(paramMongoClient, paramMongoCollection):
    print("starting product title embedding")
    mongodb_client = paramMongoClient
    mongodb_session = mongodb_client.start_session()
    mongodb_session_id = mongodb_session.session_id
    mongodb_collection = (paramMongoClient[mongodbAtlasDatabase])[paramMongoCollection]
    cursor = mongodb_collection.find({},no_cursor_timeout=True)
    refresh_timestamp = datetime.datetime.now()
    for doc in cursor:
        print("product title embedding in progress")
        if (datetime.datetime.now() - refresh_timestamp).total_seconds() > 300:
            print("refreshing session")
            mongodb_session.client.admin.command({"refreshSessions": [ mongodb_session_id]})
            refresh_timestamp = datetime.datetime.now()
        try:
            docFilter = {"_id": doc["_id"]}
            titleVectorEmbedding = model.encode(doc["title"])
            titleVectorEmbeddingList = titleVectorEmbedding.tolist()
            newFieldAttributes = {"$set":{"titleVectorEmbedding":titleVectorEmbeddingList}}
            mongodb_collection.update_one(docFilter,newFieldAttributes)
        except Exception as ex:
            print (f"Exception encountered: {ex}")
    cursor.close()
    print("product title embedding completed")

def mongodb_atlas_search_query(paramMongoClient, paramMongoCollection,paramUserQuery,paramNumOfResults):
    mongodb_collection = (paramMongoClient[mongodbAtlasDatabase])[paramMongoCollection]
    try:
        searchResult = mongodb_collection.aggregate([
        {
            '$search': {
                'index': 'searchIndex',
                'compound': {
                    'must': [{
                        'text': {
                            'path': 'description', 
                            'query': paramUserQuery
                        },
                        'text': {
                            'path': 'title', 
                            'query': paramUserQuery
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
                "description_vs_score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }, 
        {
            '$project': {
                '_id': 1, 
                'title': 1,
                'description_vs_score': 1 
            }
        }   
        ])
        return vectorSearchResult
    except OperationFailure as err:
        print (f"Error related to mongodb_atlas_vector_search_query function: {err}")

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
    return '<img src="'+ path + '"width=70">'

def correct_encoding(dictionary):
    new = {}
    for key1, val1 in dictionary.items():
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

def cleanse_image_url(param_img_url):
    try:
        clean_data = param_img_url.split('"')
        return clean_data[1]
    except:
        pass

def load_image(param_url):
    if param_url == None:
        return None
    download_image = requests.get(param_url).content
    image_name = (param_url.split('.'))[3] + '.jpg'
    with open(image_name, 'wb') as handler:
        handler.write(download_image)
    input_image = Image.open(image_name)
    resized_image = input_image.resize((224, 224))
    return resized_image.convert("RGB")

vgg16 = VGG16(weights='imagenet', include_top=False, 
              pooling='max', input_shape=(224, 224, 3))

for model_layer in vgg16.layers:
  model_layer.trainable = False

def get_image_embeddings(param_object_image):
    object_image = load_image(param_object_image)
    if object_image == None:
        return None
    image_array = np.expand_dims(image.img_to_array(object_image), axis = 0)
    image_embedding = vgg16.predict(image_array)
    return image_embedding

def main():

    #load dataset in atlas
    #open_and_load_into_mdb(xmarketHomeAndKitchenProductDataFilePath, startup_db_connection(mongodbAtlasUri), mongodbAtlasCollectionForProducts)

    #embed product description
    #data_product_embedding(startup_db_connection(mongodbAtlasUri),mongodbAtlasCollectionForProducts)

    #embed product title
    #data_product_title_embedding(startup_db_connection(mongodbAtlasUri),mongodbAtlasCollectionForProducts)

    numOfResults = 5
    user_query = input("What product are you looking for? ")
    init_result_file_html(user_query)
    user_query_embedding = model.encode(user_query)
    listOfHtmlFileResults = dict()
    pandas.set_option('mode.chained_assignment',None)
    
    #run atlas full-text search
    searchResultList = mongodb_atlas_search_query(startup_db_connection(mongodbAtlasUri),mongodbAtlasCollectionForProducts,user_query,int(numOfResults))
    searchResultDataFrame = pandas.DataFrame(list(searchResultList))
    searchResultNumpyArray = searchResultDataFrame.to_numpy()
    displaySearchResult = pandas.DataFrame(searchResultNumpyArray,columns=['ID','NAME','FTS_SCORE'])
    displaySearchResult['IMAGE'] = None
    displaySearchResult['DESCRIPTION_VS_SCORE'] = 0
    dictOfProductImg = []
    for i in range(len(displaySearchResult)):
        productId = displaySearchResult['ID'].iloc[i]
        for doc in mongodb_atlas_product_img_retrieval(startup_db_connection(mongodbAtlasUri),mongodbAtlasCollectionForProducts,productId):
            dictOfProductImg.append(doc['imgUrl'][0].split('"')[1])  
    displaySearchResult['IMAGE'] = dictOfProductImg
    
    #run vector search on description from full-text search result
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
                        "description_vs_score": {
                            "$meta": "vectorSearchScore"
                        }
                    }
                }, 
                {
                    '$project': {
                        '_id': 1, 
                        'title': 1,
                        'description_vs_score': 1 
                    }
                }
            ])
        for doc in cursor_product_vector_search_result:
            cursor_result_list.append(doc)
    cursor_product_vector_search_result_dataframe = pandas.DataFrame(cursor_result_list)
    cursor_product_vector_search_result_dataframe = pandas.DataFrame(cursor_product_vector_search_result_dataframe.to_numpy(),columns=['ID','NAME','DESCRIPTION_VS_SCORE'])
    displaySearchResult.update(cursor_product_vector_search_result_dataframe)

    #run vector search on title from full-text search result
    cursor_result_list_2 = []
    mongodb_collection_cursor_2 = ((startup_db_connection(mongodbAtlasUri))[mongodbAtlasDatabase])[mongodbAtlasCollectionForProducts]
    for searchResultDataFrameRow in searchResultDataFrame.itertuples():
        cursor_product_id_2 = searchResultDataFrame.at[searchResultDataFrameRow.Index,'_id']
        cursor_product_asin_2 = ""
        for doc in mongodb_collection_cursor_2.find({'_id':cursor_product_id_2},{'_id':1,'asin':1}):
            cursor_product_asin_2 = doc['asin']
            cursor_product_vector_search_result_2 = mongodb_collection_cursor.aggregate([
                {
                    "$vectorSearch": {
                        'index': 'vectorIndex_2',
                        'path': 'titleVectorEmbedding',
                        'filter': {'asin': {'$eq':cursor_product_asin_2}},
                        'queryVector': user_query_embedding.tolist(),
                        'numCandidates': 1,
                        "limit": 1
                    }
                },
                {
                    "$addFields": {
                        "title_vs_score": {
                            "$meta": "vectorSearchScore"
                        }
                    }
                }, 
                {
                    '$project': {
                        '_id': 1, 
                        'title': 1,
                        'title_vs_score': 1 
                    }
                }
            ])
        for doc in cursor_product_vector_search_result_2:
            cursor_result_list_2.append(doc)
    cursor_product_search_result_dataframe_2 = pandas.DataFrame(cursor_result_list_2)
    cursor_product_search_result_dataframe_2 = pandas.DataFrame(cursor_product_search_result_dataframe_2.to_numpy(),columns=['ID_2','NAME_2','TITLE_VS_SCORE'])
    display_search_result = pandas.concat([displaySearchResult,cursor_product_search_result_dataframe_2],axis=1,join='inner')
    display_search_result.drop(['ID_2','NAME_2'],axis=1,inplace=True)

    #run atlas vector search
    vectorSearchResultList = mongodb_atlas_vector_search_query(startup_db_connection(mongodbAtlasUri),mongodbAtlasCollectionForProducts,user_query_embedding.tolist(),int(numOfResults))
    vectorSearchResultDataFrame = pandas.DataFrame(list(vectorSearchResultList))
    vectorSearchResultNumpyArray = vectorSearchResultDataFrame.to_numpy()

    displayVectorSearchResult = pandas.DataFrame(vectorSearchResultNumpyArray,columns=['ID','NAME','DESCRIPTION_VS_SCORE'])
    displayVectorSearchResult['IMAGE'] = None
    dictOfProductImg = []
    for i in range(len(displayVectorSearchResult)):
        productId = displayVectorSearchResult['ID'].iloc[i]
        for doc in mongodb_atlas_product_img_retrieval(startup_db_connection(mongodbAtlasUri),mongodbAtlasCollectionForProducts,productId):
            dictOfProductImg.append(doc['imgUrl'][0].split('"')[1])  
    displayVectorSearchResult['IMAGE'] = dictOfProductImg

    #run vector search on title from vector search result
    cursor_result_list_3 = []
    mongodb_collection_cursor_3 = ((startup_db_connection(mongodbAtlasUri))[mongodbAtlasDatabase])[mongodbAtlasCollectionForProducts]
    for vectorSearchResultDataFrameRow in vectorSearchResultDataFrame.itertuples():
        cursor_product_id_3 = vectorSearchResultDataFrame.at[vectorSearchResultDataFrameRow.Index,'_id']
        cursor_product_asin_3 = ""
        for doc in mongodb_collection_cursor_3.find({'_id':cursor_product_id_3},{'_id':1,'asin':1}):
            cursor_product_asin_3 = doc['asin']
            cursor_product_vector_search_result_3 = mongodb_collection_cursor_3.aggregate([
                {
                    "$vectorSearch": {
                        'index': 'vectorIndex_2',
                        'path': 'titleVectorEmbedding',
                        'filter': {'asin': {'$eq':cursor_product_asin_3}},
                        'queryVector': user_query_embedding.tolist(),
                        'numCandidates': 1,
                        "limit": 1
                    }
                },
                {
                    "$addFields": {
                        "title_vs_score": {
                            "$meta": "vectorSearchScore"
                        }
                    }
                }, 
                {
                    '$project': {
                        '_id': 1, 
                        'title': 1,
                        'title_vs_score': 1 
                    }
                }
            ])
        for doc in cursor_product_vector_search_result_3:
            cursor_result_list_3.append(doc)
    
    cursor_product_search_result_dataframe_3 = pandas.DataFrame(cursor_result_list_3)
    cursor_product_search_result_dataframe_3 = pandas.DataFrame(cursor_product_search_result_dataframe_3.to_numpy(),columns=['ID_2','NAME_2','TITLE_VS_SCORE'])
    display_vector_search_result = pandas.concat([displayVectorSearchResult,cursor_product_search_result_dataframe_3],axis=1,join='inner')
    display_vector_search_result.drop(['ID_2','NAME_2'],axis=1,inplace=True)

    # concatenate full-text and vector search results
    agg_search_result = pandas.concat([display_search_result,display_vector_search_result],ignore_index=True)
    agg_search_result.drop_duplicates(subset='ID',inplace=True)
    agg_search_result = agg_search_result.drop(columns=['ID'])
    agg_search_result.sort_values(by=['DESCRIPTION_VS_SCORE'],ascending=False,inplace=True)
    agg_search_result = agg_search_result.reset_index(drop=True)
    agg_search_result['IMAGE_VS_SCORE'] = None

    vectorized_get_image_embeddings = np.vectorize(get_image_embeddings)
    agg_search_result['IMAGE_EMBEDDING'] = agg_search_result['IMAGE'].apply(vectorized_get_image_embeddings)
    
    for searchResultDataFrameRow in agg_search_result.itertuples():
        ref_prodcut_image = agg_search_result.at[0,'IMAGE_EMBEDDING']
        cursor_product_image = agg_search_result.at[searchResultDataFrameRow.Index,'IMAGE_EMBEDDING']
        try: 
            agg_search_result.at[searchResultDataFrameRow.Index,'IMAGE_VS_SCORE'] = cosine_similarity(ref_prodcut_image,cursor_product_image)[0][0]
        except Exception as ex:
            print (f"Exception encountered: {ex}")
    agg_search_result = agg_search_result.drop(columns=['IMAGE_EMBEDDING'])

    for row in agg_search_result.itertuples():
        try:
            if row.IMAGE_VS_SCORE < 0.65 and row.TITLE_VS_SCORE < 0.7:
                agg_search_result.drop(index=row.Index,inplace=True)
            elif row.AGG_SCORE < 0.6 and row.TITLE_VS_SCORE < 0.7:
                agg_search_result.drop(index=row.Index,inplace=True)
            elif row.IMAGE_VS_SCORE < 0.65:
                agg_search_result.drop(index=row.Index, inplace=True)
        except:
            continue
    
    agg_search_result['AGG_SCORE'] = agg_search_result['DESCRIPTION_VS_SCORE'] * agg_search_result['TITLE_VS_SCORE']
    
    display_search_result = display_search_result.drop(columns=['ID'])
    display_search_result = display_search_result.drop(columns=['FTS_SCORE'])
    display_search_result = display_search_result.drop(columns=['DESCRIPTION_VS_SCORE'])
    display_search_result = display_search_result.drop(columns=['TITLE_VS_SCORE'])
    listOfHtmlFileResults["Keyword-led Search Results"] = display_search_result.to_html(escape=False,formatters=dict(IMAGE=to_img_tag)) 

    display_vector_search_result = display_vector_search_result.drop(columns=['ID'])
    display_vector_search_result = display_vector_search_result.drop(columns=['DESCRIPTION_VS_SCORE'])
    display_vector_search_result = display_vector_search_result.drop(columns=['TITLE_VS_SCORE'])
    listOfHtmlFileResults["Vector-led Search Results"] = display_vector_search_result.to_html(escape=False,formatters=dict(IMAGE=to_img_tag))

    agg_search_result = agg_search_result.drop(columns=['FTS_SCORE'])
    agg_search_result = agg_search_result.drop(columns=['AGG_SCORE'])
    listOfHtmlFileResults["Aggregated Search Results"] = agg_search_result.to_html(escape=False,formatters=dict(IMAGE=to_img_tag))
    
    init_result_file_html(user_query)
    for key,value in listOfHtmlFileResults.items():
        insert_data_result_file(value,key)
    
if __name__ == "__main__":
    main()
