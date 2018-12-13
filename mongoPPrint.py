import pymongo
from pymongo import MongoClient

from pprint import pprint

client = MongoClient('localhost', 27017)

db = client.fastmeta

collection = db.fastmeta_log

# pprint(collection)
print(collection.count())
# for col in collection.find({}):
#     for keys in col.keys():
#         print('{', keys, ":", col[keys], '}')
