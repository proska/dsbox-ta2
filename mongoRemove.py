import pymongo
from pymongo import MongoClient

from pprint import pprint

client = MongoClient('localhost', 27017)

db = client.fastmeta

collection = db.fastmeta_log

collection.delete_many({})
