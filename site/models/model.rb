class MLModel
  include MongoMapper::Document
  key :name, String
  key :internal_name, String
  key :original_dataset, BSON::ObjectId
  key :params, Hash
end