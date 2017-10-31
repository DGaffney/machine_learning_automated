class MLModel
  include MongoMapper::Document
  key :name, String
  key :internal_name, String
  key :dataset_id, BSON::ObjectId
  key :params, Hash
  key :user_id, BSON::ObjectId
  timestamps!
end