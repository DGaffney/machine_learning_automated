class DatasetError
  include MongoMapper::Document
  key :user_id, BSON::ObjectId
  key :storage_type, String
  key :filename, String
  key :headers, Array
  key :col_classes, Array
  key :filesize_mb, Float
  key :prediction_accuracy, Integer
  key :prediction_speed, Integer
  key :prediction_column, Integer
  key :last_analyzed_at, Time
  key :current_status
  key :results, Hash
  key :row_count, Integer
  key :feature_count, Integer
  key :csv_preview_row, Array
  key :script_ran, String
  key :script_response, String
  timestamps!
end