class Dataset
  include MongoMapper::Document
  key :user_id
  key :storage_type
  key :filename
  key :headers
  key :col_classes
  key :csv_data
  key :filesize_mb
  timestamps!
  def self.add_new_validated_csv(csv_obj, user_id)
    d = Dataset.new(user_id: user_id, filename: csv_obj.filename)
    d.headers = csv_obj.headers
    d.storage_type = "database"
    d.col_classes = csv_obj.col_classes
    d.csv_data = csv_obj.csv_data
    d.filesize_mb = csv_obj.filesize_mb
    d.save!
  end
end