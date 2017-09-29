class Dataset
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
  key :latest_update, Hash
  timestamps!

  def self.add_new_validated_csv(csv_obj, user_id)
    d = Dataset.new(user_id: user_id, filename: csv_obj.filename)
    d.headers = csv_obj.headers
    d.storage_type = "database"
    d.col_classes = csv_obj.col_classes
    d.user_id = user_id
    f = File.open(SETTINGS["storage_location"]+"/csv_data/"+d.id.to_s+".gzip", "w")
    f.write(Zlib::Deflate.deflate(csv_obj.csv_data.collect{|r| CSV.generate{|csv| csv << r}}.join("")))
    f.close
    d.filesize_mb = csv_obj.filesize_mb
    d.save!
    d
  end
  
  def csv_data
    CSV.parse(Zlib::Inflate.inflate(File.read(SETTINGS["storage_location"]+"/csv_data/"+self.id.to_s+".gzip")))
  end

  def dataset_description
    if [[2,2]].include?([prediction_accuracy, prediction_speed])
      return "You chose #{pred_acc_text[prediction_accuracy]} accuracy and #{pred_spd_text[prediction_speed]} speed - this is a very hard combination so expect only a few guesses, since we'll be using fast (but not as accurate) algorithms and will have to restrict to only high-certainty cases."
    elsif prediction_accuracy == 2
      return "You chose #{pred_acc_text[prediction_accuracy]} accuracy and #{pred_spd_text[prediction_speed]} speed - precise predictions are always hard, so we'll be conserative in giving you guesses - of the guesses we return, however, the results will be associated with high certainty!"
    elsif prediction_speed == 2
      return "You chose #{pred_acc_text[prediction_accuracy]} accuracy and #{pred_spd_text[prediction_speed]} speed - fast predictions are always hard, so we'll be using fast (but not as accurate) algorithms."
    else
      return "You chose #{pred_acc_text[prediction_accuracy]} accuracy and #{pred_spd_text[prediction_speed]} speed - this is a reasonable setting and we'll be in touch when results are done!"
    end
  end
  
  def pred_acc_text
    {0 => "loose", 1 => "normal", 2 => "precise"}
  end

  def pred_spd_text
    {0 => "slow", 1 => "normal", 2 => "fast"}
  end
  
  def write_final_result(current_statement)
    f = File.open(SETTINGS["storage_location"]+"/conversion_pipelines/"+self.id.to_s+".gzip", "w")
    f.write(Zlib::Deflate.deflate(current_statement["conversion_pipeline"].to_json))
    f.close
    self.results = current_statement
    self.results.delete("conversion_pipeline")
    self.save!
  end

  def conversion_pipeline
    Zlib::Inflate.inflate(File.read(SETTINGS["storage_location"]+"/conversion_pipelines/"+self.id.to_s+".gzip"))
  end
end