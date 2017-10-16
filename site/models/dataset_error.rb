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
  key :script_response, Hash
  key :closed, Boolean
  timestamps!
  
  def self.write_new_error_dataset(dataset, error_message, script_ran)
    dataset_attrs = dataset.attributes;false
    dataset_attrs.delete("_id");false
    @de = DatasetError.new(dataset_attrs);false
    @de.script_ran = script_ran;false
    @de.script_response = error_message;false
    f = File.open(SETTINGS["storage_location"]+"/csv_data_error/"+@de.id.to_s+".gzip", "w");false
    f.write(Zlib::Deflate.deflate(dataset.csv_data.collect{|r| CSV.generate{|csv| csv << r}}.join("")));false
    f.close;false
    @de.save!
    @de
  end
  
  def csv_data
    CSV.parse(Zlib::Inflate.inflate(File.read(SETTINGS["storage_location"]+"/csv_data_error/"+self.id.to_s+".gzip")))
  end
  
  def prime
    @dataset = self
    dataset_id = @dataset.id.to_s
    @dataset.current_status = "analyzing"
    @dataset.save!
    filename = "tmp/"+dataset_id.to_s+"_"+Time.now.to_i.to_s+"_"+@dataset.filename
    csv = CSV.open(filename, "w")
    @dataset.csv_data.each do |row|
      csv << row
    end;false
    csv.close
    manifest_file = File.open(filename.gsub(".csv", "")+"_manifest.json", "w")
    manifest_file.write(@dataset.to_json)
    manifest_file.close
    script = @dataset.prediction_speed == 2 ? "predictor_fast" : "predictor_main"
    current_statement = {}
    statements = []
    puts [self.script_response["error_type"], self.script_response["message"], self.script_response["traceback"].collect{|x| x.join(" ")}.join("\n")].join("----------------------------\n")
    puts "python scripts/#{@dataset.script_ran} #{filename} #{filename.gsub(".csv", "")+"_manifest.json"} #{@dataset.col_classes[@dataset.prediction_column]} #{@dataset.prediction_speed}"
  end
end