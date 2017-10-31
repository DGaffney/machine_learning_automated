module Analyzer
  attr_accessor :current_statement
  def perform(dataset_id, dataset_model="Dataset")
    begin
      filename, manfest_filename = prep_dataset_for_analysis
      run_python_file("scripts/predictor_fast.py #{filename} #{filename.gsub(".csv", "")+"_manifest.json"} #{@dataset.col_classes[@dataset.prediction_column]} #{@dataset.prediction_speed} #{(@dataset.results["best_model"][1] rescue -100000000)}")
      if @dataset.prediction_speed == 0
        run_python_file("scripts/predictor_main.py #{filename} #{filename.gsub(".csv", "")+"_manifest.json"} #{@dataset.col_classes[@dataset.prediction_column]} #{(@dataset.results["best_model"][1] rescue -100000000)}")
        run_python_file("scripts/predictor_automl.py #{filename} #{filename.gsub(".csv", "")+"_manifest.json"} #{@dataset.col_classes[@dataset.prediction_column]} #{(@dataset.results["best_model"][1] rescue -100000000)}")
      end
    rescue => e
      binding.pry
      gg = 1
    end
  end

  def run_python_file(command)
    puts "python3.5 #{command}"
    IO.popen("python3.5 #{command}") do |io|
      io.each_line do |line|
        puts line
        @current_statement = JSON.parse(line.strip) rescue nil
        if !@current_statement.nil? && @current_statement["error"] != true
          if @current_statement["model_found"] == "true"
            @dataset.reload
            @dataset.clear_updater
            @dataset.write_final_result(@current_statement)
            @dataset.last_analyzed_at = Time.now
            @dataset.save!
          end
          @dataset.reload
          @dataset.save!
        else
          DatasetError.write_new_error_dataset(@dataset, @current_statement, "predictor_main.py")
        end
      end
    end
  end
  
  def prep_dataset_for_analysis
    @dataset = dataset_model.constantize.find(dataset_id)
    return nil if @dataset.nil?
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
    return filename, filename.gsub(".csv", "")+"_manifest.json"
  end
  
  def cleanup(filename, manifest_filename)
    @dataset = dataset_model.constantize.find(dataset_id)
    if @dataset.tipped_over?
      AnalyzeDataset.perform_async(dataset_id)
    else
      @dataset.clear_updater
      @dataset.write_final_result(@current_statement) if @current_statement["model_found"] == "true"
      @dataset.last_analyzed_at = Time.now
      @dataset.current_status = "complete"
      @dataset.save!
    end
    @dataset.wind_down(true) if @dataset.tipped_over?
    `rm #{filename}`
    `rm #{manifest_filename}`  
  end
end