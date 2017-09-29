class AnalyzeDataset
  include Sidekiq::Worker
  
  def perform(dataset_id)
    @dataset = Dataset.find(dataset_id)
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
    run_python_file("python scripts/#{script}.py #{filename} #{filename.gsub(".csv", "")+"_manifest.json"} #{@dataset.col_classes[@dataset.prediction_column]}") do |line|
      current_statement = JSON.parse(line.strip) rescue nil
      if !current_statement.nil?
        @dataset.reload
        @dataset.latest_update = current_statement if current_statement["status"] != "complete"
        @dataset.save!
      end
    end
    binding.pry
    @dataset.write_final_result(current_statement)
    @dataset.last_analyzed_at = Time.now
    @dataset.current_status = "complete"
    @dataset.save!
  end
  
  def run_python_file(command)
    Open3.popen3(command) do |stdin, stdout, stderr, wait_thr|
      while line = stdout.gets
        yield line
      end
    end
  end
end