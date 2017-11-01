class AnalyzeDatasetSpecificModel
  include Sidekiq::Worker
  include Analyzer
  sidekiq_options queue: :worker
  def perform(dataset_id, model_id)
    @model = MLModel.find(model_id)
    filename, manifest_filename = prep_dataset_for_analysis(dataset_id, "Dataset")
    model_filename = prep_model_for_analysis(@model)
    run_python_file("scripts/predictor_specific_model.py #{filename} #{manifest_filename} #{@dataset.col_classes[@dataset.prediction_column]} #{model_filename}")
    cleanup(filename, manifest_filename)
    `rm #{model_filename}`
  end

  def prep_model_for_analysis(model)
    filename = "tmp/"+model.id.to_s+"_"+Time.now.to_i.to_s+"_.json"
    manifest_file = File.open(filename, "w")
    manifest_file.write(model.to_json)
    manifest_file.close
    return filename
  end
end