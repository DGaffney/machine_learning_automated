class AnalyzeDatasetSpecificModel
  include Sidekiq::Worker
  include Analyzer
  sidekiq_options queue: :worker
  def perform(dataset_id, model_id)
    filename, manifest_filename = prep_dataset_for_analysis(dataset_id, "Dataset")
    run_python_file("scripts/predictor_specific_model.py #{filename} #{manifest_filename} #{@dataset.col_classes[@dataset.prediction_column]} #{@dataset.prediction_speed} #{(@dataset.results["best_model"][1] rescue -100000000)}")
  end
end