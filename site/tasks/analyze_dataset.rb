class AnalyzeDataset
  include Sidekiq::Worker
  include Analyzer
  sidekiq_options queue: :worker
end