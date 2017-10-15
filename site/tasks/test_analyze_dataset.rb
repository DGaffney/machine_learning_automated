class TestAnalyzeDataset
  include Sidekiq::Worker
  include Analyzer
  sidekiq_options queue: :test_worker
end