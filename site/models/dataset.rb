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
  key :results, Hash
  key :row_count, Integer
  key :feature_count, Integer
  key :csv_preview_row, Array
  timestamps!

  def self.refresh_problem_dataset(dataset_id)
    csv_data = CSV.parse(Zlib::Inflate.inflate(File.read(SETTINGS["storage_location"]+"problem_csv_data/"+dataset_id+".gzip")))
    manifest = JSON.parse(File.read(SETTINGS["storage_location"]+"problem_datasets/"+dataset_id))
    tmpname = "problem_"+rand(1000000).to_s
    csv = CSV.open("tmp/#{tmpname}.csv", "w")
    csv_data.each do |row|
      csv << row
    end;false
    csv.close
    json = File.open("tmp/#{tmpname}.json", "w")
    json.write(manifest.to_json)
    json.close
    command = "python scripts/predictor_fast.py tmp/#{tmpname}.csv tmp/#{tmpname}.json #{manifest["col_classes"][manifest["prediction_column"]]}"
    puts command
    `#{command}`
  end
  def wind_down(mail=false)
    if mail
      Mailer.send(
        User.find(self.user_id).email, 
        "Dataset #{self.filename} failed", 
        "Hey,<br/>
        Unfortunately, the dataset that you submitted didn't pass muster for the machine learners.
        Typically this happens when the dataset has many complicated data types, mixes data types in a single column,
        or has missing data in the CSV. Please double check the data for these erorrs and resubmit when you think you've cleaned it up.
        ")
    end
    `mkdir #{SETTINGS["storage_location"]+"problem_csv_data/"}`
    `mkdir #{SETTINGS["storage_location"]+"problem_conversion_pipelines/"}`
    `mkdir #{SETTINGS["storage_location"]+"problem_datasets/"}`
    f = File.open(SETTINGS["storage_location"]+"problem_datasets/"+self.id.to_s, "w")
    f.write(self.to_json)
    f.close
    binding.pry
    `mv #{SETTINGS["storage_location"]+"csv_data/"+self.id.to_s+".gzip"} #{SETTINGS["storage_location"]+"problem_csv_data/"}`
    `mv #{SETTINGS["storage_location"]+"conversion_pipelines/"+self.id.to_s+".gzip"} #{SETTINGS["storage_location"]+"problem_conversion_pipelines/"}`
    `rm -f #{SETTINGS["storage_location"]+"ml_models/"+self.id.to_s+".pkl"}`
    `rm -rf #{SETTINGS["storage_location"]+"public/images/"+self.id.to_s}`
    self.destroy
  end

  def tipped_over?
    self.results && self.results.empty? && self.current_status == "complete"
  end

  def continuous_measurement_human_language
    if self.results["diagnostic_results"]["r2"] > 0.8
      return "is extremely accurate"
    elsif self.results["diagnostic_results"]["r2"] > 0.6
      return "is very accurate"
    elsif self.results["diagnostic_results"]["r2"] > 0.4
      return "is mildly accurate"
    else self.results["diagnostic_results"]["r2"] > 0.2
      return "is weak"
    end
  end

  def binary_measurement_human_language
    if self.results["diagnostic_results"]["accuracy"] > 0.8# && self.results["diagnostic_results"]["auc"] > 0.8
      return "is very accurate"
    elsif self.results["diagnostic_results"]["accuracy"] > 0.6# && self.results["diagnostic_results"]["auc"] > 0.5
      return "is mildly accurate"
    elsif self.results["diagnostic_results"]["accuracy"] > 0.5# && self.results["diagnostic_results"]["auc"] > 0.5
      return "is better than chance"
    else self.results["diagnostic_results"]["accuracy"] > 0.5# && self.results["diagnostic_results"]["auc"] > 0.5
      return "performs poorly"
    end
  end

  def get_updated_at
    lu = self.latest_update
    if lu && lu["time"] && Time.parse(lu["time"]) > self.updated_at
      return Time.parse(lu["time"])
    else
      return self.updated_at
    end
  end

  def self.add_new_validated_csv(csv_obj, user_id)
    d = Dataset.new(user_id: user_id, filename: csv_obj.filename)
    d.headers = csv_obj.headers
    d.storage_type = "database"
    d.col_classes = csv_obj.col_classes
    d.user_id = user_id
    d.row_count = csv_obj.csv_data.count
    d.feature_count = csv_obj.csv_data[0].count-1
    csv_obj.csv_data = csv_obj.csv_data[1..-1];false if csv_obj.headers == csv_obj.csv_data[0]
    f = File.open(SETTINGS["storage_location"]+"/csv_data/"+d.id.to_s+".gzip", "w")
    f.write(Zlib::Deflate.deflate(csv_obj.csv_data.collect{|r| CSV.generate{|csv| csv << r}}.join("")))
    f.close
    d.filesize_mb = csv_obj.filesize_mb
    d.save!
    d
  end

  def percent_complete
    (self.latest_update["percent"].to_f.round(4)*100) rescue nil
  end

  def get_current_status
    if self.current_status == "complete" && self.results && !self.results.empty?
      return "Complete"
    elsif self.latest_update["percent"]
      return "≈"+self.percent_complete.to_s+"% Processed"
    elsif self.latest_update["status"]
      return self.latest_update["status"].capitalize
    end
  end

  def latest_update
    JSON.parse($redis.hget("updates", self.id.to_s)) rescue {}
  end

  def clear_updater
    $redis.hget("updates", self.id.to_s)
  end
  
  def set_update(status)
    $redis.hset("updates", self.id.to_s, status.to_json)
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
    JSON.parse(Zlib::Inflate.inflate(File.read(SETTINGS["storage_location"]+"/conversion_pipelines/"+self.id.to_s+".gzip")))
  end
  
  def model_name
    self.results["model_name"] rescue nil
  end
  
  def model_name
    self.results["model_params"] rescue nil
  end

  def model_wiki_link
    {"AdaBoostClassifier" => "https://en.wikipedia.org/wiki/AdaBoost",
    "BayesianRidge" => "https://en.wikipedia.org/wiki/Bayesian_linear_regression",
    "DecisionTreeClassifier" => "https://en.wikipedia.org/wiki/Decision_tree_learning",
    "DecisionTreeRegressor" => "https://en.wikipedia.org/wiki/Decision_tree_learning",
    "ElasticNet" => "https://en.wikipedia.org/wiki/Elastic_net_regularization",
    "GaussianNB" => "https://en.wikipedia.org/wiki/Naive_Bayes_classifier",
    "GaussianProcessRegressor" => "https://en.wikipedia.org/wiki/Kriging",
    "GradientBoostingClassifier" => "https://en.wikipedia.org/wiki/Gradient_boosting",
    "GradientBoostingRegressor" => "https://en.wikipedia.org/wiki/Gradient_boosting",
    "KNeighborsClassifier" => "https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm",
    "Lasso" => "https://en.wikipedia.org/wiki/Lasso_(statistics)",
    "LassoLars" => "https://en.wikipedia.org/wiki/Lasso_(statistics)",
    "LinearRegression" => "https://en.wikipedia.org/wiki/Linear_regression",
    "LinearSVC" => "https://en.wikipedia.org/wiki/Support_vector_machine",
    "LogisticRegression" => "https://en.wikipedia.org/wiki/Logistic_regression",
    "MLPClassifier" => "https://en.wikipedia.org/wiki/Multilayer_perceptron",
    "NearestCentroid" => "https://en.wikipedia.org/wiki/Nearest_centroid_classifier",
    "NearestNeighbors" => "https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm",
    "NuSVC" => "https://en.wikipedia.org/wiki/Support_vector_machine",
    "Perceptron" => "https://en.wikipedia.org/wiki/Perceptron",
    "RandomForestClassifier" => "https://en.wikipedia.org/wiki/Random_forest",
    "Ridge" => "https://en.wikipedia.org/wiki/Tikhonov_regularization",
    "RidgeCV" => "https://en.wikipedia.org/wiki/Tikhonov_regularization",
    "SGDClassifier" => "https://en.wikipedia.org/wiki/Gradient_descent",
    "SVC" => "https://en.wikipedia.org/wiki/Support_vector_machine",
    "SVR" => "https://en.wikipedia.org/wiki/Support_vector_machine"}
  end
  
  def predict(data)
    tmpname = self.id.to_s+rand(1000000000).to_s+".json"
    filename = SETTINGS["storage_location"]+"/predictions/"+tmpname
    f = File.open(filename, "w")
    f.write(data)
    f.close
    predictions = JSON.parse(`python scripts/predict_data.py #{self.id} #{filename}`)
    return predictions
  end
end