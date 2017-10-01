def current_user
  User.find(session[:current_user_id])
end

def current_user_id
  BSON::ObjectId(session[:current_user_id].to_s)
end

get "/datasets/:user_id/:dataset_id" do
  redirect "/" if current_user.nil?
  if params[:user_id] == current_user_id.to_s || current_user.email == "itsme@devingaffney.com"
    @dataset = Dataset.find(params[:dataset_id])
    erb :"finish"
  else
    flash[:error] = "You must be logged in as a different user to see this page"
    redirect "/"
  end
end

get "/" do
  erb :"index"
end

post "/preview" do
  redirect "/" if current_user.nil?
  csv = CSV.parse(params["file"][:tempfile].read) rescue nil
  params["file"][:filename] = params["file"][:filename].gsub(" ", "_")
  if csv.nil?
    flash[:error] = "CSV could not be read. Try again please!"
    redirect "/profile"
  else
    @csv = CSVValidator.new(csv, params["file"][:filename], params["file"][:tempfile].size/1024.0/1024)
    results = @csv.validate
    if results.class == String
      flash[:error] = results
      redirect "/profile"
    else
      @csv.csv_data = csv
      @d = Dataset.add_new_validated_csv(@csv, current_user_id)
      @csv_data = @d.csv_data
      erb :"preview"
    end
  end
end

post "/datasets/:user_id/:dataset_id" do
  redirect "/" if current_user.nil?
  @dataset = Dataset.find(params["dataset_id"])
  redirect "/profile" if @dataset.nil?
  params.select{|k,v| k.include?("header_class_")}.collect{|k,v| [k.gsub("header_class_", "").to_i, v]}.each do |row_num, classtype|
    @dataset.col_classes[row_num] = classtype if @dataset.col_classes[row_num] != classtype
  end;false
  @dataset.prediction_accuracy = params["prediction_accuracy"]
  @dataset.prediction_speed = params["prediction_speed"]
  @dataset.prediction_column = params["prediction_column"]
  prediction_example = []
  @dataset.csv_data.shuffle.first.each_with_index do |el, i|
    prediction_example << el if i != @dataset.prediction_column
  end
  @dataset.csv_preview_row = prediction_example
  @dataset.save!
  @dataset.save!
  @dataset.set_update({"status" => "queued"})
  AnalyzeDataset.perform_async(@dataset.id)
  erb :"finish"
end

get "/login" do
  erb :"login"
end

get "/logout" do
  session[:current_user_id] = nil
  redirect "/"
end

post "/login" do
  results = User.login(params)
  if results[:success] == true
    session[:current_user_id] = results[:user].id.to_s
    redirect "/profile"
  else
    flash[:error] = "Sorry, but the email/password didn't match anything on file. Care to try again?"
    redirect "/login"
  end
end

get "/reset_request" do
  erb :"reset_request"
end

post "/reset_request" do
  @user = User.first(email: params[:email])
  if @user
    @user.reset_code = BSON::ObjectId.new
    Mailer.send(params[:email], "Password Reset Link!", "<a href=\"http://machinelearning.devingaffney.com/reset?reset_code=#{@user.reset_code}\">Click to reset password</a>")
    @user.save!
    flash[:error] = "A reset code is being sent to you now. Check your email for a link to reset your password."
    redirect "/profile"
  else
    flash[:error] = "Sorry, but the email didn't match anything on file or the passwords weren't the same. Care to try again?"
    redirect "/reset"
  end
end

get "/reset" do
  @reset_code = params[:reset_code]
  erb :"reset"
end

post "/reset" do
  @user =  User.first(reset_code: BSON::ObjectId(params[:reset_code].to_s))
  if @user.nil?
    flash[:error] = "The reset code appears to be invalid. Please try to request a reset again."
    redirect "/reset_request"
  elsif params[:password] != params[:password_repeat]
    flash[:error] = "Sorry, but the passwords weren't the same. Care to try again?"
    redirect "/reset?reset_code=#{params[:reset_code]}"
  else
    @user.reset_code = nil
    @user.password = params[:password]
    @user.save!
    session[:current_user_id] = @user.id
    flash[:error] = "Successfully updated password!"
    redirect "/profile"
  end
end

get "/create_account" do
  erb :"create_account"
end

post "/create_account" do
  if params[:password].length < 8
    flash[:error] = "Password must be at least Eight characters"
    redirect "/create_account"
  end
  @user = User.new_account(params)
  if @user.class == String
    flash[:error] = @user
    redirect "/create_account"
  end
  session[:current_user_id] = @user.id.to_s
  redirect "/profile"
end

get "/profile" do
  redirect "/" if current_user.nil?
  @datasets = Dataset.where(user_id: current_user_id)
  erb :"profile"
end

get "/api/:user_id" do
  return User.find(params[:user_id]).api_response.to_json
end

get "/api/:user_id/datasets" do
  @user = User.find(params[:user_id])
  return {error: "Account not found"}.to_json if @user.nil?
  return Dataset.where(user_id: @user.id).to_a.to_json
end

get "/api/:user_id/dataset/:dataset_id" do
  @user = User.find(params[:user_id])
  return {error: "Account not found"}.to_json if @user.nil?
  @dataset = Dataset.find(params[:dataset_id])
  return JSON.parse(@dataset.to_json).merge(conversion_pipeline: @dataset.conversion_pipeline).to_json
end

post "/api/:user_id/predict" do
  @user = User.find(params[:user_id])
  return {error: "Account not found"}.to_json if @user.nil?
  @dataset = Dataset.find(params[:dataset_id])
  return @dataset.predict(params[:data]).to_json
end