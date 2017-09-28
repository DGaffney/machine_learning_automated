get "/" do
  erb :"index"
end

post "/preview" do
  user_id = 1
  csv = CSV.parse(params["file"][:tempfile].read) rescue nil
  if csv.nil?
    flash[:error] = "CSV could not be read. Try again please!"
    redirect "/"
  else
    @csv = CSVValidator.new(csv, params["file"][:filename], params["file"][:tempfile].size/1024.0/1024)
    if @csv.class == String
      flash[:error] = @csv
      redirect "/"
    else
      @d = Dataset.add_new_validated_csv(@csv, user_id)
      erb :"preview"
    end
  end
end