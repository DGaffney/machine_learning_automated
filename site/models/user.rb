class User
  include MongoMapper::Document
  key :email
  key :first_name
  key :last_name
  key :affiliation
  key :password_hash
  key :reset_code
  include BCrypt

  def self.new_account(params)
    return "User with this email already exists!" if User.first(email: params[:email])
    @user = User.first_or_create(email: params[:email])
    @user.password = params[:password]
    @user.first_name = params[:first_name]
    @user.last_name = params[:last_name]
    @user.affiliation = params[:affiliation]
    @user.save!
    @user
  end

  def password
    @password ||= Password.new(password_hash)
  end

  def password=(new_password)
    @password = Password.create(new_password)
    self.password_hash = @password
  end

  def self.login(params)
    @user = User.first(email: params[:email])
    if @user.password == params[:password]
      return {success: true, user: @user}
    else
      return {success: false, user: nil}
    end
  end
  
  def file_usage
    (Dataset.where(user_id: self.id).fields(:filesize_mb).collect(&:filesize_mb).sum.round(2)/200.0)**100
  end

  def over_limit
    file_usage/200.0 > 1 ? true : false
  end
  
  def api_response
    user = JSON.parse(self.to_json)
    user.delete("password_hash")
    user["file_usage"] = self.file_usage
    user["dataset_count"] = Dataset.where(user_id: self.id).count
    user
  end
end