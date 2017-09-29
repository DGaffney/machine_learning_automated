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
end