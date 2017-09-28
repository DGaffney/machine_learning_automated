require 'pry'
require 'mongo_mapper'
require 'hashie'
require 'oauth'
require 'sinatra'
require 'oauth'
require 'csv'
require 'dgaff'
MORE_SETTINGS = YAML.load(File.read("settings.json")) rescue {"download_path" => "#{`pwd`.strip}/../data"}
MongoMapper.connection = Mongo::MongoClient.new("localhost", 27017, :pool_size => 25, :op_timeout => 600000, :timeout => 600000, :pool_timeout => 600000)
#MongoMapper.connection["admin"].authenticate(MORE_SETTINGS["db_user"], MORE_SETTINGS["db_password"])
MongoMapper.database = "ml_automator"

Dir[File.dirname(__FILE__) + '/extensions/*.rb'].each {|file| require file }
Dir[File.dirname(__FILE__) + '/models/*.rb'].each {|file| require file }
Dir[File.dirname(__FILE__) + '/helpers/*.rb'].each {|file| require file }
Dir[File.dirname(__FILE__) + '/handlers/*.rb'].each {|file| require file }
Dir[File.dirname(__FILE__) + '/before_hooks/*.rb'].each {|file| require file }
Dir[File.dirname(__FILE__) + '/lib/*.rb'].each {|file| require file }
set :erb, :layout => :'layouts/main'
enable :sessions

helpers LayoutHelper
