require 'time'
class CSVValidator
  attr_accessor :csv_data
  attr_accessor :headers
  attr_accessor :col_classes
  attr_accessor :filename
  attr_accessor :filesize_mb
  def initialize(csv_data, filename, filesize_mb)
    self.filesize_mb = filesize_mb
    self.csv_data = csv_data;false #csv_data.shuffle.first(1000) if it needs to be shortened during validation
    self.filename = filename
  end

  def validate
    return "Must be < 10mb in order to be analyzed - larger files coming in the future" if filesize_mb > 10
    transposable = self.csv_data.transpose rescue nil ; false
    return "Must be an even CSV with same number of columns for each row!" if transposable.nil?
    cols = []
    classes = []
    transposable.each do |col|
      cols << col.collect{|v| to_something(v)}
      classes << cols[-1].collect(&:class)
    end;false
    header = contains_header(classes)
    tmpclasses = header ? classes.collect{|c| c[1..-1]} : classes
    self.headers = header ? self.csv_data[0] : 1.upto(cols.count).collect{|x| "Column #{x} (#{self.col_classes[x-1]})"}
    self.csv_data = header ? cols.collect{|c| c[1..-1]} : cols
    self.col_classes = get_classes(tmpclasses, header)    
    self.csv_data = self.csv_data.transpose
    true
  end

  def human_class(classname)
    if classname == "Fixnum"
      return "Integer"
    else
      return classname
    end
  end

  def get_classes(classes, header)
    derived_classes = []
    classes.collect(&:uniq).each_with_index do |class_set, i|
      if class_set.include?(String)
        if (Math.log(self.csv_data[i].length, 10) - Math.log(self.csv_data[i].uniq.length, 10)) > 2 || self.csv_data[i].uniq.length < 20
          derived_classes << "Categorical"
        else
          if self.csv_data.collect{|x| x.split(" ").length}.median > 2
            derived_classes << "Text"
          else
            derived_classes << "Phrase"
          end
        end
      elsif class_set.uniq.collect(&:to_s).sort == ["Fixnum", "Float"]
        derived_classes << "Float"
      elsif class_set.uniq.count == 1
        derived_classes << human_class(class_set.first.to_s)
      elsif class_set.include?(String)
        derived_classes << "Text"
      else
        derived_classes << "Text"
      end
    end
    return derived_classes
  end

  def contains_header(classes)
  self.csv_data.collect(&:first)
    classes.collect(&:first).uniq.first == String && classes.collect(&:first) != classes.collect(&:last)
  end

  def to_something(str)
    if str.to_s.downcase == "true"
      return 1
    elsif str.to_s.downcase == "false"
      return 0
    elsif (num = Integer(str) rescue Float(str) rescue nil)
      num
    else 
      tm = Chronic.parse(str) rescue nil
      return tm if !tm.nil?
      # Time.parse does not raise an error for invalid input
      str
    end
  end
end