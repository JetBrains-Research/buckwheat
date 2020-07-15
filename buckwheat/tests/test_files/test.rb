# Declaration of get_stdout function
def get_stdout(&block)
  out = $stdout
  $stdout = tmp = StringIO.new
  begin
    yield
  ensure
    $stdout = out
  end
  tmp.string
end

=begin
Deal with stderr
=end
def get_stderr(&block)
  out = $stderr
  $stderr = tmp = StringIO.new
  begin
    yield
  ensure
    $stderr = out
  end
  tmp.string
end
