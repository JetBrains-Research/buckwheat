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
