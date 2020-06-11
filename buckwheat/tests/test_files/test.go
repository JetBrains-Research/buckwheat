var (
	version = "not-set"
	commit  = "not-set"
)

func printBreakDown(out map[string][]string, buff *bytes.Buffer) {
	for name, language := range out {
		fmt.Fprintln(buff, name)
		for _, file := range language {
			fmt.Fprintln(buff, file)
		}

		fmt.Fprintln(buff)
	}
	data, err := readFile(file, limit)
	if err != nil {
		return err
	}
}
