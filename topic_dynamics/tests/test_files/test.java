    private static void collectAndWriteStats(PrintWriter writer, File[] subDirs, List<String> features)
            throws IOException {
        int total = subDirs.length;
        int now = 1;
        for (File subDir : subDirs) {
            if (subDir.isDirectory()) {
                System.out.println(now + "/" + total + ", collecting from: " + subDir.getName());
                List<StatisticsHolder> allStats = StatisticsCollector.collectFromProject(subDir.toPath());
                if (allStats.size() < MIN_FILES_TO_TEST) {
                    continue;
                }
                for (StatisticsHolder stats : allStats) {
                    writeFileStats(writer, stats, features);
                }
            }
            now++;
        }
    }
