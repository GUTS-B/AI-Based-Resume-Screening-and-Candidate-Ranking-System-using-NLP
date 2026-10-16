import java.util.*;

/** Multinomial Naive Bayes with Laplace smoothing, trained from scratch on labelled resumes. */
public class NaiveBayes {
    Map<String, Map<String, Integer>> wordCount = new HashMap<>();
    Map<String, Integer> totalWords = new HashMap<>(), docCount = new HashMap<>();
    Set<String> vocab = new HashSet<>();
    int n = 0;

    public void train(String label, List<String> tokens) {
        n++;
        docCount.merge(label, 1, Integer::sum);
        Map<String, Integer> m = wordCount.computeIfAbsent(label, k -> new HashMap<>());
        for (String t : tokens) { m.merge(t, 1, Integer::sum); totalWords.merge(label, 1, Integer::sum); vocab.add(t); }
    }

    /** Returns P(category | resume) for every category. */
    public Map<String, Double> predict(List<String> tokens) {
        Map<String, Double> logP = new HashMap<>();
        double max = -1e18;
        for (String c : docCount.keySet()) {
            double lp = Math.log(docCount.get(c) / (double) n);
            Map<String, Integer> m = wordCount.get(c);
            for (String t : tokens)
                lp += Math.log((m.getOrDefault(t, 0) + 1.0) / (totalWords.get(c) + vocab.size()));
            logP.put(c, lp); max = Math.max(max, lp);
        }
        double sum = 0;
        for (String c : logP.keySet()) { double e = Math.exp(logP.get(c) - max); logP.put(c, e); sum += e; }
        for (String c : logP.keySet()) logP.put(c, logP.get(c) / sum);
        return logP;
    }
}
