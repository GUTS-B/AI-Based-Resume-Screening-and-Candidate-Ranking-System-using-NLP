import java.util.*;

/** NLP pipeline: lowercase -> clean -> tokenize -> stop-word removal -> light stemming. */
public class Preprocessor {
    static final Set<String> STOP = new HashSet<>(Arrays.asList(
        "a","an","the","and","or","of","to","in","for","with","on","at","by","is","are","was","were","be",
        "as","from","that","this","it","i","my","we","our","have","has","had","will","can","using","used",
        "use","work","worked","working","year","years","experience","responsible","good","strong"));

    public static List<String> tokens(String text) {
        List<String> out = new ArrayList<>();
        for (String w : text.toLowerCase().replaceAll("[^a-z0-9+#. ]", " ").split("\\s+")) {
            w = w.replaceAll("^\\.+|\\.+$", "");
            if (w.length() < 2 || STOP.contains(w)) continue;
            out.add(stem(w));
        }
        return out;
    }

    /** Simple suffix-stripping stemmer (developing -> develop). */
    static String stem(String w) {
        for (String s : new String[]{"ing", "ed", "es", "s"})
            if (w.length() > s.length() + 2 && w.endsWith(s)) return w.substring(0, w.length() - s.length());
        return w;
    }
}
