import com.sun.net.httpserver.*;
import java.io.*;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

/** Trains the model at startup, serves the frontend and exposes POST /api/rank. */
public class Server {
    static NaiveBayes nb = new NaiveBayes();
    static List<String> skills = new ArrayList<>();
    static class R { String name; double score, cos, sk, cls; List<String> matched = new ArrayList<>(), missing = new ArrayList<>(); }

    public static void main(String[] args) throws Exception {
        for (String line : Files.readAllLines(Paths.get("data/training.csv"), StandardCharsets.UTF_8)) {
            int i = line.indexOf('|');
            if (i > 0) nb.train(line.substring(0, i).trim(), Preprocessor.tokens(line.substring(i + 1)));
        }
        for (String s : Files.readAllLines(Paths.get("data/skills.txt"))) if (!s.isBlank()) skills.add(s.trim().toLowerCase());
        System.out.println("Model trained on " + nb.n + " resumes, vocabulary = " + nb.vocab.size() + " words");

        HttpServer srv = HttpServer.create(new InetSocketAddress(8080), 0);
        srv.createContext("/", ex -> send(ex, 200, "text/html; charset=utf-8", Files.readAllBytes(Paths.get("web/index.html"))));
        srv.createContext("/api/rank", ex -> {
            try {
                String body = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                send(ex, 200, "application/json; charset=utf-8", rank(body).getBytes(StandardCharsets.UTF_8));
            } catch (Exception e) { send(ex, 500, "text/plain", String.valueOf(e).getBytes()); }
        });
        srv.start();
        System.out.println("Open http://localhost:8080");
    }

    static void send(HttpExchange ex, int code, String type, byte[] data) throws IOException {
        ex.getResponseHeaders().set("Content-Type", type);
        ex.sendResponseHeaders(code, data.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(data); }
    }

    /** Body format: JD \u0001 name \u0002 resumeText \u0001 name \u0002 resumeText ... */
    static String rank(String body) {
        String[] parts = body.split("\u0001");
        String jd = parts[0];
        List<String> names = new ArrayList<>(), texts = new ArrayList<>();
        for (int i = 1; i < parts.length; i++) {
            String[] p = parts[i].split("\u0002", 2);
            if (p.length == 2 && !p[1].isBlank()) { names.add(p[0]); texts.add(p[1]); }
        }
        List<List<String>> docs = new ArrayList<>();
        docs.add(Preprocessor.tokens(jd));
        for (String t : texts) docs.add(Preprocessor.tokens(t));
        int N = docs.size();
        Map<String, Integer> df = new HashMap<>();
        for (List<String> d : docs) for (String w : new HashSet<>(d)) df.merge(w, 1, Integer::sum);
        List<Map<String, Double>> vecs = new ArrayList<>();
        for (List<String> d : docs) {
            Map<String, Double> v = new HashMap<>();
            for (String w : d) v.merge(w, 1.0, Double::sum);
            for (Map.Entry<String, Double> e : v.entrySet())
                e.setValue(e.getValue() * (Math.log((N + 1.0) / (df.get(e.getKey()) + 1)) + 1));
            vecs.add(v);
        }
        Map<String, Double> jdProb = nb.predict(docs.get(0));
        String cat = Collections.max(jdProb.entrySet(), Map.Entry.comparingByValue()).getKey();

        List<String> jdSkills = new ArrayList<>();
        for (String s : skills) if (has(jd.toLowerCase(), s)) jdSkills.add(s);

        List<R> res = new ArrayList<>();
        for (int i = 0; i < texts.size(); i++) {
            R r = new R(); r.name = names.get(i);
            String low = texts.get(i).toLowerCase();
            for (String s : jdSkills) (has(low, s) ? r.matched : r.missing).add(s);
            r.cos = cosine(vecs.get(0), vecs.get(i + 1));
            r.sk = jdSkills.isEmpty() ? 0 : r.matched.size() / (double) jdSkills.size();
            r.cls = nb.predict(docs.get(i + 1)).get(cat);
            r.score = 0.5 * r.cos + 0.3 * r.sk + 0.2 * r.cls;
            res.add(r);
        }
        res.sort((a, b) -> Double.compare(b.score, a.score));

        StringBuilder sb = new StringBuilder("{\"category\":\"" + esc(cat) + "\",\"confidence\":" + f(jdProb.get(cat))
            + ",\"jdSkills\":" + arr(jdSkills) + ",\"results\":[");
        for (int i = 0; i < res.size(); i++) {
            R r = res.get(i);
            if (i > 0) sb.append(',');
            sb.append("{\"name\":\"").append(esc(r.name)).append("\",\"score\":").append(f(r.score))
              .append(",\"cosine\":").append(f(r.cos)).append(",\"skill\":").append(f(r.sk))
              .append(",\"cls\":").append(f(r.cls)).append(",\"matched\":").append(arr(r.matched))
              .append(",\"missing\":").append(arr(r.missing)).append('}');
        }
        return sb.append("]}").toString();
    }

    static double cosine(Map<String, Double> a, Map<String, Double> b) {
        double dot = 0, na = 0, nb2 = 0;
        for (Map.Entry<String, Double> e : a.entrySet()) { na += e.getValue() * e.getValue(); Double y = b.get(e.getKey()); if (y != null) dot += e.getValue() * y; }
        for (double y : b.values()) nb2 += y * y;
        return (na == 0 || nb2 == 0) ? 0 : dot / (Math.sqrt(na) * Math.sqrt(nb2));
    }
    static boolean has(String text, String skill) {
        return Pattern.compile("(?<![a-z0-9])" + Pattern.quote(skill) + "(?![a-z0-9])").matcher(text).find();
    }
    static String f(double d) { return String.format(Locale.US, "%.4f", d); }
    static String esc(String s) { return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", " ").replace("\r", ""); }
    static String arr(List<String> l) { StringBuilder s = new StringBuilder("["); for (int i = 0; i < l.size(); i++) s.append(i > 0 ? "," : "").append('"').append(esc(l.get(i))).append('"'); return s.append(']').toString(); }
}
