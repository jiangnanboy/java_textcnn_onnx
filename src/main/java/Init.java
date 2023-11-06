import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtException;
import utils.CollectionUtil;
import utils.PropertiesReader;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Optional;

/**
 * @author sy
 * @date 2023/11/03 22:57
 */
public class Init {
    public static OrtSession session;
    public static OrtEnvironment env;
    public static Map<String, Long> dict;


    static {
        try {
            initModel(PropertiesReader.get("relevance_model"));
            initDict(PropertiesReader.get("dict_path"));
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }

    /**
     * @param modelPath
     * @throws OrtException
     */
    public static void initModel(String modelPath) throws OrtException {
        System.out.println("init model...");
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(modelPath, new OrtSession.SessionOptions());
    }

    public static void closeModel() {
        System.out.println("close model...");
        if(Optional.ofNullable(session).isPresent()) {
            try {
                session.close();
            } catch (OrtException e) {
                e.printStackTrace();
            }
        }
        if(Optional.ofNullable(env).isPresent()) {
            env.close();
        }
    }

    /**
     * @param dictPath
     */
    public static void initDict(String dictPath) {
        System.out.println("init dict...");
        try(BufferedReader br = Files.newBufferedReader(Paths.get(dictPath), StandardCharsets.UTF_8)) {
            dict = CollectionUtil.newHashMap();
            String line;
            while ((line = br.readLine()) != null) {
                String[] strs = line.split(" ");
                String word = strs[0].trim();
                long index = Long.valueOf(strs[1].trim());
                dict.put(word, index);
            }
        } catch(IOException e) {
            e.printStackTrace();
        }
    }

}


