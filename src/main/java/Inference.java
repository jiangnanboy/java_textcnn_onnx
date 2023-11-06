import ai.onnxruntime.*;
import utils.CollectionUtil;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author sy
 * @date 2023/11/04 22:25
 */
public class Inference {

    /**
     * @param args
     * @throws OrtException
     */
    public static void main(String...args) throws OrtException {
        Map<String, OnnxTensor> inputMap = parse("我们热爱人工智能。");
        System.out.println(Init.session.getInputInfo());
        System.out.println(Init.session.getOutputInfo());
        double prob = infer(inputMap);
        System.out.println("prob -> " + prob);
    }

    /**
     * parse sentence
     * @param sent
     * @return
     * @throws OrtException
     */
    public static Map<String, OnnxTensor> parse(String sent) throws OrtException {
        return parse(sent, 100);
    }

    /**
     * parse sentence
     * @param sent
     * @param maxLength
     * @return
     * @throws OrtException
     */
    public static Map<String, OnnxTensor> parse(String sent, int maxLength) throws OrtException {
        List<String> tokenList = CollectionUtil.newArrayList();
        for(int i=0;i<sent.length();i++) {
            tokenList.add(sent.substring(i, i+1));
        }
        if(tokenList.size() > maxLength) {
            tokenList = tokenList.subList(0, maxLength - 1);
        } else if(tokenList.size() < maxLength) {
            int range = maxLength - tokenList.size();
            for(int i=0; i<range; i++) {
                tokenList.add("<pad>");
            }
        }
        List<Long> tokenIds = tokenList.stream().map(token -> Init.dict.getOrDefault(token, 0L)).collect(Collectors.toList());
        long[] inputIds = new long[tokenIds.size()];
        for(int i=0; i<tokenIds.size(); i++) {
            inputIds[i] = tokenIds.get(i);
        }
        long[] shape = new long[]{1, inputIds.length};
        Object ObjInputIds = OrtUtil.reshape(inputIds, shape);
        OnnxTensor inputOnnx = OnnxTensor.createTensor(Init.env, ObjInputIds);
        Map<String, OnnxTensor> inputMap = CollectionUtil.newHashMap();
        inputMap.put("input_1", inputOnnx);
        return inputMap;
    }

    /**
     * infer
     * @param inputs
     * @return
     */
    public static double infer(Map<String, OnnxTensor> inputs) {
        double prob = 0;
        try (OrtSession.Result result = Init.session.run(inputs)) {
            OnnxValue onnxValue = result.get(0);
            float[][] labels = (float[][])onnxValue.getValue();
            float[] resultLabels = labels[0];
            double[] softmaxResults = softmax(resultLabels);
            prob = getProb(softmaxResults);
        } catch (OrtException e) {
            e.printStackTrace();
        }
        return prob;
    }

    /**
     * get max prob
     * @param probabilities
     * @return
     */
    public static double getMaxProb(double[] probabilities) {
        double maxVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxVal) {
                maxVal = probabilities[i];
            }
        }
        return maxVal;
    }

    /**
     * get prob
     * @param probabilities
     * @return
     */
    public static double getProb(double[] probabilities) {
        double prob = probabilities[1];
        return prob;
    }

    /**
     * softmax
     * @param input
     * @return
     */
    private static double[] softmax(float[] input) {
        List<Float> inputList = CollectionUtil.newArrayList();
        for(int i=0; i<input.length; i++) {
            inputList.add(input[i]);
        }
        double inputSum = inputList.stream().mapToDouble(Math::exp).sum();
        double[] output = new double[input.length];
        for (int i=0; i<input.length; i++) {
            output[i] = (Math.exp(input[i]) / inputSum);
        }
        return output;
    }

}


