### java_textcnn_onnx
java is used to load the onnx format model of textcnn

### step
1.after training the textcnn model, convert it to onnx format (src/main/resource/model.onnx)

2.extract token mapping dict (src/main/resources/token.txt)

3.use java to load onnx format model and token dict (src/main/java/Init)

4.inference and prediction (src/main/java/Inference)

### example
(text binary classification)

predict：src/main/java/Inference
```
public static void main(String...args) throws OrtException {
        Map<String, OnnxTensor> inputMap = parse("我们热爱人工智能。");
        System.out.println(Init.session.getInputInfo());
        System.out.println(Init.session.getOutputInfo());
        double prob = infer(inputMap);
        System.out.println("prob -> " + prob);
    }
```

### contact

1、github：https://github.com/jiangnanboy

2、博客：https://www.cnblogs.com/little-horse/

3、邮件:2229029156@qq.com


