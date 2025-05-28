package com.aimodel.mageclassifier.service;

import com.aimodel.mageclassifier.model.PredictionResult;
import com.aimodel.mageclassifier.util.AIUtil;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.List;

@Service
public class ImageClassifierService {

    private static final int HEIGHT = 100;
    private static final int WIDTH = 100;
    private static final int CHANNELS = 3;
    private static final List<String> LABELS = List.of("cat", "dog");

    private final MultiLayerNetwork model;

    public ImageClassifierService() throws IOException {
        File modelFile = new File(AIUtil.MODEL_PATH);
        //File modelFile = new File("image-model1.zip");
        model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
    }

    public PredictionResult predict(MultipartFile file) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
        INDArray image = loader.asMatrix(file.getInputStream());

        image.divi(255); // normalize manually

        INDArray output = model.output(image);
        int predictedClass = output.argMax(1).getInt(0);
        double confidence = output.getDouble(0, predictedClass);

        return new PredictionResult(LABELS.get(predictedClass), confidence);
    }
}
