package com.aimodel.mageclassifier.ui;

import com.aimodel.mageclassifier.util.AIUtil;
import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

@Service
@Slf4j
public class ImagePredictionService {
    private MultiLayerNetwork model;
    private final NativeImageLoader loader = new NativeImageLoader(100, 100, 3);
    private final ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
    private List<File> imageFiles = new ArrayList<>();

    public ImagePredictionService() {
        try {
            File modelFile = new File(AIUtil.MODEL_PATH);
            if (!modelFile.exists()) {
                throw new RuntimeException("Model file not found at: " + modelFile.getAbsolutePath());
            }
            model = MultiLayerNetwork.load(modelFile, true);
            log.info("Model loaded successfully.");

            imageFiles = loadImageFiles();
            log.info("Loaded {} image files from dataset.", imageFiles.size());
        } catch (IOException e) {
            e.printStackTrace();
            System.err.println("Failed to load DL4J model: " + e.getMessage());
        }
    }


    public PredictionResult predictByPath(String path) {
        PredictionResult result = new PredictionResult();
        result.setImagePath(path);

        try {
            File imageFile = new File("src/main/resources/static" + path);
            INDArray image = loader.asMatrix(imageFile);
            scaler.transform(image);
            INDArray output = model.output(image);
            int prediction = output.argMax(1).getInt(0);
            result.setPrediction(prediction == 0 ? "Cat" : "Dog");
        } catch (IOException e) {
            e.printStackTrace();
            result.setPrediction("Error");
        }

        return result;
    }

    public List<PredictionResult> getImagesOnly() {
        List<File> files = this.imageFiles;
        // Shuffle the list to randomize the order
        Collections.shuffle(files);
        List<PredictionResult> results = new ArrayList<>();
        for (File file : files) {
            PredictionResult result = new PredictionResult();
            result.setImagePath("/datasets/" + file.getParentFile().getName() + "/" + file.getName());
            result.setPrediction("Pending");
            results.add(result);
        }
        return results;
    }

    private List<File> loadImageFiles() {
        File rootDir = new File(AIUtil.DATASET_PATH);
        List<File> imageFiles = new ArrayList<>();
        if (rootDir.exists()) {
            File[] subdirs = rootDir.listFiles(File::isDirectory);
            if (subdirs != null) {
                for (File dir : subdirs) {
                    File[] files = dir.listFiles(f -> f.getName().toLowerCase().matches(".*\\.(jpg|png|jpeg)"));
                    if (files != null) imageFiles.addAll(Arrays.asList(files));
                }
            }
        }
        if (imageFiles.isEmpty()) {
            throw new RuntimeException("No image files found in dataset directory: " + AIUtil.DATASET_PATH);
        }
        return imageFiles;
    }
}
