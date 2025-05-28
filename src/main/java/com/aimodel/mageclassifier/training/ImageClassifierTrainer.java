package com.aimodel.mageclassifier.training;

import com.aimodel.mageclassifier.util.AIUtil;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Random;

public class  ImageClassifierTrainer {

    private static final int HEIGHT = 100;
    private static final int WIDTH = 100;
    private static final int CHANNELS = 3;
    private static final int BATCH_SIZE = 16;
    private static final int OUTPUT_NUM = 2; // Cat or Dog
    private static final int EPOCHS = 10;
    private static final long SEED = 123;

    public static void main(String[] args) throws Exception {
        long startTime = System.currentTimeMillis();
        System.out.println("Starting image classifier training with train/test split...");

        Random rng = new Random(SEED);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        // Set up the file split to read images from the dataset directory
        FileSplit fileSplit = new FileSplit(new File(AIUtil.DATASET_PATH), NativeImageLoader.ALLOWED_FORMATS, rng);


        // Split into 80% train and 20% test
        InputSplit[] inputSplits = fileSplit.sample(new BalancedPathFilter(rng, NativeImageLoader.ALLOWED_FORMATS,labelMaker), 80, 20);
        InputSplit trainData = inputSplits[0];
        InputSplit testData = inputSplits[1];

        // Initialize train iterator
        ImageRecordReader trainRR = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
        trainRR.initialize(trainData);
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, BATCH_SIZE, 1, OUTPUT_NUM);
        trainIter.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        // Initialize test iterator
        ImageRecordReader testRR = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
        testRR.initialize(testData);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, BATCH_SIZE, 1, OUTPUT_NUM);
        testIter.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        // CNN model configuration
        MultiLayerNetwork model = new MultiLayerNetwork(multiLayerConfiguration());
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        // Train the model
        System.out.println("Training model...");
        for (int i = 0; i < EPOCHS; i++) {
            trainIter.reset();
            model.fit(trainIter);
        }

        // Evaluate on test data
        System.out.println("Evaluating model...");
        Evaluation eval = model.evaluate(testIter);
        System.out.println(eval.stats());

        // Save model
        model.save(new File(AIUtil.MODEL_PATH), true);
        System.out.println("Model saved.");

        long endTime = System.currentTimeMillis();
        System.out.println("Training and evaluation completed in " + (endTime - startTime) / 1000 + " seconds.");
    }

    /**
     * Configures the CNN model for image classification.
     *
     * @return MultiLayerConfiguration for the CNN model
     */
    private static MultiLayerConfiguration multiLayerConfiguration() {
        return new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .updater(new Adam(0.001))
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(CHANNELS)
                        .stride(1, 1)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nOut(64)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU).nOut(128).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(OUTPUT_NUM)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(HEIGHT, WIDTH, CHANNELS))
                .build();
    }
}
