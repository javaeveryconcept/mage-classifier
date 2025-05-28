package com.aimodel.mageclassifier.model;

public class PredictionResult {
    private String label;
    private double confidence;

    public PredictionResult(String label, double confidence) {
        this.label = label;
        this.confidence = confidence;
    }

    public String getLabel() {
        return label;
    }

    public double getConfidence() {
        return confidence;
    }
}
