package com.aimodel.mageclassifier.ui;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class PredictionResult {
    private String imagePath;
    private String prediction;
}