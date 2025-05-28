package com.aimodel.mageclassifier.controller;

import com.aimodel.mageclassifier.model.PredictionResult;
import com.aimodel.mageclassifier.service.ImageClassifierService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
@RequestMapping("/api/image")
@Slf4j
public class ImageClassifierController {

    @Autowired
    private ImageClassifierService imageService;

    @PostMapping("/predict")
    public ResponseEntity<PredictionResult> predict(@RequestParam("image") MultipartFile file) throws IOException {
        log.info ("Received image for prediction: {}", file.getOriginalFilename());
        PredictionResult result = imageService.predict(file);
        return ResponseEntity.ok(result);
    }
}
