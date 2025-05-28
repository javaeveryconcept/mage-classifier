package com.aimodel.mageclassifier.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class StatusController {
    @GetMapping("/")
    public String status() {
        return "DL4J Image Classifier is running!";
    }
}
