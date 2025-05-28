package com.aimodel.mageclassifier.ui;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Controller
@Slf4j
@AllArgsConstructor
public class ImagePredictionController {
    private final ImagePredictionService service;


    @GetMapping("/index")
    public String showImages(@RequestParam(defaultValue = "0") int page,
                             @RequestParam(defaultValue = "10") int size,
                             Model model) {
        log.info("Loading index page with page: {}, size: {}", page, size);
        List<PredictionResult> imageFiles  = service.getImagesOnly();

        int totalImages = imageFiles.size();
        int fromIndex = Math.min(page * size, totalImages);
        int toIndex = Math.min(fromIndex + size, totalImages);
        List<PredictionResult> pageData = imageFiles.subList(fromIndex, toIndex);
        int totalPages = (int) Math.ceil((double) totalImages / size);

        model.addAttribute("images", pageData);
        model.addAttribute("currentPage", page);
        model.addAttribute("pageSize", size);
        model.addAttribute("totalPages", totalPages);
        model.addAttribute("pageSizes", List.of(10, 20, 50));
        model.addAttribute("totalRecords", totalImages);
        model.addAttribute("paginationRange", getPaginationRange(page, totalPages));

        return "index";
    }

    @PostMapping("/predict")
    @ResponseBody
    public PredictionResult predict(@RequestParam String path) throws IOException {
        return service.predictByPath(path);
    }


    private List<Integer> getPaginationRange(int currentPage, int totalPages) {
        int start = Math.max(0, currentPage - 2);
        int end = Math.min(totalPages - 1, currentPage + 2);
        List<Integer> range = new ArrayList<>();

        if (start > 0) range.add(0);         // always show first
        if (start > 1) range.add(-1);        // -1 means ellipsis

        for (int i = start; i <= end; i++) {
            range.add(i);
        }

        if (end < totalPages - 2) range.add(-2); // -2 means ellipsis
        if (end < totalPages - 1) range.add(totalPages - 1); // always show last

        return range;
    }

}
