#include "strapYolov8.h"
#include "cmd_line_utilStrap.h"


// Runs object detection on an input image then saves the annotated image to disk.
int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath;
    std::string inputImage;

    // Parse the command line arguments
	if (!parseArguments(argc, argv, config, onnxModelPath, inputImage)) {
		return -1;
    }

    // Create the YoloV8 engine
    YoloV8 yoloV8(onnxModelPath, config);

    // Read the input image
    auto img = cv::imread(inputImage);
    if (img.empty()) {
        std::cout << "Error: Unable to read image at path '" << inputImage << "'" << std::endl;
        return -1;
    }

    // Run inference
    const auto objects = yoloV8.detectObjects(img);

    // Draw the bounding boxes on the image
    yoloV8.drawObjectLabels(img, objects);

    for (const auto& object: objects) 
    {
        if (object.label == 1)
        {
            std::cout << "Detected " << object.probability  << std::endl;
            std::cout << "Detected " << object.boxMask.cols  << std::endl;
            std::cout << "Detected " << object.boxMask.rows  << std::endl;

        }
        // Choose the color
        // int colorIndex = object.label % COLOR_LIST.size(); // We have only defined 80 unique colors
        // std::cout << "Detected " << object.label <<  std::endl;
        // std::cout << "Detected " << object.probability  << std::endl;

        // Add the mask for said object
        // mask(object.rect).setTo(color * 255, object.boxMask);
    }

    // std::cout << "Detected " << objects.size() << " objects" << std::endl;
    // std::cout << "Detected " << objects.rect() << " objects" << std::endl;

    // Save the image to disk
    const auto outputName = inputImage.substr(0, inputImage.find_last_of('.')) + "_annotated.jpg";
    cv::imwrite(outputName, img);
    std::cout << "Saved annotated image to: " << outputName << std::endl;

    return 0;
}