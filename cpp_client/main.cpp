#include <tensorflow/c/c_api.h>
#include <iostream>
#include <vector>
#include <cstring>
#include "TextPreprocessor.h"
#include "LabelUtils.h"

int main() {
    std::cout << "[DEBUG] ENTERED main()" << std::endl;

    // Set model and asset paths relative to the executable
    const char* export_dir = "../python_ml_server/model/saved_model";
    const char* tag = "serve";
    const char* input_name = "serving_default_keras_tensor";
    const char* output_name = "StatefulPartitionedCall_1";
    int input_dim = 100;

    // Initialize text preprocessor with vocab file
    std::string vocab_path = "../python_ml_server/model/word_index.txt";
    TextPreprocessor preprocessor(vocab_path, input_dim);

    // Load label file
    std::string label_path = "../python_ml_server/model/labels.txt";
    std::vector<std::string> labels = load_labels(label_path);

    // Get user input and preprocess
    std::cout << "Enter text for emotion classification: ";
    std::string user_text;
    std::getline(std::cin, user_text);
    std::vector<float> input_vals = preprocessor.preprocess(user_text);

    std::cout << "[DEBUG] Preprocessed input, size: " << input_vals.size() << std::endl;

    // Set up TensorFlow session and load model
    std::cout << "[DEBUG] Starting TensorFlow C API client..." << std::endl;
    TF_Status* status = TF_NewStatus();
    TF_Graph* graph = TF_NewGraph();
    TF_SessionOptions* opts = TF_NewSessionOptions();

    std::cout << "[DEBUG] Loading SavedModel..." << std::endl;
    TF_Session* sess = TF_LoadSessionFromSavedModel(opts, nullptr, export_dir, &tag, 1, graph, nullptr, status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "ERROR loading model: " << TF_Message(status) << std::endl;
        TF_DeleteStatus(status);
        return 1;
    }
    std::cout << "Model loaded successfully!" << std::endl;

    // Look up input and output operations
    TF_Operation* input_op_ptr = TF_GraphOperationByName(graph, input_name);
    if (!input_op_ptr) {
        std::cerr << "[ERROR] Input operation not found: " << input_name << std::endl;
        TF_DeleteStatus(status);
        return 1;
    }
    TF_Output input_op = {input_op_ptr, 0};

    TF_Operation* output_op_ptr = TF_GraphOperationByName(graph, output_name);
    if (!output_op_ptr) {
        std::cerr << "[ERROR] Output operation not found: " << output_name << std::endl;
        TF_DeleteStatus(status);
        return 1;
    }
    TF_Output output_op = {output_op_ptr, 0};

    // Create input tensor
    int64_t dims[] = {1, input_dim};
    TF_Tensor* input_tensor = TF_AllocateTensor(
        TF_FLOAT, dims, 2, sizeof(float)*input_dim
    );
    std::memcpy(TF_TensorData(input_tensor), input_vals.data(), sizeof(float)*input_dim);

    if (!input_tensor) {
        std::cerr << "[ERROR] Failed to create input tensor" << std::endl;
        TF_DeleteStatus(status);
        return 1;
    }

    TF_Tensor* output_tensor = nullptr;

    // Run inference
    TF_SessionRun(sess,
                  nullptr,
                  &input_op, &input_tensor, 1,
                  &output_op, &output_tensor, 1,
                  nullptr, 0,
                  nullptr,
                  status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "ERROR during inference: " << TF_Message(status) << std::endl;
        TF_DeleteTensor(input_tensor);
        TF_DeleteStatus(status);
        return 1;
    }

    if (!output_tensor) {
        std::cerr << "[ERROR] Output tensor is null!" << std::endl;
        TF_DeleteTensor(input_tensor);
        TF_DeleteStatus(status);
        return 1;
    }

    // Print output probabilities
    auto data = static_cast<float*>(TF_TensorData(output_tensor));
    size_t output_elements = TF_TensorByteSize(output_tensor) / sizeof(float);
    std::cout << "Output (" << output_elements << " elements): ";
    for (size_t i = 0; i < output_elements; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    // Print predicted emotion label
    if (output_elements == labels.size()) {
        size_t pred = argmax(data, output_elements);
        std::cout << "Predicted emotion: " << labels[pred] << std::endl;
    } else {
        std::cout << "Warning: Output size (" << output_elements
                  << ") does not match number of labels (" << labels.size() << ")." << std::endl;
    }

    // Clean up
    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(output_tensor);
    TF_CloseSession(sess, status);
    TF_DeleteSession(sess, status);
    TF_DeleteGraph(graph);
    TF_DeleteSessionOptions(opts);
    TF_DeleteStatus(status);

    std::cout << "[DEBUG] Done. Exiting." << std::endl;
    return 0;
}