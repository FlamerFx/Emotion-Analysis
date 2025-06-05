#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "glfw3.h"
#include "glfw3native.h"
#include <iostream>
#include <string>
#include <cstring> // for std::memcpy
#include <filesystem>

// Include your inference headers
#include "TextPreprocessor.h"
#include "LabelUtils.h"

// Include TensorFlow C API
#include "tensorflow/c/c_api.h"

// Include STB image for texture loading
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Forward declare your inference function
std::string predict_emotion(const std::string& text);

// Helper to get base directory (where executable is run)
std::string get_base_dir() {
    return std::filesystem::current_path().string();
}

GLuint LoadTextureFromFile(const char* filename) {
    int w, h, channels;
    unsigned char* data = stbi_load(filename, &w, &h, &channels, 4);
    if (!data) return 0;
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    stbi_image_free(data);
    return tex;
}

void ImGuiTextShadow(const ImVec4& color, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    char buf[512];
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    // Draw shadow (black, slightly offset)
    draw_list->AddText(ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(pos.x+2, pos.y+2), IM_COL32(0,0,0,180), buf);
    // Draw main text
    draw_list->AddText(ImGui::GetFont(), ImGui::GetFontSize(), pos, ImGui::ColorConvertFloat4ToU32(color), buf);

    // Move cursor as if text was drawn
    ImGui::Dummy(ImGui::CalcTextSize(buf));
}

enum HertaState { WELCOME, THINKING, HAPPY, SAD, ANGRY, FEAR };
HertaState herta_state = WELCOME;

int main() {
    std::string base_dir = get_base_dir();

    // Setup window
    if (!glfwInit()) return 1;
    GLFWwindow* window = glfwCreateWindow(600, 400, "Emotion Classifier - Dear ImGui", NULL, NULL);
    glfwMakeContextCurrent(window);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Setup font
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontDefault(); // Keep default for fallback
    ImFont* customFont = io.Fonts->AddFontFromFileTTF((base_dir + "/comic.ttf").c_str(), 28.0f);
    io.FontGlobalScale = 1.3f;

    static char input[256] = "";
    static std::string result = "";

    // ---- Load texture ONCE, after OpenGL context is ready ----
    GLuint herta_welcome  = LoadTextureFromFile((base_dir + "/Herta.png").c_str());
    GLuint herta_thinking = LoadTextureFromFile((base_dir + "/Herta thinkling.png").c_str());
    GLuint herta_happy    = LoadTextureFromFile((base_dir + "/Herta happy.png").c_str());
    GLuint herta_sad      = LoadTextureFromFile((base_dir + "/Herta sad.png").c_str());
    GLuint herta_angry    = LoadTextureFromFile((base_dir + "/Herta angry.png").c_str());
    GLuint herta_fear     = LoadTextureFromFile((base_dir + "/Herta fear.png").c_str());
    if (herta_welcome == 0 || herta_thinking == 0 || herta_happy == 0 ||
        herta_sad == 0 || herta_angry == 0 || herta_fear == 0) {
        std::cerr << "Failed to load one or more Herta textures!" << std::endl;
    }
    // ---------------------------------------------------------

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Gradient background
        ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
        ImVec2 size = ImGui::GetIO().DisplaySize;
        draw_list->AddRectFilledMultiColor(
            ImVec2(0, 0), size,
            IM_COL32(40, 40, 80, 255),   // Top-left color
            IM_COL32(80, 80, 160, 255),  // Top-right color
            IM_COL32(30, 30, 60, 255),   // Bottom-right color
            IM_COL32(10, 10, 30, 255)    // Bottom-left color
        );

        // Fullscreen window
        ImVec2 display_size = ImGui::GetIO().DisplaySize;
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(display_size, ImGuiCond_Always);

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 18.0f);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, IM_COL32(30, 30, 40, 220));
        ImGui::Begin("Emotion Classifier", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        // Herta image selection
        GLuint current_herta = herta_welcome;
        if (herta_state == THINKING) current_herta = herta_thinking;
        else if (herta_state == HAPPY) current_herta = herta_happy;
        else if (herta_state == SAD) current_herta = herta_sad;
        else if (herta_state == ANGRY) current_herta = herta_angry;
        else if (herta_state == FEAR) current_herta = herta_fear;

        ImGui::Columns(2, nullptr, false); // 2 columns, no border

        // --- Left column: Herta image ---
        ImGui::SetColumnWidth(0, 270); // Enough for 256px image + padding
        float t = ImGui::GetTime();
        float bounce = 10.0f * sinf(t * 2.5f);
        ImVec2 herta_pos = ImGui::GetCursorScreenPos();
        ImGui::SetCursorScreenPos(ImVec2(herta_pos.x, herta_pos.y + bounce));
        ImGui::Image((ImTextureID)current_herta, ImVec2(256, 256));
        ImGui::SetCursorScreenPos(herta_pos); // Reset for next widgets

        ImGui::NextColumn();

        // --- Right column: UI ---
        ImVec4 animatedColor = ImVec4(0.4f + 0.2f*sinf(t*2), 0.7f, 1.0f, 1.0f);
        ImGuiTextShadow(animatedColor, "Hello, I am Herta!");
        ImGui::TextWrapped("I was made by Sitanshu and Yash for a semester project.\nKeep in mind that I am just a semester project and I can make mistakes.");
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::Text("Enter text for emotion classification:");

        // Clamp input box width to 400px max
        ImVec2 input_box_size = ImVec2(400, 80);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, IM_COL32(10, 10, 30, 255));
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 255, 255));
        if (ImGui::InputTextMultiline("##input", input, IM_ARRAYSIZE(input), input_box_size)) {
            if (strlen(input) == 0)
                herta_state = WELCOME;
            else
                herta_state = THINKING;
        }
        ImGui::PopStyleColor(2);
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            input[0] = '\0';
            result.clear();
            herta_state = WELCOME;
        }
        ImGui::Spacing();

        ImGui::PushFont(customFont);
        if (ImGui::Button("Predict", ImVec2(180, 0))) {
            result = predict_emotion(input);
            if (result == "joy") herta_state = HAPPY;
            else if (result == "sadness") herta_state = SAD;
            else if (result == "anger") herta_state = ANGRY;
            else if (result == "fear") herta_state = FEAR;
            else herta_state = WELCOME;
        }
        ImGui::SameLine();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Click to predict the emotion of the entered text.");
        ImGui::PopFont();

        ImGui::Spacing();

        if (!result.empty()) {
            if (result == "error") {
                ImGuiTextShadow(ImVec4(1.0f, 0.2f, 0.2f, 1.0f), "Prediction failed, please try again.");
            } else {
                ImGuiTextShadow(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Predicted emotion: %s", result.c_str());
            }
        }

        ImGui::Columns(1); // End columns

        ImGui::End();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

// Implement this function using your inference code
std::string predict_emotion(const std::string& text) {
    std::cout << "[DEBUG] ENTERED predict_emotion()" << std::endl;

    std::string base_dir = get_base_dir();

    std::string export_dir = base_dir + "/saved_model";
    const char* tag = "serve";
    const char* input_name = "serving_default_keras_tensor";
    const char* output_name = "StatefulPartitionedCall_1";
    int input_dim = 100;

    // Initialize text preprocessor with vocab file
    std::string vocab_path = base_dir + "/word_index.txt";
    TextPreprocessor preprocessor(vocab_path, input_dim);

    // Load label file
    std::string label_path = base_dir + "/labels.txt";
    std::vector<std::string> labels = load_labels(label_path);

    // Preprocess the input text
    std::vector<float> input_vals = preprocessor.preprocess(text);

    std::cout << "[DEBUG] Preprocessed input, size: " << input_vals.size() << std::endl;

    TF_Status* status = TF_NewStatus();
    TF_Graph* graph = TF_NewGraph();
    TF_SessionOptions* opts = TF_NewSessionOptions();
    TF_Session* sess = nullptr;

    // Load the TensorFlow model
    {
        std::cout << "[DEBUG] Loading SavedModel..." << std::endl;
        sess = TF_LoadSessionFromSavedModel(opts, nullptr, export_dir.c_str(), &tag, 1, graph, nullptr, status);

        if (TF_GetCode(status) != TF_OK) {
            std::cerr << "ERROR loading model: " << TF_Message(status) << std::endl;
            TF_DeleteStatus(status);
            return "error";
        }
        std::cout << "Model loaded successfully!" << std::endl;
    }

    TF_Output input_op = {TF_GraphOperationByName(graph, input_name), 0};
    TF_Output output_op = {TF_GraphOperationByName(graph, output_name), 0};

    // Create input tensor
    TF_Tensor* input_tensor = TF_AllocateTensor(TF_FLOAT, (const int64_t[]){1, input_dim}, 2, sizeof(float) * input_dim);
    std::memcpy(TF_TensorData(input_tensor), input_vals.data(), sizeof(float) * input_dim);

    TF_Tensor* output_tensor = nullptr;

    // Run inference
    {
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
            return "error";
        }
    }

    // Process output
    std::string result;
    {
        auto data = static_cast<float*>(TF_TensorData(output_tensor));
        size_t output_elements = TF_TensorByteSize(output_tensor) / sizeof(float);

        // --- PRINT PREDICTED EMOTION LABEL ---
        if (output_elements == labels.size()) {
            size_t pred = argmax(data, output_elements);
            result = labels[pred];
        } else {
            std::cerr << "Warning: Output size (" << output_elements
                      << ") does not match number of labels (" << labels.size() << ")." << std::endl;
            result = "error";
        }
        // -------------------------------------
    }

    // Cleanup
    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(output_tensor);
    TF_CloseSession(sess, status);
    TF_DeleteSession(sess, status);
    TF_DeleteGraph(graph);
    TF_DeleteSessionOptions(opts);
    TF_DeleteStatus(status);

    std::cout << "[DEBUG] Exiting predict_emotion()" << std::endl;
    return result;
}