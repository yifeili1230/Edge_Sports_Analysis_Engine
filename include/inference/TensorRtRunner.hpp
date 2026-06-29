#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#ifndef VIDEO_ENGINE_WITH_TENSORRT
#define VIDEO_ENGINE_WITH_TENSORRT 0
#endif

#if VIDEO_ENGINE_WITH_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

namespace video_engine {

class TensorRtRunner {
public:
#if VIDEO_ENGINE_WITH_TENSORRT
    explicit TensorRtRunner(const std::string& engine_path) {
        std::ifstream input(engine_path, std::ios::binary | std::ios::ate);
        if (!input.is_open()) {
            throw std::runtime_error("Could not open TensorRT engine: " + engine_path);
        }
        const auto size = input.tellg();
        if (size <= 0) {
            throw std::runtime_error("TensorRT engine is empty: " + engine_path);
        }
        input.seekg(0);
        std::vector<char> bytes(static_cast<std::size_t>(size));
        input.read(bytes.data(), size);
        if (!input.good()) {
            throw std::runtime_error("Could not read TensorRT engine: " + engine_path);
        }

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            throw std::runtime_error("Could not create TensorRT runtime.");
        }
        engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!engine_) {
            throw std::runtime_error(
                "Could not deserialize TensorRT engine. Rebuild it on this Jetson.");
        }
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("Could not create TensorRT execution context.");
        }

        for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
            const char* name = engine_->getIOTensorName(index);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                input_name_ = name;
            } else {
                output_name_ = name;
            }
        }
        if (input_name_.empty() || output_name_.empty()) {
            throw std::runtime_error(
                "TensorRT engine must have exactly one image input and one pose output.");
        }
        if (engine_->getTensorDataType(input_name_.c_str()) !=
                nvinfer1::DataType::kFLOAT ||
            engine_->getTensorDataType(output_name_.c_str()) !=
                nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error(
                "TensorRT engine input/output must be FP32; FP16 is used internally.");
        }

        input_dims_ = engine_->getTensorShape(input_name_.c_str());
        output_dims_ = engine_->getTensorShape(output_name_.c_str());
        input_elements_ = volume(input_dims_);
        output_elements_ = volume(output_dims_);
        checkCuda(cudaStreamCreate(&stream_), "create CUDA stream");
        checkCuda(cudaMalloc(&input_device_, input_elements_ * sizeof(float)),
                  "allocate TensorRT input");
        checkCuda(cudaMalloc(&output_device_, output_elements_ * sizeof(float)),
                  "allocate TensorRT output");
        if (!context_->setTensorAddress(input_name_.c_str(), input_device_) ||
            !context_->setTensorAddress(output_name_.c_str(), output_device_)) {
            throw std::runtime_error("Could not bind TensorRT engine buffers.");
        }
        output_host_.resize(output_elements_);
    }

    ~TensorRtRunner() {
        if (input_device_ != nullptr) {
            cudaFree(input_device_);
        }
        if (output_device_ != nullptr) {
            cudaFree(output_device_);
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    TensorRtRunner(const TensorRtRunner&) = delete;
    TensorRtRunner& operator=(const TensorRtRunner&) = delete;

    cv::Mat forward(const cv::Mat& input_blob) {
        if (!input_blob.isContinuous() || input_blob.type() != CV_32F ||
            input_blob.total() != input_elements_) {
            throw std::runtime_error(
                "TensorRT input blob does not match the engine input tensor.");
        }
        checkCuda(cudaMemcpyAsync(
                      input_device_, input_blob.ptr<float>(),
                      input_elements_ * sizeof(float), cudaMemcpyHostToDevice, stream_),
                  "upload TensorRT input");
        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("TensorRT enqueueV3 failed.");
        }
        checkCuda(cudaMemcpyAsync(
                      output_host_.data(), output_device_,
                      output_elements_ * sizeof(float), cudaMemcpyDeviceToHost, stream_),
                  "download TensorRT output");
        checkCuda(cudaStreamSynchronize(stream_), "synchronize TensorRT inference");

        std::vector<int> dimensions;
        dimensions.reserve(static_cast<std::size_t>(output_dims_.nbDims));
        for (int index = 0; index < output_dims_.nbDims; ++index) {
            dimensions.push_back(output_dims_.d[index]);
        }
        return cv::Mat(dimensions, CV_32F, output_host_.data()).clone();
    }

private:
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* message) noexcept override {
            if (severity <= Severity::kWARNING) {
                std::cerr << "[TensorRT] " << message << '\n';
            }
        }
    };

    static std::size_t volume(const nvinfer1::Dims& dimensions) {
        std::size_t result = 1;
        for (int index = 0; index < dimensions.nbDims; ++index) {
            if (dimensions.d[index] <= 0) {
                throw std::runtime_error(
                    "Dynamic TensorRT shapes are not supported; build a fixed-size engine.");
            }
            result *= static_cast<std::size_t>(dimensions.d[index]);
        }
        return result;
    }

    static void checkCuda(cudaError_t status, const std::string& operation) {
        if (status != cudaSuccess) {
            throw std::runtime_error(
                "Failed to " + operation + ": " + cudaGetErrorString(status));
        }
    }

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::string input_name_;
    std::string output_name_;
    nvinfer1::Dims input_dims_{};
    nvinfer1::Dims output_dims_{};
    std::size_t input_elements_ = 0;
    std::size_t output_elements_ = 0;
    void* input_device_ = nullptr;
    void* output_device_ = nullptr;
    cudaStream_t stream_ = nullptr;
    std::vector<float> output_host_;
#else
    explicit TensorRtRunner(const std::string&) {
        throw std::runtime_error(
            "This binary was built without TensorRT. Install libnvinfer-dev and rebuild.");
    }

    cv::Mat forward(const cv::Mat&) {
        throw std::runtime_error("TensorRT is unavailable in this build.");
    }
#endif
};

}  // namespace video_engine
