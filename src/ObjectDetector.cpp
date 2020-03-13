#include <ObjectDetector.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

void drawPredictions(cv::Mat &frame, const std::vector<DetObj> &detections, bool drawLabels,
                                          const cv::Scalar &color, int thickness, int lineType, int shift)
{
    CV_Assert(!frame.empty());
    for (const auto &det : detections) {
        // draw bounding box around object
        cv::rectangle(frame, det.boundingBox(), color, thickness, lineType, shift);

        if (drawLabels) {
            // either label with classname or just present the confidence threshold
            const std::string label = det.name().empty() ? std::string(cv::format("%.2f", det.confidence()))
                                                              : det.name();
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            // draw label on grey background
	        auto &bb = det.boundingBox();
            int left = bb.x;
            int top = std::max(bb.y, labelSize.height);
            cv::rectangle(frame, cv::Point(left, top - labelSize.height),
                          cv::Point(left + labelSize.width, top + baseLine), cv::Scalar::all(255), cv::FILLED);
            cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
        }
    }
}

ObjectDetector::ObjectDetector(const cv::String &model, const cv::String &config, const cv::String &framework) {
    readNet(model, config, framework);
}

void ObjectDetector::readNet(const cv::String & model, const cv::String & config, const cv::String & framework) {
    _net = cv::dnn::readNet(model, config, framework);
    _out_names = _net.getUnconnectedOutLayersNames();
}

void ObjectDetector::setScale(double scale) {
    _scale = scale;
}

void ObjectDetector::setMean(const cv::Scalar &mean) {
    _mean = mean;
}

void ObjectDetector::setSwapRB(bool swap) {
    _swap_rb = swap;
}

void ObjectDetector::setCrop(bool crop) {
    _crop = crop;
}

void ObjectDetector::setSize(const cv::Size &size) {
    _size = size;
}

void ObjectDetector::setDDepth(int ddepth) {
    CV_Assert(ddepth == CV_8U || ddepth == CV_32F);
    _ddepth = ddepth;
}

void ObjectDetector::setNMSThreshold(float nmsThreshold) {
    CV_Assert(0.0 <= nmsThreshold && nmsThreshold <= 1.0);
    _nms_threshold = nmsThreshold;
}

void ObjectDetector::setConfidenceThreshold(float confThreshold) {
    CV_Assert(0.0 <= confThreshold && confThreshold <= 1.0);
    _conf_threshold = confThreshold;
}

void ObjectDetector::setClasses(const std::vector<std::string> &classes) {
    _classes = classes;
}

std::vector<DetObj> ObjectDetector::run(const cv::Mat &frame) {
    std::vector<DetObj> detections;
    run(frame, detections);
    return detections;
}

std::vector<DetObj>& ObjectDetector::run(const cv::Mat &frame, std::vector<DetObj> &detections) {  
    CV_Assert(!frame.empty());
    preprocess(frame);
    std::vector<cv::Mat> outs;
    _net.forward(outs, _out_names);
    postprocess(_size == cv::Size(0, 0) ? cv::Size(frame.cols, frame.rows) : _size, outs, detections);
    return detections;
}

cv::dnn::Net& ObjectDetector::getNet() {
    return _net;
}

float ObjectDetector::getConfidenceThreshold() const {
    return _conf_threshold;
}

float ObjectDetector::getNMSThreshold() const {
    return _nms_threshold;
}

void ObjectDetector::preprocess(const cv::Mat &frame) {
    static cv::Mat blob;
    static const cv::Size size(_size.width == 0 ? frame.cols : _size.width, _size.height == 0 ? frame.rows : _size.height);
    blob = cv::dnn::blobFromImage(frame, 1.0, size, cv::Scalar(), _swap_rb, _crop);

    // Faster-RCNN or R-FCN
    if (_net.getLayer(0)->outputNameToIndex("im_info") != -1) {
        cv::resize(frame, frame, size);
        cv::Mat imInfo = (cv::Mat_<float>(1, 3) << size.height, size.width, 1.6f);
        _net.setInput(imInfo, "im_info");
    } else {
        _net.setInput(blob, "", _scale, _mean);
    }
}

void ObjectDetector::postprocess(const cv::Size &size, const std::vector<cv::Mat> &outs, std::vector<DetObj> &detections) {
    static std::vector<int> outLayers = _net.getUnconnectedOutLayers();
    static std::string outLayerType = _net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    if (outLayerType == "DetectionOutput") {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        for (auto &out : outs) {
            auto data = (float *) out.data;
            for (size_t i = 0; i < out.total(); i += 7) {
                float confidence = data[i + 2];
                if (confidence > _conf_threshold) {
                    int left   = (int) data[i + 3];
                    int top    = (int) data[i + 4];
                    int right  = (int) data[i + 5];
                    int bottom = (int) data[i + 6];
                    int width  = right - left + 1;
                    int height = bottom - top + 1;
                    if (width <= 2 || height <= 2) {
                        left   = (int) (data[i + 3] * size.width);
                        top    = (int) (data[i + 4] * size.height);
                        right  = (int) (data[i + 5] * size.width);
                        bottom = (int) (data[i + 6] * size.height);
                        width  = right - left + 1;
                        height = bottom - top + 1;
                    }
                    // Skip 0th background class id.
                    classIds.push_back((int) (data[i + 1]) - 1);
                    boxes.emplace_back(cv::Rect(left, top, width, height));
                    confidences.push_back(confidence);
                }
            }
        }
    } else if (outLayerType == "Region") {
        for (auto &out : outs) {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            auto data = (float *) out.data;
            for (int j = 0; j < out.rows; ++j, data += out.cols) {
                cv::Mat scores = out.row(j).colRange(5, out.cols);
                cv::Point classIdPoint;
                double confidence;
                cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);
                if (confidence > _conf_threshold) {
                    int centerX = (int)(data[0] * size.width);
                    int centerY = (int)(data[1] * size.height);
                    int width = (int)(data[2] * size.width);
                    int height = (int)(data[3] * size.height);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.emplace_back(cv::Rect(left, top, width, height));
                }
            }
        }
    } else {
        CV_Error(cv::Error::StsNotImplemented, "unknown output layer type: " + outLayerType);
    }

    // non maximum suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, _conf_threshold, _nms_threshold, indices);

    // filter for prediction sufficing NMS threshold
    detections.clear();
    detections.reserve(indices.size());
    std::for_each(indices.begin(), indices.end(), [&](int idx) {
        detections.emplace_back(DetObj(classIds[idx], boxes[idx].x, boxes[idx].y, boxes[idx].width,
                    boxes[idx].height, confidences[idx], classIds[idx] < _classes.size() ? _classes[idx] : ""));
    });
}

