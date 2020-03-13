#ifndef __OBJECTDETECTOR_HPP
#define __OBJECTDETECTOR_HPP

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdint>

/***
 * Wrapper class for detected objects found
 * in the images
 */
class DetectedObject {
public:
    DetectedObject() = default;

    DetectedObject(int id, int left, int top, int width, int height, float confidence, const cv::String &name="") :
            _id(id), _rect(left, top, width, height), _confidence(confidence), _name(name){}

    DetectedObject(int id, const cv::Rect &bb, float confidence, const cv::String &name="") : _id(id), _rect(bb), _confidence(confidence), _name(name) {}

    uint32_t id() const { return _id; }

    const cv::Rect& boundingBox() const { return _rect; }

    float confidence() const { return _confidence; }

    const cv::String& name() const { return _name; }

private:
    uint32_t _id = -1;

    cv::Rect _rect = cv::Rect(-1, -1 , 0, 0);

    float _confidence = -1.0;

    cv::String _name;

};

// give a shorted name
typedef DetectedObject	DetObj;

/***
 * draw object detector output on frame
 * @param frame
 * @param predictions
 * @param drawLabels
 * @param color
 * @param thickness
 * @param lineType
 * @param shift
 */
void drawPredictions(cv::Mat &frame, const std::vector<DetObj> &predictions, bool drawLabels=true,
                        const cv::Scalar &color=cv::Scalar(0, 0, 255), int thickness=1, int lineType=cv::LINE_8, int shift=0);

/***
 * Wrapper class around OpenCV's DNN module to support
 * object detection models.
 * It has been tested with SSD-Mobilenet v2 and Tiny Yolo v3.
 */
class ObjectDetector {
public:

    /***
     * default construct object detector
     */
    ObjectDetector() = default;

    /***
     * read the getNetwork from the specified files
     * @param model file containing the getNetwork parameters
     * @param config file containing the getNetwork architecture
     * @param framework the framwork that was used to make the mode
     */
    explicit ObjectDetector(const cv::String &model, const cv::String &config="", const cv::String &framework="");

    /***
     * read the getNetwork from the specified files
     * @param model file containing the getNetwork parameters
     * @param config file containing the getNetwork architecture
     * @param framework the framwork that was used to make the mode
     */
    void readNet(const cv::String &model, const cv::String &config="", const cv::String &framework="");

    /***
     * set the scaling parameter, used to multiply all pixel values with
     * prior to running the image through the detector
     * @param scale
     */
    void setScale(double scale);

    /***
     * set the size that the image is scaled to before running the detector
     * if this is not set, the image size is used instead
     * this size must be an appropriate size for the network
     * @param size
     */
    void setSize(const cv::Size &size);

    /***
     * set the mean that is subtracted before the image is run through the detector
     * @param mean
     */
    void setMean(const cv::Scalar &mean);

    /***
     * indicate whether or not red and blue channel shall be swapped before
     * the network is run
     * @param swap
     */
    void setSwapRB(bool swap);

    void setCrop(bool crop);

    void setDDepth(int ddepth);

    void setNMSThreshold(float nmsThreshold);

    void setConfidenceThreshold(float confThreshold);

    void setClasses(const std::vector<std::string> &classes);

    /***
     * run detector on the input frame, based on the parameters
     * a list of DetectedObjects of objects in the image
     * @param frame image
     * @param drawPred if true the prediction are drawn as labelled bounding boxes on the image
     * @param drawLabel if predictions are drawn, then this indicated it the label should be drawn to the prediction
     * @return
     */
    std::vector<DetObj> run(const cv::Mat &frame);

    std::vector<DetObj>& run(const cv::Mat &frame, std::vector<DetObj> &detections);

    /***
     * get handle to the underlying cv::dnn::Net object
     * @return OpenCV's cv::dnn::Net
     */
    cv::dnn::Net& getNet();

    /***
     * get the confidence threshold
     * @return confidence threshold
     */
    float getConfidenceThreshold() const;

    /***
     * get the non-maximum suppression threshold
     * @return non-maximum suppression threshold
     */
    float getNMSThreshold() const;

private:

    void preprocess(const cv::Mat &frame);

    void postprocess(const cv::Size &size, const std::vector<cv::Mat> &outs, std::vector<DetObj> &detections);

    std::vector<cv::String> _out_names;

    std::vector<std::string> _classes;

    cv::dnn::Net _net;

    cv::Size _size = cv::Size(0, 0);

    double _scale = 1.0;

    cv::Scalar _mean = cv::Scalar();

    float _nms_threshold = 0.4;

    float _conf_threshold = 0.5;

    int _ddepth = CV_32F;

    bool _swap_rb = true;

    bool _crop = false;

};

#endif // __OBJECTDETECTOR_HPP
