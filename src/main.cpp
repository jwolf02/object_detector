#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <ObjectDetector.hpp>
#include <Consumable.hpp>
#include <Thread.hpp>
#include <atomic>
#include <thread>
#include <cstdlib>
#include <object_detector/DetectedObjects.h>

class DetectorThread : public Thread {
public:

	DetectorThread() = default;

	virtual void setup() {} // nothing to do

	virtual void loop() {
		frame.wait_for_update(std::chrono::milliseconds(100));
		auto &image = frame.consume();
		detector.run(image, tmp);
		frame.finish();
		detections.update(std::move(tmp));
	}

	virtual void cleanup() {} // nothing to do

	ObjectDetector detector;

	Consumable<cv::Mat> frame;

    std::vector<DetObj> tmp;

	Consumable<std::vector<DetObj>> detections;

};

DetectorThread thread;

void imageCallback(const sensor_msgs::Image::ConstPtr& msg) {
	cv_bridge::CvImageConstPtr cv_ptr;
	try {
		// get image as cv::Mat from camera node
		cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
		
		auto &image = thread.frame;
		if (image.updatable()) {
			image.update(cv_ptr->image);
		}

	} catch (cv_bridge::Exception &e) {
		ROS_ERROR("cv_bridge: %s", e.what());
	}
}

void terminate() {
	thread.terminate();
}

#define debug(x)   std::cout << x << std::endl;

int main(int argc, char **argv) {

	ros::init(argc, argv, "object_detector");

	ros::NodeHandle n("~");

	// command line parameters
	std::string model_path;
	std::string config_path;
	std::string framework;
	double scale;
	double nms_threshold;
	double conf_threshold;
	std::string target;
    bool verbose;
    bool crop;
    std::string camera_topic;

	// retrieve parameters
	if (!n.getParam("model_path", model_path)) {
        ROS_ERROR("model_path must be set");
        exit(1);
    }
	if (!n.getParam("config_path", config_path)) {
        ROS_ERROR("config_path must be set");
        exit(1);
    }
	n.param("framework", framework, std::string(""));
	n.param("scale", scale, 1.0);
	n.param("nms_threshold", nms_threshold, 0.4);
	n.param("confidence_threshold", conf_threshold, 0.3);
	n.param("target", target, std::string("cpu"));
    n.param("verbose", verbose, false);
    n.param("crop", crop, true);
    n.param("camera_topic", camera_topic, std::string("/camera/image_raw"));
    
	// set parameters
	auto &detector = thread.detector;
	detector.readNet(model_path, config_path, framework);
	detector.setConfidenceThreshold(conf_threshold);
	detector.setNMSThreshold(nms_threshold);
	detector.setCrop(crop);
	detector.setScale(scale);

    if (verbose) {
        std::cout << "target=" << target << std::endl;
    }

	// discern target
	if (target == "cpu") {
		detector.getNet().setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		detector.getNet().setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	} else if (target == "gpu") {
		detector.getNet().setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		detector.getNet().setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	} else {
		ROS_ERROR("unrecognized target: %s", target.c_str());
	}	

    if (verbose) {
        std::cout << "subscribing to camera under " << camera_topic << std::endl;
    }

	// subscribe to camera image
	ros::Subscriber sub = n.subscribe(camera_topic, 1, imageCallback);

    if (verbose) {
        std::cout << "publishing detected objects under /object_detector/detected_objects" << std::endl;
    }

	// publish objects detected
	ros::Publisher pub = n.advertise<object_detector::DetectedObjects>("/object_detector/detected_objects", 5);

    if (verbose) {
        std::cout << "running object detector in background thread" << std::endl;
    }

	std::atexit(terminate);
	thread.run();

	object_detector::DetectedObjects detobjs;
	auto &detections = detobjs.detections;
	while (true) {
		ros::spinOnce();
		// check if detector has new output
		if (thread.detections.available()) {
			detections.clear();

			// copy from DetObj to object_detector::DetectedObject type
			auto &det_list = thread.detections.consume();
			for (const auto &det : det_list) {
				object_detector::DetectedObject tmp;
				tmp.id = det.id();
                auto &bb = det.boundingBox();
				tmp.x = bb.x;
				tmp.y = bb.y;
				tmp.width = bb.width;
				tmp.height = bb.height;
				tmp.confidence = det.confidence();
				tmp.class_name = det.name();
				detections.push_back(tmp);
			}
			thread.detections.finish();

			// publish data
			pub.publish(detobjs);

            if (verbose) {
                auto &det = detobjs.detections;
                if (det.empty()) {
                    std::cout << "no object detected" << std::endl;
                } else {
                    auto &det = detobjs.detections;
                    std::cout << "detected " << det.size() << " objects: ";
                    for (auto &d : det) {
                        std::cout << d.id << '(' << d.confidence << ") ";
                    }
                    std::cout << std::endl;
                }
            }
		}
	}

	return EXIT_SUCCESS;
}

