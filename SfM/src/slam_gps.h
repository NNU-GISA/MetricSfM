// ObjectSfM - Object Based Structure-from-Motion.
// Copyright (C) 2018  Ohio State University, CEGE, GDA group
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef OBJECTSFM_SYSTEM_SLAM_GPS_H_
#define OBJECTSFM_SYSTEM_SLAM_GPS_H_

#include "basic_structs.h"
#include "graph.h"
#include "database.h"
#include "camera.h"
#include "structure.h"
#include "optimizer.h"
#include "accuracy_accessment.h"

namespace objectsfm {

// the system is designed to handle sfm from internet images, in which not
// all the focal lengths are known
class SLAMGPS
{
public:
	SLAMGPS();
	~SLAMGPS();

	IncrementalSfMOptions options_;

	// the main pipeline
	void Run(std::string fold);

	// slam information
	void ReadinSLAM(std::string file_slam);

	void ReadinGPS(std::string file_gps, std::map<int, cv::Point2d> &gps_info);

	void AssociateCameraGPS(std::string file_rgb, std::map<int, cv::Point2d> &gps_info);

	// feature extraction
	void FeatureExtraction(std::string fold);

	// feature matching
	void FeatureMatching(std::string fold);

	// triangulation
	void Triangulation(std::string fold);

	// bundle
	void FullBundleAdjustment();

	// 
	void SaveModel();

	void WriteCameraPointsOut(std::string path);

	void GrawGPS(std::string path);

	void GrawSLAM(std::string path);

	void SaveUndistortedImage(std::string fold);

	void SaveForSure(std::string fold);

	void SaveforOpenMVS(std::string file);

	void SaveforCMVS(std::string file);

	void GetAccuracy(std::string file, std::vector<CameraModel*> cam_models, std::vector<Camera*> cams, std::vector<Point3D*> pts);

	// 
	void DrawPts(std::vector<int> img_ids, std::string fold);

	void WriteOutMatches(int idx1, int idx2, std::vector<std::pair<int, int>> &matches);

	void WriteOutMatchGraph(std::vector<std::vector<int>> &match_graph);

	void WriteOutPriorInfo(std::string file, std::vector<std::vector<int>> &ids,
		std::vector<std::vector<cv::Mat>> &Fs,
		std::vector<std::vector<cv::Mat>> &Hs);

	void ReadinPriorInfo(std::string file, std::vector<std::vector<int>> &ids,
		std::vector<std::vector<cv::Mat>> &Fs,
		std::vector<std::vector<cv::Mat>> &Hs);

private:
	BundleAdjustOptions bundle_full_options_;
	Database db_;  // data base
	Graph graph_;  // graph

	int rows, cols;
	double fx, fy, cx, cy;
	double k1, k2, k3, p1, p2;

	std::vector<CameraModel*> cam_models_; // camera models
	std::vector<Camera*> cams_;  // cameras
	std::vector<Point3D*> pts_;  // 3d points from slam
	std::vector<Point3D*> pts_new_;  // 3d points from matching

	std::vector<cv::Point2d> cams_gps_;  // camera pose from gps
	std::vector<std::string> cams_name_;  // camera image name
	AccuracyAssessment* accuracer_;
	double th_outlier = 3.0;
};

}  // namespace objectsfm

#endif  // OBJECTSFM_SYSTEM_INCREMENTAL_SFM_SYSTEM_H_
