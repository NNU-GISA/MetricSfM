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
	void Run(std::string file_slam, std::string file_gps, std::string file_rgb);

	// initialize
	void ReadinSLAM(std::string file_slam);

	void ReadinGPS(std::string file_gps, std::map<int, cv::Point2d> &gps_info);

	void AssociateCameraGPS(std::string file_rgb, std::map<int, cv::Point2d> &gps_info);

	void FullBundleAdjustment();

	void SaveModel();

	void WriteCameraPointsOut(std::string path);

	void GrawGPS(std::string path);

	void GrawSLAM(std::string path);

private:
	BundleAdjustOptions bundle_full_options_;

	int rows_, cols_;
	double fx, fy, cx, cy;

	std::vector<CameraModel*> cam_models_; // camera models
	std::vector<Camera*> cams_;  // cameras
	std::vector<Point3D*> pts_;  // structures
	std::vector<cv::Point2d> cams_gps_;  // camera pose from gps
};

}  // namespace objectsfm

#endif  // OBJECTSFM_SYSTEM_INCREMENTAL_SFM_SYSTEM_H_
