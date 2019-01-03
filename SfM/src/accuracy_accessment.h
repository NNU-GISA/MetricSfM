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

#ifndef OBJECTSFM_OBJ_ACCURACY_ASSESSMENT_H_
#define OBJECTSFM_OBJ_ACCURACY_ASSESSMENT_H_

#include "basic_structs.h"
#include "graph.h"
#include "database.h"
#include "camera.h"
#include "structure.h"
#include "optimizer.h"

namespace objectsfm {

	// the system is designed to handle sfm from internet images, in which not
	// all the focal lengths are known
	class AccuracyAssessment
	{
	public:
		AccuracyAssessment();
		~AccuracyAssessment();

		void SetData(std::vector<CameraModel*> &cam_models, std::vector<Camera*> &cams, std::vector<Point3D*> &pts);

		// pts
		bool ErrorReprojectionPti(int i, double &e_avg, double &e_mse, int &n_obs);

		void ErrorReprojectionPts(double &e_avg, double &e_mse, int &n_obs, std::vector<double> &errors);

		// cameras


		// camera models
	private:
		std::vector<CameraModel*> cam_models_; // camera models
		std::vector<Camera*> cams_;  // cameras
		std::vector<Point3D*> pts_;  // structures
	};


}  // namespace objectsfm

#endif  // OBJECTSFM_OBJ_CAMERA_H_
