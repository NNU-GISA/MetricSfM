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

#ifndef OBJECTSFM_OBJ_OPTIMIZER_H_
#define OBJECTSFM_OBJ_OPTIMIZER_H_

#include <vector>
#include <Eigen/Core>
#include "ceres/ceres.h"
#include "basic_structs.h"

namespace objectsfm {

class Camera;
class Point3D;

class BundleAdjuster
{
public:
	BundleAdjuster(std::vector<Camera*> cams, std::vector<CameraModel*> cam_models, std::vector<Point3D*> pts);

	~BundleAdjuster();

	void SetOptions(BundleAdjustOptions options);

	void RunOptimizetion(bool is_seed, double weight);

	void FindOutliersPoints();

	void UpdateParameters();

private:
	ceres::Problem problem_;

	ceres::Solver::Options options_;
	
	ceres::Solver::Summary summary_;

	std::vector<Camera*> cams_;
	
	std::vector<CameraModel*> cam_models_;

	std::vector<Point3D*> pts_;

	// Move the "center" of the reconstruction to the origin, where the center is determined by computing the 
	// marginal median of the points. The reconstruction is then scaled so that the median absolute deviation 
	// of the points measured from the origin is 100.0.
	void Normalize();

	void Perturb();
};


}  // namespace objectsfm

#endif  // OBJECTSFM_OBJ_OPTIMIZER_H_
