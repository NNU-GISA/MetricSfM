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

#include "orientation/fundamental_matrix_eight_point.h"

#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/LU>

#include <ctime>

#include "utils/basic_funcs.h"

namespace objectsfm {

	bool FundamentalMatrixEightPoint::NormalizedEightPointFundamentalMatrixRANSAC( std::vector<Eigen::Vector2d>& pts1, 
		std::vector<Eigen::Vector2d>& pts2, Eigen::Matrix3d &F )
	{
		int num = pts1.size();

		std::vector<Eigen::Matrix3d> fundamental_matrices;
		if (num<8)
		{
			return false;
		}
		else if (num<16)
		{
			// inadequate points, use all for Essential Matrix estimation
			Eigen::Matrix3d F_hyp;
			if (!NormalizedEightPointFundamentalMatrix(pts1, pts2, F_hyp))
			{
				return false;
			}
			fundamental_matrices.push_back(F_hyp);
		}
		else
		{
			int ransac_times = 200;
			for (int i=0; i<ransac_times; ++i)
			{
				// get random index
				std::vector<int> idxs;
				math::RandVectorN(0, num, 8, idxs);

				std::vector<Eigen::Vector2d> pts1_partial, pts2_partial;
				for (int j=0; j<idxs.size(); ++j)
				{
					int idx = idxs[j];
					pts1_partial.push_back(pts1[idx]);
					pts2_partial.push_back(pts2[idx]);
				}

				Eigen::Matrix3d F_hyp;
				if (!NormalizedEightPointFundamentalMatrix(pts1_partial, pts2_partial, F_hyp))
				{
					continue;
				}
				fundamental_matrices.push_back(F_hyp);
			}
		}

		if (!fundamental_matrices.size())
		{
			return false;
		}

		// find the one with least error
		double error_min = 1000000.0;
		int idx_min = 0;
		for (int i=0; i<fundamental_matrices.size();++i)
		{
			double error_i = Error(fundamental_matrices[i], pts1, pts2);
			//std::cout << fundamental_matrices[i] << std::endl;
			//std::cout << error_i << std::endl;
			if (error_i < error_min)
			{
				error_min = error_i;
				idx_min = i;
			}
		}

		F = fundamental_matrices[idx_min];
	}


	// Computes the Fundamental Matrix
	// (http://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision) ) from 8
	// or more image correspondences according to the normalized 8 point algorithm
	// (Hartley and Zisserman alg 11.1 page 282). 
	//  estimate the fundamental matrix that makes pts2^t * F * pts1 = 0
	bool FundamentalMatrixEightPoint::NormalizedEightPointFundamentalMatrix(std::vector<Eigen::Vector2d>& pts1, std::vector<Eigen::Vector2d>& pts2, 
		Eigen::Matrix3d &F)
	{
		int num_pts = pts1.size();

		// normalize the image points.
		std::vector<Eigen::Vector2d> pts1_norm(pts1.size());
		std::vector<Eigen::Vector2d> pts2_norm(pts2.size());

		Eigen::Matrix3d mat1_norm, mat2_norm;
		NormalizeImagePoints(pts1, &pts1_norm, &mat1_norm);
		NormalizeImagePoints(pts2, &pts2_norm, &mat2_norm);

		//
		// Build the constraint matrix based on x2' * F * x1 = 0.
		Eigen::Matrix<double, Eigen::Dynamic, 9> constraint_matrix(num_pts, 9);
		for (int i = 0; i < num_pts; i++) 
		{
			constraint_matrix.block<1, 3>(i, 0) = pts1_norm[i].homogeneous();
			constraint_matrix.block<1, 3>(i, 0) *= pts2_norm[i].x();
			constraint_matrix.block<1, 3>(i, 3) = pts1_norm[i].homogeneous();
			constraint_matrix.block<1, 3>(i, 3) *= pts2_norm[i].y();
			constraint_matrix.block<1, 3>(i, 6) = pts1_norm[i].homogeneous();
		}

		// Solve the constraint equation for F from nullspace extraction.
		// An LU decomposition is efficient for the minimally constrained case.
		// Otherwise, use an SVD.
		Eigen::Matrix<double, 9, 1> normalized_fvector;
		if (num_pts == 8)
		{
			const auto lu_decomposition = constraint_matrix.fullPivLu();
			if (lu_decomposition.dimensionOfKernel() != 1)
			{
				return false;
			}
			normalized_fvector = lu_decomposition.kernel();
		}
		else
		{
			Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9> > cmatrix_svd(constraint_matrix, Eigen::ComputeFullV);
			normalized_fvector = cmatrix_svd.matrixV().col(8);
		}

		// NOTE: This is the transpose of a valid fundamental matrix! We implement a
		// "lazy" transpose and defer it to the SVD a few lines below.
		Eigen::Map<const Eigen::Matrix3d> normalized_fmatrix(normalized_fvector.data());

		// Find the closest singular matrix to F under frobenius norm. We can compute
		// this matrix with SVD.
		Eigen::JacobiSVD<Eigen::Matrix3d> fmatrix_svd(normalized_fmatrix.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Vector3d singular_values = fmatrix_svd.singularValues();

		// Set smallest eigen value of fmatrix into 0
		singular_values[2] = 0.0;

		// Recompute F
		F = fmatrix_svd.matrixU() * singular_values.asDiagonal() * fmatrix_svd.matrixV().transpose();

		// Correct for the point normalization.
		F = mat2_norm.transpose() * F * mat1_norm;

		return true;
	}


	// Computes the normalization matrix transformation that centers image points
	// around the origin with an average distance of sqrt(2) to the centroid.
	// Returns the transformation matrix and the transformed points. This assumes
	// that no points are at infinity.
	bool FundamentalMatrixEightPoint::NormalizeImagePoints(const std::vector<Eigen::Vector2d>& image_points,
		std::vector<Eigen::Vector2d>* normalized_image_points,
		Eigen::Matrix3d* normalization_matrix)
	{
		Eigen::Map<const Eigen::Matrix<double, 2, Eigen::Dynamic> > image_points_mat(image_points[0].data(), 2, image_points.size());

		// Allocate the output vector and map an Eigen object to the underlying data
		// for efficient calculations.
		normalized_image_points->resize(image_points.size());
		Eigen::Map<Eigen::Matrix<double, 2, Eigen::Dynamic> > normalized_image_points_mat((*normalized_image_points)[0].data(), 2, image_points.size());

		// Compute centroid.
		const Eigen::Vector2d centroid(image_points_mat.rowwise().mean());

		// Calculate average RMS distance to centroid.
		const double rms_mean_dist = sqrt((image_points_mat.colwise() - centroid).squaredNorm() / image_points.size());

		// Create normalization matrix.
		const double norm_factor = sqrt(2.0) / rms_mean_dist;
		*normalization_matrix << norm_factor, 0, -1.0 * norm_factor* centroid.x(),
			0, norm_factor, -1.0 * norm_factor * centroid.y(),
			0, 0, 1;

		// Normalize image points.
		const Eigen::Matrix<double, 3, Eigen::Dynamic> normalized_homog_points = (*normalization_matrix) * image_points_mat.colwise().homogeneous();
		normalized_image_points_mat = normalized_homog_points.colwise().hnormalized();

		return true;
	}

	double FundamentalMatrixEightPoint::Error(const Eigen::Matrix3d & F, const std::vector<Eigen::Vector2d>& pts1, const std::vector<Eigen::Vector2d>& pts2)
	{
		double total_error = 0.0;
		for (int i = 0; i<pts1.size(); ++i)
		{
			const Eigen::Vector3d epiline_x = F * pts1[i].homogeneous();
			const double numerator_sqrt = pts2[i].homogeneous().dot(epiline_x);
			const Eigen::Vector4d denominator(pts2[i].homogeneous().dot(F.col(0)),
				pts2[i].homogeneous().dot(F.col(1)),
				epiline_x[0], epiline_x[1]);

			// Finally, return the complete Sampson distance.
			total_error += numerator_sqrt * numerator_sqrt / denominator.squaredNorm();
		}

		return total_error;
	}


}  // namespace objectsfm
