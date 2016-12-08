#include <eigen3/Eigen/Dense>

#ifndef UTILS_HPP
#define UTILS_HPP

void loadMatrix(const std::string filename,
                Eigen::MatrixXd&matrix, Eigen::MatrixXd&mask);


#endif
