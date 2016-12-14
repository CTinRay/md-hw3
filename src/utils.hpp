#include <eigen3/Eigen/Dense>

#ifndef UTILS_HPP
#define UTILS_HPP

void loadMatrix(const std::string& filename,
                Eigen::MatrixXd&matrix, Eigen::MatrixXd&mask);

void loadModel(const std::string& filename,
               Eigen::MatrixXd& m1, Eigen::MatrixXd& m2);

void loadMem(const std::string& filename, Eigen::MatrixXd& mem);   

void loadFilled(const std::string& filename,
                Eigen::MatrixXd&matrix, Eigen::MatrixXd&mask);

#endif
