#ifndef MNIST_READER_H
#define MNIST_READER_H
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <string>
#include <vector>

using namespace std;


class mnist_reader
{
public:
    mnist_reader(string full_path_image, string full_path_label);
    //static const pair<Eigen::VectorXi, vector<Eigen::VectorXd> createInput(string, string);
    void ReadMNIST(vector<Eigen::VectorXf> &mnist, Eigen::VectorXi &label);

private:
    static int reverseInt (int i);
    string mFullPathImage;
    string mFullPathLabel;
};

#endif // MNIST_READER_H
