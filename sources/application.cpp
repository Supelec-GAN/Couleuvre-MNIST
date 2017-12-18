#include "headers/application.hpp"
#include <headers/rapidjson/error/en.h>

#include <math.h>
#include <fstream>

Application::Application(Batch teachingBatch, Batch testingBatch, const std::string& configFileName)
: mConfig(loadConfig(configFileName))
, mNetwork(new NeuralNetwork(mConfig.neuronLayerSizes))
, mTeacher(mNetwork)
, mTeachingBatch(teachingBatch)
, mTestingBatch(testingBatch)
, mStatsCollector()
, mTestCounter(0)
{

}

void Application::runExperiments()
{
    for(unsigned int index{0}; index < mConfig.nbExperiments; ++index)
    {
        runSingleExperiment();
        std::cout << "Exp num. " << (index+1) << " finie !" << std::endl;
        resetExperiment();
    }

    mStatsCollector.exportData(true);
}

void Application::runSingleExperiment()
{
    mStatsCollector[0].addResult(runTest());

    for(unsigned int loopIndex{0}; loopIndex < mConfig.nbLoopsPerExperiment; ++loopIndex)
    {
        std::cout << "Apprentissage num. : " << (loopIndex)*mConfig.nbTeachingsPerLoop << std::endl;
        runTeach();
        mStatsCollector[loopIndex+1].addResult(runTest());
    }
}

void Application::resetExperiment()
{
    mNetwork->reset();
}

void Application::runTeach()
{
    std::uniform_int_distribution<> distribution(0, mTeachingBatch.size()-1);
    std::mt19937 randomEngine((std::random_device())());

    for(unsigned int index{0}; index < mConfig.nbTeachingsPerLoop; index++)
    {
        Sample sample{mTeachingBatch[distribution(randomEngine)]};
        mTeacher.backProp(sample.first, sample.second, mConfig.step, mConfig.dx);
        if(index %100 == 0)
            std::cout << "+" << index << std::endl;
    }
}

float Application::runTest(bool returnErrorRate)
{
    float errorMean{0};

    if (returnErrorRate)
    {
        int maxLine, maxCol;
        for(std::vector<Sample>::iterator itr = mTestingBatch.begin(); itr != mTestingBatch.end(); ++itr)
        {
            Eigen::MatrixXf output{mNetwork->process(itr->first)};
            output.maxCoeff(&maxLine, &maxCol);
            output.setZero();
            output(maxLine, maxCol) = 1;
            errorMean += sqrt((output - itr->second).squaredNorm())/sqrt(2);
        }
    }
    else
    {
        for(std::vector<Sample>::iterator itr = mTestingBatch.begin(); itr != mTestingBatch.end(); ++itr)
        {
            Eigen::MatrixXf output{mNetwork->process(itr->first)};
            errorMean += sqrt((output - itr->second).squaredNorm());
        }
    }

    return errorMean/static_cast<float>(mTestingBatch.size());
}

Application::Config Application::loadConfig(const std::string& configFileName)
{
    std::stringstream ss;
    std::ifstream inputStream(configFileName);
    if(!inputStream)
    {
      throw std::runtime_error("Application::loadConfig Error - Failed to load " + configFileName);
    }
    ss << inputStream.rdbuf();
    inputStream.close();
    rapidjson::Document doc;
    rapidjson::ParseResult ok(doc.Parse(ss.str().c_str()));
    if(!ok)
    {
        std::cout << stderr << "JSON parse error: %s (%u)" << rapidjson::GetParseError_En(ok.Code()) << ok.Offset() << std::endl;
        exit(EXIT_FAILURE);
    }

    displayConfig();

    return getConfig(doc);
}


Application::Config Application::getConfig(rapidjson::Document& document)
{
    Config conf;

    conf.step = document["step"].GetFloat();
    conf.dx = document["dx"].GetFloat();

    conf.nbExperiments = document["nbExperiments"].GetUint();
    conf.nbLoopsPerExperiment = document["nbLoopsPerExperiment"].GetUint();
    conf.nbTeachingsPerLoop = document["nbTeachingsPerLoop"].GetUint();

    auto layersSizes = document["layersSizes"].GetArray();
    for(rapidjson::SizeType i = 0; i < layersSizes.Size(); i++)
        conf.neuronLayerSizes.push_back(layersSizes[i].GetUint());

    return conf;
}

void Application::displayConfig()
{
    *mStatsCollector.getCSVFile() << "step" << mConfig.step;
    *mStatsCollector.getCSVFile() << "dx" << mConfig.dx;
    *mStatsCollector.getCSVFile() << "nbExperiments" << mConfig.nbExperiments;
    *mStatsCollector.getCSVFile() << "nbLoopsPerExperiment" << mConfig.nbLoopsPerExperiment;
    *mStatsCollector.getCSVFile() << "nbTeachingsPerLoop" << mConfig.nbTeachingsPerLoop;
    *mStatsCollector.getCSVFile() << "neuronLayerSizes";
    for(auto c : mConfig.neuronLayerSizes)
        *mStatsCollector.getCSVFile() << c;

    *mStatsCollector.getCSVFile() << endrow;
}


