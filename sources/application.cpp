#include "headers/application.hpp"

#include <math.h>
#include <rapidjson/include/rapidjson/error/en.h>
#include <fstream>

Application::Application(NeuralNetwork::Ptr network, Batch teachingBatch, Batch testingBatch)
: mNetwork(network)
, mTeacher(mNetwork)
, mTeachingBatch(teachingBatch)
, mTestingBatch(testingBatch)
, mStatsCollector()
, mTestCounter(0)
{
    // Charge la configuration de l'application
    loadConfig();
}

Application::Application(   NeuralNetwork::Ptr network,
                            std::function<Eigen::VectorXf (Eigen::VectorXf)> modelFunction,
                            std::vector<Eigen::VectorXf> teachingInputs,
                            std::vector<Eigen::VectorXf> testingInputs)
: mNetwork(network)
, mTeacher(mNetwork)
, mStatsCollector()
, mTestCounter(0)
{
    // Charge la configuration de l'application
    loadConfig();

    // Génère le batch d'apprentissage à partir des entrées et de la fonction à modéliser
    for(size_t i{0}; i < teachingInputs.size(); ++i)
        mTeachingBatch.push_back(Sample(teachingInputs[i], modelFunction(teachingInputs[i])));
    // Génère le batch d'apprentissage à partir des entrées et de la fonction à modéliser
    for(size_t i{0}; i < testingInputs.size(); ++i)
        mTestingBatch.push_back(Sample(testingInputs[i], modelFunction(testingInputs[i])));
}

void Application::runExperiments(unsigned int nbExperiments, unsigned int nbLoops, unsigned int nbTeachingsPerLoop)
{
    for(unsigned int index{0}; index < nbExperiments; ++index)
    {
        runSingleExperiment(nbLoops, nbTeachingsPerLoop);
        std::cout << "Exp num. " << (index+1) << " finie !" << std::endl;
        resetExperiment();
    }

    mStatsCollector.exportData(true);
}

void Application::runSingleExperiment(unsigned int nbLoops, unsigned int nbTeachingsPerLoop)
{
    for(unsigned int loopIndex{0}; loopIndex < nbLoops; ++loopIndex)
    {
        runTeach(nbTeachingsPerLoop);
        mStatsCollector[loopIndex].addResult(runTest());
        std::cout << "Apprentissage num. : " << (loopIndex+1)*400 << std::endl;
    }
}

void Application::resetExperiment()
{
    mNetwork->reset();
}

void Application::runTeach(unsigned int nbTeachings)
{
    std::uniform_int_distribution<> distribution(0, mTeachingBatch.size()-1);
    std::mt19937 randomEngine((std::random_device())());

    for(unsigned int index{0}; index < nbTeachings; index++)
    {
        auto sample{mTeachingBatch[distribution(randomEngine)]};
        mTeacher.backProp(sample.first, sample.second, mConfig.step, mConfig.dx);
    }
}

float Application::runTest(int limit)
{
    float errorMean{0};

    for(auto itr = mTestingBatch.begin(); itr != mTestingBatch.end() && limit-- != 0; ++itr)
    {
        auto output{mNetwork->process(itr->first)};
        errorMean += sqrt((output - itr->second).squaredNorm());
    }

    return errorMean/static_cast<float>(mTestingBatch.size());
}

void Application::loadConfig()
{
    std::stringstream ss;
    std::ifstream inputStream("config.json");
    if(!inputStream)
    {
      throw("Failed to load file");
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

    setConfig(doc);
}


void Application::setConfig(rapidjson::Document& document)
{
    mConfig.step = document["step"].GetFloat();
    mConfig.dx = document["dx"].GetFloat();

    *mStatsCollector.getCSVFile() << "Step" << mConfig.step << "dx" << mConfig.dx << endrow;
}


