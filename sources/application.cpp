#include "headers/application.hpp"
#include <headers/rapidjson/error/en.h>

#include <math.h>
#include <fstream>

Application::Application(NeuralNetwork::Ptr network, Batch teachingBatch, Batch testingBatch, const std::string& configFileName)
: mNetwork(network)
, mTeacher(mNetwork)
, mTeachingBatch(teachingBatch)
, mTestingBatch(testingBatch)
, mStatsCollector()
, mTestCounter(0)
{
    // Charge la configuration de l'application
    loadConfig(configFileName);
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
    mStatsCollector[0].addResult(runTest());

    for(unsigned int loopIndex{0}; loopIndex < nbLoops; ++loopIndex)
    {
        std::cout << "Apprentissage num. : " << (loopIndex)*nbTeachingsPerLoop << std::endl;
        runTeach(nbTeachingsPerLoop);
        mStatsCollector[loopIndex+1].addResult(runTest());
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
        Sample sample{mTeachingBatch[distribution(randomEngine)]};
        mTeacher.backProp(sample.first, sample.second, mConfig.step, mConfig.dx);
        if(index %100 == 0)
            std::cout << "+" << index << std::endl;
    }
}

float Application::runTest(int limit, bool returnErrorRate)
{
    float errorMean{0};

    if (returnErrorRate)
    {
        int maxLine, maxCol;
        for(std::vector<Sample>::iterator itr = mTestingBatch.begin(); itr != mTestingBatch.end() && limit-- != 0; ++itr)
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
        for(std::vector<Sample>::iterator itr = mTestingBatch.begin(); itr != mTestingBatch.end() && limit-- != 0; ++itr)
        {
            Eigen::MatrixXf output{mNetwork->process(itr->first)};
            errorMean += sqrt((output - itr->second).squaredNorm());
        }
    }

    return errorMean/static_cast<float>(mTestingBatch.size());
}

void Application::loadConfig(const std::string& configFileName)
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

    setConfig(doc);
    displayConfig(doc);
}


void Application::setConfig(rapidjson::Document& document)
{
    mConfig.step = document["step"].GetFloat();
    mConfig.dx = document["dx"].GetFloat();

    mConfig.nbExperiments = document["nbExperiments"].GetUint();
    mConfig.nbLoopsPerExperiment = document["nbLoopsPerExperiment"].GetUint();
    mConfig.nbTeachingsPerLoop = document["nbTeachingsPerLoop"].GetUint();
}

void Application::displayConfig(rapidjson::Document &doc)
{
    for(auto mItr = doc.MemberBegin(); mItr != doc.MemberEnd(); ++mItr)
    {
        auto key = (*mItr).name.GetString();

        if(doc[key].IsArray())
        {
            *mStatsCollector.getCSVFile() << key;
            for(rapidjson::SizeType i = 0; i < doc[key].Size(); i++)
                *mStatsCollector.getCSVFile() << doc[key].GetFloat();
        }
        else if (doc[key].IsFloat())
        {
            *mStatsCollector.getCSVFile() << key << doc[key].GetFloat();
        }
        else if (doc[key].IsUint())
        {
            *mStatsCollector.getCSVFile() << key << doc[key].GetUint();
        }
        else if (doc[key].IsBool())
        {
            *mStatsCollector.getCSVFile() << key << doc[key].GetBool();
        }
    }

    *mStatsCollector.getCSVFile() << endrow;
}


