#include "argvParser.h"

vector<string> split(const string& s, char delimiter)
{
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

void parseClient(string detectParams, map<string, double> *outputParams)
{
    vector<string> splittedLine = split(detectParams, ' ');
    for (int i = 0; i < splittedLine.size(); i+=2)
    {
        (*outputParams)[splittedLine[i]] = strtod(splittedLine[i+1].c_str(), nullptr);
    }
}