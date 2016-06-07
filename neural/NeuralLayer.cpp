#include "NeuralLayer.h"


int NeuralLayer::GroupCount;
int NeuralLayer::EffectiveGroupCount;
int NeuralLayer::Step;

NeuralLayer::NeuralLayer()
{
}


NeuralLayer::~NeuralLayer()
{
	deleteData();
	if (WeightMatrix) { delete WeightMatrix; }
	if (BiasVector) { delete BiasVector; }
}


void NeuralLayer::deleteData()
{
	if (InputMatrix) { delete InputMatrix; }
	if (OutputMatrix) { delete OutputMatrix; }
	if (DeltaMatrix) { delete DeltaMatrix; }
	if (ExpectMatrix) { delete ExpectMatrix; }
	if (_asBiasVector) { delete _asBiasVector; }
}




