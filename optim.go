package main

import "math"

const EPSILON = 1e-8

func FeedForward(net *SeqNet, input *[]float64) *[]float64 {
	var output = (*net.layers)[0].FeedForward(input)
	for i := 1; i < len(*net.layers); i++ {
		output = (*net.layers)[i].FeedForward(output)
	}
	return output
}

func BackPropagate(net *SeqNet, optimizer *Optimizer, input *[]float64, expectedOutput *[]float64) {
	var numLayers = len(*net.layers)
	var numOut = len(*expectedOutput)
	var output = (*net.layers)[0].CachedFeedForward(input)
	for i := 1; i < len(*net.layers); i++ {
		output = (*net.layers)[i].CachedFeedForward(output)
	}
	var outError = make([]float64, numOut)
	for i := 0; i < numOut; i++ {
		outError[i] = (*output)[i] - (*expectedOutput)[i]
	}
	var steps = make([]*[]float64, numLayers)
	for i := numLayers - 1; i > 0; i-- {
		steps[i] = (*net.layers)[i].GetStep(&outError)
		outError = *(*net.layers)[i].BackPropagate(&outError)
	}
	steps[0] = (*net.layers)[0].GetStep(&outError)
	(*optimizer).Update(&steps)
}

func Loss(net *SeqNet, inputs *[][]float64, outputs *[][]float64) float64 {
	var sumError float64 = 0
	for i := 0; i < len(*inputs); i++ {
		var output = FeedForward(net, &(*inputs)[i])
		for j := 0; j < len(*output); j++ {
			var diff = (*output)[j] - (*outputs)[i][j]
			sumError += diff * diff
		}
	}
	return sumError
}

type Optimizer interface {
	Update(steps *[]*[]float64)
	Step(net *SeqNet)
}

type SGD struct {
	sumSteps     *[]*[]float64
	learningRate float64
}

func NewSGD(net *SeqNet, learningRate float64) Optimizer {
	var sumSteps = make([]*[]float64, len(*net.layers))
	for i := 0; i < len(*net.layers); i++ {
		var step = make([]float64, (*net.layers)[i].Parameters())
		sumSteps[i] = &step
	}
	return SGD{
		sumSteps:     &sumSteps,
		learningRate: learningRate,
	}
}

func (sgd SGD) Update(steps *[]*[]float64) {
	for i := 0; i < len(*steps); i++ {
		for j := 0; j < len(*(*steps)[i]); j++ {
			(*(*sgd.sumSteps)[i])[j] += (*(*steps)[i])[j] * sgd.learningRate
		}
	}
}

func (sgd SGD) Step(net *SeqNet) {
	for i := 0; i < len(*net.layers); i++ {
		(*net.layers)[i].Step((*sgd.sumSteps)[i])
		for j := 0; j < len(*(*sgd.sumSteps)[i]); j++ {
			(*(*sgd.sumSteps)[i])[j] = 0
		}
	}
}

type SGDMomentum struct {
	sumSteps    *[]*[]float64
	cachedSteps *[]*[]float64
	momentum    float64
	momentumI   float64
}

func NewSGDMomentum(net *SeqNet, learningRate float64, momentum float64) Optimizer {
	var length = len(*net.layers)
	var sumSteps = make([]*[]float64, length)
	var cachedSteps = make([]*[]float64, length)
	for i := 0; i < length; i++ {
		var step = make([]float64, (*net.layers)[i].Parameters())
		var cache = make([]float64, (*net.layers)[i].Parameters())
		sumSteps[i] = &step
		cachedSteps[i] = &cache
	}
	return SGDMomentum{
		sumSteps:    &sumSteps,
		cachedSteps: &cachedSteps,
		momentum:    momentum * learningRate,
		momentumI:   (1 - momentum) * learningRate,
	}
}

func (sgd SGDMomentum) Update(steps *[]*[]float64) {
	for i := 0; i < len(*steps); i++ {
		for j := 0; j < len(*(*steps)[i]); j++ {
			(*(*sgd.cachedSteps)[i])[j] += (*(*steps)[i])[j]
		}
	}
}

func (sgd SGDMomentum) Step(net *SeqNet) {
	for i := 0; i < len(*sgd.sumSteps); i++ {
		for j := 0; j < len(*(*sgd.sumSteps)[i]); j++ {
			(*(*sgd.sumSteps)[i])[j] += (*(*sgd.sumSteps)[i])[j]*sgd.momentum + (*(*sgd.cachedSteps)[i])[j]*sgd.momentumI
			(*(*sgd.cachedSteps)[i])[j] = 0
		}
		(*net.layers)[i].Step((*sgd.sumSteps)[i])
	}
}

type RMSProp struct {
	learningRate float64
	sumSteps     *[]*[]float64
	cachedSteps  *[]*[]float64
	beta0        float64
	beta1        float64
}

func newRMSProp(net *SeqNet, learningRate float64, beta float64) Optimizer {
	var length = len(*net.layers)
	var sumSteps = make([]*[]float64, length)
	var cachedSteps = make([]*[]float64, length)
	for i := 0; i < length; i++ {
		var step = make([]float64, (*net.layers)[i].Parameters())
		var cache = make([]float64, (*net.layers)[i].Parameters())
		sumSteps[i] = &step
		cachedSteps[i] = &cache
	}
	return RMSProp{
		sumSteps:     &sumSteps,
		cachedSteps:  &cachedSteps,
		learningRate: learningRate,
		beta0:        beta,
		beta1:        1 - beta,
	}
}

func (rmsProp RMSProp) Update(steps *[]*[]float64) {
	for i := 0; i < len(*steps); i++ {
		for j := 0; j < len(*(*steps)[i]); j++ {
			(*(*rmsProp.cachedSteps)[i])[j] += (*(*steps)[i])[j]
		}
	}
}

func (rmsProp RMSProp) Step(net *SeqNet) {
	for i := 0; i < len(*rmsProp.sumSteps); i++ {
		var length = len(*(*rmsProp.sumSteps)[i])
		var steps = make([]float64, length)
		for j := 0; j < len(*(*rmsProp.sumSteps)[i]); j++ {
			var cachedStep = (*(*rmsProp.cachedSteps)[i])[j]
			(*(*rmsProp.sumSteps)[i])[j] = (*(*rmsProp.sumSteps)[i])[j]*rmsProp.beta0 + cachedStep*cachedStep*rmsProp.beta1
			steps[j] = cachedStep / (math.Sqrt((*(*rmsProp.sumSteps)[i])[j]) + EPSILON) * rmsProp.learningRate
			(*(*rmsProp.cachedSteps)[i])[j] = 0
		}
		(*net.layers)[i].Step(&steps)
	}
}
