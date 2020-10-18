package main

import (
	"math"
	"math/rand"
)

type SeqNet struct {
	layers *[]Layer
}

type Layer interface {
	FeedForward(input *[]float64) *[]float64

	CachedFeedForward(input *[]float64) *[]float64

	BackPropagate(error *[]float64) *[]float64

	GetStep(delta *[]float64) *[]float64

	Step(step *[]float64)

	Parameters() int
}

type Dense struct {
	in          int
	out         int
	connections *[]float64
	lastInput   []float64
}

func NewDenseRand(in int, out int, min float64, max float64) *Dense {
	var numConnections = in * out
	var diff = max - min
	var connections = make([]float64, numConnections)
	for i := 0; i < numConnections; i++ {
		connections[i] = min + rand.Float64()*diff
	}
	return &Dense{
		in:          in,
		out:         out,
		connections: &connections,
		lastInput:   make([]float64, in),
	}
}

func (dense Dense) FeedForward(input *[]float64) *[]float64 {
	var output = make([]float64, dense.out)
	for i := 0; i < dense.out; i++ {
		var offset = i * dense.in
		for j := 0; j < dense.in; j++ {
			output[i] += (*input)[j] * (*dense.connections)[offset+j]
		}
	}
	return &output
}

func (dense Dense) CachedFeedForward(input *[]float64) *[]float64 {
	copy(dense.lastInput, *input)
	return dense.FeedForward(input)
}

func (dense Dense) BackPropagate(error *[]float64) *[]float64 {
	var prevDelta = make([]float64, dense.in)
	for i := 0; i < dense.out; i++ {
		var offset = i * dense.in
		for j := 0; j < dense.in; j++ {
			prevDelta[j] += (*error)[i] * (*dense.connections)[offset+j]
		}
	}
	return &prevDelta
}

func (dense Dense) GetStep(delta *[]float64) *[]float64 {
	var step = make([]float64, len(*dense.connections))
	for i := 0; i < dense.out; i++ {
		var offset = i * dense.in
		for j := 0; j < dense.in; j++ {
			step[offset+j] = (*delta)[i] * dense.lastInput[j]
		}
	}
	return &step
}

func (dense Dense) Step(step *[]float64) {
	for i := 0; i < len(*step); i++ {
		(*dense.connections)[i] -= (*step)[i]
	}
}

func (dense Dense) Parameters() int {
	return len(*dense.connections)
}

type Bias struct {
	connections *[]float64
}

func NewBiasRand(out int, min float64, max float64) *Bias {
	var diff = max - min
	var connections = make([]float64, out)
	for i := 0; i < out; i++ {
		connections[i] = min + rand.Float64()*diff
	}
	return &Bias{
		connections: &connections,
	}
}

func (bias Bias) FeedForward(input *[]float64) *[]float64 {
	var output = make([]float64, len(*bias.connections))
	for i := 0; i < len(*bias.connections); i++ {
		output[i] = (*input)[i] + (*bias.connections)[i]
	}
	return &output
}

func (bias Bias) CachedFeedForward(input *[]float64) *[]float64 {
	return bias.FeedForward(input)
}

func (bias Bias) BackPropagate(error *[]float64) *[]float64 {
	var copied = make([]float64, len(*error))
	copy(copied, *error)
	return &copied
}

func (bias Bias) GetStep(delta *[]float64) *[]float64 {
	var copied = make([]float64, len(*delta))
	copy(copied, *delta)
	return &copied
}

func (bias Bias) Step(step *[]float64) {
	for i := 0; i < len(*step); i++ {
		(*bias.connections)[i] -= (*step)[i]
	}
}

func (bias Bias) Parameters() int {
	return len(*bias.connections)
}

type Function interface {
	out(in float64) float64

	dOut(in float64) (float64, float64)
}

type Activation struct {
	out      int
	function Function
	cache    []float64
}

func NewActivation(out int, function Function) *Activation {
	return &Activation{
		out:      out,
		function: function,
		cache:    make([]float64, out),
	}
}

func (activation Activation) FeedForward(input *[]float64) *[]float64 {
	var output = make([]float64, activation.out)
	for i := 0; i < activation.out; i++ {
		output[i] = activation.function.out((*input)[i])
	}
	return &output
}

func (activation Activation) CachedFeedForward(input *[]float64) *[]float64 {
	var output = make([]float64, activation.out)
	for i := 0; i < activation.out; i++ {
		output[i], activation.cache[i] = activation.function.dOut((*input)[i])
	}
	return &output
}

func (activation Activation) BackPropagate(error *[]float64) *[]float64 {
	var out = make([]float64, activation.out)
	for i := 0; i < activation.out; i++ {
		out[i] = (*error)[i] * activation.cache[i]
	}
	return &out
}

func (activation Activation) GetStep(delta *[]float64) *[]float64 {
	return &[]float64{}
}

func (activation Activation) Step(step *[]float64) {

}

func (activation Activation) Parameters() int {
	return 0
}

type ReLU struct{}

func (relu ReLU) out(in float64) float64 {
	if in < 0 {
		return 0
	} else {
		return in
	}
}

func (relu ReLU) dOut(in float64) (float64, float64) {
	if in < 0 {
		return 0, 0
	} else {
		return in, 1
	}
}

type Sigmoid struct{}

func (sigmoid Sigmoid) out(in float64) float64 {
	return 1 / (1 + math.Exp(-in))
}

func (sigmoid Sigmoid) dOut(in float64) (float64, float64) {
	var val = sigmoid.out(in)
	return val, val * (1 - val)
}
