package main

import (
	"fmt"
	"math/rand"
	"time"
)

func main() {
	rand.Seed(time.Now().UTC().UnixNano())
	var dense0 = NewDenseRand(2, 5, -0.1, 0.1)
	var bias0 = NewBiasRand(5, 0.0, 0.6)
	var relu = NewActivation(5, ReLU{})
	var dense1 = NewDenseRand(5, 1, 0.0, 0.6)
	var bias1 = NewBiasRand(1, 0.0, 0.6)
	var sigmoid = NewActivation(1, Sigmoid{})
	var layers = []Layer{dense0, bias0, relu, dense1, bias1, sigmoid}
	var net = SeqNet{
		layers: &layers,
	}

	var optimizer = newRMSProp(&net, 1e-3, 0.999)

	var in = [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	var out = [][]float64{{0}, {1}, {1}, {0}}

	for i := 0; i < len(in); i++ {
		fmt.Println("in: ", in[i])
		fmt.Println("out: ", FeedForward(&net, &in[i]))
	}

	//5145128172
	//2067972000
	var startTime = time.Now().UTC().UnixNano()
	for i := 0; i < 1000000; i++ {
		var index = int(rand.Float64() * 4)
		BackPropagate(&net, &optimizer, &in[index], &out[index])
		optimizer.Step(&net)
	}
	fmt.Println(time.Now().UTC().UnixNano() - startTime)

	for i := 0; i < len(in); i++ {
		fmt.Println("in: ", in[i])
		fmt.Println("out: ", FeedForward(&net, &in[i]))
	}
}
