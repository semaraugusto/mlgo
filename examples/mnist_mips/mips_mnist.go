package main

import (
	"errors"
	"fmt"

	"mlgo/common"
	"mlgo/ml"
)

type mnistHParams struct {
	nInput   int32
	nHidden  int32
	nClasses int32
}

type mnistModel struct {
	hparams mnistHParams

	fc1Weight *ml.Tensor
	fc1Bias   *ml.Tensor

	fc2Weight *ml.Tensor
	fc2Bias   *ml.Tensor
}

const (
	// ReadFromBigEndian constant
	ReadFromBigEndian = true
	// OutputToBigEndian constant
	OutputToBigEndian = true
	// InputSize constant
	InputSize = 14
)

func mipsMnistModelLoad(model *mnistModel) error {
	fmt.Println("start MipsMnistModelLoad")
	modelBytes := common.ReadBytes(common.MODEL_ADDR, ReadFromBigEndian)
	index := 0
	fmt.Println("modelBytes len: ", len(modelBytes))

	// verify magic
	{
		magic := common.ReadInt32FromBytes(modelBytes, &index, ReadFromBigEndian)
		fmt.Printf("magic: %x\n", magic)
		if magic != 0x67676d6c {
			return errors.New("invalid model file (bad magic)")
		}
	}

	// Read FC1 layer 1
	{
		fmt.Println("reading fc1")
		nDims := int32(common.ReadInt32FromBytes(modelBytes, &index, ReadFromBigEndian))
		fmt.Println("nDims: ", nDims)
		neWeight := make([]int32, 0)
		for i := int32(0); i < nDims; i++ {
			neWeight = append(neWeight, int32(common.ReadInt32FromBytes(modelBytes, &index, ReadFromBigEndian)))
		}
		fmt.Println("neWeight: ", neWeight)
		// FC1 dimensions taken from file, eg. 768x500
		model.hparams.nInput = neWeight[0]
		model.hparams.nHidden = neWeight[1]

		if ReadFromBigEndian {
			fc1WeightDataSize := model.hparams.nInput * model.hparams.nHidden
			fc1WeightData := common.DecodeFloat32List(modelBytes[index : index+4*int(fc1WeightDataSize)])
			index += 4 * int(fc1WeightDataSize)
			model.fc1Weight = ml.NewTensor2DWithData(nil, ml.TYPE_F32, uint32(model.hparams.nInput), uint32(model.hparams.nHidden), fc1WeightData)
		} else {
			model.fc1Weight = ml.NewTensor2D(nil, ml.TYPE_F32, uint32(model.hparams.nInput), uint32(model.hparams.nHidden))
			fmt.Println("len(model.fc1Weight.Data): ", len(model.fc1Weight.Data))
			for i := 0; i < len(model.fc1Weight.Data); i++ {
				model.fc1Weight.Data[i] = common.ReadFP32FromBytes(modelBytes, &index, ReadFromBigEndian)
				if i%10000 == 0 {
					fmt.Println("loading fc1Weight: ", i)
				}
			}
		}

		fmt.Println("index: ", index)

		neBias := make([]int32, 0)
		for i := 0; i < int(nDims); i++ {
			neBias = append(neBias, int32(common.ReadInt32FromBytes(modelBytes, &index, ReadFromBigEndian)))
		}

		if ReadFromBigEndian {
			fc1BiasDataSize := int(model.hparams.nHidden)
			fc1BiasData := common.DecodeFloat32List(modelBytes[index : index+4*fc1BiasDataSize])
			index += 4 * fc1BiasDataSize
			model.fc1Bias = ml.NewTensor1DWithData(nil, ml.TYPE_F32, uint32(model.hparams.nHidden), fc1BiasData)
		} else {
			model.fc1Bias = ml.NewTensor1D(nil, ml.TYPE_F32, uint32(model.hparams.nHidden))
			fmt.Println("len(model.fc1Bias.Data): ", len(model.fc1Bias.Data))
			for i := 0; i < len(model.fc1Bias.Data); i++ {
				model.fc1Bias.Data[i] = common.ReadFP32FromBytes(modelBytes, &index, ReadFromBigEndian)
				if i%10000 == 0 {
					fmt.Println("loading fc1Bias: ", i)
				}
			}
		}

	}

	// Read Fc2 layer 2
	{
		fmt.Println("reading fc2")
		nDims := int32(common.ReadInt32FromBytes(modelBytes, &index, ReadFromBigEndian))
		neWeight := make([]int32, 0)
		for i := 0; i < int(nDims); i++ {
			neWeight = append(neWeight, int32(common.ReadInt32FromBytes(modelBytes, &index, ReadFromBigEndian)))
		}

		// FC1 dimensions taken from file, eg. 10x500
		model.hparams.nClasses = neWeight[1]

		if ReadFromBigEndian {
			fc2WeightDataSize := int(model.hparams.nHidden * model.hparams.nClasses)
			fc2WeightData := common.DecodeFloat32List(modelBytes[index : index+4*fc2WeightDataSize])
			index += 4 * fc2WeightDataSize
			model.fc2Weight = ml.NewTensor2DWithData(nil, ml.TYPE_F32, uint32(model.hparams.nHidden), uint32(model.hparams.nClasses), fc2WeightData)
		} else {
			model.fc2Weight = ml.NewTensor2D(nil, ml.TYPE_F32, uint32(model.hparams.nHidden), uint32(model.hparams.nClasses))
			for i := 0; i < len(model.fc2Weight.Data); i++ {
				model.fc2Weight.Data[i] = common.ReadFP32FromBytes(modelBytes, &index, ReadFromBigEndian)
			}
		}

		neBias := make([]int32, 0)
		for i := 0; i < int(nDims); i++ {
			neBias = append(neBias, int32(common.ReadInt32FromBytes(modelBytes, &index, ReadFromBigEndian)))
		}

		if ReadFromBigEndian {
			fc2BiasDataSize := int(model.hparams.nClasses)
			fc2BiasData := common.DecodeFloat32List(modelBytes[index : index+4*fc2BiasDataSize])
			index += 4 * fc2BiasDataSize
			model.fc2Bias = ml.NewTensor1DWithData(nil, ml.TYPE_F32, uint32(model.hparams.nClasses), fc2BiasData)
		} else {
			model.fc2Bias = ml.NewTensor1D(nil, ml.TYPE_F32, uint32(model.hparams.nClasses))
			for i := 0; i < len(model.fc2Bias.Data); i++ {
				model.fc2Bias.Data[i] = common.ReadFP32FromBytes(modelBytes, &index, ReadFromBigEndian)
			}
		}

		ml.PrintTensor(model.fc2Bias, "model.fc2Bias")
	}

	fmt.Println("current index: ", index)

	return nil
}

// input is 784 bytes
func mipsInputProcess() []float32 {
	fmt.Println("start MipsInputProcess")
	buf := common.ReadBytes(common.INPUT_ADDR, ReadFromBigEndian)
	fmt.Println("buf len: ", len(buf))
	digits := make([]float32, InputSize*InputSize)

	// render the digit in ASCII
	var c string
	for row := 0; row < InputSize; row++ {
		for col := 0; col < InputSize; col++ {
			// fmt.Printf("(%d, %d), value: %.18f\n", row, col, float32(buf[row*InputSize+col]))
			digits[row*InputSize+col] = float32(buf[row*InputSize+col]) / 255
			if buf[row*InputSize+col] > 230 {
				c += "*"
			} else {
				c += "_"
			}
		}
		c += "\n"
	}
	fmt.Println(c)

	return digits
}

func mipsMnistEval(model *mnistModel, digit []float32) int {
	fmt.Println("start MIPSMnistEval")
	ctx0 := &ml.Context{}
	graph := ml.Graph{ThreadsCount: 1}

	input := ml.NewTensor1D(ctx0, ml.TYPE_F32, uint32(model.hparams.nInput))
	copy(input.Data, digit)

	// fc1 MLP = Ax + b
	fc1 := ml.Add(ctx0, ml.MulMat(ctx0, model.fc1Weight, input), model.fc1Bias)
	fc2 := ml.Add(ctx0, ml.MulMat(ctx0, model.fc2Weight, ml.Relu(ctx0, fc1)), model.fc2Bias)

	// final := fc2
	// softmax
	final := ml.SoftMax(ctx0, fc2)

	// run the computation
	ml.BuildForwardExpand(&graph, final)
	ml.GraphCompute(ctx0, &graph)

	ml.PrintTensor(final, "final tensor")

	maxIndex := 0
	for i := 0; i < 10; i++ {
		if final.Data[i] > final.Data[maxIndex] {
			maxIndex = i
		}
	}
	return maxIndex
}

func mipsStoreInMemory(ret int) {
	retBytes := common.IntToBytes(ret, OutputToBigEndian)
	common.Output(retBytes, OutputToBigEndian)
}

// MIPS_MNIST function
func MIPS_MNIST() {
	// func MipsMnist() {
	fmt.Println("Start MIPS MNIST")
	input := mipsInputProcess()
	model := new(mnistModel)
	err := mipsMnistModelLoad(model)
	if err != nil {
		fmt.Println(err)
		common.Halt()
	}
	ret := mipsMnistEval(model, input)
	fmt.Println("Predicted digit is ", ret)
	mipsStoreInMemory(ret)
}
