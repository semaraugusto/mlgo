package test

import (
	// "errors"
	// "math"
	// "errors"

	"fmt"
	"io"
	"math"
	"reflect"
	"unsafe"

	// "mlgo/common"
	"os"
	"testing"

	"mlgo/ml"

	"github.com/x448/float16"
)

type MatMulHParams struct {
	nInput   int32
	nClasses int32
	loaded   bool
}

type MatMulModel struct {
	hparams MatMulHParams

	weight *ml.Tensor
	bias   *ml.Tensor
}

type MatMulInputs struct {
	model  *MatMulModel
	input  *ml.Tensor
	output *ml.Tensor
}

func (model *MatMulModel) loadHParams(nIn, nOut int) {
	model.hparams.nInput = int32(nIn)
	model.hparams.nClasses = int32(nOut)
	model.hparams.loaded = true
}

func TestMatMulLoadHParams(t *testing.T) {
	ml.SINGLE_THREAD = true
	model := new(MatMulModel)
	const nIn = 25
	const nOut = 10
	model.loadHParams(nIn, nOut)
	if model.hparams.nInput != int32(nIn) {
		t.Fatalf("Expected %d, got %d", nIn, model.hparams.nInput)
	}
	if model.hparams.nClasses != int32(nOut) {
		t.Fatalf("Expected %d, got %d", nOut, model.hparams.nClasses)
	}
	if model.hparams.loaded != true {
		t.Fatalf("Expected %t, got %t", true, model.hparams.loaded)
	}
}

// NB! INT = 32 bits
func readUInt(file *os.File) uint32 {
	buf := make([]byte, 4)
	if count, err := file.Read(buf); err != nil || count != 4 {
		return 0
	}
	return uint32(buf[3])<<24 | uint32(buf[2])<<16 | uint32(buf[1])<<8 | uint32(buf[0])
}

// NB! INT = 32 bits
func readInt(file *os.File) uint32 {
	buf := make([]byte, 4)
	if count, err := file.Read(buf); err != nil || count != 4 {
		return 0
	}
	return uint32(buf[3])<<24 | uint32(buf[2])<<16 | uint32(buf[1])<<8 | uint32(buf[0])
}

func readString(file *os.File, len uint32) string {
	buf := make([]byte, len)
	if count, err := file.Read(buf); err != nil || count != int(len) {
		return ""
	}
	return string(buf)
}

func safeReadFP16ToFP32(file *os.File) (float32, error) {
	buf := make([]byte, 2)
	if count, err := file.Read(buf); err != nil || count != 2 {
		return 0.0, err
	}
	bits := uint16(buf[1])<<8 | uint16(buf[0])
	f16 := float16.Frombits(bits)
	return f16.Float32(), nil
}

func readFP16ToFP32(file *os.File) float32 {
	buf := make([]byte, 2)
	if count, err := file.Read(buf); err != nil || count != 2 {
		return 0.0
	}
	bits := uint16(buf[1])<<8 | uint16(buf[0])
	f16 := float16.Frombits(bits)
	return f16.Float32()
}

func readFP32(file *os.File) float32 {
	buf := make([]byte, 4)
	if count, err := file.Read(buf); err != nil || count != 4 {
		return 0.0
	}
	bits := uint32(buf[3])<<24 | uint32(buf[2])<<16 | uint32(buf[1])<<8 | uint32(buf[0])
	return math.Float32frombits(bits)
}

func (model *MatMulModel) loadTensors(filePath string) (*ml.Tensor, error) {
	if model.hparams.loaded != true {
		return fmt.Errorf("HParams not loaded")
	}

	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	numDimensions := readInt(file)
	fmt.Printf("Num Dimensions: %d\n", numDimensions)

	nameLen := readInt(file)
	fmt.Printf("nameLen: %d\n", nameLen)

	tensorType := readInt(file)
	fmt.Printf("tensorType: %d\n", tensorType)

	ne0 := readInt(file)
	fmt.Printf("ne0: %d\n", ne0)

	ne1 := readInt(file)
	fmt.Printf("ne1: %d\n", ne1)
	tensorSize := ne0 * ne1

	name := readString(file, nameLen)
	fmt.Printf("name: %s\n", name)

	alignment := int64(32)
	offset, _ := file.Seek(0, io.SeekCurrent)
	for ; offset%alignment != 0; offset++ {
	}
	file.Seek(offset, io.SeekStart)

	var fake []byte
	tensor := ml.NewTensor2D(nil, ml.TYPE_F32, ne0, ne1)

	fakeHeader := (*reflect.SliceHeader)(unsafe.Pointer(&fake))
	// NB! unsafe.Pointer(tensor.Data) for *Data VS unsafe.Pointer(&tensor.Data) for Data
	dataHeader := (*reflect.SliceHeader)(unsafe.Pointer(&tensor.Data))

	fakeHeader.Data = dataHeader.Data
	fakeHeader.Len = int(tensorSize * 4)
	fakeHeader.Cap = int(tensorSize * 4)

	// fmt.Printf("\n== FAKE []BYTE LEN = %d", len(fake))
	if count, err := io.ReadFull(file, fake); err != nil || count != int(tensorSize*4) {
		fmt.Printf("\n[ERROR] Failed to read BIG FP32 chunk from model!")
		fmt.Printf("\n[ERROR] COUNT = %d | ERR = %s", count, err.Error())
		os.Exit(1)
	}
	fmt.Printf("\n== FAKE []BYTE LEN = %d", len(fake))
	fmt.Printf("\n== TENSOR []BYTE LEN = %d", len(tensor.Data))
	fmt.Printf("\n== TENSOR.DATA[0] = %.10f", tensor.Data[0])
	fmt.Printf("\n== TENSOR.DATA[1] = %.10f", tensor.Data[1])
	return tensor, nil
}

func TestLoadBias1(t *testing.T) {
	ml.SINGLE_THREAD = true
	model := new(MatMulModel)
	// weightsPath := "models/l2_bias.txt"
	tensorPath := "../../test_tensors/q_cur_src0.tensor"
	const nIn = 25
	const nOut = 10
	model.loadHParams(nIn, nOut)
	if tensor, err := model.loadTensors(tensorPath); err != nil {
		t.Fatalf(err.Error())
		return
	}
	expected := [nOut]float32{
		-2.848905324935913086e-01,
		2.043375670909881592e-01,
		1.892944127321243286e-01,
		-1.301911473274230957e-01,
		1.793343722820281982e-01,
		1.507467478513717651e-01,
		-2.965211570262908936e-01,
		1.678309142589569092e-01,
		-2.104278355836868286e-01,
		1.698205918073654175e-01,
	}
	if model.bias.Dims != 1 {
		t.Fatalf("Expected 2 dimensions, got %d", model.bias.Dims)
	}

	if len(model.bias.Data) != int(model.hparams.nClasses) {
		t.Fatalf("Expected 25 values, got %d", len(model.bias.Data))
	}

	for i := 0; i < len(model.bias.Data); i++ {
		if model.bias.Data[i] != expected[i] {
			t.Fatalf("ERROR: Expected: '%f'\nGot: '%f'", expected[i], model.bias.Data[i])
		}
	}
}

// // func (m *MatMulModel) linear_eval(threadCount int, digit []float32) int {
// // func (model *MatMulModel) linearEval(threadCount int, digit []float32) []float32 {
// func (model *MatMulModel) linearEval(threadCount int, digit []float32) ([]float32, []float32) {
// 	fmt.Println("START EVAL")
// 	ctx0 := &ml.Context{}
// 	graph := ml.Graph{ThreadsCount: threadCount}
// 	fmt.Println("m.hparams.nInput: ", model.hparams.nInput)
//
// 	input := ml.NewTensor1D(ctx0, ml.TYPE_F32, uint32(model.hparams.nInput))
// 	copy(input.Data, digit)
//
// 	// fc1 MLP = Ax + b
// 	fmt.Println("BEFORE MULMAT")
// 	fmt.Println("weight: ", model.weight)
// 	fmt.Println("input: ", input)
// 	mulmat := ml.MulMat(ctx0, model.weight, input)
// 	fmt.Println("MULMAT")
// 	fc := ml.Add(ctx0, mulmat, model.bias)
// 	final := ml.SoftMax(ctx0, fc)
// 	fmt.Println("OPS DEFINED")
//
// 	// run the computation
// 	// ml.BuildForwardExpand(&graph, fc)
// 	ml.BuildForwardExpand(&graph, final)
// 	ml.GraphCompute(ctx0, &graph)
//
// 	// ml.PrintTensor(mulmat, "mulmat")
// 	fmt.Println("mulmat", mulmat)
// 	// ml.PrintTensor(fc, "fc")
// 	fmt.Println("fc: ", fc)
// 	// ml.PrintTensor(final, "final tensor")
// 	// fmt.Println("final: ", final)
//
// 	return fc.Data, final.Data
// 	// return fc.Data
// }

// func TestModelLoad(t *testing.T) {
// 	const nIn = 25
// 	const nOut = 10
// 	model := new(MatMulModel)
// 	tInputs := new(MatMulInputs)
// 	model.loadHParams(nIn, nOut)
// 	fmt.Println("!!model: ", model)
// 	tInputs.model = model
// 	biasPath := "models/l2_bias.txt"
// 	weightsPath := "models/l2_weights.txt"
// 	fmt.Println("tInputs.!model: ", tInputs.model)
// 	if err := loadModel(tInputs.model, weightsPath, biasPath); err != nil {
// 		t.Fatalf(err.Error())
// 	}
// 	// if err := tInputs.model.loadWeights(weightsPath); err != nil {
// 	// 	t.Fatalf(err.Error())
// 	// }
// 	// if err := tInputs.model.loadBias(biasPath); err != nil {
// 	// 	t.Fatalf(err.Error())
// 	// 	// return
// 	// }
// }

// func TestModelEval(t *testing.T) {
// 	const nIn = 25
// 	const nOut = 10
// 	expected := []float32{
// 		3.339550495147705078e+00, -1.150749111175537109e+01, 9.716508984565734863e-01, -5.528323650360107422e+00, 2.966210126876831055e+00, 7.380880713462829590e-01, 3.723233222961425781e+00, -5.654765605926513672e+00, 6.697512269020080566e-01, -3.081719398498535156e+00,
// 	}
// 	model := new(MatMulModel)
// 	tInputs := new(MatMulInputs)
// 	model.loadHParams(nIn, nOut)
// 	tInputs.model = model
// 	biasPath := "models/l2_bias.txt"
// 	weightsPath := "models/l2_weights.txt"
// 	if err := tInputs.model.loadWeights(weightsPath); err != nil {
// 		t.Fatalf(err.Error())
// 	}
// 	if err := tInputs.model.loadBias(biasPath); err != nil {
// 		t.Fatalf(err.Error())
// 		// return
// 	}
// 	inputPath := "models/relu_out.txt"
// 	if err := tInputs.loadInput(inputPath); err != nil {
// 		t.Fatalf(err.Error())
// 		return
// 	}
// 	outputPath := "models/l2_out.txt"
// 	if err := tInputs.loadOutput(outputPath); err != nil {
// 		t.Fatalf(err.Error())
// 		return
// 	}
// 	fmt.Println("tInputs.input: ", tInputs.input)
// 	// fmt.Println("tInputs.output: ", tInputs.output)
// 	fmt.Println("tInputs.model: ", tInputs.model)
//
// 	for i := 0; i < len(tInputs.output.Data); i++ {
// 		if tInputs.output.Data[i] != expected[i] {
// 			t.Fatalf("ERROR: Expected: '%f'\nGot: '%f'", expected[i], tInputs.input.Data[i])
// 		}
// 	}
//
// 	// fmt.Println("tInputs.model.weight: ", tInputs.model.weight)
//
// 	fc, pred := tInputs.model.linearEval(1, tInputs.input.Data)
// 	// pred := tInputs.model.linearEval(1, tInputs.input.Data)
// 	// fc := pred
// 	index := 0
// 	maxVal := float32(0)
// 	// for p := range pred {
// 	for i := 0; i < len(pred); i++ {
// 		p := pred[i]
// 		if p > maxVal {
// 			maxVal = p
// 			index = i
// 		}
// 	}
//
// 	fmt.Println("Predicted: ", index)
// 	fmt.Println("expected: ", expected)
// 	fmt.Println("fc   : ", fc)
// 	fmt.Println("logits   : ", pred)
// 	failed := false
// 	sumError := float32(0.0)
// 	countError := 0
// 	for i := 0; i < len(tInputs.output.Data); i++ {
// 		if fc[i] != expected[i] {
// 			failed = true
// 			sumError += fc[i] - expected[i]
// 			countError++
// 		}
// 	}
// 	if failed {
// 		t.Fatalf("ERRORS: difference in %d outputs: mean error %.10f", countError, sumError/float32(countError))
// 	}
// }
