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
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	numDimensions := readInt(file)
	// fmt.Printf("Num Dimensions: %d\n", numDimensions)

	nameLen := readInt(file)
	// fmt.Printf("nameLen: %d\n", nameLen)

	_ = readInt(file)
	// tensorType := readInt(file)
	// fmt.Printf("tensorType: %d\n", tensorType)

	ne := []uint32{}
	tensorSize := uint32(1)
	for i := uint32(0); i < numDimensions; i++ {
		val := readInt(file)
		ne = append(ne, val)
		tensorSize *= val
	}

	// ne1 := readInt(file)
	// fmt.Printf("ne1: %d\n", ne[1])
	// tensorSize := ne[0] * ne[1]

	// _ = readString(file, nameLen)
	name := readString(file, nameLen)
	fmt.Printf("loading tensor named: %s - with shape %v\n", name, ne)

	alignment := int64(32)
	offset, _ := file.Seek(0, io.SeekCurrent)
	for ; offset%alignment != 0; offset++ {
	}
	_, err = file.Seek(offset, io.SeekStart)
	if err != nil {
		return nil, err
	}

	var fake []byte
	var tensor *ml.Tensor
	switch numDimensions {
	case 1:
		tensor = ml.NewTensor1D(nil, ml.TYPE_F32, ne[0])
	case 2:
		tensor = ml.NewTensor2D(nil, ml.TYPE_F32, ne[0], ne[1])
	case 3:
		tensor = ml.NewTensor3D(nil, ml.TYPE_F32, ne[0], ne[1], ne[2])
	case 4:
		tensor = ml.NewTensor4D(nil, ml.TYPE_F32, ne[0], ne[1], ne[2], ne[3])
	default:
		fmt.Printf("\n[ERROR] Unsupported number of dimensions: %d", numDimensions)
		os.Exit(1)
	}

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
	// fmt.Printf("\n== FAKE []BYTE LEN = %d", len(fake))
	// fmt.Printf("\n== TENSOR []BYTE LEN = %d", len(tensor.Data))
	// fmt.Printf("\n== TENSOR.DATA[0] = %.10f", tensor.Data[0])
	// fmt.Printf("\n== TENSOR.DATA[1] = %.10f", tensor.Data[1])
	return tensor, nil
}

func TestLoadTensor(t *testing.T) {
	ml.SINGLE_THREAD = true
	model := new(MatMulModel)
	// weightsPath := "models/l2_bias.txt"
	tensorPath := "../../test_tensors/q_cur_src0.tensor"
	// const nIn = 25
	// const nOut = 10
	// model.loadHParams(nIn, nOut)
	tensor, err := model.loadTensors(tensorPath)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}
	model.weight = tensor
	// fmt.Printf("\n== TENSOR.NE = %+v", tensor.NE)
	// fmt.Printf("\n== TENSOR.DATA[0] = %.10f", tensor.Data[0])
	// fmt.Printf("\n== TENSOR.DATA[1] = %.10f", tensor.Data[1])
	if model.weight.Dims != 2 {
		t.Fatalf("Expected 2 dimensions, got %d", model.weight.Dims)
	}

	if len(model.weight.Data) != int(4096*4096) {
		t.Fatalf("Expected 25 values, got %d", len(model.weight.Data))
	}
}

// func (model *MatMulModel) MatMulEval(threadCount int, src0 *ml.Tensor, src1 *ml.Tensor) []float32 {
func (model *MatMulModel) MatMulEval(threadCount int, src0 *ml.Tensor, src1 *ml.Tensor) *ml.Tensor {
	ctx0 := &ml.Context{}
	graph := ml.Graph{ThreadsCount: threadCount}
	res := ml.MulMat(ctx0, src0, src1)
	ml.BuildForwardExpand(&graph, res)
	ml.GraphCompute(ctx0, &graph)

	return res
}

func TestMatMulSmall(t *testing.T) {
	// t.Skip("Skipping test")
	ml.SINGLE_THREAD = true
	model := new(MatMulModel)
	// weightsPath := "models/l2_bias.txt"
	filePathSrc0 := "../../test_tensors/x0.bin"
	filePathSrc1 := "../../test_tensors/x1.bin"
	filePathRes := "../../test_tensors/y.bin"
	// op is x1 * x0 = y
	B, err := model.loadTensors(filePathSrc0)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}
	A, err := model.loadTensors(filePathSrc1)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}
	C, err := model.loadTensors(filePathRes)
	// const nIn = 25
	// const nOut = 10
	// model.loadHParams(nIn, nOut)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}

	actual := model.MatMulEval(1, A, B)

	if A.Dims != 4 || B.Dims != 4 || C.Dims != 4 {
		t.Fatalf("Expected 4 dimensions tensors read, got %d", A.Dims)
	}

	if actual.Dims != 4 {
		t.Fatalf("Expected 4 dimensions result, got %d", actual.Dims)
	}

	if len(actual.Data) != len(C.Data) {
		t.Fatalf("LENGTH IS DIFFERENT: EXPECTED %d, GOT %d", len(C.Data), len(actual.Data))
	}
	checkOutput := CheckMatMul(actual, A, B)
	checkExpected := CheckMatMul(C, A, B)
	fmt.Printf("actual shape %v\n", actual.NE)
	fmt.Printf("Y shape      %v\n", C.NE)
	fmt.Printf("CHECK_MATMUL_Expected: %v\n", checkExpected)
	if checkOutput != true || checkExpected != true {
		fmt.Printf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
		t.Fatalf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
	}

	// sumError := 0.
	// countError := 0
	// for i := 0; i < 10; i++ {
	// 	if actual.Data[i] != C.Data[i] {
	// 		fmt.Printf("actual %.10f - expected: %.10f = %.10f\n", actual.Data[i], C.Data[i], sumError)
	// 		sumError += math.Abs(float64(actual.Data[i]) - float64(C.Data[i]))
	// 		countError++
	// 	}
	// }
	//
	// // checkOriginal := CheckMatMul(C, A, B)
	// fmt.Printf("SUM_ERROR: %.10f", sumError)
	// t.Fatalf("MEAN_ERROR: %.10f", sumError/float64(countError))
}

func TestMatMulSmall2(t *testing.T) {
	// t.Skip("Skipping test")
	ml.SINGLE_THREAD = true
	model := new(MatMulModel)
	// weightsPath := "models/l2_bias.txt"
	filePathSrc0 := "../../test_tensors/x10.bin"
	filePathSrc1 := "../../test_tensors/x11.bin"
	filePathRes := "../../test_tensors/y1.bin"
	// op is x1 * x0 = y
	B, err := model.loadTensors(filePathSrc0)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}
	A, err := model.loadTensors(filePathSrc1)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}
	C, err := model.loadTensors(filePathRes)
	// const nIn = 25
	// const nOut = 10
	// model.loadHParams(nIn, nOut)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}

	actual := model.MatMulEval(1, A, B)

	if A.Dims != 2 || B.Dims != 2 || C.Dims != 2 {
		t.Fatalf("Expected 4 dimensions tensors read, got %d", A.Dims)
	}

	if actual.Dims != 2 {
		t.Fatalf("Expected 4 dimensions result, got %d", actual.Dims)
	}

	if len(actual.Data) != len(C.Data) {
		t.Fatalf("LENGTH IS DIFFERENT: EXPECTED %d, GOT %d", len(C.Data), len(actual.Data))
	}
	checkOutput := CheckMatMul(actual, A, B)
	checkExpected := CheckMatMul(C, A, B)
	fmt.Printf("actual shape %v\n", actual.NE)
	fmt.Printf("Y shape      %v\n", C.NE)
	fmt.Printf("CHECK_MATMUL_Expected: %v\n", checkExpected)
	if checkOutput != true || checkExpected != true {
		fmt.Printf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
		t.Fatalf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
	}

	// sumError := 0.
	// countError := 0
	// for i := 0; i < 10; i++ {
	// 	if actual.Data[i] != C.Data[i] {
	// 		fmt.Printf("actual %.10f - expected: %.10f = %.10f\n", actual.Data[i], C.Data[i], sumError)
	// 		sumError += math.Abs(float64(actual.Data[i]) - float64(C.Data[i]))
	// 		countError++
	// 	}
	// }
	//
	// // checkOriginal := CheckMatMul(C, A, B)
	// fmt.Printf("SUM_ERROR: %.10f", sumError)
	// t.Fatalf("MEAN_ERROR: %.10f", sumError/float64(countError))
}

func TestMatMulSmall3(t *testing.T) {
	ml.SINGLE_THREAD = true
	model := new(MatMulModel)
	// weightsPath := "models/l2_bias.txt"
	filePathSrc0 := "../../test_tensors/x20.bin"
	filePathSrc1 := "../../test_tensors/x21.bin"
	filePathRes := "../../test_tensors/y2.bin"
	// op is x1 * x0 = y
	B, err := model.loadTensors(filePathSrc0)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}
	A, err := model.loadTensors(filePathSrc1)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}
	C, err := model.loadTensors(filePathRes)
	// const nIn = 25
	// const nOut = 10
	// model.loadHParams(nIn, nOut)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}

	actual := model.MatMulEval(1, A, B)

	if A.Dims != 1 || B.Dims != 2 || C.Dims != 2 {
		t.Fatalf("wrong dimensions tensors read, got %d", A.Dims)
	}

	if actual.Dims != 1 {
		t.Fatalf("Expected 4 dimensions result, got %d", actual.Dims)
	}

	if len(actual.Data) != len(C.Data) {
		t.Fatalf("LENGTH IS DIFFERENT: EXPECTED %d, GOT %d", len(C.Data), len(actual.Data))
	}
	checkOutput := CheckMatMul(actual, A, B)
	checkExpected := CheckMatMul(C, A, B)
	fmt.Printf("actual shape %v\n", actual.NE)
	fmt.Printf("Y shape      %v\n", C.NE)
	fmt.Printf("CHECK_MATMUL_Expected: %v\n", checkExpected)
	if checkOutput != true || checkExpected != true {
		fmt.Printf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
		t.Fatalf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
	}
}

func TestMatMulSmall4(t *testing.T) {
	ml.SINGLE_THREAD = true
	model := new(MatMulModel)
	// weightsPath := "models/l2_bias.txt"
	filePathSrc0 := "../../test_tensors/x30.bin"
	filePathSrc1 := "../../test_tensors/x31.bin"
	filePathRes := "../../test_tensors/y3.bin"
	// op is x1 * x0 = y
	B, err := model.loadTensors(filePathSrc0)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}
	A, err := model.loadTensors(filePathSrc1)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}
	C, err := model.loadTensors(filePathRes)
	// const nIn = 25
	// const nOut = 10
	// model.loadHParams(nIn, nOut)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}

	actual := model.MatMulEval(1, A, B)

	if A.Dims != 4 || B.Dims != 4 || C.Dims != 4 {
		t.Fatalf("wrong dimensions tensors read, got %d", A.Dims)
	}

	if actual.Dims != 4 {
		t.Fatalf("Expected 4 dimensions result, got %d", actual.Dims)
	}

	if len(actual.Data) != len(C.Data) {
		t.Fatalf("LENGTH IS DIFFERENT: EXPECTED %d, GOT %d", len(C.Data), len(actual.Data))
	}
	checkOutput := CheckMatMul(actual, A, B)
	checkExpected := CheckMatMul(C, A, B)
	fmt.Printf("actual shape %v\n", actual.NE)
	fmt.Printf("Y shape      %v\n", C.NE)
	fmt.Printf("CHECK_MATMUL_Expected: %v\n", checkExpected)
	if checkOutput != true || checkExpected != true {
		fmt.Printf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
		t.Fatalf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
	}
}

func TestMatMulSmall4Transposed(t *testing.T) {
	ml.SINGLE_THREAD = true
	model := new(MatMulModel)
	// weightsPath := "models/l2_bias.txt"
	filePathSrc0 := "../../test_tensors/x30_transposed.bin"
	filePathSrc1 := "../../test_tensors/x31_transposed.bin"
	filePathRes := "../../test_tensors/y3_transposed.bin"
	// op is x1 * x0 = y
	B, err := model.loadTensors(filePathSrc0)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}
	A, err := model.loadTensors(filePathSrc1)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}
	C, err := model.loadTensors(filePathRes)
	// const nIn = 25
	// const nOut = 10
	// model.loadHParams(nIn, nOut)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}

	actual := model.MatMulEval(1, A, B)

	if A.Dims != 4 || B.Dims != 4 || C.Dims != 4 {
		t.Fatalf("wrong dimensions tensors read, got %d", A.Dims)
	}

	if actual.Dims != 4 {
		t.Fatalf("Expected 4 dimensions result, got %d", actual.Dims)
	}

	if len(actual.Data) != len(C.Data) {
		t.Fatalf("LENGTH IS DIFFERENT: EXPECTED %d, GOT %d", len(C.Data), len(actual.Data))
	}
	checkOutput := CheckMatMul(actual, A, B)
	checkExpected := CheckMatMul(C, A, B)
	fmt.Printf("actual shape %v\n", actual.NE)
	fmt.Printf("Y shape      %v\n", C.NE)
	fmt.Printf("CHECK_MATMUL_Expected: %v\n", checkExpected)
	if checkOutput != true || checkExpected != true {
		fmt.Printf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
		t.Fatalf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
	}
}

func TestMatMul(t *testing.T) {
	ml.SINGLE_THREAD = true
	model := new(MatMulModel)
	// weightsPath := "models/l2_bias.txt"
	// filePathSrc0 := "../../test_tensors/q_cur_src0.tensor"
	// filePathSrc1 := "../../test_tensors/q_cur_src1.tensor"
	// filePathRes := "../../test_tensors/q_cur_result.tensor"
	// filePathSrc0 := "../../test_tensors/k_cur_src0.tensor"
	// filePathSrc1 := "../../test_tensors/k_cur_src1.tensor"
	// filePathRes := "../../test_tensors/k_cur_result.tensor"
	filePathSrc0 := "../../test_tensors/output_src0.tensor"
	filePathSrc1 := "../../test_tensors/output_src1.tensor"
	filePathRes := "../../test_tensors/output_result.tensor"
	A, err := model.loadTensors(filePathSrc0)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}
	B, err := model.loadTensors(filePathSrc1)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}
	C, err := model.loadTensors(filePathRes)
	// const nIn = 25
	// const nOut = 10
	// model.loadHParams(nIn, nOut)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}

	fmt.Printf("\n== A.NE = %+v", A.NE)
	fmt.Printf("\n== B.NE = %+v", B.NE)
	fmt.Printf("\n== C.NE = %+v", C.NE)
	fmt.Printf("\n== A.DATA[0] = %.10f", A.Data[0])
	fmt.Printf("\n== A.DATA[1] = %.10f", A.Data[1])

	fmt.Printf("\n== B.DATA[0] = %.10f", B.Data[0])
	fmt.Printf("\n== B.DATA[1] = %.10f", B.Data[1])

	fmt.Printf("\n== C.DATA[0] = %.10f", C.Data[0])
	fmt.Printf("\n== C.DATA[1] = %.10f\n", C.Data[1])
	actual := model.MatMulEval(1, A, B)

	if A.Dims != 2 || B.Dims != 2 || C.Dims != 2 {
		t.Fatalf("Expected 2 dimensions, got %d", A.Dims)
	}

	if actual.Dims != 2 {
		t.Fatalf("Expected 2 dimensions, got %d", actual.Dims)
	}

	if len(actual.Data) != len(C.Data) {
		t.Fatalf("LENGTH IS DIFFERENT: EXPECTED %d, GOT %d", len(C.Data), len(actual.Data))
	}

	// sumError := 0.
	// countError := 0
	// for i := 0; i < 10; i++ {
	// 	if actual.Data[i] != C.Data[i] {
	// 		fmt.Printf("actual %.10f - expected: %.10f = %.10f\n", actual.Data[i], C.Data[i], sumError)
	// 		sumError += math.Abs(float64(actual.Data[i]) - float64(C.Data[i]))
	// 		countError++
	// 	}
	// }

	// checkOriginal := CheckMatMul(C, A, B)
	checkOutput := CheckMatMul(actual, A, B)
	checkExpected := CheckMatMul(C, A, B)

	// for i := 0; i < 32; i++ {
	// 	for j := 0; j < 2; j++ {
	// 		fmt.Printf("actual: %3.10f ", MatGet(actual, uint32(i), uint32(j), 0, 0))
	// 	}
	// 	fmt.Printf("\n")
	// }
	// fmt.Printf("\n")

	for i := 0; i < 32; i++ {
		for j := 0; j < 2; j++ {
			fmt.Printf("A: %3.10f ", MatGet(A, uint32(i), uint32(j), 0, 0))
		}
		fmt.Printf("\n")
	}
	fmt.Printf("\n")

	if checkOutput != true || checkExpected != true {
		fmt.Printf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
		t.Fatalf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
	}
	fmt.Printf("CHECK_MATMUL_OUTPUT: %v\n", checkOutput)
}

// func TestMatMul2(t *testing.T) {
// 	ml.SINGLE_THREAD = true
// 	model := new(MatMulModel)
// 	// weightsPath := "models/l2_bias.txt"
// 	filePathSrc0 := "../../test_tensors/k_cur_src0.tensor"
// 	filePathSrc1 := "../../test_tensors/k_cur_src1.tensor"
// 	filePathRes := "../../test_tensors/k_cur_result.tensor"
// 	A, err := model.loadTensors(filePathSrc0)
// 	if err != nil {
// 		t.Fatalf(err.Error())
// 		return
// 	}
// 	B, err := model.loadTensors(filePathSrc1)
// 	if err != nil {
// 		t.Fatalf(err.Error())
// 		return
// 	}
// 	C, err := model.loadTensors(filePathRes)
// 	// const nIn = 25
// 	// const nOut = 10
// 	// model.loadHParams(nIn, nOut)
// 	if err != nil {
// 		t.Fatalf(err.Error())
// 		return
// 	}
//
// 	// fmt.Printf("\n== A.NE = %+v", A.NE)
// 	// fmt.Printf("\n== B.NE = %+v", B.NE)
// 	// fmt.Printf("\n== C.NE = %+v", C.NE)
// 	// fmt.Printf("\n== A.DATA[0] = %.10f", A.Data[0])
// 	// fmt.Printf("\n== A.DATA[1] = %.10f", A.Data[1])
// 	//
// 	// fmt.Printf("\n== B.DATA[0] = %.10f", B.Data[0])
// 	// fmt.Printf("\n== B.DATA[1] = %.10f", B.Data[1])
// 	//
// 	// fmt.Printf("\n== C.DATA[0] = %.10f", C.Data[0])
// 	// fmt.Printf("\n== C.DATA[1] = %.10f\n", C.Data[1])
// 	actual := model.MatMulEval(1, A, B)
//
// 	if A.Dims != 2 || B.Dims != 2 || C.Dims != 2 {
// 		t.Fatalf("Expected 2 dimensions, got %d", A.Dims)
// 	}
//
// 	if actual.Dims != 2 {
// 		t.Fatalf("Expected 2 dimensions, got %d", actual.Dims)
// 	}
//
// 	if len(actual.Data) != len(C.Data) {
// 		t.Fatalf("WRONG RESULT")
// 	}
//
// 	sumError := 0.
// 	countError := 0
// 	for i := 0; i < 10; i++ {
// 		if actual.Data[i] != C.Data[i] {
// 			// fmt.Printf("actual %.10f - expected: %.10f = %.10f\n", actual.Data[i], C.Data[i], sumError)
// 			sumError += math.Abs(float64(actual.Data[i]) - float64(C.Data[i]))
// 			countError++
// 		}
// 	}
// 	// t.Fatalf("SUM_ERROR: %.10f", sumError)
// 	t.Fatalf("MEAN_ERROR: %.10f", sumError/float64(countError))
// }

func MatGet(t *ml.Tensor, i0, i1, i2, i3 uint32) float32 {
	nb0 := t.NB[0] / 4
	nb1 := t.NB[1] / 4
	nb2 := t.NB[2] / 4
	nb3 := t.NB[3] / 4

	return t.Data[i0*nb0+i1*nb1+i2*nb2+i3*nb3]
}

func NaiveMatMul(y, x0, x1 *ml.Tensor) bool {
	n00 := x0.NE[0]
	n10 := x0.NE[1]
	n20 := x0.NE[2]
	n30 := x0.NE[3]

	n01 := x1.NE[0]
	n11 := x1.NE[1]
	n21 := x1.NE[2]
	n31 := x1.NE[3]

	n02 := y.NE[0]
	n12 := y.NE[1]
	n22 := y.NE[2]
	n32 := y.NE[3]

	fmt.Printf("x0: [%d , %d , %d , %d ]\n", n00, n10, n20, n30)

	for j := uint32(0); j < n10; j++ {
		for i := uint32(0); i < n00; i++ {
			fmt.Printf("%6.3f ", MatGet(x0, i, j, 0, 0))
		}
		fmt.Printf("\n")
	}
	fmt.Printf("\n")
	fmt.Printf("x1: [%d , %d , %d , %d ]\n", n01, n11, n21, n31)
	for j := uint32(0); j < n11; j++ {
		for i := uint32(0); i < n01; i++ {
			fmt.Printf("%6.3f ", MatGet(x1, i, j, 0, 0))
		}
		fmt.Printf("\n")
	}
	fmt.Printf("\n")

	fmt.Printf("y: [%d , %d , %d , %d ]\n", n02, n12, n22, n32)
	for j := uint32(0); j < n12; j++ {
		for i := uint32(0); i < n02; i++ {
			fmt.Printf("%6.3f ", MatGet(y, i, j, 0, 0))
		}
		fmt.Printf("\n")
	}
	fmt.Printf("loop0 limits: [%d , loop1 limits: %d , loop2 limits: %d , loop3 limits: %d kloop limits: %d]\n", n32, n22, n12, n02, n00)

	for i3 := uint32(0); i3 < n32; i3++ {
		for i2 := uint32(0); i2 < n22; i2++ {
			for i1 := uint32(0); i1 < n12; i1++ {
				for i0 := uint32(0); i0 < n02; i0++ {
					sum := float32(0.0)
					for k := uint32(0); k < n00; k++ {
						sum += MatGet(x0, k, i0, i2, i3) * MatGet(x1, k, i1, i2, i3)
					}
					if math.Abs(float64(sum)-float64(MatGet(y, i0, i1, i2, i3))) > 1e-5 {
						fmt.Printf("error: i0=%d, i1=%d, i2=%d, i3=%d, sum=%f, y=%f\n", i0, i1, i2, i3, sum, MatGet(y, i0, i1, i2, i3))
						// os.Exit(1)
						return false
					}
				}
			}
		}
	}

	return true
}

func CheckMatMul(y, x0, x1 *ml.Tensor) bool {
	n00 := x0.NE[0]
	// n10 := x0.NE[1]
	// n20 := x0.NE[2]
	// n30 := x0.NE[3]
	//
	// n01 := x1.NE[0]
	// n11 := x1.NE[1]
	// n21 := x1.NE[2]
	// n31 := x1.NE[3]

	n02 := y.NE[0]
	n12 := y.NE[1]
	n22 := y.NE[2]
	n32 := y.NE[3]

	// fmt.Printf("x0: [%d , %d , %d , %d ]\n", n00, n10, n20, n30)

	// for j := uint32(0); j < n10; j++ {
	// 	for i := uint32(0); i < n00; i++ {
	// 		fmt.Printf("%6.3f ", MatGet(x0, i, j, 0, 0))
	// 	}
	// 	fmt.Printf("\n")
	// }
	// fmt.Printf("\n")
	// fmt.Printf("x1: [%d , %d , %d , %d ]\n", n01, n11, n21, n31)
	// for j := uint32(0); j < n11; j++ {
	// 	for i := uint32(0); i < n01; i++ {
	// 		fmt.Printf("%6.3f ", MatGet(x1, i, j, 0, 0))
	// 	}
	// 	fmt.Printf("\n")
	// }
	// fmt.Printf("\n")

	// fmt.Printf("y: [%d , %d , %d , %d ]\n", n02, n12, n22, n32)
	// for j := uint32(0); j < n12; j++ {
	// 	for i := uint32(0); i < n02; i++ {
	// 		fmt.Printf("%6.3f ", MatGet(y, i, j, 0, 0))
	// 	}
	// 	fmt.Printf("\n")
	// }

	// fmt.Printf("loop0 limits: [%d , loop1 limits: %d , loop2 limits: %d , loop3 limits: %d kloop limits: %d]\n", n32, n22, n12, n02, n00)
	success := true
	for i3 := uint32(0); i3 < n32; i3++ {
		for i2 := uint32(0); i2 < n22; i2++ {
			for i1 := uint32(0); i1 < n12; i1++ {
				for i0 := uint32(0); i0 < n02; i0++ {
					sum := float32(0.0)
					for k := uint32(0); k < n00; k++ {
						sum += MatGet(x0, k, i0, i2, i3) * MatGet(x1, k, i1, i2, i3)
					}
					if math.Abs(float64(sum)-float64(MatGet(y, i0, i1, i2, i3))) > 1e-4 {
						fmt.Printf("error: i0=%d, i1=%d, i2=%d, i3=%d, sum=%f, y=%f\n", i0, i1, i2, i3, sum, MatGet(y, i0, i1, i2, i3))
						// os.Exit(1)
						// success = false
						return false
					}
				}
			}
		}
	}

	return success
}

// static void ggml_compute_forward_mul_mat(
//         const struct ggml_compute_params * params,
//               struct ggml_tensor * dst) {
//
//     const struct ggml_tensor * src0 = dst->src[0];
//     const struct ggml_tensor * src1 = dst->src[1];
//
//     int64_t t0 = ggml_perf_time_us();
//     UNUSED(t0);
//
//     GGML_TENSOR_BINARY_OP_LOCALS
//
//     // ne0x -> src0
//     // ne1x -> src1
//     // nex -> dst
//
//     const int ith = params->ith;
//     const int nth = params->nth;
//
//     const enum ggml_type type = src0->type;
//
//     const bool src1_cont = ggml_is_contiguous(src1);
//
//     ggml_vec_dot_t    const vec_dot               = type_traits[type].vec_dot;
//     enum ggml_type    const vec_dot_type          = type_traits[type].vec_dot_type;
//     ggml_from_float_t const from_float_to_vec_dot = type_traits[vec_dot_type].from_float;
//     int64_t           const vec_dot_num_rows      = type_traits[type].nrows;
//
//     GGML_ASSERT(ne0 == ne01);
//     GGML_ASSERT(ne1 == ne11);
//     GGML_ASSERT(ne2 == ne12);
//     GGML_ASSERT(ne3 == ne13);
//
//     // we don't support permuted src0 or src1
//     GGML_ASSERT(nb00 == ggml_type_size(type));
//     GGML_ASSERT(nb10 == ggml_type_size(src1->type));
//
//     // dst cannot be transposed or permuted
//     GGML_ASSERT(nb0 == sizeof(float));
//     GGML_ASSERT(nb0 <= nb1);
//     GGML_ASSERT(nb1 <= nb2);
//     GGML_ASSERT(nb2 <= nb3);
//
//     // broadcast factors
//     const int64_t r2 = ne12/ne02;
//     const int64_t r3 = ne13/ne03;
//
//     // nb01 >= nb00 - src0 is not transposed
//     //   compute by src0 rows
//
// #if defined(GGML_USE_CLBLAST)
//     if (ggml_cl_can_mul_mat(src0, src1, dst)) {
//         if (params->ith == 0 && params->type == GGML_TASK_TYPE_COMPUTE) {
//             ggml_cl_mul_mat(src0, src1, dst, params->wdata, params->wsize);
//         }
//         return;
//     }
// #endif
//
// #if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS)
//     if (ggml_compute_forward_mul_mat_use_blas(dst)) {
//         const int64_t ne_plane      = ne01*ne00;
//         const size_t  desired_wsize = ne13*ne12*ne_plane*sizeof(float);
//         UNUSED(desired_wsize);
//
//         if (params->type == GGML_TASK_TYPE_INIT) {
//             if (type != GGML_TYPE_F32) {
//                 assert(params->wsize >= desired_wsize);
//                 // parallelize by src0 rows
//                 for (int64_t i13 = 0; i13 < ne13; i13++) {
//                     for (int64_t i12 = 0; i12 < ne12; i12++) {
//                         // broadcast src0 into src1 across 2nd,3rd dimension
//                         const int64_t i03 = i13/r3;
//                         const int64_t i02 = i12/r2;
//
//                         const void           *       x        = (char *)  src0->data    + i02*nb02          + i03*nb03;
//                               float          * const wdata    = (float *) params->wdata + i13*ne12*ne_plane + i12*ne_plane;
//                               ggml_to_float_t  const to_float = type_traits[type].to_float;
//
//                         for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
//                             to_float((const char *) x + i01*nb01, wdata + i01*ne00, ne00);
//                         }
//                     }
//                 }
//             }
//             return;
//         }
//
//         if (params->type == GGML_TASK_TYPE_FINALIZE) {
//             return;
//         }
//
//         // perform sgemm, parallelization controlled by blas lib
//         if (ith != 0) {
//             return;
//         }
//
//         //const int64_t tgemm0 = ggml_perf_time_us();
//         for (int64_t i13 = 0; i13 < ne13; i13++) {
//             for (int64_t i12 = 0; i12 < ne12; i12++) {
//                 const int64_t i03 = i13/r3;
//                 const int64_t i02 = i12/r2;
//
//                 const void  * x = (char *)            src0->data + i02*nb02 + i03*nb03;
//                 const float * y = (float *) ((char *) src1->data + i12*nb12 + i13*nb13);
//                       float * d = (float *) ((char *)  dst->data + i12*nb2  + i13*nb3);
//
//                 if (type != GGML_TYPE_F32) {
//                     x = (float *) params->wdata + i13*ne12*ne_plane + i12*ne_plane;
//                 }
//
//                 cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
//                           ne1, ne01, ne10,
//                          1.0f,    y, ne10,
//                                   x, ne00,
//                          0.0f,    d, ne01);
//             }
//         }
//         //printf("cblas_sgemm = %.3f ms, %lld flops\n", (ggml_perf_time_us() - tgemm0)/1000.0, ne13*ne12*ne1*ne01*ne10*2);
//
//         //printf("CBLAS = %f ms, %d x %d x %d x %d\n", (ggml_perf_time_us() - t0)/1000.0, ne0, ne1, ne2, ne3);
//
//         return;
//     }
// #endif
//
//     if (params->type == GGML_TASK_TYPE_INIT) {
//         if (ith != 0) {
//             return;
//         }
//         if (src1->type != vec_dot_type) {
//             char * wdata = params->wdata;
//             const size_t row_size = ggml_row_size(vec_dot_type, ne10);
//
//             assert(params->wsize >= ne11*ne12*ne13*row_size);
//             GGML_ASSERT(src1->type == GGML_TYPE_F32);
//
//             for (int64_t i13 = 0; i13 < ne13; ++i13) {
//                 for (int64_t i12 = 0; i12 < ne12; ++i12) {
//                     for (int64_t i11 = 0; i11 < ne11; ++i11) {
//                         from_float_to_vec_dot((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11), (void *) wdata, ne10);
//                         wdata += row_size;
//                     }
//                 }
//             }
//         }
//
//         return;
//     }
//
//     if (params->type == GGML_TASK_TYPE_FINALIZE) {
//         return;
//     }
//
//     const void * wdata    = (src1->type == vec_dot_type) ? src1->data : params->wdata;
//     const size_t row_size = ggml_row_size(vec_dot_type, ne10);
//
//     const int64_t nr0 = ne01;          // src0 rows
//     const int64_t nr1 = ne1*ne12*ne13; // src1 rows
//
//     //printf("nr0 = %lld, nr1 = %lld\n", nr0, nr1);
//
//     // distribute the thread work across the inner or outer loop based on which one is larger
//
//     const int64_t nth0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
//     const int64_t nth1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
//
//     const int64_t ith0 = ith % nth0;
//     const int64_t ith1 = ith / nth0;
//
//     const int64_t dr0 = (nr0 + nth0 - 1)/nth0;
//     const int64_t dr1 = (nr1 + nth1 - 1)/nth1;
//
//     const int64_t ir010 = dr0*ith0;
//     const int64_t ir011 = MIN(ir010 + dr0, nr0);
//
//     const int64_t ir110 = dr1*ith1;
//     const int64_t ir111 = MIN(ir110 + dr1, nr1);
//
//     //printf("ir010 = %6lld, ir011 = %6lld, ir110 = %6lld, ir111 = %6lld\n", ir010, ir011, ir110, ir111);
//
//     // threads with no work simply yield (not sure if it helps)
//     if (ir010 >= ir011 || ir110 >= ir111) {
//         sched_yield();
//         return;
//     }
//
//     assert(ne12 % ne02 == 0);
//     assert(ne13 % ne03 == 0);
//
//     // block-tiling attempt
//     const int64_t blck_0 = 16;
//     const int64_t blck_1 = 16;
//
//     // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
//     int64_t nrc = vec_dot_num_rows;
//     // TODO: currently the mmla kernels support only even numbered rows/cols.
//     // this check can be removed once they are extended to support odd numbered rows/cols too
//     if ((nr0 % 2 != 0) || (ne11 % 2 != 0)) {
//         nrc = 1;
//     }
//
//     const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;
//
//     // attempt to reduce false-sharing (does not seem to make a difference)
//     // 16 * 2, accounting for mmla kernels
//     float tmp[32];
//
//     for (int64_t iir1 = ir110; iir1 < ir111; iir1 += blck_1) {
//         for (int64_t iir0 = ir010; iir0 < ir011; iir0 += blck_0) {
//             for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir111; ir1 += nrc) {
//                 const int64_t i13 = (ir1/(ne12*ne1));
//                 const int64_t i12 = (ir1 - i13*ne12*ne1)/ne1;
//                 const int64_t i11 = (ir1 - i13*ne12*ne1 - i12*ne1);
//
//                 // broadcast src0 into src1
//                 const int64_t i03 = i13/r3;
//                 const int64_t i02 = i12/r2;
//
//                 const int64_t i1 = i11;
//                 const int64_t i2 = i12;
//                 const int64_t i3 = i13;
//
//                 const char * src0_row = (const char *) src0->data + (0 + i02*nb02 + i03*nb03);
//
//                 // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
//                 //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
//                 //       the original src1 data pointer, so we should index using the indices directly
//                 // TODO: this is a bit of a hack, we should probably have a better way to handle this
//                 const char * src1_col = (const char *) wdata +
//                     (src1_cont || src1->type != vec_dot_type
//                      ? (i11      + i12*ne11 + i13*ne12*ne11)*row_size
//                      : (i11*nb11 + i12*nb12 + i13*nb13));
//                 float * dst_col = (float *) ((char *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3));
//
//                 //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir011; ++ir0) {
//                 //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
//                 //}
//
//                 for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir011; ir0 += nrc) {
//                     vec_dot(ne00, &tmp[ir0 - iir0], (nrc>1 ? 16 : 0), src0_row + ir0*nb01, (nrc>1 ? nb01 : 0), src1_col, (nrc>1 ? src1_col_stride : 0), nrc);
//                 }
//
//                 for (int cn = 0; cn < nrc; ++cn) {
//                     memcpy(&dst_col[iir0 + cn*nb1/nb0], tmp + (cn*16), (MIN(iir0 + blck_0, ir011) - iir0)*sizeof(float));
//                 }
//             }
//         }
//     }
// }

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
