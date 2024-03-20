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

	_ = readString(file, nameLen)
	// name := readString(file, nameLen)
	// fmt.Printf("loading tensor named: %s - with shape %v\n", name, ne)

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

	if count, err := io.ReadFull(file, fake); err != nil || count != int(tensorSize*4) {
		fmt.Printf("\n[ERROR] Failed to read BIG FP32 chunk from model!")
		fmt.Printf("\n[ERROR] COUNT = %d | ERR = %s", count, err.Error())
		return nil, err
	}
	return tensor, nil
}

func TestLoadTensor(t *testing.T) {
	ml.SINGLE_THREAD = true
	model := new(MatMulModel)
	tensorPath := "../../test_tensors/qcur_src0.tensor"
	tensor, err := model.loadTensors(tensorPath)
	if err != nil {
		t.Fatalf(err.Error())
		return
	}
	model.weight = tensor
	if model.weight.Dims != 2 {
		t.Fatalf("Expected 2 dimensions, got %d", model.weight.Dims)
	}

	if len(model.weight.Data) != int(4096*4096) {
		t.Fatalf("Expected %d values, got %d", 4096*4096, len(model.weight.Data))
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
	if checkOutput != true {
		t.Fatalf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
		// fmt.Printf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
	}
	if checkExpected != true {
		// fmt.Printf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
		t.Fatalf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
	}
}

func TestMatMulSmall2(t *testing.T) {
	// t.Skip("Skipping test")
	ml.SINGLE_THREAD = true
	model := new(MatMulModel)
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
	if checkOutput != true {
		t.Fatalf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
		// fmt.Printf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
	}
	if checkExpected != true {
		// fmt.Printf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
		t.Fatalf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
	}
}

func TestMatMulSmall3(t *testing.T) {
	ml.SINGLE_THREAD = true
	model := new(MatMulModel)
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
	if checkOutput != true {
		t.Fatalf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
		// fmt.Printf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
	}
	if checkExpected != true {
		// fmt.Printf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
		t.Fatalf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
	}
}

func TestMatMulSmall4(t *testing.T) {
	ml.SINGLE_THREAD = true
	model := new(MatMulModel)
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
	if checkOutput != true {
		t.Fatalf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
		// fmt.Printf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
	}
	if checkExpected != true {
		// fmt.Printf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
		t.Fatalf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
	}
}

func TestMatMulSmall4Transposed(t *testing.T) {
	ml.SINGLE_THREAD = true
	model := new(MatMulModel)
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
	if checkOutput != true {
		t.Fatalf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
		// fmt.Printf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
	}
	if checkExpected != true {
		// fmt.Printf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
		t.Fatalf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
	}
}

func TestMatMul(t *testing.T) {
	ml.SINGLE_THREAD = true
	model := new(MatMulModel)
	filePathSrc0 := "../../test_tensors/qcur_src0.tensor"
	filePathSrc1 := "../../test_tensors/qcur_src1.tensor"
	filePathRes := "../../test_tensors/qcur_result.tensor"
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
	if err != nil {
		t.Fatalf(err.Error())
		return
	}

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

	checkOutput := CheckMatMul(actual, A, B)
	checkExpected := CheckMatMul(C, A, B)

	if checkOutput != true {
		t.Fatalf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
		// fmt.Printf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
	}
	if checkExpected != true {
		// fmt.Printf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
		t.Fatalf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
	}
}

func TestMatMul2(t *testing.T) {
	ml.SINGLE_THREAD = true
	model := new(MatMulModel)
	filePathSrc0 := "../../test_tensors/kcur_src0.tensor"
	filePathSrc1 := "../../test_tensors/kcur_src1.tensor"
	filePathRes := "../../test_tensors/kcur_result.tensor"
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
	if err != nil {
		t.Fatalf(err.Error())
		return
	}

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

	checkOutput := CheckMatMul(actual, A, B)
	checkExpected := CheckMatMul(C, A, B)

	if checkOutput != true {
		// t.Fatalf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
		t.Fatalf("[ERROR] CHECK_MATMUL_OUTPUT: true got %v\n", checkOutput)
	}
	if checkExpected != true {
		t.Fatalf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
		// t.Fatalf("[ERROR] CHECK_MATMUL_EXPECTED: true got %v\n", checkExpected)
	}
}

func MatGet(t *ml.Tensor, i0, i1, i2, i3 uint32) float32 {
	nb0 := t.NB[0] / 4
	nb1 := t.NB[1] / 4
	nb2 := t.NB[2] / 4
	nb3 := t.NB[3] / 4

	return t.Data[i0*nb0+i1*nb1+i2*nb2+i3*nb3]
}

func CheckMatMul(y, x0, x1 *ml.Tensor) bool {
	n00 := x0.NE[0]

	n02 := y.NE[0]
	n12 := y.NE[1]
	n22 := y.NE[2]
	n32 := y.NE[3]

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
						return false
					}
				}
			}
		}
	}

	return success
}
