package test

import (
	// "errors"
	// "math"
	// "errors"
	"fmt"
	"strconv"
	// "mlgo/common"
	"mlgo/ml"
	"os"
	"strings"
	"testing"
)

type LinearHParams struct {
	n_input   int32
	n_classes int32
	loaded    bool
}

type LinearModel struct {
	hparams LinearHParams

	weight *ml.Tensor
	bias   *ml.Tensor
}

type TestInputs struct {
	model  *LinearModel
	input  *ml.Tensor
	output *ml.Tensor
}

func parseLinearModelWeightsTxt(weights_str string, model *LinearModel) error {
	return nil
}

//	func linearModelLoad(model *LinearModel) error {
//		weightsPath := "models/l2_weights.txt"
//
//		// weightsFile, err := os.Open(weightsPath)
//		weightsFile, err := os.ReadFile(weightsPath)
//		if err != nil {
//			return err
//		}
//		// fmt.Print(string(weights))
//		weightsStr := string(weightsFile)
//		layerWeightsStr := strings.Split(weightsStr, "\n")
//		layerWeights := make([][]string, 0)
//		for i := 0; i < NOut; i++ {
//			layerWeights = append(layerWeights, strings.Split(layerWeightsStr[i], " "))
//
//		}
//
//		fmt.Println(layerWeights[0][0])
//		ft64, err := strconv.ParseFloat(layerWeights[0][0], 32)
//		if err != nil {
//			return err
//		}
//		// fmt.Println("float32: %.18f", ft32)
//		fmt.Printf("float64: %.20f\n", ft64)
//		fmt.Printf("float32: %.20f\n", float32(ft64))
//		fmt.Println(len(layerWeights))
//		fmt.Println("HERE")
//		// weights := make([10][25]float32, 0)
//		// weights := [NOut][NIn]float32{}
//		weights := []float32{}
//		for i := 0; i < NOut; i++ {
//			for j := 0; j < NIn; j++ {
//				ft32, err := strconv.ParseFloat(layerWeights[i][j], 32)
//				if err != nil {
//					return err
//				}
//				// weights[i][j] = float32(ft32)
//				weights = append(weights, float32(ft32))
//				// layerWeights[i] = append(layerWeights[, strings.Split(layerWeightsStr[i], " "))
//			}
//
//		}
//		fmt.Println("THERE")
//		// fmt.Println("WEIGHTS: ", weights)
//		// fmt.Println(len(layerWeights[10]))
//		// weightsStr.split(" ")
//
//		// Read FC1 layer 1
//		{
//			// n_dims := int32(common.ReadInt32FromFile(weightsFile))
//			// ne_weight := make([]int32, 0)
//			// for i := int32(0); i < n_dims; i++ {
//			// 	ne_weight = append(ne_weight, int32(common.ReadInt32FromFile(weightsFile)))
//			// }
//			// FC1 dimensions taken from file, eg. 768x500
//			model.hparams.n_input = 25
//			model.hparams.n_hidden = 1
//			fmt.Println("model.hparams.n_input", model.hparams.n_input)
//			fmt.Println("model.hparams.n_hidden", model.hparams.n_hidden)
//
//			model.weight = ml.NewTensor2D(nil, ml.TYPE_F32, uint32(model.hparams.n_input), uint32(model.hparams.n_hidden))
//			for i := 0; i < len(model.weight.Data); i++ {
//				// model.weight.Data[i] = common.ReadFP32FromFile(weightsFile)
//				model.weight.Data[i] = weights[i]
//			}
//
//			// ne_bias := make([]int32, 0)
//			// for i := 0; i < int(n_dims); i++ {
//			// 	ne_bias = append(ne_bias, int32(common.ReadInt32FromFile(file)))
//			// }
//			//
//			// model.fc_bias = ml.NewTensor1D(nil, ml.TYPE_F32, uint32(model.hparams.n_hidden))
//			// for i := 0; i < len(model.fc_bias.Data); i++ {
//			// 	model.fc_bias.Data[i] = common.ReadFP32FromFile(file)
//			// }
//		}
//		fmt.Println("linearModelLoadUnchecked")
//		fmt.Println("model: ", model)
//		fmt.Println("model weights: ", model.weight)
//
//		return nil
//	}
func (model *LinearModel) loadHParams(nIn, nOut int) {
	model.hparams.n_input = int32(nIn)
	model.hparams.n_classes = int32(nOut)
	model.hparams.loaded = true
}

func TestLoadHParams(t *testing.T) {
	ml.SINGLE_THREAD = true
	model := new(LinearModel)
	const nIn = 25
	const nOut = 10
	model.loadHParams(nIn, nOut)
	if model.hparams.n_input != int32(nIn) {
		t.Fatalf("Expected %d, got %d", nIn, model.hparams.n_input)
	}
	if model.hparams.n_classes != int32(nOut) {
		t.Fatalf("Expected %d, got %d", nOut, model.hparams.n_classes)
	}
	if model.hparams.loaded != true {
		t.Fatalf("Expected %t, got %t", true, model.hparams.loaded)
	}
}

// func loadWeights(model *LinearModel, weightsPath string, nIn, nOut int) error {
func (model *LinearModel) loadWeights(weightsPath string) error {
	if model.hparams.loaded != true {
		return fmt.Errorf("HParams not loaded")
	}
	file, err := os.ReadFile(weightsPath)
	if err != nil {
		return err
	}

	// weightsStr := string(weightsFile)
	str := strings.Trim(string(file), " ")
	weightsStr := strings.Split(str, "\n")

	layerWeights := [][]string{}
	for i := 0; i < int(model.hparams.n_classes); i++ {
		split := strings.Split(weightsStr[i], " ")
		layerWeights = append(layerWeights, split)

	}

	weights := []float32{}
	for i := 0; i < int(model.hparams.n_classes); i++ {
		data, err := strListToF32List(layerWeights[i])
		if err != nil {
			return err
		}
		weights = append(weights, data...)

	}

	model.weight = ml.NewTensor2D(nil, ml.TYPE_F32, uint32(model.hparams.n_input), uint32(model.hparams.n_classes))
	for i := 0; i < len(model.weight.Data); i++ {
		model.weight.Data[i] = weights[i]
	}

	return nil
}

func TestLoadWeights(t *testing.T) {
	const LOADED = 75
	expected := [LOADED]float32{
		-2.377904802560806274e-01, 5.536662936210632324e-01, -1.078299522399902344e+00, 6.318405866622924805e-01, -8.699013590812683105e-01, -8.142797350883483887e-01, -2.761250734329223633e-01, -3.304014205932617188e-01, 3.081635236740112305e-01, -4.025681316852569580e-02, 5.408433675765991211e-01, -7.181265950202941895e-01, -6.969867348670959473e-01, -6.016466021537780762e-01, -4.845047295093536377e-01, 5.513049364089965820e-01, 6.159024834632873535e-01, -1.422609090805053711e-01, 2.319888621568679810e-01, -7.162545919418334961e-01, 1.324884742498397827e-01, 6.312111616134643555e-01, 3.444145023822784424e-01, -8.397390842437744141e-01, 4.852400124073028564e-01, 4.840023815631866455e-01, -9.432380795478820801e-01, 8.071793317794799805e-01, -7.150199413299560547e-01, -4.020605608820915222e-02, -7.737609148025512695e-01, 5.271476507186889648e-01, -9.400956034660339355e-01, 8.877672255039215088e-02, -1.434337347745895386e-01, 1.433019876480102539e+00, 6.270796060562133789e-02, -8.493579626083374023e-01, 3.317112326622009277e-01, 4.183668494224548340e-01, -5.561479330062866211e-01, -1.437873840332031250e-01, 4.098496735095977783e-01, 6.715497374534606934e-01, 4.574362337589263916e-01, 1.165987133979797363e+00, -3.557277321815490723e-01, -9.827654361724853516e-01, 3.027272522449493408e-01, -6.643827557563781738e-01,
		2.798346579074859619e-01, 4.990698397159576416e-02, -2.478509694337844849e-01, -6.024796366691589355e-01, 7.464453577995300293e-02, -3.927443325519561768e-01, -1.801786571741104126e-01, 4.111442267894744873e-01, 5.660344362258911133e-01, -3.778194785118103027e-01, -4.662587940692901611e-01, 6.136606335639953613e-01, 2.672267854213714600e-01, -9.641145169734954834e-02, 9.147436618804931641e-01, -1.941628694534301758e+00, 9.007028341293334961e-01, -1.293321251869201660e-01, -6.844352483749389648e-01, -7.810074687004089355e-01, -4.052036702632904053e-01, 3.838924467563629150e-01, 3.618322610855102539e-01, 4.628591835498809814e-01, -1.615414172410964966e-01,
	}

	ml.SINGLE_THREAD = true
	model := new(LinearModel)
	nIn := 25
	nOut := 10
	model.loadHParams(nIn, nOut)

	filePath := "models/l2_weights.txt"
	if err := model.loadWeights(filePath); err != nil {
		t.Fatalf(err.Error())
		return
	}
	if model.weight.Dims != 2 {
		t.Fatalf("Expected 2 dimensions, got %d", model.weight.Dims)
	}
	if len(model.weight.Data) != 250 {
		t.Fatalf("Expected 250 values, got %d", len(model.weight.Data))
	}

	// for i := 0; i < len(model.weight.Data); i++ {
	for i := 0; i < LOADED; i++ {
		if model.weight.Data[i] != expected[i] {
			t.Fatalf("ERROR: Expected: '%f'\nGot: '%f'", expected[i], model.weight.Data[i])
		}
	}
}

func (m *LinearModel) loadBias(filePath string) error {
	if m.hparams.loaded != true {
		return fmt.Errorf("HParams not loaded")
	}
	file, err := os.ReadFile(filePath)
	if err != nil {
		return err
	}

	// str := string(file)
	sep := "\n"
	str := strings.Trim(string(file), " ")
	//
	bias, err := strToF32List(str, sep)
	if err != nil {
		return err
	}
	if len(bias) != int(m.hparams.n_classes) {
		return fmt.Errorf("Expected %d values, got %d", m.hparams.n_classes, len(bias))
	}

	m.bias = ml.NewTensor1D(nil, ml.TYPE_F32, uint32(m.hparams.n_classes))
	m.bias.Data = bias
	return nil
}

func TestLoadBias(t *testing.T) {
	ml.SINGLE_THREAD = true
	model := new(LinearModel)
	weightsPath := "models/l2_bias.txt"
	const nIn = 25
	const nOut = 10
	model.loadHParams(nIn, nOut)
	if err := model.loadBias(weightsPath); err != nil {
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

	if len(model.bias.Data) != int(model.hparams.n_classes) {
		t.Fatalf("Expected 25 values, got %d", len(model.bias.Data))
	}

	for i := 0; i < len(model.bias.Data); i++ {
		if model.bias.Data[i] != expected[i] {
			t.Fatalf("ERROR: Expected: '%f'\nGot: '%f'", expected[i], model.bias.Data[i])
		}
	}
}

func loadModel(m *LinearModel, weightsPath string, biasPath string) error {
	ml.SINGLE_THREAD = true
	model := new(LinearModel)
	const nIn = 25
	const nOut = 10
	// model.loadHParams(nIn, nOut)
	if err := model.loadWeights(weightsPath); err != nil {
		return fmt.Errorf(err.Error())
	}
	if err := model.loadBias(biasPath); err != nil {
		return fmt.Errorf(err.Error())
		// return
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
		return fmt.Errorf("Expected 2 dimensions, got %d", model.bias.Dims)
	}

	if len(model.bias.Data) != int(model.hparams.n_classes) {
		return fmt.Errorf("Expected 25 values, got %d", len(model.bias.Data))
	}

	for i := 0; i < len(model.bias.Data); i++ {
		if model.bias.Data[i] != expected[i] {
			fmt.Errorf("ERROR: Expected: '%f'\nGot: '%f'", expected[i], model.bias.Data[i])
		}
	}
	const LOADED = 75
	expectedWeights := [LOADED]float32{
		-2.377904802560806274e-01, 5.536662936210632324e-01, -1.078299522399902344e+00, 6.318405866622924805e-01, -8.699013590812683105e-01, -8.142797350883483887e-01, -2.761250734329223633e-01, -3.304014205932617188e-01, 3.081635236740112305e-01, -4.025681316852569580e-02, 5.408433675765991211e-01, -7.181265950202941895e-01, -6.969867348670959473e-01, -6.016466021537780762e-01, -4.845047295093536377e-01, 5.513049364089965820e-01, 6.159024834632873535e-01, -1.422609090805053711e-01, 2.319888621568679810e-01, -7.162545919418334961e-01, 1.324884742498397827e-01, 6.312111616134643555e-01, 3.444145023822784424e-01, -8.397390842437744141e-01, 4.852400124073028564e-01, 4.840023815631866455e-01, -9.432380795478820801e-01, 8.071793317794799805e-01, -7.150199413299560547e-01, -4.020605608820915222e-02, -7.737609148025512695e-01, 5.271476507186889648e-01, -9.400956034660339355e-01, 8.877672255039215088e-02, -1.434337347745895386e-01, 1.433019876480102539e+00, 6.270796060562133789e-02, -8.493579626083374023e-01, 3.317112326622009277e-01, 4.183668494224548340e-01, -5.561479330062866211e-01, -1.437873840332031250e-01, 4.098496735095977783e-01, 6.715497374534606934e-01, 4.574362337589263916e-01, 1.165987133979797363e+00, -3.557277321815490723e-01, -9.827654361724853516e-01, 3.027272522449493408e-01, -6.643827557563781738e-01,
		2.798346579074859619e-01, 4.990698397159576416e-02, -2.478509694337844849e-01, -6.024796366691589355e-01, 7.464453577995300293e-02, -3.927443325519561768e-01, -1.801786571741104126e-01, 4.111442267894744873e-01, 5.660344362258911133e-01, -3.778194785118103027e-01, -4.662587940692901611e-01, 6.136606335639953613e-01, 2.672267854213714600e-01, -9.641145169734954834e-02, 9.147436618804931641e-01, -1.941628694534301758e+00, 9.007028341293334961e-01, -1.293321251869201660e-01, -6.844352483749389648e-01, -7.810074687004089355e-01, -4.052036702632904053e-01, 3.838924467563629150e-01, 3.618322610855102539e-01, 4.628591835498809814e-01, -1.615414172410964966e-01,
	}
	if model.weight.Dims != 2 {
		return fmt.Errorf("Expected 2 dimensions, got %d", model.weight.Dims)
	}
	if len(model.weight.Data) != 250 {
		return fmt.Errorf("Expected 250 values, got %d", len(model.weight.Data))
	}

	// for i := 0; i < len(model.weight.Data); i++ {
	for i := 0; i < LOADED; i++ {
		if model.weight.Data[i] != expectedWeights[i] {
			return fmt.Errorf("ERROR: Expected: '%f'\nGot: '%f'", expected[i], model.weight.Data[i])
		}
	}
	fmt.Println("model: ", model)
	fmt.Println("model.weight: ", model.weight)
	return nil

}

func strToF32List(contents, sep string) ([]float32, error) {
	str := strings.Trim(contents, " ")
	strSplit := strings.Split(str, sep)

	data, err := strListToF32List(strSplit)
	return data, err
}
func strListToF32List(strList []string) ([]float32, error) {
	size := len(strList)
	if strList[size-1] == "" {
		size = size - 1
	}
	t := make([]float32, size)
	for i := 0; i < size; i++ {
		if strList[i] == "" {
			continue
		}
		trimmed := strings.Trim(strList[i], " ")
		trimmed = strings.Trim(strList[i], "\n")
		ft32, err := strconv.ParseFloat(trimmed, 32)
		if err != nil {
			fmt.Println("ERROR: ", trimmed)
			return nil, err
		}
		t[i] = float32(ft32)
	}
	return t, nil
}

func fileToStr(filePath string) (string, error) {
	file, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}

	return strings.Trim(string(file), " "), nil
}

func (ti *TestInputs) loadInput(filePath string) error {
	if ti.model.hparams.loaded != true {
		return fmt.Errorf("HParams not loaded")
	}
	str, err := fileToStr(filePath)
	if err != nil {
		return err
	}
	sep := " "
	input, err := strToF32List(str, sep)
	if err != nil {
		return err
	}

	ti.input = ml.NewTensor1D(nil, ml.TYPE_F32, uint32(ti.model.hparams.n_input))
	ti.input.Data = input

	return nil
}

func (ti *TestInputs) loadOutput(filePath string) error {
	if ti.model.hparams.loaded != true {
		return fmt.Errorf("HParams not loaded")
	}
	str, err := fileToStr(filePath)
	if err != nil {
		return err
	}
	sep := " "
	input, err := strToF32List(str, sep)
	if err != nil {
		return err
	}
	ti.output = ml.NewTensor1D(nil, ml.TYPE_F32, uint32(ti.model.hparams.n_classes))
	ti.output.Data = input

	return nil
}

func TestLoadInput(t *testing.T) {
	const nIn = 25
	const nOut = 10
	model := new(LinearModel)
	tInputs := new(TestInputs)
	tInputs.model = model
	model.loadHParams(nIn, nOut)
	expected := [nIn]float32{
		0.000000000000000000e+00,
		2.918797016143798828e+00,
		0.000000000000000000e+00,
		2.899260044097900391e+00,
		0.000000000000000000e+00,
		5.984142422676086426e-02,
		4.986535906791687012e-01,
		1.960299134254455566e+00,
		2.902335166931152344e+00,
		0.000000000000000000e+00,
		0.000000000000000000e+00,
		2.519920587539672852e+00,
		3.270778417587280273e+00,
		3.656535148620605469e-01,
		0.000000000000000000e+00,
		0.000000000000000000e+00,
		0.000000000000000000e+00,
		0.000000000000000000e+00,
		3.391362905502319336e+00,
		5.931361317634582520e-01,
		0.000000000000000000e+00,
		1.682612895965576172e+00,
		1.680085539817810059e+00,
		0.000000000000000000e+00,
		4.993188381195068359e+00,
	}
	ml.SINGLE_THREAD = true
	filePath := "models/relu_out.txt"

	if err := tInputs.loadInput(filePath); err != nil {
		t.Fatalf(err.Error())
		return
	}

	for i := 0; i < len(tInputs.input.Data); i++ {
		if tInputs.input.Data[i] != expected[i] {
			t.Fatalf("ERROR: Expected: '%f'\nGot: '%f'", expected[i], tInputs.input.Data[i])
		}
	}
}

func TestLoadOutput(t *testing.T) {
	const nIn = 25
	const nOut = 10
	model := new(LinearModel)
	tInputs := new(TestInputs)
	tInputs.model = model
	model.loadHParams(nIn, nOut)
	expected := [nOut]float32{
		3.339550495147705078e+00, -1.150749111175537109e+01, 9.716508984565734863e-01, -5.528323650360107422e+00, 2.966210126876831055e+00, 7.380880713462829590e-01, 3.723233222961425781e+00, -5.654765605926513672e+00, 6.697512269020080566e-01, -3.081719398498535156e+00,
	}
	ml.SINGLE_THREAD = true
	filePath := "models/l2_out.txt"

	if err := tInputs.loadOutput(filePath); err != nil {
		t.Fatalf(err.Error())
		return
	}

	for i := 0; i < len(tInputs.output.Data); i++ {
		if tInputs.output.Data[i] != expected[i] {
			t.Fatalf("ERROR: Expected: '%f'\nGot: '%f'", expected[i], tInputs.input.Data[i])
		}
	}
}
// func (m *LinearModel) linear_eval(threadCount int, digit []float32) int {
func (m *LinearModel) linear_eval(threadCount int, digit []float32) []float32 {

	fmt.Println("START EVAL")
	ctx0 := &ml.Context{}
	graph := ml.Graph{ThreadsCount: threadCount}
	fmt.Println("m.hparams.n_input: ", m.hparams.n_input)

	input := ml.NewTensor1D(ctx0, ml.TYPE_F32, uint32(m.hparams.n_input))
	copy(input.Data, digit)

	// fc1 MLP = Ax + b
	fmt.Println("BEFORE MULMAT")
	fmt.Println("weight: ", m.weight)
	fmt.Println("input: ", input)
	mulmat := ml.MulMat(ctx0, m.weight, input)
	fmt.Println("MULMAT")
	fc := ml.Add(ctx0, mulmat, m.bias)
	// final := ml.Relu(ctx0, fc1)
	fmt.Println("OPS DEFINED")

	// run the computation
	ml.BuildForwardExpand(&graph, fc)
	ml.GraphCompute(ctx0, &graph)

	ml.PrintTensor(mulmat, "mulmat")
	ml.PrintTensor(fc, "fc")
	// ml.PrintTensor(final, "final tensor")

	return fc.Data

}

func TestModelLoad(t *testing.T) {
	const nIn = 25
	const nOut = 10
	model := new(LinearModel)
	tInputs := new(TestInputs)
	model.loadHParams(nIn, nOut)
	fmt.Println("!!model: ", model)
	tInputs.model = model
	biasPath := "models/l2_bias.txt"
	weightsPath := "models/l2_weights.txt"
	fmt.Println("tInputs.!model: ", tInputs.model)
	if err := tInputs.model.loadWeights(weightsPath); err != nil {
		t.Fatalf(err.Error())
	}
	if err := tInputs.model.loadBias(biasPath); err != nil {
		t.Fatalf(err.Error())
		// return
	}
}

func TestModelEval(t *testing.T) {
	const nIn = 25
	const nOut = 10
	expected := []float32 {
		3.339550495147705078e+00, -1.150749111175537109e+01, 9.716508984565734863e-01, -5.528323650360107422e+00, 2.966210126876831055e+00, 7.380880713462829590e-01, 3.723233222961425781e+00, -5.654765605926513672e+00, 6.697512269020080566e-01, -3.081719398498535156e+00,
	}
	model := new(LinearModel)
	tInputs := new(TestInputs)
	model.loadHParams(nIn, nOut)
	tInputs.model = model
	biasPath := "models/l2_bias.txt"
	weightsPath := "models/l2_weights.txt"
	if err := tInputs.model.loadWeights(weightsPath); err != nil {
		t.Fatalf(err.Error())
	}
	if err := tInputs.model.loadBias(biasPath); err != nil {
		t.Fatalf(err.Error())
		// return
	}
	inputPath := "models/relu_out.txt"
	if err := tInputs.loadInput(inputPath); err != nil {
		t.Fatalf(err.Error())
		return
	}
	outputPath := "models/l2_out.txt"
	if err := tInputs.loadOutput(outputPath); err != nil {
		t.Fatalf(err.Error())
		return
	}
	fmt.Println("tInputs.input: ", tInputs.input)
	// fmt.Println("tInputs.output: ", tInputs.output)
	fmt.Println("tInputs.model: ", tInputs.model)

	for i := 0; i < len(tInputs.output.Data); i++ {
		if tInputs.output.Data[i] != expected[i] {
			t.Fatalf("ERROR: Expected: '%f'\nGot: '%f'", expected[i], tInputs.input.Data[i])
		}
	}
	fmt.Println("tInputs.model.weight: ", tInputs.model.weight)

	pred := tInputs.model.linear_eval(1, tInputs.input.Data)

	for i := 0; i < len(tInputs.output.Data); i++ {
		if pred[i] != expected[i] {
			t.Fatalf("ERROR: Expected: '%.18f'\nGot: '%.18f'. diff: %.18f", expected[i], pred[i], pred[i] - expected[i])
		}
	}
}
