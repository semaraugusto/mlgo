package llama

import (
	"container/ring"
	"fmt"
	"mlgo/ml"
	"os"
	"testing"
)

type ModelParams struct {
	seed         int
	threadsCount int
	predictCount uint32 // new tokens to predict
	repeatLastN  uint32 // last n tokens to penalize
	partsCount   int    // amount of model parts (-1 = determine from model dimensions)
	ctxSize      uint32 // context size
	batchSize    uint32 // batch size for prompt processing

	// --- sampling parameters

	topK          uint32  // 40
	topP          float32 // 0.95
	temp          float32 // 0.80
	repeatPenalty float32 // 1.10

	model  string // model path
	prompt string

	antiprompt []string // string upon seeing which more user input is prompted

	memoryFP16 bool // use f16 instead of f32 for memory kv
	// randomPrompt bool // do not randomize prompt if none provided
	// useColor     bool // use color to distinguish generations and inputs
	interactive bool // interactive mode

	// verbosePrompt bool
}

func asciiToString(str string) string {
	runes := []rune(str)

	var result []int

	for i := 0; i < len(runes); i++ {
		result = append(result, int(runes[i]))
	}

	fmt.Printf("CONVERTING: %s to %v\n", str, result)
	return str
}

func TestTokenizerGGUF(t *testing.T) {
	// t.Skip("Skipping. Test is known to be failing")
	// modelFile := "../model/llama-7b-fp32.gguf"
	modelFile := "../model/state_dict/ggml-model-f32.gguf"
	params := ModelParams{
		model: modelFile,
	}

	// --- load the model
	fmt.Println("[INFO] Starting load")

	ctx, err := LoadModelGGUF(params.model, false, true)
	ctx.Vocab.SpecialAddBos = 1

	if err != nil {
		_, err := Colorize("\n[magenta][ ERROR ][white] Failed to load GGUF model [light_magenta]\"%s\"\n\n", params.model)
		if err != nil {
			return
		}
		os.Exit(0)
	}

	// tokenize the prompt
	// prompt := "Why Golang is so popular?"
	// prompt := " "
	prompt := "hey llama, why is golang so popular?"
	tokens := ml.Tokenize(ctx.Vocab, prompt, false)
	fmt.Println("Tokens: ", tokens)
	fmt.Println("len(Tokens): ", len(tokens))
	output := ""
	expected := []uint32{28139, 8814, 2786, 28725, 2079, 349, 20918, 602, 579, 4387, 28804}
	fmt.Println("len(expected): ", len(expected))
	if len(tokens) != len(expected) {
		t.Fatalf("Expected %v, got %v\n", expected, output)
	}
	for i, tokenID := range tokens {
		tokenStr := ml.Token2Str(ctx.Vocab, tokenID)
		if tokenID != expected[i] {
			fmt.Printf("Expected %v, got %v\n", tokenID, expected[i])
			t.Errorf("Expected %v, got %v\n", tokenID, expected[i])
		}
		fmt.Println("tokenID: ", tokenID, " Token: ", tokenStr)
		output += tokenStr
	}
}

func TestTokenizerBin(t *testing.T) {
	modelFile := "../model/llama-7b-fp32.bin"
	params := ModelParams{
		model: modelFile,
	}

	// --- load the model
	fmt.Println("\n[INFO] Starting load ")

	ctx, err := LoadModel(params.model, false, true)
	ctx.Vocab.SpecialAddBos = 1

	if err != nil {
		_, err := Colorize("\n[magenta][ ERROR ][white] Failed to load GGUF model [light_magenta]\"%s\"\n\n", params.model)
		if err != nil {
			return
		}
		os.Exit(0)
	}

	// tokenize the prompt
	// prompt := "Why Golang is so popular?"
	prompt := " "
	tokens := ml.Tokenize(ctx.Vocab, prompt, false)
	fmt.Println("Tokens: ", tokens)

	output := ""
	expected := " "
	for _, tokenID := range tokens {
		tokenStr := ml.Token2Str(ctx.Vocab, tokenID)
		// s := asciiToString(tokenStr)
		// fmt.Println("tokenID: ", tokenID, " Token: ", tokenStr, "converted: ", s)
		fmt.Println("tokenID: ", tokenID, " Token: ", tokenStr)
		output += tokenStr
	}
	if expected != output {
		fmt.Printf("Expected `%s`, got `%s`\n", expected, output)
		t.Errorf("Expected `%s`, got `%s`\n", expected, output)
	}

}

func TestLLaMAFixedTokensGGUF(t *testing.T) {
	t.Skip("Skipping. Test is known to be failing")
	// modelFile := "../model/llama-7b-fp32.gguf"
	modelFile := "../model/state_dict/ggml-model-f32.gguf"
	// prompt := "hey llama, why golang is so popular?"
	params := ModelParams{
		model:       modelFile,
		interactive: false,

		ctxSize:      512,
		seed:         -1,
		threadsCount: 1,
		predictCount: 1,
		repeatLastN:  64,
		partsCount:   -1,
		batchSize:    8,

		topK:          40,
		topP:          0.95,
		temp:          0.8,
		repeatPenalty: 1.10,

		memoryFP16: true,
	}
	threadCount := 32
	ctx, err := LoadModelGGUF(modelFile, true, false)
	fmt.Println("Load Model Finish")
	if err != nil {
		fmt.Println("load model error: ", err)
		return
	}
	lastNTokens := ring.New(int(params.ctxSize))
	for i := 0; i < int(params.ctxSize); i++ {
		lastNTokens.Value = uint32(0)
		lastNTokens = lastNTokens.Next()
	}
	// A function to append a token to the ring buffer
	appendToken := func(token uint32) {
		fmt.Println("Appending token: ", token)
		lastNTokens.Value = token
		lastNTokens = lastNTokens.Next()
	}
	// embd := ml.Tokenize(ctx.Vocab, prompt, true)
	embd := []uint32{28139, 8814, 2786, 28725, 2079, 349, 20918, 602, 579, 4387, 28804}
	NumPredict := 10
	for i := 0; i < NumPredict; i++ {
		err = Eval(ctx, embd, uint32(len(embd)), 0, threadCount)
		for i, id := range embd {
			token := ml.Token2Str(ctx.Vocab, id)
			fmt.Printf("[INFO] %d token: (id: %d, str: `%s`)\n", i, id, token)
		}
		fmt.Println("Eval Model Finish")
		id := SampleTopPTopK(ctx,
			lastNTokens, params.repeatLastN,
			params.topK, params.topP, params.temp, params.repeatPenalty)
		appendToken(id)
		fmt.Printf("RESULT: id: %d, `%s`\n", id, ml.Token2Str(ctx.Vocab, id))
		embd = []uint32{id}
	}
}

// func TestLLaMA(t *testing.T) {
// 	modelFile := "../model/llama-7b-fp32.bin"
// 	prompt := "Why Golang is so popular?"
// 	threadCount := 32
// 	ctx, err := LoadModel(modelFile, true)
// 	fmt.Println("Load Model Finish")
// 	if err != nil {
// 		fmt.Println("load model error: ", err)
// 		return
// 	}
// 	embd := ml.Tokenize(ctx.Vocab, prompt, true)
// 	err = Eval(ctx, embd, uint32(len(embd)), 0, threadCount)
// 	fmt.Println("Eval Model Finish")
// }
//
// func TestLLaMAEvalGraph(t *testing.T) {
// 	modelFile := "../models/llama-7b-fp32.bin"
// 	prompt := "Why Golang is so popular?"
// 	threadCount := 32
// 	ctx, err := LoadModel(modelFile, true)
// 	fmt.Println("Load Model Finish")
// 	if err != nil {
// 		fmt.Println("load model error: ", err)
// 		return
// 	}
// 	embd := ml.Tokenize(ctx.Vocab, prompt, true)
// 	graph, mlctx, err := ExpandGraph(ctx, embd, uint32(len(embd)), 0, threadCount)
// 	nodeID := int(graph.NodesCount) - 1
// 	ml.GraphComputeByNodes(mlctx, graph, nodeID)
// 	ml.PrintTensor(graph.Nodes[nodeID], "before")
//
// 	envBytes := ml.SaveComputeNodeEnvToBytes(uint32(nodeID), graph.Nodes[nodeID], graph, true)
// 	nodeID_, tensorGraphList_ , err := ml.DecodeComputeNodeEnv(envBytes, true, false)
// 	// save bytes from mips
// 	{
// 		fout, err := os.Create(fmt.Sprintf("../data/node_%v", nodeID))
// 		if err != nil {
// 			fmt.Println(err)
// 			return
// 		}
// 		defer fout.Close()
// 		_, err = fout.Write(envBytes)
// 		if err != nil {
// 			fmt.Println(err)
// 			return
// 		}
// 	}
// 	// save => tensorOnGraph[]
// 	tensorGraphList := ml.SaveComputeNodeEnv(graph.Nodes[nodeID], graph)
// 	fmt.Println("nodeID Equal: ", nodeID_ == uint32(nodeID))
// 	fmt.Println("tensorGraphList: ", reflect.DeepEqual(tensorGraphList_, tensorGraphList))
//
// 	// reconstruct
// 	tensorList := make([]*ml.Tensor, 0)
// 	tensorMap := make(map[uint32]*ml.Tensor)
// 	for i := 0; i < len(tensorGraphList); i++ {
// 		tensor := tensorGraphList[i].ToTensor(nil)
// 		tensorMap[tensorGraphList[i].NodeID] = tensor
// 		tensorList = append(tensorList, tensor)
// 	}
// 	// fill in the nodeid
// 	for i := 0; i < len(tensorList); i++ {
// 		tensor := tensorList[i]
// 		tensorG := tensorGraphList[i]
// 		if src0, ok := tensorMap[tensorG.Src0NodeID]; ok {
// 			tensor.Src0 = src0
// 		}
// 		if src1, ok := tensorMap[tensorG.Src1NodeID]; ok {
// 			tensor.Src1 = src1
// 		}
// 	}
//
// 	// compute
// 	ml.ComputeNodeForward(tensorMap[uint32(nodeID)])
//
// 	// ml.ComputeNodeForward(graph.Nodes[nodeID])
// 	ml.PrintTensor(tensorMap[uint32(nodeID)], "after")
//
// 	fmt.Println("graph node number: ", graph.NodesCount)
// 	fmt.Println("Eval Model Finish")
// }

