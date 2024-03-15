package llama

import (
	"container/ring"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"reflect"
	"runtime"
	"sort"
	"unsafe"

	"github.com/abrander/gguf"
	"github.com/mattn/go-colorable"
	"github.com/mitchellh/colorstring"
	"github.com/schollz/progressbar/v3"
	"github.com/x448/float16"
	"golang.org/x/exp/slices"

	"mlgo/ml"
)

const (
	LLAMA_FILE_VERSION           = 1
	LLAMA_FILE_MAGIC             = 0x67676a74 // 'ggjt' in hex
	LLAMA_FILE_MAGIC_OLD         = 0x67676d66 // 'ggmf' in hex
	LLAMA_FILE_MAGIC_UNVERSIONED = 0x67676d6c // 'ggml' pre-versioned files

	SPLIT_NONE       = 0
	SPLIT_BY_COLUMNS = 1
	SPLIT_BY_ROWS    = 2
)

var (
	// determine number of model parts based on the dimension
	LLAMA_N_PARTS = map[uint32]int{
		4096: 1,
		5120: 2,
		6656: 4,
		8192: 8,
	}
)

type pair struct {
	first  float32
	second uint32
}

type Context struct {
	Model *Model
	Vocab *ml.Vocab

	// decode output (2-dimensional array: [n_tokens][n_vocab])
	Logits    []float32
	LogitsAll bool

	// input embedding (1-dimensional array: [n_embd])
	Embedding []float32
}

func NewContext() *Context {
	return &Context{
		Model:     NewModel(),
		Vocab:     ml.NewVocab(0),
		Logits:    make([]float32, 0, 0), // NewFloatSlice(0, 0),
		Embedding: make([]float32, 0, 0), // NewFloatSlice(0, 0),
	}
}

// struct llama_context_params {
type ContextParams struct {
	CtxSize    uint32 // text context
	PartsCount int    // -1 for default
	Seed       int    // RNG seed, 0 for random
	LogitsAll  bool   // the llama_eval() call computes all logits, not just the last one
	VocabOnly  bool   // only load the vocabulary, no weights
	UseLock    bool   // force system to keep model in RAM
	Embedding  bool   // embedding mode only
}

type Layer struct {

	// normalization
	attentionNorm *ml.Tensor

	// attention
	wq *ml.Tensor
	wk *ml.Tensor
	wv *ml.Tensor
	wo *ml.Tensor

	// normalization
	ffn_norm *ml.Tensor

	// ff
	w1 *ml.Tensor
	w2 *ml.Tensor
	w3 *ml.Tensor
}

// default hparams (LLaMA 7B)
type HParams struct {
	ctxSize      uint32 // 512
	vocabSize    uint32 // 32000
	embdSize     uint32 // 4096
	multSize     uint32 // 256
	headsCount   uint32 // 32
	kvHeadsCount uint32 // 32
	layersCount  uint32 // 32
	rotCount     uint32 // 64
	f16          uint32 // 1
}

type ModelType uint8

// available llama models
const (
	MODEL_UNKNOWN ModelType = iota
	MODEL_7B
	MODEL_13B
	MODEL_30B
	MODEL_65B
)

type KVCache struct {
	K *ml.Tensor
	V *ml.Tensor

	N uint32 // number of tokens currently in the cache
}

type Model struct {
	Type    ModelType
	ctx     *ml.Context
	hparams HParams

	tokEmbeddings *ml.Tensor
	norm          *ml.Tensor
	output        *ml.Tensor

	layers []Layer
	kvSelf KVCache // key + value cache for the self attention

	loadedCount uint32
	tensors     map[string]*ml.Tensor
}

func NewModel() *Model {
	return &Model{
		hparams: HParams{
			ctxSize:     512,
			vocabSize:   32000,
			embdSize:    4096,
			multSize:    256,
			headsCount:  32,
			layersCount: 32,
			rotCount:    64,
			f16:         1,
		},
		layers:  make([]Layer, 0),
		tensors: make(map[string]*ml.Tensor),
		kvSelf: KVCache{
			K: &ml.Tensor{},
			V: &ml.Tensor{},
		},
	}
}

func debug(format string, args ...interface{}) {
	fmt.Printf(colorstring.Color("[green]"+format+"[reset]\n"), args...)
}

func min(a, b int) int {
	if a <= b {
		return a
	}
	return b
}

// Resize() (safe) for using instead of C++ std::vector:resize()
// https://go.dev/play/p/VlQ7N75E5AD
func Resize(slice []float32, size int) []float32 {
	newSlice := make([]float32, size)
	for i := 0; i < min(size, len(slice)); i++ {
		newSlice[i] = slice[i]
	}
	return newSlice
}

// NB! This do not clear the underlying array when resizing
// https://go.dev/play/p/DbK4dFqwrZn
func ResizeInplace(slice *[]float32, size int) {
	if len(*slice) == size {
		return
	} else if size < len(*slice) {
		*slice = (*slice)[:size]
	} else {
		*slice = slices.Grow(*slice, size)
		*slice = (*slice)[:size]
	}
}

// evaluate the transformer
//
//   - lctx:      llama context
//   - tokens:    new batch of tokens to process
//   - n_past:    the context size so far
//   - n_threads: number of threads to use
//

func Eval(

	lctx *Context,
	tokens []uint32,
	tokensCount uint32,
	pastCount uint32,
	threadsCount int) error {

	N := tokensCount
	model := lctx.Model
	kvSelf := model.kvSelf

	embdSize := model.hparams.embdSize
	layersCount := model.hparams.layersCount
	ctxSize := model.hparams.ctxSize
	headsCount := model.hparams.headsCount
	vocabSize := model.hparams.vocabSize
	kvHeadsCount := model.hparams.kvHeadsCount
	rotCount := model.hparams.embdSize / model.hparams.headsCount
	nEmbdHead := embdSize / headsCount
	numKVGroups := headsCount / kvHeadsCount
	cacheSize := kvHeadsCount * nEmbdHead
	debug("ctxSize: %d", ctxSize)
	debug("numKVGroups: %d", numKVGroups)

	ctx0 := &ml.Context{} //ctx0 := ml.Init(ml.InitParams{})

	graph := ml.Graph{ThreadsCount: threadsCount}

	// Convert the tokens to a []float32 slice
	tokensFloat32 := make([]float32, len(tokens))
	for i, token := range tokens {
		tokensFloat32[i] = float32(token)
	}

	// Initialize the embd tensor with the tokensFloat32 data
	embd := ml.NewTensor(ctx0, ml.TYPE_F32, 1, uint32(len(tokens)), 1, 1, 1, tokensFloat32)
	inpL := ml.GetRows(ctx0, model.tokEmbeddings, embd)

	for il := uint32(0); il < layersCount; il++ {
		inpSA := inpL
		cur := &ml.Tensor{}

		// norm
		cur = ml.RMSNorm(ctx0, inpL)

		// cur = attention_norm*cur
		rep := ml.Repeat(ctx0, model.layers[il].attentionNorm, cur)

		cur = ml.Mul(ctx0, rep, cur)

		// self-attention
		{
			Qcur := ml.MulMat(ctx0, model.layers[il].wq, cur)
			Kcur := ml.MulMat(ctx0, model.layers[il].wk, cur)
			Vcur := ml.MulMat(ctx0, model.layers[il].wv, cur)

			// store key and value to memory
			if N >= 1 {

				////struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, N*n_embd, (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past));
				////struct ggml_tensor * v = ggml_view_1d(ctx0, kv_self.v, N*n_embd, (ggml_element_size(kv_self.v)*n_embd)*(il*n_ctx + n_past));

				////ggml_build_forward_expand(&graph, ggml_cpy(ctx0, Kcur, k));
				////ggml_build_forward_expand(&graph, ggml_cpy(ctx0, Vcur, v));

				// NB! ggml_element_size(kv_self.k) = 2 for FP16
				origNe0 := N * cacheSize
				origOffset := cacheSize * (il*ctxSize + pastCount)
				ne0 := N * cacheSize
				if ne0 != origNe0 {
					fmt.Println("Error: ne0 != origNe0")
					os.Exit(1)
				}
				k := ml.View1D(ctx0, kvSelf.K, origNe0, origOffset)
				v := ml.View1D(ctx0, kvSelf.V, origNe0, origOffset)

				ml.BuildForwardExpand(&graph, ml.Copy(ctx0, Kcur, k))
				ml.BuildForwardExpand(&graph, ml.Copy(ctx0, Vcur, v))
			}

			Q :=
				ml.Permute(ctx0,
					ml.Rope(ctx0,
						ml.Copy(ctx0,
							Qcur,
							ml.NewTensor3D(ctx0, ml.TYPE_F32, embdSize/headsCount, headsCount, N)),
						pastCount, rotCount, 0),
					0, 2, 1, 3)

			// K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
			viewNE0 := (pastCount + N) * cacheSize
			viewOffset := (il * ctxSize * cacheSize)
			view := ml.View1D(ctx0, kvSelf.K, viewNE0, viewOffset)

			cacheNE0 := cacheSize / kvHeadsCount
			cacheNE1 := kvHeadsCount
			cacheNE2 := pastCount + N

			reshape := ml.Reshape3D(ctx0,
				////ggml_view_1d(ctx0, kv_self.k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(kv_self.k)*n_embd),
				////n_embd/n_head, n_head, n_past + N),
				view,
				cacheNE0, cacheNE1, cacheNE2)

			K :=
				ml.Permute(ctx0,
					ml.Rope(ctx0,
						reshape,
						pastCount, rotCount, 1),
					0, 2, 1, 3)

			KQ := ml.MulMat(ctx0, K, Q)

			KQScaled :=
				ml.Scale(ctx0,
					KQ,
					ml.NewFP32(ctx0, float32(1.0/math.Sqrt(float64(embdSize)/float64(headsCount)))),
				)

			// KQ_masked = mask_past(KQ_scaled)
			////struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);
			KQMasked := ml.DiagMaskInf(ctx0, KQScaled, pastCount)

			// KQ = soft_max(KQ_masked)
			////struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);
			KQSoftMax := ml.SoftMax(ctx0, KQMasked)

			view = ml.View1D(ctx0, kvSelf.V, viewNE0, viewOffset)
			VTrans :=
				ml.Copy(ctx0,
					ml.Permute(ctx0,
						ml.Reshape3D(ctx0,
							view,
							cacheNE0, cacheNE1, cacheNE2),
						1, 2, 0, 3),
					ml.NewTensor3D(ctx0, ml.TYPE_F32 /* kv_self.v->type */, cacheNE2, cacheNE0, cacheNE1))

			KQV := ml.MulMat(ctx0, VTrans, KQSoftMax)
			KQVMerged := ml.Permute(ctx0, KQV, 0, 2, 1, 3)
			cur = ml.Copy(ctx0,
				KQVMerged,
				ml.NewTensor2D(ctx0, ml.TYPE_F32, embdSize, N))

			// projection (no bias)
			cur = ml.MulMat(ctx0,
				model.layers[il].wo,
				cur)

		}

		inpFF := ml.Add(ctx0, cur, inpSA)

		// feed-forward network
		{
			// norm
			{
				cur = ml.RMSNorm(ctx0, inpFF)

				// cur = ffn_norm*cur
				cur = ml.Mul(ctx0,
					ml.Repeat(ctx0, model.layers[il].ffn_norm, cur),
					cur)
			}

			tmp := ml.MulMat(ctx0,
				model.layers[il].w3,
				cur)

			cur = ml.MulMat(ctx0,
				model.layers[il].w1,
				cur)

			// SILU activation
			cur = ml.Silu(ctx0, cur)

			cur = ml.Mul(ctx0, cur, tmp)

			cur = ml.MulMat(ctx0,
				model.layers[il].w2,
				cur)
		}

		cur = ml.Add(ctx0, cur, inpFF)

		// input for next layer
		inpL = cur

	}

	// used at the end to optionally extract the embeddings
	////var embeddings *ml.Tensor

	// --- norm

	inpL = ml.RMSNorm(ctx0, inpL)

	// inpL = norm*inpL
	inpL = ml.Mul(ctx0,
		ml.Repeat(ctx0, model.norm, inpL),
		inpL)

	embeddings := inpL

	// lm_head
	inpL = ml.MulMat(ctx0, model.output, inpL)

	// logits -> probs
	// COMMENTED inpL = ggml_soft_max(ctx0, inpL);

	// run the computation
	ml.BuildForwardExpand(&graph, inpL)

	ml.GraphCompute(ctx0, &graph)

	// --- extract logits

	if lctx.LogitsAll {
		fmt.Print("\n[HALT] Not Expected: lctx.LogitsAll == true")
		os.Exit(1)

		/*
			// Copy inpL.Data to lctx.Logits
			for i := uint32(0); i < vocabSize*N; i++ {
				if i >= uint32(len(lctx.Logits)) || i >= uint32(len(inpL.Data)) {
					fmt.Println("Error: Index out of bounds during Logits copy")
					os.Exit(1)
				}
				lctx.Logits[i] = inpL.Data[i]
			}
		*/
	} else {
		// Copy only the relevant part of inpL.Data to lctx.Logits
		for i := uint32(0); i < vocabSize; i++ {
			srcIndex := vocabSize*(N-1) + i
			if i >= uint32(len(lctx.Logits)) || srcIndex >= uint32(len(inpL.Data)) {
				fmt.Println("Error: Index out of bounds during Logits copy")
				os.Exit(1)
			}
			lctx.Logits[i] = inpL.Data[srcIndex]
		}
	}

	if ml.DEBUG {
		printTensor(inpL, "INPL")

		fmt.Printf("\n\n=== LOGITS === %d ===\n", len(lctx.Logits)) // DEBUG
		for ii := 0; ii < 13; ii++ {
			fmt.Printf("%.4f  ", lctx.Logits[ii])
		}
	}

	// --- extract embeddings

	if len(lctx.Embedding) > 0 {
		////memcpy(embedding_out.data(), (float *) ggml_get_data(embeddings) + (n_embd*(N - 1)), sizeof(float)*n_embd);
		for i := uint32(0); i < embdSize; i++ {
			lctx.Embedding[i] = embeddings.Data[(embdSize*(N-1))+i] // FIXME ASAP
		}
	}

	return nil
}

func printTensor(tensor *ml.Tensor, name string) {
	var dt string
	if tensor.Type == ml.TYPE_F16 {
		dt = "FP16"
	}
	if tensor.Type == ml.TYPE_F32 {
		dt = "FP32"
	}
	if tensor.Type == ml.TYPE_Q4_0 {
		dt = "INT4"
	}

	fmt.Printf("\n\n=== [ %s | %s | %d:%d:%d ] ===\n",
		name, dt, tensor.NE[0], tensor.NE[1], tensor.NE[2])

	for nn := 0; nn < min(12, int(tensor.NE[1])); nn++ {
		fmt.Printf("\n %d x %d ...\t", nn, tensor.NE[0])
		for ii := 0; ii < min(12, int(tensor.NE[0])); ii++ {
			fmt.Printf("%.3f\t", tensor.Data[nn*int(tensor.NE[0])+ii])
		}
	}
}

func sampleTopK(logitsID []pair, topK uint32) []pair {
	// find the top K tokens

	// std::partial_sort
	// Rearranges elements such that the range [first, middle) contains
	// the sorted middle − first smallest elements in the range [first, last).
	// The order of equal elements is not guaranteed to be preserved.
	// The order of the remaining elements in the range [middle, last) is unspecified.

	/*std::partial_sort(
	        logits_id.begin(),
	        logits_id.begin() + top_k, logits_id.end(),
	        [](const std::pair<double, gpt_vocab::id> & a, const std::pair<double, gpt_vocab::id> & b) {
	    return a.first > b.first;
	});*/

	//keys := make([]double, 0, len(logitsID))
	//for k := range logitsID {
	//	keys = append(keys, k)
	//}
	//sort.Float64s(keys)

	sort.Slice(
		logitsID[:topK],
		func(i, j int) bool {
			return logitsID[i].first < logitsID[j].first // FIXME ASAP We need bigger elements first
		})

	// logits_id.resize(top_k);
	//for i := uint32(0); i < len(keys)-topK; i++ {
	//delete(logitsID, keys[i])
	//}

	ret := make([]pair, 0, topK)
	copy(ret, logitsID)

	return ret
}

// llama_sample_top_p_top_k
// sample next token given probabilities for each embedding
//
//   - consider only the top K tokens
//   - from them, consider only the top tokens with cumulative probability > P
//

// std::mt19937 = A Mersenne Twister pseudo-random generator of 32-bit numbers with a state size of 19937 bits.
func SampleTopPTopK(
	lctx *Context,
	// lastNTokens []uint32,
	lastNTokens *ring.Ring,
	lastNTokensSize uint32, // FIXME Remove
	topK uint32,
	topP float32,
	temp float32,
	repeatPenalty float32,
) uint32 {

	////auto & rng = lctx.rng;
	////logitsCount := uint32(len(vocab.ID2Token))
	logitsCount := lctx.Model.hparams.vocabSize
	logits := lctx.Logits

	if ml.DEBUG {
		fmt.Printf("\n\n>>> SampleTopPTopK <<<\n")
		fmt.Printf("\n=== LOGITS | %d ===\n", len(logits))
		for i := 0; i < 8; i++ {
			fmt.Printf("%.4f ", logits[i])
		}
		fmt.Printf(" ... ")
		for i := int(len(logits)) - 1; i >= int(len(logits))-8; i-- {
			fmt.Printf("%.4f ", logits[i])
		}
		/*
			fmt.Printf("\n=== LAST N TOKENS | %d ===\n", len(lastNTokens))
			for i := 0; i < int(lastNTokensSize); i++ {
				fmt.Printf("%d ", lastNTokens[i])
			}
		*/
		extractedTokens := ExtractTokens(lastNTokens.Move(-int(lastNTokensSize)), int(lastNTokensSize))
		fmt.Printf("\n=== LAST N TOKENS | %d ===\n", len(extractedTokens))
		for i := 0; i < int(lastNTokensSize); i++ {
			fmt.Printf("%d ", extractedTokens[i])
		}
	}

	////if (temp <= 0) {
	////    // select the token with the highest logit directly
	////    float max_logit = plogits[0];
	////    llama_vocab::id max_id = 0;
	////
	////    for (int i = 1; i < n_logits; ++i) {
	////        if (plogits[i] > max_logit) {
	////            max_logit = plogits[i];
	////            max_id = i;
	////        }
	////    }
	////    return max_id;
	////}

	////const auto * plogits = logits.data() + logits.size() - n_logits;
	//plogits := logits[len(logits)-int(logitsCount):] // FIXME ASAP
	plogits := logits[:]

	////std::vector<std::pair<double, llama_vocab::id>> logits_id;
	////logits_id.reserve(n_logits);
	logitsID := make([]pair, 0, logitsCount) // FIXME LEN vs CAP

	{
		scale := float32(1.0 / temp)
		for i := uint32(0); i < logitsCount; i++ {

			// Repetition penalty from ctrl paper (https://arxiv.org/abs/1909.05858)
			// Credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main

			// Check if the i-th token is present in the last_n_tokens ring buffer
			tokenExists := false
			lastNTokens.Do(func(p interface{}) {
				if p.(uint32) == i {
					tokenExists = true
				}
			})

			// If lastNTokens already contains i-th token, append it with repeat penalty
			if tokenExists {
				// If score < 0, then repetition penalty has to be multiplied to reduce the previous token probability
				if plogits[i] < 0.0 {
					logitsID = append(logitsID, pair{plogits[i] * scale * repeatPenalty, i})
				} else {
					logitsID = append(logitsID, pair{plogits[i] * scale / repeatPenalty, i})
				}
				// Else append pair to logitsID, scaling probability
			} else {
				logitsID = append(logitsID, pair{plogits[i] * scale, i})
			}
		}
	}

	if ml.DEBUG {
		fmt.Printf("\n=== LOGITS ID AFTER | %d ===\n", len(logitsID))
		for i := 0; i < min(6, len(logitsID)); i++ {
			fmt.Printf("{ %.3f | %d }", logitsID[i].first, logitsID[i].second)
		}
		fmt.Printf(" ... ")
		for i := len(logitsID) - 6; i < len(logitsID)-1; i++ {
			fmt.Printf("{ %.3f | %d } ", logitsID[i].first, logitsID[i].second)
		}
	}

	// sort logitsID slice and return only top K elements
	//// sampleTopK(logitsID, topK)

	// NB! Inline logic for [sampleTopK] right here

	//// std::partial_sort(
	////	logits_id.begin(),
	////	logits_id.begin() + top_k, logits_id.end(),
	////	[](const std::pair<float, llama_vocab::id> & a, const std::pair<float, llama_vocab::id> & b) {
	//// return a.first > b.first;
	//// });
	//// logits_id.resize(top_k);

	sort.Slice(
		logitsID, // logitsID[:topK],
		func(a, b int) bool {
			return logitsID[a].first > logitsID[b].first
		})

	if ml.DEBUG {
		fmt.Printf("\n=== LOGITS ID SORTED | TOP K = %d ===\n", topK)
		for i := 0; i < min(6, len(logitsID)); i++ {
			fmt.Printf("{ %.3f | %d }", logitsID[i].first, logitsID[i].second)
		}
		fmt.Printf(" ... ")
		for i := len(logitsID) - 6; i < len(logitsID)-1; i++ {
			fmt.Printf("{ %.3f | %d } ", logitsID[i].first, logitsID[i].second)
		}
	}

	logitsID = logitsID[:topK]

	if ml.DEBUG {
		fmt.Printf("\n=== LOGITS ID RESIZED | %d ===\n", len(logitsID))
		for i := 0; i < min(6, len(logitsID)); i++ {
			fmt.Printf("{ %.3f | %d }", logitsID[i].first, logitsID[i].second)
		}
		fmt.Printf(" ... ")
		for i := len(logitsID) - 6; i < len(logitsID)-1; i++ {
			fmt.Printf("{ %.3f | %d } ", logitsID[i].first, logitsID[i].second)
		}
	}

	// FIXME Why loop? We've already SORTED logitsID and the MAX is just the FIRST element
	////double maxl = -INFINITY;
	maxl := float32(math.Inf(-1))
	for _, kv := range logitsID {
		//// maxl = std::max(maxl, kv.first);
		maxl = max(maxl, kv.first)
	}

	// compute probs for the top k tokens
	////probs.reserve(logits_id.size());
	probs := make([]float32, 0, len(logitsID)) // FIXME LEN vs CAP

	sum := float64(0.0)
	for _, kv := range logitsID {
		p := math.Exp(float64(kv.first - maxl))
		probs = append(probs, float32(p))
		sum += p
	}

	if ml.DEBUG {
		fmt.Printf("\n=== PROBS | %d ===\n", len(probs))
		for i := 0; i < min(6, len(probs)); i++ {
			fmt.Printf("%.3f  ", probs[i])
		}
		fmt.Printf(" ... ")
		for i := len(logitsID) - 6; i < len(probs)-1; i++ {
			fmt.Printf("%.3f  ", probs[i])
		}
	}

	// normalize the probs
	for i := range probs {
		probs[i] /= float32(sum)
	}

	if ml.DEBUG {
		fmt.Printf("\n=== PROBS NORM | %d ===\n", len(probs))
		for i := 0; i < min(6, len(probs)); i++ {
			fmt.Printf("%.3f  ", probs[i])
		}
		fmt.Printf(" ... ")
		for i := len(logitsID) - 6; i < len(probs)-1; i++ {
			fmt.Printf("%.3f  ", probs[i])
		}
	}

	if topP < 1.0 {

		cumsum := float32(0.0) // TODO float64 for better math?
		for i := uint32(0); i < uint32(len(probs)); i++ {
			cumsum += probs[i]
			if cumsum >= topP {
				probs = probs[:i+1]
				logitsID = logitsID[:i+1]
				break
			}
		}

		cumsum = 1.0 / cumsum
		for i := uint32(0); i < uint32(len(probs)); i++ {
			probs[i] *= cumsum
		}
	}

	if ml.DEBUG {
		if len(probs) > 6 {
			fmt.Printf("\n=== PROBS POST | %d ===\n", len(probs))
			for i := 0; i < min(6, len(probs)); i++ {
				fmt.Printf("%.3f  ", probs[i])
			}
			fmt.Printf(" ... ")
			for i := len(logitsID) - 6; i < len(probs)-1; i++ {
				fmt.Printf("%.3f  ", probs[i])
			}
		}
	}

	////std::discrete_distribution<> dist(probs.begin(), probs.end());
	////int idx = dist(rng);
	////return logits_id[idx].second;

	// --- discrete distribution
	//     TODO Do we need something better than hand-crafted math here?

	// seed := time.Now().UnixNano()
	seed := int64(0x1cc)
	source := rand.NewSource(seed)

	for i := 0; i < len(probs); i++ {
		f := float32(source.Int63()) / (1 << 63)
		probs[i] = probs[i] * probs[i] * f * f
	}

	idx := 0
	maxProb := probs[0]
	for i := 1; i < len(probs); i++ {
		if probs[i] > maxProb {
			idx = i
			maxProb = probs[i]
		}
	}

	if ml.DEBUG {
		fmt.Printf("\nidx = %d", idx)
		fmt.Printf("\nlogitsID = %d | weight = %f", logitsID[idx].second, logitsID[idx].first)
	}

	return logitsID[idx].second
}

// evaluate the transformer calculated by NodeID and return graphs
//
//   - lctx:      llama context
//   - tokens:    new batch of tokens to process
//   - n_past:    the context size so far
//   - n_threads: number of threads to use
func ExpandGraph(

	lctx *Context,
	tokens []uint32,
	tokensCount uint32,
	pastCount uint32,
	threadsCount int) (*ml.Graph, *ml.Context, error) {

	N := tokensCount
	model := lctx.Model
	kvSelf := model.kvSelf

	embdSize := model.hparams.embdSize
	layersCount := model.hparams.layersCount
	ctxSize := model.hparams.ctxSize
	headsCount := model.hparams.headsCount
	rotCount := model.hparams.embdSize / model.hparams.headsCount

	ctx0 := &ml.Context{} //ctx0 := ml.Init(ml.InitParams{})

	// for big prompts, if BLAS is enabled, it is better to use only one thread
	// otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
	graph := ml.Graph{ThreadsCount: threadsCount}

	// Convert the tokens to a []float32 slice
	tokensFloat32 := make([]float32, len(tokens))
	for i, token := range tokens {
		tokensFloat32[i] = float32(token)
	}

	// Initialize the embd tensor with the tokensFloat32 data
	embd := ml.NewTensor(ctx0, ml.TYPE_F32, 1, uint32(len(tokens)), 1, 1, 1, tokensFloat32)
	inpL := ml.GetRows(ctx0, model.tokEmbeddings, embd)

	for il := uint32(0); il < layersCount; il++ {

		//if il > 0 {
		//	break // DEBUG
		//}

		inpSA := inpL
		cur := &ml.Tensor{}

		// norm
		cur = ml.RMSNorm(ctx0, inpL)

		// cur = attention_norm*cur
		rep := ml.Repeat(ctx0, model.layers[il].attentionNorm, cur)

		cur = ml.Mul(ctx0, rep, cur)

		// self-attention
		{
			Qcur := ml.MulMat(ctx0, model.layers[il].wq, cur)
			Kcur := ml.MulMat(ctx0, model.layers[il].wk, cur)
			Vcur := ml.MulMat(ctx0, model.layers[il].wv, cur)

			// store key and value to memory
			if N >= 1 {

				////struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, N*n_embd, (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past));
				////struct ggml_tensor * v = ggml_view_1d(ctx0, kv_self.v, N*n_embd, (ggml_element_size(kv_self.v)*n_embd)*(il*n_ctx + n_past));

				////ggml_build_forward_expand(&graph, ggml_cpy(ctx0, Kcur, k));
				////ggml_build_forward_expand(&graph, ggml_cpy(ctx0, Vcur, v));

				// NB! ggml_element_size(kv_self.k) = 2 for FP16
				k := ml.View1D(ctx0, kvSelf.K, N*embdSize, embdSize*(il*ctxSize+pastCount))
				v := ml.View1D(ctx0, kvSelf.V, N*embdSize, embdSize*(il*ctxSize+pastCount))

				ml.BuildForwardExpand(&graph, ml.Copy(ctx0, Kcur, k))
				ml.BuildForwardExpand(&graph, ml.Copy(ctx0, Vcur, v))
			}

			// Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
			Q :=
				ml.Permute(ctx0,
					ml.Rope(ctx0,
						ml.Copy(ctx0,
							Qcur,
							ml.NewTensor3D(ctx0, ml.TYPE_F32, embdSize/headsCount, headsCount, N)),
						pastCount, rotCount, 0),
					0, 2, 1, 3)

			// K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
			K :=
				ml.Permute(ctx0,
					ml.Rope(ctx0,
						ml.Reshape3D(ctx0,
							////ggml_view_1d(ctx0, kv_self.k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(kv_self.k)*n_embd),
							////n_embd/n_head, n_head, n_past + N),
							ml.View1D(ctx0, kvSelf.K, (pastCount+N)*embdSize, il*ctxSize*embdSize),
							embdSize/headsCount, headsCount, pastCount+N),
						pastCount, rotCount, 1),
					0, 2, 1, 3)

			// K * Q
			////struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
			KQ := ml.MulMat(ctx0, K, Q)

			// KQ_scaled = KQ / sqrt(n_embd/n_head)
			KQScaled :=
				ml.Scale(ctx0,
					KQ,
					ml.NewFP32(ctx0, float32(1.0/math.Sqrt(float64(embdSize)/float64(headsCount)))),
				)

			// KQ_masked = mask_past(KQ_scaled)
			////struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);
			KQMasked := ml.DiagMaskInf(ctx0, KQScaled, pastCount)

			// KQ = soft_max(KQ_masked)
			////struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);
			KQSoftMax := ml.SoftMax(ctx0, KQMasked)

			// V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
			VTrans :=
				ml.Copy(ctx0,
					ml.Permute(ctx0,
						ml.Reshape3D(ctx0,
							ml.View1D(ctx0, kvSelf.V, (pastCount+N)*embdSize, il*ctxSize*embdSize),
							embdSize/headsCount, headsCount, pastCount+N),
						1, 2, 0, 3),
					ml.NewTensor3D(ctx0, ml.TYPE_F32 /* kv_self.v->type */, pastCount+N, embdSize/headsCount, headsCount))

			// KQV = transpose(V) * KQ_soft_max
			KQV := ml.MulMat(ctx0, VTrans, KQSoftMax)

			// KQV_merged = KQV.permute(0, 2, 1, 3)
			KQVMerged := ml.Permute(ctx0, KQV, 0, 2, 1, 3)

			// cur = KQV_merged.contiguous().view(n_embd, N)
			cur = ml.Copy(ctx0,
				KQVMerged,
				ml.NewTensor2D(ctx0, ml.TYPE_F32, embdSize, N))

			// projection (no bias)
			cur = ml.MulMat(ctx0,
				model.layers[il].wo,
				cur)
		}

		inpFF := ml.Add(ctx0, cur, inpSA)

		// feed-forward network
		{
			// norm
			{
				cur = ml.RMSNorm(ctx0, inpFF)

				// cur = ffn_norm*cur
				cur = ml.Mul(ctx0,
					ml.Repeat(ctx0, model.layers[il].ffn_norm, cur),
					cur)
			}

			tmp := ml.MulMat(ctx0,
				model.layers[il].w3,
				cur)

			cur = ml.MulMat(ctx0,
				model.layers[il].w1,
				cur)

			// SILU activation
			cur = ml.Silu(ctx0, cur)

			cur = ml.Mul(ctx0, cur, tmp)

			cur = ml.MulMat(ctx0,
				model.layers[il].w2,
				cur)
		}

		cur = ml.Add(ctx0, cur, inpFF)

		// input for next layer
		inpL = cur

	}

	// used at the end to optionally extract the embeddings
	////var embeddings *ml.Tensor

	// --- norm

	inpL = ml.RMSNorm(ctx0, inpL)

	// inpL = norm*inpL
	inpL = ml.Mul(ctx0,
		ml.Repeat(ctx0, model.norm, inpL),
		inpL)

	// lm_head
	inpL = ml.MulMat(ctx0, model.output, inpL)

	// add an mock op here
	zeroTensor := ml.NewTensor2D(ctx0, inpL.Type, inpL.NE[0], inpL.NE[1])
	inpL = ml.Add(ctx0, inpL, zeroTensor)

	// logits -> probs
	// COMMENTED inpL = ggml_soft_max(ctx0, inpL);

	// run the computation
	ml.BuildForwardExpand(&graph, inpL)

	// ml.GraphComputeByNodes(ctx0, &graph, nodeID)

	return &graph, ctx0, nil
}

// llama_model_load
// load the model's weights from a file
// see convert-pth-to-ggml.py for details on format

func LoadModel(
	fileName string,
	//partsCount int,
	silent bool,
	vocabOnly bool,
) (*Context, error) {

	lctx := NewContext()

	file, err := os.Open(fileName)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// --- check header magic and format version

	magic := readInt(file)

	if magic == LLAMA_FILE_MAGIC_UNVERSIONED || magic == LLAMA_FILE_MAGIC_OLD {
		fmt.Printf("\n[ERROR] Invalid model file '%s'! Too old, regenerate!", fileName)
		return nil, fmt.Errorf("invalid model file")
	}

	if magic != LLAMA_FILE_MAGIC {
		fmt.Printf("\n[ERROR] Invalid model file '%s'! Wrong MAGIC in header", fileName)
		return nil, fmt.Errorf("invalid model file")
	}

	version := readInt(file)

	if version != LLAMA_FILE_VERSION {
		fmt.Printf("\n[ERROR] Invalid model file '%s'! Unsupported version", fileName)
		return nil, fmt.Errorf("invalid model file")
	}

	// --- load hparams

	vocabSize := readInt(file)   // vocab_size
	embdSize := readInt(file)    // dim
	multSize := readInt(file)    // multiple_of
	headsCount := readInt(file)  // n_heads
	layersCount := readInt(file) // n_layers
	rotCount := readInt(file)    // rot = dim // n_heads [obsolete]
	fmt.Println("\n[info] rotCount: ", rotCount)
	f16 := readInt(file) // ftype

	model := lctx.Model

	model.hparams.vocabSize = vocabSize
	model.hparams.embdSize = embdSize
	model.hparams.multSize = multSize
	model.hparams.headsCount = headsCount
	model.hparams.kvHeadsCount = headsCount
	model.hparams.layersCount = layersCount
	model.hparams.rotCount = rotCount
	model.hparams.f16 = f16

	// --- init cache
	//KVCacheInit(&lctx.Model.hparams, &lctx.Model.kvSelf, ml.TYPE_F32)
	dt := ml.TYPE_F32
	size := embdSize * layersCount * 512 /*ctxSize*/ // FIXME ctxSize
	lctx.Model.kvSelf.K = ml.NewTensor1D(nil, dt, size)
	lctx.Model.kvSelf.V = ml.NewTensor1D(nil, dt, size)
	// fmt.Printf("\n[info] DT  = %v", dt)
	// fmt.Printf("\n[info] size  = %v", size)
	// fmt.Printf("\n[info] kvSelf.K  = %v", lctx.Model.kvSelf.K.NE)
	// fmt.Printf("\n[info] kvSelf.v   = %v", lctx.Model.kvSelf.V.NE)
	// fmt.Println("\n[info] SIZE: ", size)
	// fmt.Println("\n[info] embdSize * layersCount: ", embdSize*layersCount)

	// NB! Do not try to resize / relocate secondary pointers
	lctx.Vocab = ml.NewVocab(vocabSize)
	vocab := lctx.Vocab

	// FIXME Reserve extra space for tokensCount (N) = 8 (as with LogitsAll == true)
	//lctx.Logits = make([]float32, vocabSize*8, vocabSize*8) // NewFloatSlice(vocabSize, vocabSize) // FIXME ASAP
	lctx.Logits = make([]float32, vocabSize, vocabSize) // use just vocab size as CPP version does by default

	if ml.DEBUG {
		fmt.Printf("\nvocab  = %d", vocabSize)
		fmt.Printf("\nembd   = %d", embdSize)
		fmt.Printf("\nmult   = %d", multSize)
		fmt.Printf("\nheads  = %d", headsCount)
		fmt.Printf("\nlayers = %d", layersCount)
		fmt.Printf("\nrot    = %d", rotCount)
		fmt.Printf("\nf16    = %d", f16)
	}

	//fmt.Printf("\nctx   = %d", hparamsCtx)
	//fmt.Printf("\nn_ff    = %d", n_ff)

	n_ff := ((2*(4*embdSize)/3 + multSize - 1) / multSize) * multSize

	// --- load vocab

	if !silent && runtime.GOOS == "windows" {
		Colorize("[magenta][ INIT ][white] Loading vocab...")
	}

	vocabBar := progressbar.NewOptions(
		int(vocabSize),
		progressbar.OptionFullWidth(),
		//progressbar.OptionSetWidth(40),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionSetPredictTime(false),
		progressbar.OptionSetElapsedTime(false),
		progressbar.OptionSetDescription("[light_magenta][ INIT ][light_blue] Loading model vocab...  [light_cyan]"),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer:        "[light_magenta]▒[reset]",
			SaucerHead:    "[white]▒[reset]",
			SaucerPadding: "[dark_gray]▒[reset]",
			BarStart:      "[dark_gray]║[reset]",
			BarEnd:        "[dark_gray]║[reset]",
		}))

	for i := uint32(0); i < vocabSize; i++ {

		if !silent && runtime.GOOS != "windows" && i%100 == 0 {
			vocabBar.Set(int(i))
		}

		length := readInt(file)
		token := readString(file, length)
		score := readFP32(file)

		vocab.Token2ID[token] = i
		vocab.ID2Token[i] = ml.TokenScore{Token: token, Score: score}
	}

	if !silent && runtime.GOOS != "windows" {
		vocabBar.Finish()
		fmt.Printf("\n")
	}
	if vocabOnly {
		return lctx, nil
	}

	ctx := model.ctx

	// --- prepare memory for the weights
	{
		model.tokEmbeddings = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embdSize, vocabSize)

		model.norm = ml.NewTensor1D(ctx, ml.TYPE_F32, embdSize)
		model.output = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embdSize, vocabSize)

		// map by name
		model.tensors["tok_embeddings.weight"] = model.tokEmbeddings

		model.tensors["norm.weight"] = model.norm
		model.tensors["output.weight"] = model.output

		model.layers = make([]Layer, layersCount)
		for i := uint32(0); i < layersCount; i++ {
			//auto & layer = model.layers[i];

			model.layers[i].attentionNorm = ml.NewTensor1D(ctx, ml.TYPE_F32, embdSize)

			model.layers[i].wq = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embdSize, embdSize)
			model.layers[i].wk = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embdSize, embdSize)
			model.layers[i].wv = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embdSize, embdSize)
			model.layers[i].wo = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embdSize, embdSize)

			model.layers[i].ffn_norm = ml.NewTensor1D(ctx, ml.TYPE_F32, embdSize)

			model.layers[i].w1 = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embdSize, n_ff)
			model.layers[i].w2 = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, n_ff, embdSize)
			model.layers[i].w3 = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embdSize, n_ff)

			// map by name
			prefix := fmt.Sprintf("layers.%d.", i)

			model.tensors[prefix+"attention_norm.weight"] = model.layers[i].attentionNorm

			model.tensors[prefix+"attention.wq.weight"] = model.layers[i].wq
			model.tensors[prefix+"attention.wk.weight"] = model.layers[i].wk
			model.tensors[prefix+"attention.wv.weight"] = model.layers[i].wv
			model.tensors[prefix+"attention.wo.weight"] = model.layers[i].wo

			model.tensors[prefix+"ffn_norm.weight"] = model.layers[i].ffn_norm

			model.tensors[prefix+"feed_forward.w1.weight"] = model.layers[i].w1
			model.tensors[prefix+"feed_forward.w2.weight"] = model.layers[i].w2
			model.tensors[prefix+"feed_forward.w3.weight"] = model.layers[i].w3
		}
	}

	if !silent && runtime.GOOS == "windows" {
		Colorize("\n[magenta][ INIT ][white] Loading model - please wait ...")
	}

	// https://pkg.go.dev/github.com/schollz/progressbar/v3#Option
	bar := progressbar.NewOptions(int(layersCount*9),
		progressbar.OptionFullWidth(),
		//progressbar.OptionSetWidth(40),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionSetPredictTime(false),
		progressbar.OptionSetElapsedTime(false),
		progressbar.OptionSetDescription("[light_magenta][ INIT ][light_blue] Loading model weights...[light_cyan]"),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer:        "[light_magenta]▒[reset]",
			SaucerHead:    "[white]▒[reset]",
			SaucerPadding: "[dark_gray]▒[reset]",
			BarStart:      "[dark_gray]║[reset]",
			BarEnd:        "[dark_gray]║[reset]",
		}))

	// --- load weights

	var tensorsCount uint32
	for {

		dims := readInt(file)
		if dims < 1 || dims > 2 { // TODO Check for EOF
			break
		}

		nameLength := readInt(file)
		shardType := ml.DType(readInt(file))

		nelements := 1
		ne := [2]uint32{1, 1}
		for i := 0; i < int(dims); i++ {
			ne[i] = readInt(file)
			nelements *= int(ne[i])
		}

		name := readString(file, nameLength)
		if _, ok := model.tensors[name]; !ok {
			fmt.Printf("\n[ERROR] Unknown tensor '%s' in model file", name)
			os.Exit(1)
		}

		if ml.DEBUG {
			typeStr := "FP32"
			if shardType == ml.TYPE_F16 {
				typeStr = "FP16"
			}
			memStr := fmt.Sprintf("%dM", nelements*4/1024/1024)
			fmt.Printf("\n=== LAYER #%d === %s | %s | %s ===", tensorsCount, typeStr, name, memStr)
		}

		/* The latest GGJT format is always ONE-PART-NO-SPLIT-TENSORS binary file, so the parsing is really streamlined

		    partsCount := LLAMA_N_PARTS[embdSize]
			splitType := SPLIT_NONE
			if partsCount > 1 && dims > 1 {
				splitType = SPLIT_BY_COLUMNS
				if strings.Contains(name, "output") {
					splitType = SPLIT_NONE
				} else if strings.Contains(name, "layers") &&
					!strings.Contains(name, "attention.wo.weight") &&
					!strings.Contains(name, "feed_forward.w2.weight") {
					splitType = SPLIT_NONE
				}
			}
		*/

		tensor := model.tensors[name]
		tensorSize := tensor.Nelements()

		// --- all tensors in file are aligned for 32 bytes

		alignment := int64(32)
		offset, _ := file.Seek(0, io.SeekCurrent)
		for ; offset%alignment != 0; offset++ {
		}
		file.Seek(offset, io.SeekStart)

		// --- read tensor into memory

		if shardType == ml.TYPE_F16 {
			// FIXME Single-dimension tensors always presented as FP32
			// after conversion from PyTorch even for FP16 models
			for n := uint32(0); n < tensorSize; n++ {
				tensor.Data[n] = readFP16ToFP32(file)
			}
		} else if shardType == ml.TYPE_F32 {
			var fake []byte
			fakeHeader := (*reflect.SliceHeader)(unsafe.Pointer(&fake))
			// NB! unsafe.Pointer(tensor.Data) for *Data VS unsafe.Pointer(&tensor.Data) for Data
			dataHeader := (*reflect.SliceHeader)(unsafe.Pointer(&tensor.Data))

			fakeHeader.Data = dataHeader.Data
			fakeHeader.Len = int(tensorSize * 4)
			fakeHeader.Cap = int(tensorSize * 4)

			//fmt.Printf("\n== FAKE []BYTE LEN = %d", len(fake))
			if count, err := io.ReadFull(file, fake); err != nil || count != int(tensorSize*4) {
				fmt.Printf("\n[ERROR] Failed to read BIG FP32 chunk from model!")
				fmt.Printf("\n[ERROR] COUNT = %d | ERR = %s", count, err.Error())
				os.Exit(1)
			}
		} else {
			fmt.Printf("\n[ERROR] Tensor data type is not supported yet!")
			os.Exit(0)
		}

		tensorsCount++
		model.loadedCount++
		if !silent && runtime.GOOS != "windows" {
			bar.Add(1)
		}
	}

	if !silent && runtime.GOOS != "windows" {
		bar.Finish()
	}

	return lctx, nil
}

func loadTensor(
	modelTensor *ml.Tensor,
	ggufTensor *gguf.TensorInfo,
) error {
	ggufReader, err := ggufTensor.Reader()
	fileReader := ggufReader.(*os.File)
	if err != nil {

		return fmt.Errorf("could not find tensor for %s", ggufTensor.Name)
		// Do something
	}
	// if tensor.Dimensions != val.NE {
	// 	return nil, fmt.Errorf("Expected %v, FOund %v", tensor.Dimensions, val.NE)
	// }
	var elemSize int
	if ggufTensor.Type.String() == "float16" {
		elemSize = 2
	} else if ggufTensor.Type.String() == "float32" {
		elemSize = 4
	} else {
		return fmt.Errorf("type not supported: %s", ggufTensor.Type.String())
	}

	blockSize := int(ggufTensor.Size()) / elemSize / 100
	s := blockSize
	percent := 0
	nElements := modelTensor.Nelements()
	if ggufTensor.Type.String() == "float32" {
		// modelTensor.Data[i], err = safeReadFP16ToFP32(fileReader)
		// var fake []byte
		fake := make([]byte, nElements*4)
		fakeHeader := (*reflect.SliceHeader)(unsafe.Pointer(&fake))
		// NB! unsafe.Pointer(tensor.Data) for *Data VS unsafe.Pointer(&tensor.Data) for Data
		dataHeader := (*reflect.SliceHeader)(unsafe.Pointer(&modelTensor.Data))
		tensorSize := modelTensor.Nelements()

		fakeHeader.Data = dataHeader.Data
		fakeHeader.Len = int(tensorSize * 4)
		fakeHeader.Cap = int(tensorSize * 4)

		// fmt.Printf("\n== FAKE []BYTE LEN = %d", len(fake))
		if count, err := io.ReadFull(fileReader, fake); err != nil || count != int(tensorSize*4) {
			fmt.Printf("\n[ERROR] Failed to read BIG FP32 chunk from model!")
			fmt.Printf("\n[ERROR] COUNT = %d | ERR = %s", count, err.Error())
			fmt.Println("\n[ERROR] NAME: ", ggufTensor.Name)
			fmt.Println("\n[ERROR] EXPECTED DIMS: ", ggufTensor.Dimensions)
			fmt.Println("\n[ERROR] ACTUAL   DIMS: ", modelTensor.NE)
			os.Exit(1)
		}
	} else if ggufTensor.Type.String() == "float16" {
		for i := 0; i < len(modelTensor.Data); i++ {
			if i == int(s) {
				s += blockSize
				percent++
				fmt.Printf("loading %s: at %d percent: \n", ggufTensor.Name, percent)

			}
			// modelTensor.Data[i], err = safeReadFP32(fileReader)
			modelTensor.Data[i], err = safeReadFP16ToFP32(fileReader)
			if err != nil {
				fmt.Printf("Failed to read value for %s", ggufTensor.Name)
			}
		}
	}
	return nil
}

// LoadModelGGUF function
// llama_model_load
// load the model's weights from a file
// see convert-pth-to-ggml.py for details on format
func LoadModelGGUF(
	fileName string,
	// partsCount int,
	silent bool,
	vocabOnly bool,
) (*Context, error) {
	lctx := NewContext()
	g, _ := gguf.OpenFile(fileName)
	for k := range g.Metadata {
		fmt.Printf("%s\n", k)
	}

	fmt.Println("---------------------------------")
	fmt.Printf("Got GGUF file version: %d\n", g.Version)
	arch, _ := g.Metadata.String("general.architecture")
	if arch != "llama" {
		os.Exit(1)
	}
	embdSize, _ := g.Metadata.Int("llama.embedding_length")
	layersCount, _ := g.Metadata.Int("llama.block_count")
	headCount, _ := g.Metadata.Int("llama.attention.head_count")
	nEmbdHead := embdSize / headCount
	kvHeadCount, _ := g.Metadata.Int("llama.attention.head_count_kv")
	multSize, _ := g.Metadata.Int("llama.feed_forward_length")
	tokensAny, err := g.Metadata.Any("tokenizer.ggml.tokens")

	switch tt := tokensAny.(type) {
	case []string:
		fmt.Println("Tokens: ", tt[0])
	default:
		fmt.Println("[ERROR] Tokens are not a string array!")
		os.Exit(1)
	}

	if err != nil {
		return nil, err
	}
	tokens := tokensAny.([]string)
	vocabSize := uint32(len(tokens))
	scoresAny, err := g.Metadata.Any("tokenizer.ggml.scores")
	if err != nil {
		return nil, err
	}
	scores := scoresAny.([]float32)
	model := lctx.Model
	model.hparams.vocabSize = vocabSize
	model.hparams.kvHeadsCount = uint32(kvHeadCount)

	model.hparams.embdSize = uint32(embdSize)
	model.hparams.multSize = uint32(multSize)
	model.hparams.headsCount = uint32(headCount)
	model.hparams.kvHeadsCount = uint32(kvHeadCount)
	model.hparams.layersCount = uint32(layersCount)
	model.hparams.rotCount = uint32(embdSize / headCount)
	model.hparams.f16 = 1 // TODO: FIX THIS

	dt := ml.TYPE_F32
	size := uint32(embdSize * kvHeadCount * 512) /*ctxSize*/ // FIXME ctxSize
	model.kvSelf.K = ml.NewTensor1D(nil, dt, size)
	model.kvSelf.V = ml.NewTensor1D(nil, dt, size)

	lctx.Vocab = ml.NewVocab(uint32(vocabSize))
	vocab := lctx.Vocab

	// FIXME Reserve extra space for tokensCount (N) = 8 (as with LogitsAll == true)
	//lctx.Logits = make([]float32, vocabSize*8, vocabSize*8) // NewFloatSlice(vocabSize, vocabSize) // FIXME ASAP
	lctx.Logits = make([]float32, vocabSize, vocabSize) // use just vocab size as CPP version does by default

	vocabBar := progressbar.NewOptions(
		int(vocabSize),
		progressbar.OptionFullWidth(),
		// progressbar.OptionSetWidth(40),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionSetPredictTime(false),
		progressbar.OptionSetElapsedTime(false),
		progressbar.OptionSetDescription("[light_magenta][ INIT ][light_blue] Loading model vocab...  [light_cyan]"),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer:        "[light_magenta]▒[reset]",
			SaucerHead:    "[white]▒[reset]",
			SaucerPadding: "[dark_gray]▒[reset]",
			BarStart:      "[dark_gray]║[reset]",
			BarEnd:        "[dark_gray]║[reset]",
		}))

	// default special tokens
	// vocab.special_bos_id = 1;
	// vocab.special_eos_id = 2;
	// vocab.special_unk_id = 0;
	// vocab.special_sep_id = -1;
	// vocab.special_pad_id = -1;

	for i := uint32(0); i < uint32(vocabSize); i++ {
		if !silent && runtime.GOOS != "windows" && i%100 == 0 {
			err = vocabBar.Set(int(i))
			if err != nil {
				return nil, err
			}
		}
		// fmt.Println("Loading token: ", tokens[i])
		// fmt.Println("Score: ", scores[i])
		vocab.Token2ID[tokens[i]] = i
		vocab.ID2Token[i] = ml.TokenScore{Token: tokens[i], Score: scores[i]}
	}
	if vocabOnly {
		return lctx, nil
	}

	// model := lctx.Model
	nFf := ((2*(4*embdSize)/3 + multSize - 1) / multSize) * multSize
	ctx := model.ctx

	{
		model.tokEmbeddings = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, uint32(embdSize), vocabSize)
		model.norm = ml.NewTensor1D(ctx, ml.TYPE_F32, uint32(embdSize))
		model.output = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, uint32(embdSize), vocabSize)

		// map by name
		model.tensors["token_embd.weight"] = model.tokEmbeddings

		model.tensors["output_norm.weight"] = model.norm
		model.tensors["output.weight"] = model.output

		model.layers = make([]Layer, layersCount)
		// layersCountUint := uint32(2)
		for i := uint32(0); i < uint32(layersCount); i++ {
			// for i := uint32(0); i < layersCountUint; i++ {
			// auto & layer = model.layers[i];

			model.layers[i].attentionNorm = ml.NewTensor1D(ctx, ml.TYPE_F32, uint32(embdSize))

			model.layers[i].wq = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, uint32(embdSize), uint32(embdSize))
			model.layers[i].wk = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, uint32(embdSize), uint32(kvHeadCount*nEmbdHead))
			model.layers[i].wv = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, uint32(embdSize), uint32(kvHeadCount*nEmbdHead))
			model.layers[i].wo = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, uint32(embdSize), uint32(embdSize))

			model.layers[i].ffn_norm = ml.NewTensor1D(ctx, ml.TYPE_F32, uint32(embdSize))

			model.layers[i].w1 = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, uint32(embdSize), uint32(nFf))
			model.layers[i].w2 = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, uint32(nFf), uint32(embdSize))
			model.layers[i].w3 = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, uint32(embdSize), uint32(nFf))

			// map by name
			prefix := fmt.Sprintf("blk.%d.", i)

			model.tensors[prefix+"attn_norm.weight"] = model.layers[i].attentionNorm

			model.tensors[prefix+"attn_q.weight"] = model.layers[i].wq
			model.tensors[prefix+"attn_k.weight"] = model.layers[i].wk
			model.tensors[prefix+"attn_v.weight"] = model.layers[i].wv
			model.tensors[prefix+"attn_output.weight"] = model.layers[i].wo

			model.tensors[prefix+"ffn_norm.weight"] = model.layers[i].ffn_norm

			model.tensors[prefix+"ffn_gate.weight"] = model.layers[i].w1
			model.tensors[prefix+"ffn_down.weight"] = model.layers[i].w2
			model.tensors[prefix+"ffn_up.weight"] = model.layers[i].w3
		}
	}

	if !silent && runtime.GOOS == "windows" {
		Colorize("\n[magenta][ INIT ][white] Loading model - please wait ...")
	}

	// https://pkg.go.dev/github.com/schollz/progressbar/v3#Option
	bar := progressbar.NewOptions(int(layersCount*9),
		progressbar.OptionFullWidth(),
		// progressbar.OptionSetWidth(40),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionSetPredictTime(false),
		progressbar.OptionSetElapsedTime(false),
		progressbar.OptionSetDescription("[light_magenta][ INIT ][light_blue] Loading model weights...[light_cyan]"),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer:        "[light_magenta]▒[reset]",
			SaucerHead:    "[white]▒[reset]",
			SaucerPadding: "[dark_gray]▒[reset]",
			BarStart:      "[dark_gray]║[reset]",
			BarEnd:        "[dark_gray]║[reset]",
		}))

	// --- load weights
	// TODO: add parallelism
	loadedTensors := 0
	fmt.Println("loading g.Tensors - num tensors = ", len(g.Tensors))
	for t := range g.Tensors {
		ggufTensor := g.Tensors[t]
		// fmt.Printf("\n[INFO] Loading tensor named: %s dims %v", ggufTensor.Name, ggufTensor.Dimensions)
		// fmt.Println("Loading ggufTensor shape: ", ggufTensor.Dimensions)
		modelTensor, ok := model.tensors[ggufTensor.Name]
		if !ok {
			return nil, fmt.Errorf("could not find tensor for %s", ggufTensor.Name)
		}
		err = loadTensor(modelTensor, &ggufTensor)
		if err != nil {
			return nil, err
		}
		loadedTensors++
		if !silent && runtime.GOOS != "windows" {
			bar.Add(1)
		}
	}
	if !silent && runtime.GOOS != "windows" {
		bar.Finish()
	}
	fmt.Println("Loaded Tensors: ", loadedTensors)

	fmt.Println("---------------------------------")
	fmt.Println("END LOAD MODEL GGUF")
	return lctx, nil
}

func max(a, b float32) float32 {
	if a >= b {
		return a
	}
	return b
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

// ExtractTokens is a function to extract a slice of tokens from the ring buffer
func ExtractTokens(r *ring.Ring, count int) []uint32 {
	tokens := make([]uint32, count)
	for i := 0; i < count; i++ {
		tokens[i] = r.Value.(uint32)
		r = r.Next()
	}
	return tokens
}

func Colorize(format string, opts ...interface{}) (n int, err error) {
	var DefaultOutput = colorable.NewColorableStdout()
	return fmt.Fprintf(DefaultOutput, colorstring.Color(format), opts...)
}
