package test

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

// Float64FromBytes function
func Float64FromBytes(bytes []byte) float64 {
	bits := binary.LittleEndian.Uint64(bytes)
	float := math.Float64frombits(bits)
	return float
}

// Float64Bytes function
func Float64Bytes(float float64) []byte {
	bits := math.Float64bits(float)
	bytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(bytes, bits)
	return bytes
}

// Float32FromBytes function
func Float32FromBytes(bytes []byte) float32 {
	bits := binary.LittleEndian.Uint32(bytes)
	float := math.Float32frombits(bits)
	return float
}

// Float32Bytes function
func Float32Bytes(float float32) []byte {
	bits := math.Float32bits(float)
	bytes := make([]byte, 4)
	binary.LittleEndian.PutUint32(bytes, bits)
	return bytes
}

// StrListToF32List function
func StrListToF32List(strList []string) ([]float32, error) {
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
		trimmed = strings.Trim(trimmed, "\n")
		ft32, err := strconv.ParseFloat(trimmed, 32)
		if err != nil {
			fmt.Println("ERROR: ", trimmed)
			return nil, err
		}
		t[i] = float32(ft32)
	}
	return t, nil
}

// StrToF32List function
func StrToF32List(contents, sep string) ([]float32, error) {
	str := strings.Trim(contents, " ")
	strSplit := strings.Split(str, sep)

	data, err := StrListToF32List(strSplit)
	return data, err
}

// FileToStr function
func FileToStr(filePath string) (string, error) {
	file, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}

	return strings.Trim(string(file), " "), nil
}
