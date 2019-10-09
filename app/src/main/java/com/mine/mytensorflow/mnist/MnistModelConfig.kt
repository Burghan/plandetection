package com.mine.mytensorflow.mnist

import java.util.Arrays
import java.util.Collections

/**
 * The most of those information can be found in MNIST.ipynb
 */
object MnistModelConfig {
    var MODEL_FILENAME = "mnist_model.tflite"

    val INPUT_IMG_SIZE_WIDTH = 28
    val INPUT_IMG_SIZE_HEIGHT = 28
    val FLOAT_TYPE_SIZE = 4
    val PIXEL_SIZE = 1
    val MODEL_INPUT_SIZE = FLOAT_TYPE_SIZE * INPUT_IMG_SIZE_WIDTH * INPUT_IMG_SIZE_HEIGHT * PIXEL_SIZE

    val OUTPUT_LABELS = Collections.unmodifiableList(
        Arrays.asList("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    )

    val MAX_CLASSIFICATION_RESULTS = 3
    val CLASSIFICATION_THRESHOLD = 0.1f
}
