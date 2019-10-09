package com.mine.mytensorflow.mnist

import android.content.res.AssetManager
import android.graphics.Bitmap
import com.mine.mytensorflow.mnist.MnistModelConfig.CLASSIFICATION_THRESHOLD
import com.mine.mytensorflow.mnist.MnistModelConfig.MAX_CLASSIFICATION_RESULTS
import org.tensorflow.lite.Interpreter

import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.ArrayList
import java.util.PriorityQueue


class MnistClassifier private constructor(private val interpreter: Interpreter) {

    fun recognizeImage(bitmap: Bitmap): List<Classification> {
        val byteBuffer = convertBitmapToByteBuffer(bitmap)
        val result = Array(1) { FloatArray(MnistModelConfig.OUTPUT_LABELS.size) }
        interpreter.run(byteBuffer, result)
        return getSortedResult(result)
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(MnistModelConfig.MODEL_INPUT_SIZE)
        byteBuffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(MnistModelConfig.INPUT_IMG_SIZE_WIDTH * MnistModelConfig.INPUT_IMG_SIZE_HEIGHT)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (pixel in pixels) {
            val rChannel = (pixel shr 16 and 0xFF).toFloat()
            val gChannel = (pixel shr 8 and 0xFF).toFloat()
            val bChannel = (pixel and 0xFF).toFloat()
            val pixelValue = (rChannel + gChannel + bChannel) / 3f / 255f
            byteBuffer.putFloat(pixelValue)
        }
        return byteBuffer
    }

    private fun getSortedResult(resultsArray: Array<FloatArray>): List<Classification> {
        val sortedResults = PriorityQueue<Classification>(
            MAX_CLASSIFICATION_RESULTS
        ) { lhs, rhs -> rhs.confidence.compareTo(lhs.confidence) }

        for (i in MnistModelConfig.OUTPUT_LABELS.indices) {
            val confidence = resultsArray[0][i]
            if (confidence > CLASSIFICATION_THRESHOLD) {
                MnistModelConfig.OUTPUT_LABELS.size
                sortedResults.add(
                    Classification(
                        MnistModelConfig.OUTPUT_LABELS[i],
                        confidence
                    )
                )
            }
        }

        return ArrayList<Classification>(sortedResults)
    }

    companion object {

        @Throws(IOException::class)
        fun classifier(assetManager: AssetManager, modelPath: String): MnistClassifier {
            val byteBuffer =
                loadModelFile(assetManager, modelPath)
            val interpreter = Interpreter(byteBuffer)
            return MnistClassifier(interpreter)
        }

        @Throws(IOException::class)
        private fun loadModelFile(assetManager: AssetManager, modelPath: String): ByteBuffer {
            val fileDescriptor = assetManager.openFd(modelPath)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }
    }
}
