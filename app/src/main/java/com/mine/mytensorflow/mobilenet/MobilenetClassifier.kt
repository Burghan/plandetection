package com.mine.mytensorflow.mobilenet

import android.content.res.AssetManager
import android.graphics.Bitmap

import org.tensorflow.lite.Interpreter

import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.ArrayList
import java.util.PriorityQueue;
import com.mine.mytensorflow.mnist.Classification
import com.mine.mytensorflow.mobilenet.MobilenetModelConfig.CLASSIFICATION_THRESHOLD
import com.mine.mytensorflow.mobilenet.MobilenetModelConfig.IMAGE_MEAN
import com.mine.mytensorflow.mobilenet.MobilenetModelConfig.IMAGE_STD
import com.mine.mytensorflow.mobilenet.MobilenetModelConfig.INPUT_IMG_SIZE_HEIGHT
import com.mine.mytensorflow.mobilenet.MobilenetModelConfig.INPUT_IMG_SIZE_WIDTH
import com.mine.mytensorflow.mobilenet.MobilenetModelConfig.MAX_CLASSIFICATION_RESULTS
import com.mine.mytensorflow.mobilenet.MobilenetModelConfig.MODEL_INPUT_SIZE
import java.lang.Float


class MobilenetClassifier private constructor(private val interpreter: Interpreter) {

    fun recognizeImage(bitmap: Bitmap): List<Classification> {
        val byteBuffer = convertBitmapToByteBuffer(bitmap)
        val result = Array(1) { FloatArray(MobilenetModelConfig.OUTPUT_LABELS.size) }
        interpreter.run(byteBuffer, result)
        return getSortedResult(result)
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(MODEL_INPUT_SIZE)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(INPUT_IMG_SIZE_WIDTH * INPUT_IMG_SIZE_HEIGHT)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until INPUT_IMG_SIZE_WIDTH) {
            for (j in 0 until INPUT_IMG_SIZE_HEIGHT) {
                val `val` = intValues[pixel++]
                byteBuffer.putFloat(((`val` shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                byteBuffer.putFloat(((`val` shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                byteBuffer.putFloat(((`val` and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
            }
        }
        return byteBuffer
    }

    private fun getSortedResult(resultsArray: Array<FloatArray>): List<Classification> {
        val sortedResults = PriorityQueue<Classification>(
            MAX_CLASSIFICATION_RESULTS
        ) { lhs, rhs -> Float.compare(rhs.confidence, lhs.confidence) }

        for (i in MobilenetModelConfig.OUTPUT_LABELS.indices) {
            val confidence = resultsArray[0][i]
            if (confidence > CLASSIFICATION_THRESHOLD) {
                MobilenetModelConfig.OUTPUT_LABELS.size
                sortedResults.add(Classification(MobilenetModelConfig.OUTPUT_LABELS[i], confidence))
            }
        }

        return ArrayList<Classification>(sortedResults)
    }

    companion object {

        @Throws(IOException::class)
        fun classifier(assetManager: AssetManager, modelPath: String): MobilenetClassifier {
            val byteBuffer = loadModelFile(assetManager, modelPath)
            val interpreter = Interpreter(byteBuffer)
            return MobilenetClassifier(interpreter)
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